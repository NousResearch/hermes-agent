"""Messaging controls for gateway-hosted Group Chats."""

from __future__ import annotations

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway import (
    hosted_room_controls,
    hosted_room_driver,
    hosted_room_messaging,
    hosted_rooms,
)
from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
from gateway.hosted_room_messaging import (
    MessagingRoomBackend,
    RoomControlError,
    format_room_bot_detail,
    format_room_bot_list,
    format_room_detail,
    format_room_list,
    list_messaging_rooms,
    messaging_actor,
    messaging_event_id,
    parse_room_command,
    relay_provenance_is_unknown,
    resolve_room,
    resolve_room_picker_choice,
    room_bot_picker_choices,
    room_picker_choices,
    retry_room,
    send_to_room,
    stop_room,
)
from gateway.platforms.base import MessageEvent, MessageType, SendResult
from gateway.session import SessionSource
from hermes_cli.commands import resolve_command
from tui_gateway.hosted_room_service import HostedRoomService


class _FakeService:
    def __init__(self, db_path):
        self.db_path = db_path
        self.sent = []
        self.stopped = []
        self.retried = []
        self.room_status = {"running": True, "working": False, "blocked": False}

    def status(self, _room_id):
        return dict(self.room_status)

    def send(self, **kwargs):
        self.sent.append(kwargs)
        return {"seq": 1}

    def stop_room(self, room_id, *, cancel_id):
        self.stopped.append((room_id, cancel_id))
        return 2

    def retry_room_task(self, room_id, *, task_id, retry_id=None):
        self.retried.append((room_id, task_id, retry_id))
        return {"status": "queued"}


class _TestHostedRoomService(HostedRoomService):
    """Hosted service with a fixed two-profile test roster."""

    def __init__(self, db_path):
        super().__init__(ModuleType("test_hosted_room_server"), db_path=db_path)

    def local_profiles(self) -> tuple[str, ...]:
        return ("default", "ops")


class _ImmediateRPC:
    """In-process room worker transport that settles without a model call."""

    def __init__(self):
        self.sessions = {}

    def resolve_exact(self, *, profile, title, source):
        return self.sessions.get((profile, title))

    def create(self, *, profile, title, source):
        session = {"session_id": f"{profile}-session", "title": title}
        self.sessions[(profile, title)] = session
        return session

    def resume(self, *, profile, session_id, source):
        return {"session_id": session_id}

    def submit(
        self,
        *,
        profile,
        session_id,
        prompt,
        source,
        task,
        execution_generation,
        on_terminal,
    ):
        on_terminal({"status": "settled", "text": f"reply from {profile}"})
        return {"accepted": True}

    def history(self, *, profile, session_id, source):
        return []

    def info(self, *, profile, session_id, source):
        return {"active": False, "task_id": None}

    def interrupt(self, *, profile, session_id, source, expected_task_id):
        return {"interrupted": True}


class _HoldingRPC(_ImmediateRPC):
    """Keep a turn active until the owner observes a durable stop intent."""

    def __init__(self):
        super().__init__()
        self.active = {}
        self.interrupts = []

    def create(self, *, profile, title, source):
        session = super().create(profile=profile, title=title, source=source)
        self.active[session["session_id"]] = {"active": False, "task_id": None}
        return session

    def submit(
        self,
        *,
        profile,
        session_id,
        prompt,
        source,
        task,
        execution_generation,
        on_terminal,
    ):
        self.active[session_id] = {"active": True, "task_id": task.task_id}
        return {"accepted": True}

    def info(self, *, profile, session_id, source):
        return dict(self.active[session_id])

    def interrupt(self, *, profile, session_id, source, expected_task_id):
        state = self.active[session_id]
        if state["task_id"] != expected_task_id:
            return {"interrupted": False}
        state["active"] = False
        self.interrupts.append(expected_task_id)
        return {"interrupted": True}


def _wait_for(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition was not reached")


def _seed_rooms(tmp_path):
    db = tmp_path / "state.db"
    authority = "install:test-gateway"
    first = hosted_rooms.create_room(
        db,
        room_id="release-room",
        name="Release room",
        members=[
            {"member_id": "default", "handle": "hermes"},
            {"member_id": "ops", "handle": "ops", "display_name": "Operations"},
        ],
        authority_gateway_id=authority,
    )
    second = hosted_rooms.create_room(
        db,
        room_id="research-room",
        name="Research room",
        members=[
            {"member_id": "default", "handle": "hermes"},
            {"member_id": "research", "handle": "research"},
        ],
        authority_gateway_id=authority,
    )
    return db, first, second


def test_secondary_profile_only_lists_rooms_in_its_frozen_roster(tmp_path):
    db, _, _ = _seed_rooms(tmp_path)
    service = _FakeService(db)

    assert {room["room_id"] for room in list_messaging_rooms(service)} == {
        "release-room",
        "research-room",
    }
    assert [
        room["room_id"] for room in list_messaging_rooms(service, profile="ops")
    ] == ["release-room"]
    assert [
        room["room_id"] for room in list_messaging_rooms(service, profile="research")
    ] == ["research-room"]


def _event(
    text: str,
    *,
    platform: Platform = Platform.SIGNAL,
    user_id: str = "user-1",
    message_id: str = "message-1",
    media: bool = False,
    media_urls: list[str] | None = None,
    media_types: list[str] | None = None,
    is_bot: bool = False,
    chat_type: str = "dm",
    is_one_to_one: bool | None = True,
) -> MessageEvent:
    source = SessionSource(
        platform=platform,
        chat_id=f"chat-{platform.value}",
        chat_type=chat_type,
        user_id=user_id,
        user_name="Display Name",
        is_bot=is_bot,
    )
    source.is_one_to_one = is_one_to_one
    return MessageEvent(
        text=text,
        message_type=MessageType.COMMAND,
        user_id=user_id,
        user_name="Display Name",
        message_id=message_id,
        media_urls=media_urls
        if media_urls is not None
        else ["/tmp/image.png"]
        if media
        else [],
        media_types=media_types
        if media_types is not None
        else ["image/png"]
        if media
        else [],
        source=source,
    )


def _runner(*, platform: Platform = Platform.SIGNAL, extra=None, talk_users=None):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    effective_extra = {"allow_admin_from": ["user-1"]} if extra is None else extra
    if extra is None and talk_users is None:
        talk_users = ["user-1"]
    if talk_users is not None:
        effective_extra = {**effective_extra, "allow_from": list(talk_users),
                           "group_allow_from": list(talk_users)}
    runner.config = GatewayConfig(
        platforms={
            platform: PlatformConfig(
                enabled=True,
                token="***",
                extra=effective_extra,
            )
        }
    )
    runner.adapters = {
        platform: SimpleNamespace(
            config=runner.config.platforms[platform],
            typed_command_prefix="!"
            if platform in {Platform.MATRIX, Platform.SLACK}
            else "/"
        )
    }
    return runner


class _PickerAdapter:
    typed_command_prefix = "/"

    def __init__(self):
        self.calls = []

    async def send_choice_picker(self, **kwargs):
        self.calls.append(kwargs)
        return SendResult(success=True, message_id="picker-1")


def _seed_classic_projection(home, *, room_id="classic-room"):
    import yaml

    home.mkdir(parents=True, exist_ok=True)
    (home / "profile.yaml").write_text(
        yaml.safe_dump(
            {
                "ui_meta": {
                    "hermes-bots-groups": {
                        "version": 3,
                        "updatedAt": 2000,
                        "rooms": {
                            f"id:{room_id}": {
                                "name": "Desktop planning",
                                "roomId": room_id,
                                "hosted": None,
                                "desktopAuthorityHash": hashlib.sha256(
                                    b"authority:test"
                                ).hexdigest(),
                                "members": [
                                    {"name": "default", "handle": "hermes"},
                                    {"name": "reviewer", "handle": "reviewer"},
                                ],
                                "log": [
                                    {
                                        "id": "message-1",
                                        "from": {"kind": "user", "name": "You"},
                                        "text": "Review the plan",
                                        "at": 1000,
                                        "thread": "thread-1",
                                    },
                                    {
                                        "id": "message-2",
                                        "from": {"kind": "member", "name": "Reviewer"},
                                        "text": "The rollout needs a rollback step.",
                                        "at": 2000,
                                        "thread": "thread-1",
                                    },
                                ],
                            }
                        },
                    }
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_room_commands_are_gateway_dispatchable_without_interrupting_agent():
    group = resolve_command("group")
    assert group is not None and group.gateway_only and group.busy_policy == "dispatch"
    assert group.subcommands == ()
    assert resolve_command("groups") is None
    assert resolve_command("rooms") is None


def test_classic_room_projection_is_listed_with_recent_activity(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    _seed_classic_projection(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    service = _FakeService(tmp_path / "state.db")

    rooms = list_messaging_rooms(service)

    assert len(rooms) == 1
    assert rooms[0]["_room_mode"] == "desktop"
    assert rooms[0]["desktop_available"] is False
    detail = format_room_detail(service, rooms[0])
    assert "💬 **Desktop planning**" in detail
    assert "🟡 waiting for Desktop" in detail
    assert "• **Reviewer:** The rollout needs a rollback step." in detail
    listing = format_room_list(service)
    assert listing.startswith("👥 **Group Chats**\n")
    assert "🧭 **Controls**\nCheck: `/group <number>`" in listing
    assert "Send: `/group <number> send <message>`" in listing
    assert "Stop: `/group <number> stop`" in listing


@pytest.mark.asyncio
async def test_routed_profile_lists_its_own_classic_group_chats(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    _seed_classic_projection(home / "profiles" / "worker", room_id="worker-room")
    monkeypatch.setenv("HERMES_HOME", str(home))
    service = _FakeService(tmp_path / "state.db")
    monkeypatch.setattr(
        "gateway.hosted_room_messaging.current_room_backend", lambda: service
    )
    event = _event("/group")
    event.source.profile = "worker"
    runner = _runner()
    receiver = runner.adapters[event.source.platform]
    receiver.config = runner.config.platforms[event.source.platform]
    event.source._transport_adapter_ref = lambda: receiver

    listing = await runner._handle_rooms_command(event)

    assert "Desktop planning" in listing
    assert "No Group Chats yet" not in listing


@pytest.mark.asyncio
async def test_name_keyed_legacy_group_chat_cannot_be_mutated_by_stale_number(
    tmp_path, monkeypatch
):
    import yaml

    home = tmp_path / "hermes"
    _seed_classic_projection(home)
    profile = home / "profile.yaml"
    raw = yaml.safe_load(profile.read_text(encoding="utf-8"))
    snapshot = raw["ui_meta"]["hermes-bots-groups"]
    snapshot["version"] = 2
    legacy = snapshot["rooms"].pop("id:classic-room")
    legacy.pop("roomId")
    snapshot["rooms"] = {"Desktop planning": legacy}
    profile.write_text(yaml.safe_dump(raw), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    service = _FakeService(tmp_path / "state.db")
    monkeypatch.setattr(
        "gateway.hosted_room_messaging.current_room_backend", lambda: service
    )

    result = await _runner()._handle_rooms_command(
        _event("/group 1 send do not misroute", message_id="legacy-number")
    )

    assert result == (
        "Open this older Group Chat once in the latest Hermes Desktop before "
        "changing it from messaging."
    )


def test_participant_gateway_lists_reads_and_controls_remote_room(
    tmp_path, monkeypatch
):
    db = tmp_path / "state.db"
    now = time.time()
    hosted_room_controls.save_peer_control_link(
        db,
        room_id="remote-room",
        member_id="reviewer",
        room_name="Release planning",
        member_count=2,
        home_url="https://home.example.test",
        authority_gateway_id="install:home",
        authority_epoch=3,
        control_token="A" * 43,
        expires_at=now + 600,
        now=now,
    )
    calls = []

    class _RemoteClient:
        def __init__(self, link):
            assert link.control_token == "A" * 43

        def summary(self):
            return {
                "room": {
                    "room_id": "remote-room",
                    "name": "Release planning",
                    "members": [
                        {"member_id": "author", "display_name": "Author"},
                        {"member_id": "reviewer", "display_name": "Reviewer"},
                    ],
                    "authority_gateway_id": "install:home",
                    "authority_epoch": 3,
                },
                "status": {"working": True, "blocked": False, "counts": {}},
                "events": [
                    {
                        "kind": "message.member",
                        "actor": {"id": "reviewer"},
                        "payload": {"member_id": "reviewer", "text": "Ready."},
                    }
                ],
            }

        def mutate(self, **kwargs):
            calls.append(kwargs)
            if kwargs["action"] == "retry":
                return {"action": "retry", "processed": 1}
            return {"action": kwargs["action"]}

    monkeypatch.setattr(hosted_room_messaging, "RoomControlHTTPClient", _RemoteClient)
    monkeypatch.setattr(
        hosted_rooms,
        "local_authority_gateway_id",
        lambda: "install:participant",
    )
    service = _FakeService(db)

    rooms = list_messaging_rooms(service)
    assert [
        (room["name"], room["_room_mode"], room["member_count"]) for room in rooms
    ] == [("Release planning", "remote", 2)]
    assert "⚪ **1. Release planning** · connected · 2 Bots" in format_room_list(
        service
    )
    detail = format_room_detail(service, rooms[0])
    assert "💬 **Release planning**" in detail
    assert "🟡 work queued or running" in detail
    assert "• **Reviewer:** Ready." in detail
    assert "A" * 43 not in repr(rooms)
    assert (
        send_to_room(
            service,
            rooms[0],
            _event("/group 1 send hello", message_id="remote-send"),
            "hello",
        )
        == "Queued in Release planning."
    )
    assert (
        stop_room(
            service,
            rooms[0],
            _event("/group 1 stop", message_id="remote-stop"),
        )
        == "Stop requested for Release planning. Active work will stop safely."
    )
    assert (
        retry_room(
            service,
            rooms[0],
            _event("/group 1 retry", message_id="remote-retry"),
        )
        == "Retry checked for Release planning (1 task)."
    )
    assert [call["action"] for call in calls] == ["send", "stop", "retry"]
    assert calls[0]["actor_display_name"] == "Display Name via Signal"


def test_legacy_projection_uses_the_same_name_identity_as_new_desktop(
    tmp_path, monkeypatch
):
    import yaml

    home = tmp_path / "hermes"
    home.mkdir(parents=True)
    (home / "profile.yaml").write_text(
        yaml.safe_dump({
            "ui_meta": {
                "hermes-bots-groups": {
                    "version": 2,
                    "rooms": {
                        "Legacy planning": {
                            "name": "Legacy planning",
                            "members": [{"name": "default"}],
                            "log": [],
                        }
                    },
                }
            }
        }),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))

    room = list_messaging_rooms(_FakeService(tmp_path / "state.db"))[0]

    assert room["room_id"] == "name:Legacy planning"


def test_more_than_128_classic_rooms_list_without_disabling_controls(
    tmp_path, monkeypatch
):
    import yaml

    home = tmp_path / "hermes"
    home.mkdir(parents=True)
    rooms = {
        f"id:room-{index}": {
            "name": f"Group chat {index}",
            "roomId": f"room-{index}",
            "members": [{"name": "default"}],
            "log": [],
        }
        for index in range(260)
    }
    (home / "profile.yaml").write_text(
        yaml.safe_dump({
            "ui_meta": {"hermes-bots-groups": {"version": 3, "rooms": rooms}}
        }),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))

    listed = list_messaging_rooms(_FakeService(tmp_path / "state.db"))

    assert len(listed) == 260
    assert len({room["messaging_ref"] for room in listed}) == 260


def test_malformed_projected_room_does_not_hide_healthy_group_chats(
    tmp_path, monkeypatch
):
    import yaml

    home = tmp_path / "hermes"
    _seed_classic_projection(home)
    profile = home / "profile.yaml"
    raw = yaml.safe_load(profile.read_text(encoding="utf-8"))
    raw["ui_meta"]["hermes-bots-groups"]["rooms"]["id:broken"] = {
        "name": "Broken",
        "roomId": "x" * 201,
        "desktopAuthorityHash": hashlib.sha256(b"authority:broken").hexdigest(),
        "members": [{"name": "default"}],
        "log": [],
    }
    profile.write_text(yaml.safe_dump(raw), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))

    rooms = list_messaging_rooms(_FakeService(tmp_path / "state.db"))

    assert [room["room_id"] for room in rooms] == ["classic-room"]


def test_classic_room_send_and_stop_wait_for_desktop(tmp_path, monkeypatch):
    from gateway import desktop_room_mailbox

    home = tmp_path / "hermes"
    _seed_classic_projection(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(
        hosted_rooms,
        "local_authority_gateway_id",
        lambda: "install:test-gateway",
    )
    service = _FakeService(tmp_path / "state.db")
    room = list_messaging_rooms(service)[0]

    sent = send_to_room(
        service,
        room,
        _event("/group 1 send hello", message_id="send-1"),
        "hello",
    )

    assert (
        sent == "Saved for Desktop planning. Open or update Hermes Desktop to continue."
    )
    commands = desktop_room_mailbox.claim_commands(
        desktop_room_mailbox.default_db_path(),
        consumer_id="desktop:test",
        room_authorities=[
            {
                "room_id": "classic-room",
                "authority_token": "authority:test",
            }
        ],
    )
    assert [(item["action"], item["payload"]) for item in commands] == [
        (
            "send",
            {
                "actor_display_name": "Display Name via Signal",
                "message": "hello",
                "recipients": [
                    {"handle": "hermes", "name": "default"},
                    {"handle": "reviewer", "name": "reviewer"},
                ],
            },
        )
    ]
    stopped = stop_room(
        service,
        room,
        _event("/group 1 stop", message_id="stop-1"),
    )
    assert stopped == (
        "Stop saved for Desktop planning. Open or update Hermes Desktop to apply it."
    )
    stop_commands = desktop_room_mailbox.claim_commands(
        desktop_room_mailbox.default_db_path(),
        consumer_id="desktop:test",
        room_authorities=[
            {
                "room_id": "classic-room",
                "authority_token": "authority:test",
            }
        ],
        actions=["stop"],
    )
    assert [(item["action"], item["payload"]) for item in stop_commands] == [
        (
            "stop",
            {
                "target_command_id": commands[0]["command_id"],
                "target_thread_id": "thread-1",
            },
        )
    ]


def test_legacy_desktop_room_control_requests_one_current_desktop_open(
    tmp_path, monkeypatch
):
    import yaml

    home = tmp_path / "hermes"
    _seed_classic_projection(home)
    profile = home / "profile.yaml"
    raw = yaml.safe_load(profile.read_text(encoding="utf-8"))
    del raw["ui_meta"]["hermes-bots-groups"]["rooms"]["id:classic-room"][
        "desktopAuthorityHash"
    ]
    profile.write_text(yaml.safe_dump(raw), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    room = list_messaging_rooms(_FakeService(tmp_path / "state.db"))[0]

    with pytest.raises(RoomControlError, match="Open this Group Chat once"):
        send_to_room(
            _FakeService(tmp_path / "state.db"),
            room,
            _event("/group 1 send hello", message_id="legacy-send"),
            "hello",
        )


def test_classic_stop_targets_the_latest_user_message_not_only_its_thread(
    tmp_path, monkeypatch
):
    from gateway import desktop_room_mailbox

    home = tmp_path / "hermes"
    _seed_classic_projection(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    service = _FakeService(tmp_path / "state.db")
    room = list_messaging_rooms(service)[0]

    stop_room(
        service,
        room,
        _event("/group 1 stop", message_id="stop-message-fence"),
    )
    claimed = desktop_room_mailbox.claim_commands(
        desktop_room_mailbox.default_db_path(),
        consumer_id="desktop:test",
        room_authorities=[
            {"room_id": "classic-room", "authority_token": "authority:test"}
        ],
        actions=["stop"],
    )

    assert claimed[0]["payload"] == {
        "target_message_id": "message-1",
        "target_thread_id": "thread-1",
    }


def test_classic_room_detail_surfaces_failed_command_recovery(tmp_path, monkeypatch):
    from gateway import desktop_room_mailbox

    home = tmp_path / "hermes"
    _seed_classic_projection(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    db = desktop_room_mailbox.default_db_path()
    desktop_room_mailbox.enqueue_command(
        db,
        command_id="messaging:failed",
        room_id="classic-room",
        authority_hash=hashlib.sha256(b"authority:test").hexdigest(),
        action="send",
        payload={"message": "hello"},
    )
    claimed = desktop_room_mailbox.claim_commands(
        db,
        consumer_id="desktop:test",
        room_authorities=[
            {
                "room_id": "classic-room",
                "authority_token": "authority:test",
            }
        ],
    )[0]
    desktop_room_mailbox.complete_command(
        db,
        consumer_id="desktop:test",
        command_id="messaging:failed",
        lease_token=claimed["lease_token"],
        success=False,
        result={"message": "route missing"},
    )
    service = _FakeService(tmp_path / "state.db")
    room = list_messaging_rooms(service)[0]

    detail = format_room_detail(service, room)

    assert "💬 **Desktop planning**" in detail
    assert "⚠️ needs attention" in detail
    assert "The latest command could not be applied." in detail


def test_classic_retry_requeues_all_expired_commands_and_replays_receipt(
    tmp_path, monkeypatch
):
    from gateway import desktop_room_mailbox

    home = tmp_path / "hermes"
    _seed_classic_projection(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    db = desktop_room_mailbox.default_db_path()
    now = [100.0]
    for index in range(2):
        desktop_room_mailbox.enqueue_command(
            db,
            command_id=f"messaging:expired-{index}",
            room_id="classic-room",
            authority_hash=hashlib.sha256(b"authority:test").hexdigest(),
            action="send",
            payload={"message": str(index)},
            clock=lambda: now[0],
        )
        now[0] += 1
    now[0] += desktop_room_mailbox.PENDING_TTL_SECONDS + 1
    assert (
        desktop_room_mailbox.claim_commands(
            db,
            consumer_id="desktop:test",
            room_authorities=[
                {
                    "room_id": "classic-room",
                    "authority_token": "authority:test",
                }
            ],
            clock=lambda: now[0],
        )
        == []
    )

    service = _FakeService(db)
    room = list_messaging_rooms(service)[0]
    event = _event("/group 1 retry", message_id="retry-expired")
    result = retry_room(service, room, event)
    replay = retry_room(service, room, event)

    assert result == "Retry queued for Desktop planning (2 commands)."
    assert replay == result


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('1 send "hi there"', ("send", "1", '"hi there"')),
        ('1 send "a" and "b"', ("send", "1", '"a" and "b"')),
        ("1 send --literal-prefix", ("send", "1", "--literal-prefix")),
        ("1 send -- quoted style", ("send", "1", "-- quoted style")),
        ("1 stop", ("stop", "1", "")),
        ("1 retry", ("retry", "1", "")),
        ("1 approve", ("approve", "1", "")),
        ("1 approve A1B2C3D4", ("approve", "1", "A1B2C3D4")),
        ("1 deny", ("deny", "1", "")),
    ],
)
def test_parse_room_command_keeps_names_and_message_content(raw, expected):
    parsed = parse_room_command(raw)
    assert (parsed.action, parsed.room_query, parsed.message) == expected


def test_empty_classic_retry_keeps_actionable_user_error(tmp_path, monkeypatch):
    from gateway.desktop_room_mailbox import DesktopRoomMailboxError

    service = _FakeService(tmp_path / "state.db")
    room = {
        "room_id": "classic-room",
        "name": "Desktop planning",
        "_room_mode": "desktop",
    }

    def no_retryable_commands(*_args, **_kwargs):
        raise DesktopRoomMailboxError("no failed Group Chat command needs retry")

    monkeypatch.setattr(
        "gateway.desktop_room_mailbox.retryable_command_ids",
        no_retryable_commands,
    )

    with pytest.raises(RoomControlError, match="no failed work to retry"):
        retry_room(service, room, _event("/group 1 retry", message_id="empty-retry"))


@pytest.mark.parametrize("raw", ["", "send room", "send -- hello", "stop"])
def test_parse_room_command_returns_actionable_usage(raw):
    with pytest.raises(RoomControlError, match="Use `/group"):
        parse_room_command(raw)


@pytest.mark.asyncio
async def test_messaging_approval_command_uses_exact_pending_coordinates(
    tmp_path, monkeypatch
):
    db, _release, _research = _seed_rooms(tmp_path)
    service = MessagingRoomBackend(db_path=db)
    captured = {}
    monkeypatch.setattr(
        "gateway.hosted_room_messaging.current_room_backend",
        lambda: service,
    )

    def submit(_service, room, **kwargs):
        captured.update({"room": room, **kwargs})
        return (
            1,
            {
                "member_id": "ops",
                "task_id": "task-1",
                "execution_generation": 2,
                "request_id": "request-1",
            },
            {"queued": False, "choice": "once"},
        )

    monkeypatch.setattr(
        "gateway.hosted_room_messaging_approvals.submit_room_approval",
        submit,
    )

    result = await _runner()._handle_rooms_command(
        _event("/group 1 approve A1B2C3D4", message_id="approval-message-1")
    )

    assert result == "Approved once for Operations. Check: `/group 1`."
    assert captured["room"]["room_id"] == "release-room"
    assert captured["choice"] == "once"
    assert captured["selection"] == "A1B2C3D4"
    assert str(captured["command_id"]).startswith("approval:messaging:")


@pytest.mark.asyncio
async def test_approval_redelivery_returns_original_terminal_receipt_before_room_number(
    tmp_path,
    monkeypatch,
):
    from gateway import hosted_room_messaging_approvals as approvals

    db, release, _ = _seed_rooms(tmp_path)
    service = MessagingRoomBackend(db_path=db)
    pending = approvals.persist_pending_approval(
        db,
        room_id="release-room",
        member_id="ops",
        action={
            "kind": "approval",
            "authority_gateway_id": release["authority_gateway_id"],
            "authority_epoch": release["authority_epoch"],
            "task_id": "task-1",
            "execution_generation": 1,
            "request_id": "request-1",
            "approval": {"choices": ["once", "deny"]},
        },
    )
    event = _event(
        f"/group 1 approve {approvals.approval_reference(pending)}",
        message_id="terminal-redelivery",
    )
    command_id = f"approval:{messaging_event_id(event)}"
    approvals.begin_approval_command(
        db,
        command_id=command_id,
        pending=pending,
        choice="once",
    )
    approvals.complete_approval_command(
        db,
        command_id=command_id,
        result="Approval expired with the original Group Chat.",
    )
    hosted_rooms.disband_room(
        db,
        room_id="release-room",
        expected_gateway_id=str(release["authority_gateway_id"]),
        expected_epoch=int(release["authority_epoch"]),
    )
    monkeypatch.setattr(
        "gateway.hosted_room_messaging.current_room_backend",
        lambda: service,
    )

    result = await _runner()._handle_rooms_command(event)

    assert result == "Approval expired with the original Group Chat."


def test_room_resolution_is_exact_then_unique_prefix_then_substring(tmp_path):
    db, _, _ = _seed_rooms(tmp_path)
    rooms = hosted_rooms.list_rooms(db)
    assert resolve_room(rooms, "release-room")["name"] == "Release room"
    assert resolve_room(rooms, "Research")["room_id"] == "research-room"
    assert resolve_room(rooms, "lease")["room_id"] == "release-room"


def test_room_numbers_stay_stable_and_are_not_reused_after_disband(tmp_path):
    db, first, second = _seed_rooms(tmp_path)
    service = _FakeService(db)
    initial = {
        room["room_id"]: room["messaging_ref"] for room in list_messaging_rooms(service)
    }

    hosted_rooms.disband_room(
        db,
        room_id=first["room_id"],
        expected_gateway_id="install:test-gateway",
        expected_epoch=1,
    )
    third = hosted_rooms.create_room(
        db,
        room_id="stop-signals",
        name="Stop signals",
        members=[],
        authority_gateway_id="install:test-gateway",
    )
    current = list_messaging_rooms(service)
    refs = {room["room_id"]: room["messaging_ref"] for room in current}

    assert refs[second["room_id"]] == initial[second["room_id"]]
    assert refs[third["room_id"]] > max(initial.values())
    assert resolve_room(current, str(refs[third["room_id"]]))["name"] == "Stop signals"


def test_missing_room_number_fails_closed_without_matching_a_numeric_name(tmp_path):
    db, _, _ = _seed_rooms(tmp_path)
    hosted_rooms.create_room(
        db,
        room_id="roadmap-2026",
        name="2026 roadmap",
        members=[],
        authority_gateway_id="install:test-gateway",
    )
    rooms = list_messaging_rooms(_FakeService(db))

    with pytest.raises(RoomControlError, match="numbered 2026"):
        resolve_room(rooms, "2026")


def test_numeric_internal_id_requires_an_explicit_advanced_escape(tmp_path):
    db, _, _ = _seed_rooms(tmp_path)
    numeric_id = hosted_rooms.create_room(
        db,
        room_id="1",
        name="Legacy numeric ID",
        members=[],
        authority_gateway_id="install:test-gateway",
    )
    rooms = list_messaging_rooms(_FakeService(db))

    assert resolve_room(rooms, "1")["room_id"] != numeric_id["room_id"]
    assert resolve_room(rooms, "id:1")["room_id"] == numeric_id["room_id"]


def test_room_numbers_allocate_once_across_concurrent_gateway_processes(tmp_path):
    db, _, _ = _seed_rooms(tmp_path)

    def load_refs(_index):
        service = _FakeService(db)
        return {
            room["room_id"]: room["messaging_ref"]
            for room in list_messaging_rooms(service)
        }

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(load_refs, range(24)))

    assert all(result == results[0] for result in results)
    assert len(set(results[0].values())) == 2


def test_room_resolution_explains_ambiguity_and_missing_names(tmp_path):
    db, _, _ = _seed_rooms(tmp_path)
    hosted_rooms.create_room(
        db,
        room_id="release-notes",
        name="Release notes",
        members=[],
        authority_gateway_id="install:test-gateway",
    )
    rooms = hosted_rooms.list_rooms(db)
    with pytest.raises(RoomControlError, match="matches several group chats"):
        resolve_room(rooms, "release")
    with pytest.raises(RoomControlError, match="No group chat"):
        resolve_room(rooms, "missing")


@pytest.mark.asyncio
async def test_reserved_word_room_name_opens_detail_without_triggering_stop(
    tmp_path, monkeypatch
):
    db, _, _ = _seed_rooms(tmp_path)
    hosted_rooms.create_room(
        db,
        room_id="stop-signals",
        name="Stop signals",
        members=[],
        authority_gateway_id="install:test-gateway",
    )
    service = _FakeService(db)
    monkeypatch.setattr(
        "gateway.hosted_room_messaging.current_room_backend", lambda: service
    )

    result = await _runner()._handle_rooms_command(_event("/group Stop signals"))

    assert result.startswith("💬 **Stop signals**\n")
    assert service.stopped == []


@pytest.mark.asyncio
async def test_numeric_action_command_wins_over_a_command_shaped_room_name(
    tmp_path, monkeypatch
):
    db, _, _ = _seed_rooms(tmp_path)
    hosted_rooms.create_room(
        db,
        room_id="command-shaped-room",
        name="1 stop",
        members=[
            {"member_id": "default", "handle": "hermes"},
            {"member_id": "ops", "handle": "ops"},
        ],
        authority_gateway_id="install:test-gateway",
    )
    service = _FakeService(db)
    monkeypatch.setattr(
        "gateway.hosted_room_messaging.current_room_backend", lambda: service
    )

    result = await _runner()._handle_rooms_command(
        _event("/group 1 stop", message_id="stop-command-shaped")
    )

    assert result.startswith("Stop requested for Release room")
    assert service.stopped[0][0] == "release-room"


@pytest.mark.asyncio
async def test_group_list_keyword_uses_the_same_helpful_listing(tmp_path, monkeypatch):
    db, _, _ = _seed_rooms(tmp_path)
    monkeypatch.setattr(
        "gateway.hosted_room_messaging.current_room_backend",
        lambda: _FakeService(db),
    )

    result = await _runner()._handle_rooms_command(_event("/group list"))

    assert result.startswith("👥 **Group Chats**\n")
    assert "🧭 **Controls**\nCheck: `/group <number>`" in result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "platform",
    [Platform.TELEGRAM, Platform.DISCORD, Platform.MATRIX],
)
async def test_bare_group_uses_native_picker_and_selection_refreshes_detail(
    tmp_path,
    monkeypatch,
    platform,
):
    db, _, _ = _seed_rooms(tmp_path)
    service = _FakeService(db)
    monkeypatch.setattr(
        "gateway.hosted_room_messaging.current_room_backend",
        lambda: service,
    )
    adapter = _PickerAdapter()
    runner = _runner(platform=platform)
    runner.adapters[platform] = adapter
    adapter.config = runner.config.platforms[platform]
    runner._thread_metadata_for_source = lambda source, anchor=None: {}
    runner._reply_anchor_for_event = lambda event: None

    result = await runner._handle_rooms_command(_event("/group", platform=platform))

    assert result is None
    assert len(adapter.calls) == 1
    call = adapter.calls[0]
    assert call["title"].startswith("👥 Group Chats\n")
    assert all(choice["value"].startswith("room-") for choice in call["choices"])
    assert len({choice["value"] for choice in call["choices"]}) == 2
    release_value = next(
        choice["value"] for choice in call["choices"] if "Release" in choice["label"]
    )

    detail = await call["on_choice_selected"]("chat-telegram", release_value)

    assert "💬 **Release room**" in detail
    assert "🤖 **Bots**" in detail
    assert "🧭 **Controls**" in detail

    hosted_rooms.disband_room(
        db,
        room_id="release-room",
        expected_gateway_id="install:test-gateway",
        expected_epoch=1,
    )
    missing = await call["on_choice_selected"]("chat-telegram", release_value)
    assert missing == "This Group Chat is no longer available. Run the command again."


@pytest.mark.asyncio
async def test_group_bots_drills_into_native_participant_picker(
    tmp_path,
    monkeypatch,
):
    db, _, _ = _seed_rooms(tmp_path)
    service = _FakeService(db)
    monkeypatch.setattr(
        "gateway.hosted_room_messaging.current_room_backend",
        lambda: service,
    )
    adapter = _PickerAdapter()
    runner = _runner(platform=Platform.TELEGRAM)
    runner.adapters[Platform.TELEGRAM] = adapter
    adapter.config = runner.config.platforms[Platform.TELEGRAM]
    runner._thread_metadata_for_source = lambda source, anchor=None: {}
    runner._reply_anchor_for_event = lambda event: None

    result = await runner._handle_rooms_command(
        _event("/group 1 bots", platform=Platform.TELEGRAM)
    )

    assert result is None
    call = adapter.calls[0]
    assert call["title"].startswith("🤖 Bots\n")
    assert all(choice["value"].startswith("p=") for choice in call["choices"])
    ops_value = next(
        choice["value"] for choice in call["choices"] if "Operations" in choice["label"]
    )
    detail = await call["on_choice_selected"]("chat-telegram", ops_value)
    assert detail.startswith("🤖 **Operations**")
    assert "`@ops`" in detail


@pytest.mark.asyncio
async def test_group_approvals_use_native_one_tap_choices(tmp_path, monkeypatch):
    from gateway import hosted_room_messaging_approvals as approvals

    db, release, _ = _seed_rooms(tmp_path)
    service = MessagingRoomBackend(db_path=db)
    approvals.persist_pending_approval(
        db,
        room_id="release-room",
        member_id="ops",
        action={
            "kind": "approval",
            "authority_gateway_id": str(release["authority_gateway_id"]),
            "authority_epoch": int(release["authority_epoch"]),
            "task_id": "task-approval-1",
            "execution_generation": 2,
            "request_id": "request-approval-1",
            "approval": {
                "description": "Run focused tests",
                "command": "pytest -q tests/focused",
                "choices": ["once", "deny"],
            },
        },
    )
    monkeypatch.setattr(
        "gateway.hosted_room_messaging.current_room_backend",
        lambda: service,
    )
    adapter = _PickerAdapter()
    runner = _runner(platform=Platform.TELEGRAM)
    runner.adapters[Platform.TELEGRAM] = adapter
    adapter.config = runner.config.platforms[Platform.TELEGRAM]
    runner._thread_metadata_for_source = lambda source, anchor=None: {}
    runner._reply_anchor_for_event = lambda event: None

    result = await runner._handle_rooms_command(
        _event("/group 1 approvals", platform=Platform.TELEGRAM)
    )

    assert result is None
    call = adapter.calls[0]
    assert call["title"].startswith("⚠️ **Approval needed**\n")
    assert "**Operations**: Run focused tests" in call["title"]
    assert call["choices"][0]["label"] == "✓ 1. Approve once · Operations · ＠ops"
    assert call["choices"][1]["label"] == "✕ 1. Deny · Operations · ＠ops"
    denied = await call["on_choice_selected"](
        "chat-telegram",
        call["choices"][1]["value"],
    )
    assert denied == "Decision sent for Operations."
    commands = approvals.list_pending_approval_commands(
        db,
        room_id="release-room",
    )
    assert [(command["choice"], command["request_id"]) for command in commands] == [
        ("deny", "request-approval-1")
    ]


@pytest.mark.asyncio
async def test_group_bot_controls_fall_back_to_rich_text(tmp_path, monkeypatch):
    db, _, _ = _seed_rooms(tmp_path)
    monkeypatch.setattr(
        "gateway.hosted_room_messaging.current_room_backend",
        lambda: _FakeService(db),
    )

    listing = await _runner()._handle_rooms_command(_event("/group 1 bots"))
    detail = await _runner()._handle_rooms_command(_event("/group 1 bot 2"))

    assert "🤖 **Bots in Release room**" in listing
    assert "🧭 **Controls**" in listing
    assert detail.startswith("🤖 **Operations**")
    assert "Message this Bot: `/group 1 send @ops <message>`" in detail


@pytest.mark.asyncio
async def test_native_group_picker_hides_unexpected_callback_details(
    tmp_path,
    monkeypatch,
):
    db, _, _ = _seed_rooms(tmp_path)
    service = _FakeService(db)
    monkeypatch.setattr(
        "gateway.hosted_room_messaging.current_room_backend",
        lambda: service,
    )

    def fail_detail(*_args, **_kwargs):
        raise RuntimeError("private path: /opt/data/state.db")

    monkeypatch.setattr(hosted_room_messaging, "format_room_detail", fail_detail)
    adapter = _PickerAdapter()
    runner = _runner(platform=Platform.TELEGRAM)
    runner.adapters[Platform.TELEGRAM] = adapter
    adapter.config = runner.config.platforms[Platform.TELEGRAM]
    runner._thread_metadata_for_source = lambda source, anchor=None: {}
    runner._reply_anchor_for_event = lambda event: None

    await runner._handle_rooms_command(_event("/group", platform=Platform.TELEGRAM))
    result = await adapter.calls[0]["on_choice_selected"](
        "chat-telegram",
        adapter.calls[0]["choices"][0]["value"],
    )

    assert result == "Couldn’t load that Group Chat. Run `/group` again."
    assert "/opt/data" not in result


@pytest.mark.asyncio
async def test_group_list_pages_keep_every_stable_number_reachable(
    tmp_path, monkeypatch
):
    db, _, _ = _seed_rooms(tmp_path)
    for index in range(3, 11):
        hosted_rooms.create_room(
            db,
            room_id=f"room-{index}",
            name=f"Room {index}",
            members=[
                {"member_id": "default", "handle": "hermes"},
                {"member_id": f"bot-{index}", "handle": f"bot-{index}"},
            ],
            authority_gateway_id="install:test-gateway",
        )
    monkeypatch.setattr(
        "gateway.hosted_room_messaging.current_room_backend",
        lambda: _FakeService(db),
    )

    first = await _runner()._handle_rooms_command(_event("/group list"))
    second = await _runner()._handle_rooms_command(_event("/group list 2"))

    assert "page 1 of 2" in first
    assert "More: `/group list 2`" in first
    assert "page 2 of 2" in second
    assert "9. Room 9" in second
    assert "10. Room 10" in second


@pytest.mark.asyncio
async def test_mutating_room_commands_require_the_stable_number(tmp_path, monkeypatch):
    db, _, _ = _seed_rooms(tmp_path)
    service = _FakeService(db)
    monkeypatch.setattr(
        "gateway.hosted_room_messaging.current_room_backend", lambda: service
    )

    result = await _runner()._handle_room_command(_event("/group stop Release room"))

    assert result == (
        "Use `/group <number> send <message>`, `/group <number> retry`, or "
        "`/group <number> stop`."
    )
    assert service.stopped == []


@pytest.mark.asyncio
async def test_retry_requeues_only_retryable_hosted_tasks(tmp_path, monkeypatch):
    db, _, _ = _seed_rooms(tmp_path)
    service = _FakeService(db)
    service.room_status = {
        "running": True,
        "working": False,
        "blocked": True,
        "pending_actions": [
            {"kind": "retry", "task_id": "task-1"},
            {"kind": "approval", "task_id": "task-2"},
        ],
    }
    monkeypatch.setattr(
        "gateway.hosted_room_messaging.current_room_backend", lambda: service
    )

    event = _event("/group 1 retry", message_id="retry-1")
    result = await _runner()._handle_room_command(event)
    replay = await _runner()._handle_room_command(event)

    assert result.startswith("Retry queued for Release room (1 task).")
    assert replay == result
    assert service.retried == [
        (
            "release-room",
            "task-1",
            hosted_room_controls.control_retry_attempt_id(
                f"retry:{messaging_event_id(event)}",
                "task-1",
            ),
        )
    ]


@pytest.mark.asyncio
async def test_retry_is_durably_handed_to_the_active_worker_process(
    tmp_path, monkeypatch
):
    db, _, _ = _seed_rooms(tmp_path)
    service = _FakeService(db)
    service.room_status = {
        "running": True,
        "working": False,
        "blocked": True,
        "pending_actions": [{"kind": "retry", "task_id": "task-1"}],
    }

    def lease_held(_room_id, *, task_id, retry_id=None):
        assert task_id == "task-1"
        assert retry_id == hosted_room_controls.control_retry_attempt_id(
            f"retry:{messaging_event_id(event)}",
            task_id,
        )
        raise hosted_room_driver.LeaseHeldError(
            "room driver lease is held by another generation"
        )

    service.retry_room_task = lease_held
    monkeypatch.setattr(
        "gateway.hosted_room_messaging.current_room_backend", lambda: service
    )
    event = _event("/group 1 retry", message_id="retry-cross-process")

    result = await _runner()._handle_room_command(event)
    replay = await _runner()._handle_room_command(event)

    assert result.startswith("Retry queued for Release room (1 task).")
    assert replay == result
    pending = hosted_room_controls.load_pending_control_retries(
        db,
        room_id="release-room",
    )
    assert len(pending) == 1
    assert pending[0].task_ids == ("task-1",)
    assert pending[0].command_id.startswith("worker:retry:")


def test_room_list_and_detail_are_bounded_and_user_facing(tmp_path):
    db, release, _ = _seed_rooms(tmp_path)
    service = _FakeService(db)
    hosted_rooms.append_event(
        db,
        room_id=release["room_id"],
        event_id="user-1",
        kind="message.user",
        actor={"kind": "user", "id": "messaging:signal:abc", "display_name": "Signal"},
        authority_gateway_id="install:test-gateway",
        authority_epoch=1,
        payload={"text": "Please inspect the release", "thread_id": "thread-1"},
    )
    hosted_rooms.append_event(
        db,
        room_id=release["room_id"],
        event_id="member-1",
        kind="message.member",
        actor={"kind": "member", "id": "ops"},
        payload={
            "member_id": "ops",
            "text": "The release is ready",
            "thread_id": "thread-1",
            "task_id": "task-1",
            "turn_id": "turn-1",
            "round_index": 0,
        },
        authority_gateway_id="install:test-gateway",
        authority_epoch=1,
    )

    listing = format_room_list(service)
    assert "👥 **Group Chats**" in listing
    assert "🟡 **1. Release room** · waiting for its Bots · 2 Bots" in listing
    assert "🧭 **Controls**\nCheck: `/group <number>`" in listing
    assert "Send: `/group <number> send <message>`" in listing
    assert "Retry: `/group <number> retry`" in listing
    assert "Stop: `/group <number> stop`" in listing
    detail = format_room_detail(service, release)
    assert "**Signal:** Please inspect the release" in detail
    assert "**Operations:** The release is ready" in detail
    assert "🤖 **Bots**" in detail
    assert "• Operations (`@ops`)" in detail
    assert "Send: `/group 1 send <message>`" in detail
    assert "\n\n────────\n🧭 **Controls**\nSend:" in detail
    assert "Retry:" not in detail
    assert "Stop:" not in detail
    assert "Message one Bot: `/group 1 send @handle <message>`" in detail


def test_room_picker_choices_are_bounded_stable_and_user_facing(tmp_path):
    db, _, _ = _seed_rooms(tmp_path)
    service = _FakeService(db)
    rooms = list_messaging_rooms(service)

    choices = room_picker_choices(service, rooms)

    assert all(choice["value"].startswith("room-") for choice in choices)
    assert len({choice["value"] for choice in choices}) == 2
    assert {choice["label"] for choice in choices} == {
        "🟢 1. Release room (2)",
        "🟢 2. Research room (2)",
    }
    assert all(choice["full_width"] is True for choice in choices)
    assert all("release-room" not in choice["label"] for choice in choices)
    replacement = {**rooms[0], "room_id": "replacement-room", "messaging_ref": 1}
    with pytest.raises(RoomControlError, match="no longer available"):
        resolve_room_picker_choice([replacement], choices[0]["value"])


def test_group_bot_picker_and_details_expose_only_useful_controls(tmp_path):
    db, release, _ = _seed_rooms(tmp_path)
    service = _FakeService(db)

    choices = room_bot_picker_choices(service, release)
    listing = format_room_bot_list(service, release)
    detail = format_room_bot_detail(service, release, "@ops")

    assert all(choice["value"].startswith("p=") for choice in choices)
    assert len({choice["value"] for choice in choices}) == 2
    assert [choice["label"] for choice in choices] == [
        "🤖 hermes · hermes",
        "🤖 Operations · ops",
    ]
    assert "🤖 **Bots in Release room**" in listing
    assert "2. **Operations** · `@ops`" in listing
    assert "Bot details: `/group 1 bot <number>`" in listing
    assert detail.startswith("🤖 **Operations**")
    assert "Message this Bot: `/group 1 send @ops <message>`" in detail
    assert "Stop" not in detail
    with pytest.raises(RoomControlError, match="No Bot"):
        format_room_bot_detail(service, release, "9")


def test_room_picker_keeps_recent_rooms_reachable_when_roster_is_large(tmp_path):
    db, _, _ = _seed_rooms(tmp_path)
    latest = None
    for index in range(3, 12):
        latest = hosted_rooms.create_room(
            db,
            room_id=f"room-{index}",
            name=f"Room {index}",
            members=[
                {"member_id": "default", "handle": "hermes"},
                {"member_id": f"bot-{index}", "handle": f"bot-{index}"},
            ],
            authority_gateway_id="install:test-gateway",
        )
    assert latest is not None
    hosted_rooms.append_event(
        db,
        room_id=latest["room_id"],
        event_id="latest-user-message",
        kind="message.user",
        actor={"kind": "user", "id": "messaging:test", "display_name": "Owner"},
        authority_gateway_id="install:test-gateway",
        authority_epoch=1,
        payload={"text": "Newest work", "thread_id": "thread-latest"},
    )
    service = _FakeService(db)

    choices = room_picker_choices(service, list_messaging_rooms(service))

    assert len(choices) == 8
    assert str(latest["name"]) in choices[0]["label"]


def test_group_detail_escapes_untrusted_markup(tmp_path):
    db = tmp_path / "state.db"
    room = hosted_rooms.create_room(
        db,
        room_id="markup-room",
        name="**Admin**",
        members=[
            {"member_id": "default", "handle": "hermes"},
            {"member_id": "ops", "handle": "ops", "display_name": "*System*"},
        ],
        authority_gateway_id="install:test-gateway",
    )
    hosted_rooms.append_event(
        db,
        room_id=room["room_id"],
        event_id="markup-message",
        kind="message.member",
        actor={"kind": "member", "id": "ops"},
        authority_gateway_id="install:test-gateway",
        authority_epoch=1,
        payload={
            "member_id": "ops",
            "text": "*not a command* `deploy_a` > {prod} @ops",
            "thread_id": "thread-markup",
        },
    )

    detail = format_room_detail(_FakeService(db), room)
    room_choices = room_picker_choices(
        _FakeService(db),
        list_messaging_rooms(_FakeService(db)),
    )
    bot_choices = room_bot_picker_choices(_FakeService(db), room)

    assert "💬 **Admin**" in detail
    assert "• System (`@ops`)" in detail
    safe_preview = "＊not a command＊ ｀deploy＿a｀ ＞ ｛prod｝ ＠ops"
    assert safe_preview in detail
    assert room_choices[0]["label"] == "🟢 1. Admin (2)"
    assert bot_choices[1]["label"] == "🤖 System · ops"
    unsafe_room = {**room, "name": "@room [Docs](https://invalid.example)"}
    unsafe_choice = room_picker_choices(_FakeService(db), [unsafe_room])[0]
    assert "@room" not in unsafe_choice["label"]
    assert "[" not in unsafe_choice["label"]

    from gateway.platforms.signal_format import markdown_to_signal
    from gateway.platforms.whatsapp_common import WhatsAppBehaviorMixin
    from plugins.platforms.slack.adapter import SlackAdapter
    from plugins.platforms.telegram.adapter import TelegramAdapter

    telegram = object.__new__(TelegramAdapter).format_message(detail)
    whatsapp_adapter = object.__new__(WhatsAppBehaviorMixin)
    whatsapp_adapter._sanitize_outbound_text = lambda text: text
    whatsapp = whatsapp_adapter.format_message(detail)
    signal, _styles = markdown_to_signal(detail)
    slack = object.__new__(SlackAdapter).format_message(detail)
    for rendered in (telegram, whatsapp, signal, slack):
        assert "**Admin**" not in rendered
        assert r"\*\*Admin" not in rendered
        assert safe_preview in rendered


def test_bot_controls_preserve_valid_colon_handles_without_collision(tmp_path):
    db = tmp_path / "state.db"
    long_handle = "a" * 100
    room = hosted_rooms.create_room(
        db,
        room_id="colon-room",
        name="Operations",
        members=[
            {"member_id": "prod", "handle": "ops:prod", "display_name": "Prod"},
            {"member_id": "plain", "handle": "opsprod", "display_name": "Plain"},
            {"member_id": "numeric", "handle": "1", "display_name": "Numeric"},
            {"member_id": "long", "handle": long_handle, "display_name": "Long"},
        ],
        authority_gateway_id="install:test-gateway",
    )
    service = _FakeService(db)

    choices = room_bot_picker_choices(service, room)
    detail = format_room_bot_detail(service, room, "ops:prod")
    numeric_handle = format_room_bot_detail(service, room, "@1")
    first_index = format_room_bot_detail(service, room, "1")
    long_detail = format_room_bot_detail(service, room, long_handle)

    assert all(choice["value"].startswith("p=") for choice in choices)
    assert len({choice["value"] for choice in choices}) == 4
    assert any("ops:prod" in choice["label"] for choice in choices)
    assert "Handle: `@ops:prod`" in detail
    assert "send @ops:prod <message>" in detail
    assert numeric_handle.startswith("🤖 **Numeric**")
    assert first_index.startswith("🤖 **Prod**")
    assert f"Handle: `@{long_handle}`" in long_detail

    collision_room = hosted_rooms.create_room(
        db,
        room_id="collision-room",
        name="Collision check",
        members=[
            {"member_id": "a:b", "handle": "c", "display_name": "Same"},
            {"member_id": "a", "handle": "b:c", "display_name": "Same"},
        ],
        authority_gateway_id="install:test-gateway",
    )
    collision_choices = room_bot_picker_choices(service, collision_room)
    assert len({choice["value"] for choice in collision_choices}) == 2


def test_duplicate_bot_picker_tokens_fail_closed(tmp_path, monkeypatch):
    db, release, _ = _seed_rooms(tmp_path)
    service = _FakeService(db)
    monkeypatch.setattr(
        hosted_room_messaging,
        "_room_member_picker_value",
        lambda *_args: "p=duplicate",
    )

    with pytest.raises(RoomControlError, match="No Bot"):
        format_room_bot_detail(service, release, "p=duplicate")


def test_group_detail_never_invents_a_missing_bot_handle(tmp_path):
    room = {
        "room_id": "classic-room",
        "name": "Planning",
        "members": [{"name": "CEO Assistant"}, {"name": "Review Bot"}],
        "messaging_ref": 1,
        "_room_mode": "desktop",
        "desktop_available": True,
        "log": [],
    }

    detail = format_room_detail(_FakeService(tmp_path / "state.db"), room)

    assert "• CEO Assistant" in detail
    assert "@CEOAssistant" not in detail
    assert "Message one Bot:" not in detail
    choices = room_bot_picker_choices(
        _FakeService(tmp_path / "state.db"),
        room,
    )
    token = choices[1]["value"]
    reordered = {**room, "members": list(reversed(room["members"]))}
    selected = format_room_bot_detail(
        _FakeService(tmp_path / "state.db"),
        reordered,
        token,
    )
    assert selected.startswith("🤖 **Review Bot**")


@pytest.mark.parametrize("remote", [False, True])
@pytest.mark.parametrize("working", [False, True])
def test_group_detail_deferred_attention_preserves_healthy_work(tmp_path, monkeypatch, remote, working):
    db, release, _ = _seed_rooms(tmp_path)
    service = _FakeService(db)
    service.room_status = {
        "working": working, "blocked": False,
        "pending_actions": [{"kind": "retry", "task_id": "deferred-task"}],
    }
    room = release
    if remote:
        room = {**release, "_room_mode": "remote"}
        monkeypatch.setattr(hosted_room_messaging, "_remote_summary", lambda *_: {
            "room": release, "status": service.room_status, "events": [],
        })
    backend = MessagingRoomBackend(db_path=db, service=service)
    assert backend.status(release["room_id"])["needs_attention"] is True
    detail = format_room_detail(backend, room)
    assert "Retry: `/group 1 retry`" in detail
    assert ("needs attention" in detail) is not working
    assert ("work queued or running" in detail) is working
    assert ("Stop: `/group 1 stop`" in detail) is working
    service.room_status = {"working": False, "blocked": False, "pending_actions": []}
    assert backend.status(release["room_id"])["needs_attention"] is False
    assert "needs attention" not in format_room_detail(backend, room)


def test_group_detail_only_offers_actions_that_match_current_state(tmp_path):
    db, release, _ = _seed_rooms(tmp_path)
    service = _FakeService(db)

    idle = format_room_detail(service, release)
    assert "Send: `/group 1 send <message>`" in idle
    assert "Retry:" not in idle
    assert "Stop:" not in idle

    service.room_status = {"running": True, "working": True, "blocked": False}
    working = format_room_detail(service, release)
    assert "Stop: `/group 1 stop`" in working
    assert "Retry:" not in working

    service.room_status = {"running": True, "working": False, "blocked": True}
    blocked = format_room_detail(service, release)
    assert "Retry:" not in blocked
    assert "Stop:" not in blocked

    service.room_status = {
        "running": True,
        "working": False,
        "blocked": True,
        "pending_actions": [{"kind": "retry", "task_id": "task-1"}],
    }
    retryable = format_room_detail(service, release)
    assert "Retry: `/group 1 retry`" in retryable

    service.room_status = {
        "running": True,
        "working": True,
        "blocked": True,
        "counts": {"stopping": 1},
    }
    stopping = format_room_detail(service, release)
    assert "🟡 stopping" in stopping
    assert "needs attention" not in stopping
    assert "Stop:" not in stopping

    service.room_status = {
        "running": False,
        "working": False,
        "blocked": False,
        "peer_routes": [{"member_id": "remote", "status": "needs_reauthorization"}],
    }
    assert (
        MessagingRoomBackend(db_path=db, service=service).status("release-room")[
            "blocked"
        ]
        is True
    )

    classic = {
        "room_id": "classic-room",
        "name": "Desktop work",
        "members": [{"name": "worker", "handle": "worker"}],
        "messaging_ref": 3,
        "_room_mode": "desktop",
        "desktop_available": False,
        "desktop_command": {"action": "send", "state": "pending"},
        "log": [],
    }
    pending = format_room_detail(service, classic)
    assert "Stop: `/group 3 stop`" in pending


def test_group_detail_surfaces_exact_pending_approval_commands(tmp_path):
    from gateway import hosted_room_messaging_approvals as approvals

    db, release, _ = _seed_rooms(tmp_path)
    approvals.persist_pending_approval(
        db,
        room_id="release-room",
        member_id="ops",
        action={
            "kind": "approval",
            "authority_gateway_id": str(release["authority_gateway_id"]),
            "authority_epoch": int(release["authority_epoch"]),
            "task_id": "task-approval-1",
            "execution_generation": 2,
            "request_id": "request-approval-1",
            "approval": {
                "description": "Run focused tests",
                "command": "pytest -q tests/focused",
                "choices": ["once", "deny"],
            },
        },
    )
    detail = format_room_detail(
        MessagingRoomBackend(db_path=db),
        release,
        show_approvals=True,
    )

    assert "⚠️ **Approval needed**" in detail
    assert "1. **Operations** · Run focused tests" in detail
    assert "Actions: `/group 1 approvals`" in detail
    assert "Approve once: `/group 1 approve <approval code>`" in detail
    assert "Deny: `/group 1 deny <approval code>`" in detail


def test_empty_group_list_points_to_the_only_available_next_step(tmp_path):
    listing = format_room_list(_FakeService(tmp_path / "state.db"))

    assert listing.startswith("👥 **No Group Chats yet**")
    assert "Create one in Hermes Desktop first." in listing
    assert "<number>" not in listing
