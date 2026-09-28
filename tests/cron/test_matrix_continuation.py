"""Canonical cron destinations must use the same owner and session as replies."""

from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from cron.scheduler_delivery import _deliver_result
from gateway.config import Platform, PlatformConfig, load_gateway_config
from gateway.run import GatewayRunner, _profile_runtime_scope
from gateway.session import SessionSource, SessionStore
from plugins.platforms.matrix.adapter import MatrixAdapter
from tools.send_message_tool import _send_to_platform


class _MissingEncryption(Exception):
    errcode = "M_NOT_FOUND"


@pytest.mark.parametrize("caller", ["tool", "cron"])
@pytest.mark.parametrize(
    "case",
    [
        "changed",
        "unknown",
        "isolated_absent",
        "isolated_existing",
        "shared_absent",
        "group_shared_absent",
        "flat_absent",
        "native_unavailable",
        "encrypted_unavailable",
        "media_unavailable",
    ],
)
def test_native_continuation_requires_fresh_membership_and_existing_participant(
    destinations, monkeypatch, tmp_path, caller, case
):
    import logging
    from mautrix.client.state_store.memory import MemoryStateStore
    from mautrix.client.store_updater import StoreUpdatingAPI
    from mautrix.types import MemberStateEventContent, Membership, RoomID, UserID
    from gateway.session_context import clear_session_vars, set_session_vars
    from tools.send_message_tool import send_message_tool

    runner, owners = destinations
    home, adapter, room = owners["default"]
    store = adapter._session_store
    store.config.thread_sessions_per_user = case != "shared_absent"
    store.config.group_sessions_per_user = case != "group_shared_absent"
    runner.session_store = store
    monkeypatch.setattr("gateway.run._gateway_runner_ref", lambda: runner)
    monkeypatch.setattr("model_tools._run_async", asyncio.run)
    if caller == "tool":
        monkeypatch.setattr(
            "tools.send_message_tool._send_to_platform", _send_to_platform
        )
    monkeypatch.delattr(adapter, "_get_room_members")
    thread = None if case == "flat_absent" else "$existing"
    if thread is None:
        adapter.config.extra["cron_continuable_surface"] = "in_channel"
        config_path = home / "config.yaml"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["platforms"]["matrix"]["extra"]["cron_continuable_surface"] = (
            "in_channel"
        )
        config_path.write_text(json.dumps(config), encoding="utf-8")
    transition = case in {"changed", "unknown"}
    members = {adapter._user_id, "@alice:remote.test"}
    if not transition:
        members.add("@bob:remote.test")
    state_store = MemoryStateStore()

    async def populate():
        await state_store.set_members(
            RoomID(room),
            {
                UserID(member): MemberStateEventContent(membership=Membership.JOIN)
                for member in members
            },
        )

    asyncio.run(populate())
    accepted = False
    fresh_reads = []

    async def request(method, path, **kwargs):
        assert str(path).endswith("joined_members"), path
        fresh_reads.append(accepted)
        if accepted and case == "unknown":
            raise ValueError("membership unavailable after accepted send")
        return {"joined": {member: {} for member in members}}

    sdk = StoreUpdatingAPI(
        mxid=adapter._user_id,
        state_store=state_store,
        api=SimpleNamespace(request=request, log=logging.getLogger("matrix-test")),
    )
    adapter._client.state_store = state_store
    adapter._client.get_joined_members = sdk.get_joined_members
    adapter._joined_rooms.add(room)
    source = SessionSource(
        platform=Platform.MATRIX,
        chat_id=room,
        thread_id=thread,
        chat_type="dm" if transition else "group",
        user_id="@alice:remote.test",
        profile="default",
    )
    unavailable = case.endswith("unavailable")
    alice = (
        store.get_or_create_session(source)
        if case in {"changed", "unknown", "isolated_existing"} or unavailable
        else None
    )
    bob = (
        store.get_or_create_session(
            SessionSource(
                platform=Platform.MATRIX,
                chat_id=room,
                thread_id=thread,
                chat_type="group",
                user_id="@bob:remote.test",
                profile="default",
            )
        )
        if case not in {"shared_absent", "group_shared_absent"}
        else None
    )
    before = dict(store._entries)
    asyncio.run(adapter._resolve_room_identity(room))

    async def send_event(*args):
        nonlocal accepted
        if case == "changed":
            members.add("@bob:remote.test")
        accepted = True
        return "$sent"

    adapter._client.send_message_event.side_effect = send_event
    brief = "Confirmed native brief"
    http_calls = []
    if unavailable:
        from plugins.platforms.matrix.standalone import _MatrixAPIError
        from tools import send_message_senders

        monkeypatch.setattr(
            send_message_senders, "_live_adapter", lambda *args, **kwargs: (None, None)
        )
        monkeypatch.setattr(MatrixAdapter, "connect", AsyncMock(return_value=False))
        adapter.send = AsyncMock(side_effect=ValueError("native unavailable"))
        monkeypatch.setattr(
            "tools.send_message_tool._send_to_platform", _send_to_platform
        )
        config_path = home / "config.yaml"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["platforms"]["matrix"].update(token="token")
        config["platforms"]["matrix"]["extra"]["homeserver"] = "http://matrix.test"
        config_path.write_text(json.dumps(config), encoding="utf-8")

        async def http_request(self, method, path, **kwargs):
            http_calls.append((method, path, kwargs.get("json")))
            if path.endswith("m.room.encryption"):
                if case == "encrypted_unavailable":
                    return {"algorithm": "m.megolm.v1.aes-sha2"}
                raise _MatrixAPIError(404, {"errcode": "M_NOT_FOUND"})
            if path.endswith("joined_members"):
                return {"joined": {member: {} for member in members}}
            assert method == "PUT" and "/send/" in path, path
            return {"event_id": await send_event()}

        monkeypatch.setattr(
            "plugins.platforms.matrix.standalone._HTTPDelivery.request", http_request
        )
        if case == "media_unavailable":
            attachment = tmp_path / "report.txt"
            attachment.write_bytes(b"Native-only encrypted media")
            brief += f"\nMEDIA:{attachment}"
    with _profile_runtime_scope(home, {}):
        if caller == "tool":
            tokens = set_session_vars(platform="matrix", user_id="@alice:remote.test")
            try:
                result = json.loads(
                    send_message_tool({
                        "target": f"matrix:{room}" + (f"/{thread}" if thread else ""),
                        "message": brief,
                    })
                )
            finally:
                clear_session_vars(tokens)
            if case in {"encrypted_unavailable", "media_unavailable"}:
                assert result.get("error"), result
                assert not accepted
                assert not any(method == "PUT" for method, _, _ in http_calls)
                assert store._entries == before
                return
            assert result.get("success"), result
            if transition:
                assert result["chat_type"] == "unknown", result
        else:
            error = _deliver_result(
                {
                    "id": "fresh-membership",
                    "deliver": "origin",
                    "attach_to_session": True,
                    "origin": source.to_dict(),
                },
                brief,
                runner.adapters,
                SimpleNamespace(is_running=lambda: True),
            )
            if case in {"encrypted_unavailable", "media_unavailable"}:
                assert error and "native unavailable" in error
                assert not accepted
                assert not any(method == "PUT" for method, _, _ in http_calls)
                assert store._entries == before
                return
            assert error is None, error
        should_create = caller == "cron" and case in {
            "shared_absent",
            "group_shared_absent",
        }
        if should_create:
            key = store._generate_session_key(source)
            assert set(store._entries) == {key}
            alice = store.lookup_by_session_key(key)
        else:
            assert store._entries == before
        if alice is not None:
            assert [
                (turn["role"], turn["content"])
                for turn in store.load_transcript(alice.session_id)
            ] == (
                []
                if transition
                else [
                    (
                        "assistant" if caller == "tool" else "user",
                        brief
                        if caller == "tool"
                        else f"[Cron delivery: fresh-membership]\n{brief}",
                    )
                ]
            )
        if bob is not None:
            assert store.load_transcript(bob.session_id) == []
    assert accepted
    if unavailable:
        assert [
            payload["m.relates_to"]
            for method, _, payload in http_calls
            if method == "PUT"
        ] == [
            {
                "rel_type": "m.thread",
                "event_id": thread,
                "is_falling_back": True,
                "m.in_reply_to": {"event_id": thread},
            }
        ]
    if transition:
        assert True in fresh_reads


def _schedule(coro, loop):
    future = Future()
    try:
        future.set_result(asyncio.run(coro))
    except Exception as exc:
        future.set_exception(exc)
    return future


@pytest.fixture
def destinations(tmp_path, monkeypatch):
    import hermes_state
    from agent.secret_scope import set_multiplex_active

    monkeypatch.setattr(
        hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH
    )
    home = tmp_path / ".hermes"
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("asyncio.run_coroutine_threadsafe", _schedule)
    monkeypatch.setattr(
        "plugins.platforms.matrix.adapter.time.time", lambda: 1_800_000_000
    )
    monkeypatch.setattr(
        "tools.send_message_tool._send_to_platform",
        AsyncMock(return_value={"error": "standalone unavailable"}),
    )
    runner = object.__new__(GatewayRunner)
    runner._primary_profile_name = "default"
    runner._profile_adapters = {}
    owners = {}
    for profile in ("default", "secondary"):
        target_home = home if profile == "default" else home / "profiles" / profile
        target_home.mkdir(parents=True)
        (target_home / "config.yaml").write_text(
            json.dumps({
                "cron": {"wrap_response": False},
                "platforms": {
                    "matrix": {"enabled": True, "extra": {"e2ee_mode": "off"}}
                },
            }),
            encoding="utf-8",
        )
        with _profile_runtime_scope(target_home, {}):
            config = load_gateway_config()
            config.multiplex_profiles = True
            adapter = MatrixAdapter(
                PlatformConfig(enabled=True, extra={"e2ee_mode": "off"})
            )
            room = f"!{profile}:remote.test"
            adapter._client = SimpleNamespace(
                resolve_room_alias=AsyncMock(
                    return_value=SimpleNamespace(room_id=room, servers=["route.test"])
                ),
                join_room=AsyncMock(return_value=room),
                get_state_event=AsyncMock(side_effect=_MissingEncryption()),
                send_message_event=AsyncMock(
                    side_effect=["$opened", "$sent", "$opened-again", "$sent-again"]
                ),
                crypto=None,
            )
            adapter._refresh_dm_cache = AsyncMock()
            adapter._user_id = "@bot:remote.test"
            adapter._get_room_members = AsyncMock(return_value={adapter._user_id, "@alice:remote.test", "@bob:remote.test"})
            adapter._get_room_member_profiles = AsyncMock(return_value=None)
            adapter._get_room_state_value = AsyncMock(return_value=None)
            adapter.gateway_runner = runner
            adapter.set_owner_profile(profile)
            adapter.set_session_store(SessionStore(target_home / "sessions", config))
            owners[profile] = (target_home, adapter, room)
    runner.adapters = {Platform.MATRIX: owners["default"][1]}
    runner._profile_adapters["secondary"] = {Platform.MATRIX: owners["secondary"][1]}
    runner.config = config
    set_multiplex_active(True)
    return runner, owners


@pytest.mark.parametrize("thread", [None, "$existing"])
@pytest.mark.parametrize("chat_type", ["group", "dm"])
def test_alias_delivery_seeds_the_owning_reply_session(destinations, thread, chat_type):
    runner, owners = destinations
    loop = SimpleNamespace(is_running=lambda: True)
    for index, profile in enumerate(("default", "secondary", "default")):
        home, adapter, room = owners[profile]
        adapter._get_room_members.return_value = {adapter._user_id, "@alice:remote.test"}
        if chat_type == "group":
            adapter._get_room_members.return_value.add("@bob:remote.test")
        brief = f"Scheduled brief {index} for {profile}"
        alias = "#scheduled:remote.test"
        target = f"matrix:{alias}" + (f"/{thread}" if thread else "")
        with _profile_runtime_scope(home, {}):
            error = _deliver_result(
                {"id": f"brief-{index}", "deliver": target, "attach_to_session": True},
                brief,
                runner.adapters,
                loop,
            )
            assert error is None, error
            args = adapter._client.send_message_event.await_args.args
            delivered_room = str(args[0])
            delivered_thread = args[2]["m.relates_to"]["event_id"]
            source = SessionSource(
                platform=Platform.MATRIX,
                chat_id=delivered_room,
                thread_id=delivered_thread,
                chat_type=chat_type,
                user_id="@alice:remote.test",
                profile=profile,
            )
            store = adapter._session_store
            reply_key = store._generate_session_key(source)
            assert delivered_room == room
            assert reply_key in store._entries, list(store._entries)
            entry = store.get_or_create_session(source)
            assert brief in json.dumps(store.load_transcript(entry.session_id))
            assert all(alias not in key for key in store._entries)
    assert [
        owner[1]._client.resolve_room_alias.await_count for owner in owners.values()
    ] == [2, 1]


@pytest.mark.parametrize(
    "failure", ["lookup", "join", "encryption", "send", "media", "timeout"]
)
def test_failed_alias_delivery_creates_no_continuation(
    destinations, failure, tmp_path, monkeypatch
):
    from gateway.platforms.base import SendResult

    runner, owners = destinations
    home, adapter, _ = owners["secondary"]
    client = adapter._client
    if failure == "lookup":
        client.resolve_room_alias.side_effect = ValueError("directory forbidden")
    if failure == "join":
        client.join_room.side_effect = ValueError("join forbidden")
    if failure == "encryption":
        client.get_state_event.side_effect = None
        client.get_state_event.return_value = {"algorithm": "m.megolm.v1.aes-sha2"}
    if failure == "send":
        client.send_message_event.side_effect = ValueError("send forbidden")
    brief = "Unseen cron brief"
    if failure == "media":
        image = tmp_path / "brief.png"
        image.write_bytes(b"image")
        brief = f"MEDIA:{image}"
        adapter.send_image_file = AsyncMock(
            return_value=SendResult(success=False, error="upload forbidden")
        )
    if failure == "timeout":
        monkeypatch.setattr(
            "cron.scheduler_delivery._live_send_text",
            lambda *a, **kw: (True, True, None),
        )
    alias = "#scheduled:remote.test"
    with _profile_runtime_scope(home, {}):
        error = _deliver_result(
            {
                "id": "failed-brief",
                "deliver": f"matrix:{alias}/$existing",
                "attach_to_session": True,
            },
            brief,
            runner.adapters,
            SimpleNamespace(is_running=lambda: True),
        )
        if failure != "timeout":
            assert error and alias in error
        assert adapter._session_store._entries == {}
    runner.adapters[Platform.MATRIX]._client.send_message_event.assert_not_awaited()


@pytest.mark.parametrize("destination", ["room", "alias"])
@pytest.mark.parametrize("has_origin", [True, False])
def test_per_user_thread_delivery_preserves_the_originating_participant(
    destinations, destination, has_origin
):
    runner, owners = destinations
    for index, profile in enumerate(("default", "secondary", "default")):
        home, adapter, room = owners[profile]
        store = adapter._session_store
        store.config.thread_sessions_per_user = True
        config_path = home / "config.yaml"
        saved_config = json.loads(config_path.read_text(encoding="utf-8"))
        saved_config["thread_sessions_per_user"] = True
        config_path.write_text(json.dumps(saved_config), encoding="utf-8")
        source = SessionSource(
            platform=Platform.MATRIX, chat_id=room, thread_id="$existing",
            chat_type="group", user_id="@alice:remote.test", profile=profile,
        )
        alice = store.get_or_create_session(source)
        bob = store.get_or_create_session(SessionSource(
            platform=Platform.MATRIX, chat_id=room, thread_id="$existing",
            chat_type="group", user_id="@bob:remote.test", profile=profile,
        ))
        original_entries = dict(store._entries)
        with _profile_runtime_scope(home, {}):
            store.append_to_transcript(alice.session_id, {
                "role": "assistant", "content": f"Alice's cached turn {index}",
            })
            before = deepcopy(store.load_transcript(alice.session_id))
            target = room if destination == "room" else "#scheduled:remote.test"
            brief = f"Alice's scheduled brief {index}"
            error = _deliver_result(
                {
                    "id": f"alice-brief-{index}",
                    "deliver": f"matrix:{target}/$existing",
                    "attach_to_session": True,
                    "origin": source.to_dict() if has_origin else None,
                },
                brief, runner.adapters, SimpleNamespace(is_running=lambda: True),
            )
            assert error is None, error
            assert store._entries == original_entries
            transcript = store.load_transcript(alice.session_id)
            if not has_origin:
                assert transcript == before
                assert store.load_transcript(bob.session_id) == []
                continue
            assert transcript[:-1] == before
            assert (transcript[-1]["role"], transcript[-1]["content"]) == (
                "user", f"[Cron delivery: alice-brief-{index}]\n{brief}",
            )
            assert store.load_transcript(bob.session_id) == []


@pytest.mark.parametrize("chat_type, unknown_state", [("dm", "membership"), ("group", "membership"), ("dm", "identity")])
@pytest.mark.parametrize("thread", [None, "$existing"])
def test_unknown_membership_defers_continuation_until_identity_recovers(
    destinations, chat_type, thread, unknown_state
):
    runner, owners = destinations
    home, adapter, room = owners["default"]
    adapter._user_id = "@bot:remote.test"
    members = {adapter._user_id, "@alice:remote.test"}
    if unknown_state == "identity":
        adapter._user_id = ""
    if chat_type == "group":
        members.add("@bob:remote.test")
    known = False
    adapter._get_room_members = AsyncMock(
        side_effect=lambda *args, **kwargs: (
            None if unknown_state == "membership" and not known else members
        )
    )
    adapter._get_room_member_profiles = AsyncMock(return_value=None)
    adapter._get_room_state_value = AsyncMock(return_value=None)
    source = SessionSource(
        platform=Platform.MATRIX, chat_id=room, thread_id=thread,
        chat_type=chat_type, user_id="@alice:remote.test", profile="default",
    )
    store = adapter._session_store
    entry = store.get_or_create_session(source)
    store.append_to_transcript(entry.session_id, {"role": "assistant", "content": "Prior turn"})
    before = deepcopy(store.load_transcript(entry.session_id))
    store._db.update_system_prompt(entry.session_id, "Cached DM or group prefix")
    keys = list(store._entries)
    with _profile_runtime_scope(home, {}):
        for known in (False, True):
            if known:
                adapter._user_id = "@bot:remote.test"
            brief = "Confirmed brief" if known else "Unclassified brief"
            error = _deliver_result(
                {"id": "membership", "deliver": f"matrix:{room}" + (f"/{thread}" if thread else ""),
                 "attach_to_session": True, "origin": source.to_dict()},
                brief, runner.adapters, SimpleNamespace(is_running=lambda: True),
            )
            assert error is None, error
            assert store._db.get_session(entry.session_id)["system_prompt"] == "Cached DM or group prefix"
            if not known:
                assert list(store._entries) == keys
                assert store.load_transcript(entry.session_id) == before
                continue
            sent = adapter._client.send_message_event.await_args.args[2]
            delivered_thread = sent.get("m.relates_to", {}).get("event_id")
            reply = SessionSource(
                platform=Platform.MATRIX, chat_id=room, thread_id=delivered_thread,
                chat_type=chat_type, user_id=source.user_id, profile="default",
            )
            reply_entry = store.get_or_create_session(reply)
            transcript = store.load_transcript(reply_entry.session_id)
            assert "Confirmed brief" in json.dumps(transcript)
            assert "Unclassified brief" not in json.dumps(transcript)
            assert store.load_transcript(entry.session_id)[:len(before)] == before


@pytest.mark.parametrize("outcome", ["confirmed", "origin_only", "failed", "unknown", "other_room"])
@pytest.mark.parametrize("alice_exists", [False, True])
@pytest.mark.parametrize("sender", ["native", "http"])
def test_fallback_uses_confirmed_canonical_destination_and_participant(
    destinations, monkeypatch, outcome, alice_exists, sender
):
    from tools.send_message_senders import _matrix_send_core

    runner, owners = destinations
    home, adapter, room = owners["secondary"]
    store = adapter._session_store
    store.config.thread_sessions_per_user = True
    (home / "config.yaml").write_text(json.dumps({
        "thread_sessions_per_user": True,
        "cron": {"wrap_response": False, "mirror_delivery": outcome == "origin_only"},
        "platforms": {"matrix": {"enabled": True, "extra": {"e2ee_mode": "off"}}},
    }), encoding="utf-8")
    source = SessionSource(
        platform=Platform.MATRIX, chat_id=room, thread_id="$existing",
        chat_type="group", user_id="@alice:remote.test", profile="secondary",
    )
    alice = store.get_or_create_session(source) if alice_exists else None
    bob = store.get_or_create_session(SessionSource(
        platform=Platform.MATRIX, chat_id=room, thread_id=source.thread_id,
        chat_type="group", user_id="@bob:remote.test", profile="secondary",
    ))
    keys = list(store._entries)
    adapter._client.resolve_room_alias.side_effect = [
        ValueError("transient directory failure"),
        SimpleNamespace(room_id=room if outcome != "other_room" else "!other:remote.test", servers=["route.test"]),
    ]
    if outcome == "other_room":
        adapter._client.join_room.return_value = "!other:remote.test"
    if outcome == "failed":
        adapter._client.send_message_event.side_effect = ValueError("send rejected")
    if outcome == "unknown":
        adapter._get_room_members = AsyncMock(return_value=None)
        adapter._get_room_state_value = AsyncMock(return_value=None)

    async def request(self, method, path, **kwargs):
        target_room = room if outcome != "other_room" else "!other:remote.test"
        if path.startswith("directory/") or path.startswith("join/"):
            return {"room_id": target_room, "servers": ["route.test"]}
        if path.endswith("m.room.encryption"):
            from plugins.platforms.matrix.standalone import _MatrixAPIError
            raise _MatrixAPIError(404, {"errcode": "M_NOT_FOUND"})
        if path.endswith("joined_members"):
            if outcome == "unknown":
                raise ValueError("membership unavailable")
            return {"joined": {"@bot:remote.test": {}, source.user_id: {}, "@bob:remote.test": {}}}
        if "/send/" in path:
            if outcome == "failed":
                raise ValueError("send rejected")
            return {"event_id": "$sent"}
        raise AssertionError(path)

    monkeypatch.setattr("plugins.platforms.matrix.standalone._HTTPDelivery.request", request)

    async def fallback(platform, pconfig, chat_id, message, *, thread_id=None, **kwargs):
        if sender == "http":
            from plugins.platforms.matrix.standalone import standalone_send
            return await standalone_send(
                PlatformConfig(token="token", extra={"homeserver": "http://matrix.test", "e2ee_mode": "off"}),
                chat_id, message, thread_id=thread_id,
            )
        return await _matrix_send_core(adapter, chat_id, message, [], {"thread_id": thread_id})

    monkeypatch.setattr("tools.send_message_tool._send_to_platform", fallback)
    with _profile_runtime_scope(home, {}):
        error = _deliver_result(
            {"id": "fallback", "deliver": "matrix:#scheduled:remote.test/$existing",
             "attach_to_session": None if outcome == "origin_only" else True, "origin": source.to_dict()},
            "Fallback brief", runner.adapters, SimpleNamespace(is_running=lambda: True),
        )
        assert bool(error) == (outcome == "failed"), error
        assert list(store._entries) == keys
        assert store.load_transcript(bob.session_id) == []
        if alice is not None:
            transcript = store.load_transcript(alice.session_id)
            assert [(turn["role"], turn["content"]) for turn in transcript] == (
                [("user", "[Cron delivery: fallback]\nFallback brief")]
                if outcome in {"confirmed", "origin_only"} else []
            )

@pytest.mark.parametrize("sender", ["tool", "cron", "fallback", "http"])
@pytest.mark.parametrize(
    "transition, alice_exists", [(False, True), (True, True), (True, False)]
)
def test_continuation_uses_live_store_policy_and_current_membership(
    destinations, monkeypatch, sender, transition, alice_exists
):
    from gateway.session_context import clear_session_vars, set_session_vars
    from tools.send_message_senders import _matrix_send_core
    from tools.send_message_tool import send_message_tool

    runner, owners = destinations
    store = owners["default"][1]._session_store
    store.config.thread_sessions_per_user = True
    config_path = owners["default"][0] / "config.yaml"
    saved_config = json.loads(config_path.read_text(encoding="utf-8"))
    saved_config["thread_sessions_per_user"] = True
    config_path.write_text(json.dumps(saved_config), encoding="utf-8")
    runner.session_store = store
    monkeypatch.setattr("gateway.run._gateway_runner_ref", lambda: runner)
    monkeypatch.setattr("model_tools._run_async", asyncio.run)
    for home, adapter, _ in owners.values():
        adapter.set_session_store(store)

    async def fallback(
        platform, pconfig, chat_id, message, *, thread_id=None, **kwargs
    ):
        if sender == "tool":
            from tools.send_message_senders import _send_matrix_via_adapter

            return await _send_matrix_via_adapter(
                pconfig, chat_id, message, thread_id=thread_id
            )
        from hermes_cli.profiles import get_active_profile_name

        if sender == "http":
            from plugins.platforms.matrix.standalone import standalone_send

            return await standalone_send(
                PlatformConfig(
                    token="token",
                    extra={"homeserver": "http://matrix.test", "e2ee_mode": "off"},
                ),
                chat_id,
                message,
                thread_id=thread_id,
            )
        adapter = runner._authorization_adapter(platform, get_active_profile_name())
        return await _matrix_send_core(
            adapter, chat_id, message, [], {"thread_id": thread_id}
        )

    if sender in {"tool", "fallback", "http"}:
        monkeypatch.setattr("tools.send_message_tool._send_to_platform", fallback)
    for index, profile in enumerate(("default", "secondary", "default")):
        home, adapter, room = owners[profile]
        members = {adapter._user_id, "@alice:remote.test"}
        if not transition:
            members.add("@bob:remote.test")
        adapter._get_room_members.return_value = members
        adapter._invalidate_room_identities(room)
        source = SessionSource(
            platform=Platform.MATRIX,
            chat_id=room,
            thread_id="$existing",
            chat_type="dm" if transition else "group",
            user_id="@alice:remote.test",
            profile=profile,
        )
        with _profile_runtime_scope(home, {}):
            alice = store.get_or_create_session(source) if alice_exists else None
            bob = store.get_or_create_session(
                SessionSource(
                    platform=Platform.MATRIX,
                    chat_id=room,
                    thread_id="$existing",
                    chat_type="group",
                    user_id="@bob:remote.test",
                    profile=profile,
                )
            )
            store.append_to_transcript(
                bob.session_id,
                {"role": "user", "content": f"Bob's private turn {index}"},
            )
            store._db.update_system_prompt(bob.session_id, "Bob's cached prefix")
            bob_before = deepcopy(store.load_transcript(bob.session_id))
            before = []
            if alice is not None:
                store.append_to_transcript(
                    alice.session_id,
                    {
                        "role": "user" if sender == "tool" else "assistant",
                        "content": f"Prior request {index}",
                    },
                )
                store._db.update_system_prompt(
                    alice.session_id, "Cached participant prefix"
                )
                before = deepcopy(store.load_transcript(alice.session_id))
            keys = list(store._entries)

            async def send_event(*args):
                if transition:
                    members.add("@bob:remote.test")
                    await adapter._on_room_state(
                        SimpleNamespace(room_id=room, sender="@bob:remote.test")
                    )
                return "$sent"

            adapter._client.send_message_event.side_effect = send_event
            if sender in {"fallback", "http"}:
                adapter._client.resolve_room_alias.side_effect = [
                    ValueError("transient lookup"),
                    SimpleNamespace(room_id=room, servers=[]),
                ]

            async def request(self, method, path, **kwargs):
                if path.startswith("directory/") or path.startswith("join/"):
                    return {"room_id": room, "servers": []}
                if path.endswith("m.room.encryption"):
                    from plugins.platforms.matrix.standalone import _MatrixAPIError

                    raise _MatrixAPIError(404, {"errcode": "M_NOT_FOUND"})
                if path.endswith("joined_members"):
                    return {"joined": {member: {} for member in members}}
                if path == "account/whoami":
                    return {"user_id": adapter._user_id}
                if "/send/" in path:
                    return {"event_id": await send_event()}
                raise AssertionError(path)

            monkeypatch.setattr(
                "plugins.platforms.matrix.standalone._HTTPDelivery.request", request
            )
            brief = f"Brief {index}"
            if sender == "tool":
                tokens = set_session_vars(
                    platform="matrix", user_id="@alice:remote.test"
                )
                try:
                    result = json.loads(
                        send_message_tool({
                            "target": "matrix:#scheduled:remote.test/$existing",
                            "message": brief,
                        })
                    )
                finally:
                    clear_session_vars(tokens)
                assert result.get("success"), result
                assert (result["chat_id"], result["thread_id"]) == (room, "$existing")
                assert bool(result.get("mirrored")) == (not transition)
            else:
                error = _deliver_result(
                    {
                        "id": "ownership",
                        "deliver": "matrix:#scheduled:remote.test/$existing",
                        "attach_to_session": True,
                        "origin": source.to_dict(),
                    },
                    brief,
                    runner.adapters,
                    SimpleNamespace(is_running=lambda: True),
                )
                assert error is None, error
            expected = [(turn["role"], turn["content"]) for turn in before]
            if not transition:
                expected.append((
                    "assistant" if sender == "tool" else "user",
                    brief
                    if sender == "tool"
                    else f"[Cron delivery: ownership]\n{brief}",
                ))
            if alice is not None:
                assert [
                    (turn["role"], turn["content"])
                    for turn in store.load_transcript(alice.session_id)
                ] == expected
                assert (
                    store._db.get_session(alice.session_id)["system_prompt"]
                    == "Cached participant prefix"
                )
            assert list(store._entries) == keys
            assert store.load_transcript(bob.session_id) == bob_before
            assert (
                store._db.get_session(bob.session_id)["system_prompt"]
                == "Bob's cached prefix"
            )
            reply = asyncio.run(adapter.resolve_delivery_target(source))
            assert (reply.chat_id, reply.thread_id, reply.user_id, reply.chat_type) == (
                room,
                "$existing",
                source.user_id,
                "group",
            )
