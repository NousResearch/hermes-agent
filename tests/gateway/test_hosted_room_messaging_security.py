"""Authorization, dispatch, and lifecycle tests for Group Chat messaging."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway import hosted_room_driver, hosted_room_messaging, hosted_rooms
from gateway.config import HomeChannel, Platform, PlatformConfig
from gateway.hosted_room_messaging import (
    MessagingRoomBackend,
    RoomControlError,
    messaging_actor,
    messaging_event_id,
    relay_provenance_is_unknown,
)
from gateway.session import SessionSource
from hermes_cli.commands import resolve_command
from tests.gateway.test_hosted_room_messaging import (
    _FakeService,
    _HoldingRPC,
    _ImmediateRPC,
    _TestHostedRoomService,
    _event,
    _runner,
    _seed_rooms,
    _wait_for,
)


def test_messaging_identity_is_stable_private_and_edit_safe():
    first = _event("/group 1 send hello", platform=Platform.TELEGRAM)
    first.platform_update_id = 123
    second = _event("/group 1 send edited", platform=Platform.TELEGRAM)
    second.platform_update_id = 124
    actor = messaging_actor(first, gateway_id="install:test-gateway")
    assert actor["display_name"] == "Display Name via Telegram"
    assert "user-1" not in actor["id"]
    assert messaging_event_id(first) == messaging_event_id(second)
    assert messaging_event_id(first) == messaging_event_id(first)


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", [Platform.SLACK, Platform.MATRIX])
async def test_dm_label_without_one_to_one_proof_cannot_control_group_chats(
    tmp_path, monkeypatch, platform
):
    service = _FakeService(tmp_path / "state.db")
    monkeypatch.setattr(
        "gateway.hosted_room_messaging.current_room_backend", lambda: service
    )

    result = await _runner(platform=platform)._handle_rooms_command(
        _event("/group", platform=platform, is_one_to_one=None)
    )

    assert result == (
        "Group Chat controls are private. Use your authorized one-to-one Hermes chat."
    )


@pytest.mark.asyncio
async def test_unknown_chat_type_fails_closed_even_with_private_transport_proof(
    tmp_path, monkeypatch
):
    service = _FakeService(tmp_path / "state.db")
    monkeypatch.setattr(
        "gateway.hosted_room_messaging.current_room_backend", lambda: service
    )

    result = await _runner()._handle_rooms_command(
        _event("/group", chat_type="", is_one_to_one=True)
    )

    assert result == (
        "Group Chat controls are private. Use your authorized one-to-one Hermes chat."
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "platform",
    [Platform.WHATSAPP_CLOUD, Platform.EMAIL, Platform.SMS],
)
async def test_native_distinct_dm_proves_private_owner_surface(
    tmp_path, monkeypatch, platform
):
    service = _FakeService(tmp_path / "state.db")
    monkeypatch.setattr(
        "gateway.hosted_room_messaging.current_room_backend", lambda: service
    )

    result = await _runner(
        platform=platform,
        extra={"allow_admin_from": ["user-1"]},
    )._handle_rooms_command(
        _event(
            "/group list",
            platform=platform,
            is_one_to_one=None,
        )
    )

    assert result.startswith("👥 **No Group Chats yet**")


@pytest.mark.asyncio
async def test_edited_message_cannot_start_group_chat_work(tmp_path, monkeypatch):
    db, _, _ = _seed_rooms(tmp_path)
    service = _FakeService(db)
    monkeypatch.setattr(
        "gateway.hosted_room_messaging.current_room_backend", lambda: service
    )
    event = _event("/group 1 send changed", message_id="message-1")
    event.source.message_is_edit = True

    result = await _runner()._handle_rooms_command(event)

    assert (
        result == "Edited messages can’t run Group Chat commands. Send a new message."
    )
    assert service.sent == []


@pytest.mark.asyncio
async def test_mutating_group_chat_commands_have_a_per_sender_rate_limit(
    tmp_path, monkeypatch
):
    db, _, _ = _seed_rooms(tmp_path)
    service = _FakeService(db)
    runner = _runner()
    monkeypatch.setattr(
        "gateway.hosted_room_messaging.current_room_backend", lambda: service
    )
    monkeypatch.setattr("gateway.group_chat_slash._GROUP_CHAT_MUTATION_RATE_LIMIT", 2)

    first = await runner._handle_rooms_command(
        _event("/group 1 send first", message_id="rate-1")
    )
    second = await runner._handle_rooms_command(
        _event("/group 1 send second", message_id="rate-2")
    )
    limited = await runner._handle_rooms_command(
        _event("/group 1 send third", message_id="rate-3")
    )

    assert first.startswith("Queued in")
    assert second.startswith("Queued in")
    assert limited == "Too many Group Chat commands. Wait a moment and try again."
    assert len(service.sent) == 2


@pytest.mark.asyncio
async def test_rate_limits_are_isolated_by_receiving_adapter_profile(
    tmp_path, monkeypatch
):
    db, _, _ = _seed_rooms(tmp_path)
    service = _FakeService(db)
    runner = _runner()
    monkeypatch.setattr(
        "gateway.hosted_room_messaging.current_room_backend", lambda: service
    )
    monkeypatch.setattr("gateway.group_chat_slash._GROUP_CHAT_MUTATION_RATE_LIMIT", 1)
    secondary = _event("/group 1 send from secondary", message_id="rate-secondary")
    secondary.source.profile = "default"
    secondary.source._transport_adapter_ref = lambda: SimpleNamespace(
        _owner_profile="ops"
    )
    primary = _event("/group 1 send from primary", message_id="rate-primary")
    primary.source._transport_adapter_ref = lambda: SimpleNamespace(_owner_profile=None)

    first = await runner._handle_rooms_command(secondary)
    second = await runner._handle_rooms_command(primary)

    assert first.startswith("Queued in")
    assert second.startswith("Queued in")
    assert len(service.sent) == 2


@pytest.mark.parametrize(
    "raw_message",
    [
        {"timestamp_ms": 1770000000000},
        {"trigger_id": "slack-trigger-1"},
        SimpleNamespace(id="discord-interaction-1"),
    ],
)
def test_real_channel_raw_ids_are_stable_without_normalized_message_id(raw_message):
    event = _event("/group 1 send hello")
    event.message_id = None
    event.source.message_id = None
    event.raw_message = raw_message
    assert messaging_event_id(event) == messaging_event_id(event)


def test_mutation_fails_closed_without_a_transport_redelivery_id():
    event = _event("/group 1 send hello")
    event.message_id = None
    event.source.message_id = None
    with pytest.raises(RoomControlError, match="stable message ID"):
        messaging_event_id(event)


def test_signal_group_idempotency_includes_sender_identity():
    first = _event("/group 1 send hello", user_id="sender-a")
    second = _event("/group 1 send hello", user_id="sender-b")
    for event in (first, second):
        event.message_id = None
        event.source.message_id = None
        event.raw_message = {"timestamp_ms": 1770000000000}
    assert messaging_event_id(first) != messaging_event_id(second)


def test_messaging_idempotency_includes_profile_and_workspace_scope():
    default = _event("/group 1 send hello", platform=Platform.TELEGRAM)
    secondary = _event("/group 1 send hello", platform=Platform.TELEGRAM)
    secondary.source.profile = "ops"
    assert messaging_event_id(default) != messaging_event_id(secondary)

    first_workspace = _event("/group 1 send hello", platform=Platform.SLACK)
    second_workspace = _event("/group 1 send hello", platform=Platform.SLACK)
    first_workspace.source.scope_id = "workspace-a"
    second_workspace.source.scope_id = "workspace-b"
    assert messaging_event_id(first_workspace) != messaging_event_id(second_workspace)


def test_messaging_idempotency_includes_receiving_adapter_profile():
    default = _event("/group 1 send hello", platform=Platform.TELEGRAM)
    secondary = _event("/group 1 send hello", platform=Platform.TELEGRAM)
    default.source.profile = secondary.source.profile = "default"
    default.source._transport_adapter_ref = lambda: SimpleNamespace(_owner_profile=None)
    secondary.source._transport_adapter_ref = lambda: SimpleNamespace(
        _owner_profile="ops"
    )

    assert messaging_event_id(default) != messaging_event_id(secondary)


def test_secondary_profile_adapter_supplies_the_typed_command_prefix():
    runner = _runner(platform=Platform.SLACK)
    runner.adapters = {}
    runner._profile_adapters = {
        "ops": {Platform.SLACK: SimpleNamespace(typed_command_prefix="!")}
    }
    source = _event("!group", platform=Platform.SLACK).source
    source.profile = "ops"

    assert runner._typed_command_prefix_for(source) == "!"


@pytest.mark.parametrize("platform", [Platform.SLACK, Platform.MATRIX])
def test_relayed_platform_keeps_its_usable_typed_command_prefix(platform):
    runner = _runner(platform=platform)
    source = _event("!group", platform=platform).source
    source.delivered_via_upstream_relay = True

    assert runner._typed_command_prefix_for(source) == "!"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command",
    ["/group 1 approvals", "/group 1 approve ABCDEF"],
)
async def test_secondary_profile_cannot_view_or_apply_group_chat_approvals(
    tmp_path, monkeypatch, command
):
    db, _, _ = _seed_rooms(tmp_path)
    service = _FakeService(db)
    monkeypatch.setattr(
        "gateway.hosted_room_messaging.current_room_backend", lambda: service
    )
    event = _event(command)
    event.source.profile = "ops"

    result = await _runner()._handle_rooms_command(event)

    assert "installation owner" in result


@pytest.mark.asyncio
async def test_secondary_adapter_routed_to_default_cannot_apply_approval(
    tmp_path, monkeypatch
):
    db, _, _ = _seed_rooms(tmp_path)
    service = _FakeService(db)
    monkeypatch.setattr(
        "gateway.hosted_room_messaging.current_room_backend", lambda: service
    )
    event = _event("/group 1 approve ABCDEF")
    event.source.profile = "default"
    event.source._transport_adapter_ref = lambda: SimpleNamespace(_owner_profile="ops")

    result = await _runner()._handle_rooms_command(event)

    assert "installation owner" in result


@pytest.mark.asyncio
async def test_secondary_profile_room_detail_hides_approval_command_and_code(
    tmp_path, monkeypatch
):
    db, _, _ = _seed_rooms(tmp_path)
    service = _FakeService(db)
    service.room_status = {
        "running": True,
        "working": False,
        "blocked": True,
        "pending_actions": [
            {
                "kind": "approval",
                "authority_gateway_id": "install:test-gateway",
                "authority_epoch": 1,
                "member_id": "ops",
                "task_id": "task-secret",
                "execution_generation": 1,
                "request_id": "request-secret",
                "approval": {
                    "description": "Read a private deployment key",
                    "command": "cat /private/deployment-key",
                    "choices": ["once", "deny"],
                },
            }
        ],
    }
    monkeypatch.setattr(
        "gateway.hosted_room_messaging.current_room_backend", lambda: service
    )
    secondary = _event("/group 1")
    secondary.source.profile = "ops"

    hidden = await _runner()._handle_rooms_command(secondary)
    visible = await _runner()._handle_rooms_command(_event("/group 1"))

    assert "deployment-key" not in hidden
    assert "Approval needed" not in hidden
    assert "deployment-key" in visible
    assert "Approval needed" in visible


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", [p for p in Platform if p is not Platform.LOCAL])
async def test_send_handler_is_shared_by_every_gateway_channel(
    tmp_path, monkeypatch, platform
):
    db, _, _ = _seed_rooms(tmp_path)
    service = _FakeService(db)
    monkeypatch.setattr(
        "gateway.hosted_room_messaging.current_room_backend", lambda: service
    )
    event = _event(
        "/group 1 send hello from messaging",
        platform=platform,
        message_id=f"message-{platform.value}",
    )
    runner = _runner(platform=platform)
    result = await runner._handle_room_command(event)
    rooms_command = f"{runner._typed_command_prefix_for(platform)}group"
    assert result == f"Queued in Release room. Check: `{rooms_command} 1`."
    assert service.sent[-1]["payload"]["text"] == "hello from messaging"
    platform_label = platform.value.replace("_", " ").title()
    assert service.sent[-1]["actor"]["display_name"] == (
        f"Display Name via {platform_label}"
    )


@pytest.mark.asyncio
async def test_entity_first_group_send_is_dispatched(tmp_path, monkeypatch):
    db, _, _ = _seed_rooms(tmp_path)
    service = _FakeService(db)
    monkeypatch.setattr(
        "gateway.hosted_room_messaging.current_room_backend", lambda: service
    )

    result = await _runner()._handle_rooms_command(
        _event("/group 1 send hello from Signal", message_id="entity-first-1")
    )

    assert result == "Queued in Release room. Check: `/group 1`."
    assert service.sent[-1]["payload"]["text"] == "hello from Signal"


@pytest.mark.asyncio
async def test_group_send_preserves_unicode_dashes_in_message_text(
    tmp_path, monkeypatch
):
    db, _, _ = _seed_rooms(tmp_path)
    service = _FakeService(db)
    monkeypatch.setattr(
        "gateway.hosted_room_messaging.current_room_backend", lambda: service
    )
    message = "Ship—today; validate range 1–3"

    result = await _runner()._handle_rooms_command(
        _event(f"/group 1 send {message}", message_id="unicode-dash-send")
    )

    assert result.startswith("Queued in Release room")
    assert service.sent[-1]["payload"]["text"] == message


@pytest.mark.asyncio
async def test_send_rejects_attachments_instead_of_silently_dropping_them(
    tmp_path, monkeypatch
):
    db, _, _ = _seed_rooms(tmp_path)
    service = _FakeService(db)
    monkeypatch.setattr(
        "gateway.hosted_room_messaging.current_room_backend", lambda: service
    )
    result = await _runner()._handle_room_command(
        _event("/group 1 send inspect this", media=True)
    )
    assert "Attachments from messaging chats aren’t supported yet" in result
    assert service.sent == []

    ignored = _event("/group 1 send inspect this")
    ignored.source.message_had_attachments = True
    result = await _runner()._handle_room_command(ignored)
    assert "Attachments from messaging chats aren’t supported yet" in result
    assert service.sent == []


@pytest.mark.asyncio
async def test_bot_authored_controls_are_rejected_to_prevent_bridge_loops(
    tmp_path, monkeypatch
):
    db, _, _ = _seed_rooms(tmp_path)
    service = _FakeService(db)
    monkeypatch.setattr(
        "gateway.hosted_room_messaging.current_room_backend", lambda: service
    )
    result = await _runner(platform=Platform.DISCORD)._handle_room_command(
        _event(
            "/group 1 send repeat this",
            platform=Platform.DISCORD,
            is_bot=True,
        )
    )
    assert result == "Group Chat controls are only available to people."
    assert service.sent == []


@pytest.mark.asyncio
async def test_raw_webhook_bot_marker_is_rejected_even_without_source_flag(
    tmp_path, monkeypatch
):
    db, _, _ = _seed_rooms(tmp_path)
    service = _FakeService(db)
    monkeypatch.setattr(
        "gateway.hosted_room_messaging.current_room_backend", lambda: service
    )
    event = _event("/group 1 send repeat this")
    event.raw_message = {"subtype": "bot_message", "bot_id": "B123"}
    result = await _runner()._handle_rooms_command(event)
    assert result == "Group Chat controls are only available to people."
    assert service.sent == []


def test_relay_and_session_roundtrip_preserve_bot_provenance():
    from gateway.relay.ws_transport import _event_from_wire

    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="chat-1",
        user_id="bot-1",
        is_bot=True,
    )
    assert SessionSource.from_dict(source.to_dict()).is_bot is True
    wire_source = source.to_dict()
    wire_source["message_is_edit"] = False
    relayed = _event_from_wire({
        "text": "/group",
        "message_type": "command",
        "source": wire_source,
    })
    assert relayed.source.is_bot is True
    assert relay_provenance_is_unknown(relayed) is False


def test_legacy_relay_without_author_classification_fails_closed():
    from gateway.relay.ws_transport import _event_from_wire

    relayed = _event_from_wire({
        "text": "/group",
        "message_type": "command",
        "source": {
            "platform": "discord",
            "chat_id": "chat-1",
            "chat_type": "dm",
            "user_id": "user-1",
        },
    })
    assert relay_provenance_is_unknown(relayed) is True


@pytest.mark.parametrize(
    ("platform", "verified", "expected"),
    [
        ("signal", None, True),
        ("telegram", None, True),
        ("whatsapp", None, True),
        ("slack", None, None),
        ("slack", True, True),
        ("matrix", False, False),
    ],
)
def test_authenticated_relay_preserves_one_to_one_privacy_proof(
    platform,
    verified,
    expected,
):
    from gateway.relay.ws_transport import _event_from_wire

    source = {
        "platform": platform,
        "chat_id": "private-chat",
        "chat_type": "dm",
        "user_id": "user-1",
        "is_bot": False,
        "message_is_edit": False,
    }
    if verified is not None:
        source["one_to_one_verified"] = verified
    relayed = _event_from_wire({
        "text": "/group",
        "message_type": "command",
        "source": source,
    })

    assert relayed.source.is_one_to_one is expected
    assert relay_provenance_is_unknown(relayed) is False


@pytest.mark.asyncio
async def test_classified_relay_dm_with_explicit_admin_can_control_group_chats(
    tmp_path,
    monkeypatch,
):
    from gateway.relay.ws_transport import _event_from_wire

    db, _, _ = _seed_rooms(tmp_path)
    service = _FakeService(db)
    monkeypatch.setattr(
        "gateway.hosted_room_messaging.current_room_backend",
        lambda: service,
    )
    event = _event_from_wire({
        "text": "/group list",
        "message_type": "command",
        "source": {
            "platform": "signal",
            "chat_id": "chat-signal",
            "chat_type": "dm",
            "user_id": "user-1",
            "is_bot": False,
            "message_is_edit": False,
        },
    })

    result = await _runner(
        extra={"allow_admin_from": ["user-1"]}
    )._handle_rooms_command(event)

    assert result.startswith("👥 **Group Chats**")


@pytest.mark.asyncio
async def test_stop_requires_admin_when_operator_enabled_slash_gating(
    tmp_path, monkeypatch
):
    db, _, _ = _seed_rooms(tmp_path)
    service = _FakeService(db)
    monkeypatch.setattr(
        "gateway.hosted_room_messaging.current_room_backend", lambda: service
    )
    extra = {
        "allow_admin_from": ["admin"],
        "user_allowed_commands": ["groups"],
    }
    result = await _runner(extra=extra)._handle_room_command(
        _event("/group 1 stop", user_id="member")
    )
    assert result.startswith("This chat can’t control Group Chats")
    assert service.stopped == []

    result = await _runner(extra=extra)._handle_room_command(
        _event("/group 1 stop", user_id="admin", message_id="stop-1")
    )
    assert result == (
        "Stop requested for Release room. Active work will stop safely. "
        "Check: `/group 1`."
    )
    assert service.stopped[0][0] == "release-room"


@pytest.mark.asyncio
async def test_even_an_admin_cannot_expose_room_history_in_a_shared_chat(
    tmp_path, monkeypatch
):
    db, _, _ = _seed_rooms(tmp_path)
    service = _FakeService(db)
    monkeypatch.setattr(
        "gateway.hosted_room_messaging.current_room_backend", lambda: service
    )
    event = _event("/group list", user_id="admin", chat_type="group")

    result = await _runner(
        extra={"group_allow_admin_from": ["admin"]}
    )._handle_rooms_command(event)

    assert result == (
        "Group Chat controls are private. Use your authorized one-to-one Hermes chat."
    )
    assert "Release room" not in result


@pytest.mark.asyncio
async def test_room_history_and_mutation_require_an_explicit_admin(
    tmp_path, monkeypatch
):
    db, _, _ = _seed_rooms(tmp_path)
    service = _FakeService(db)
    monkeypatch.setattr(
        "gateway.hosted_room_messaging.current_room_backend", lambda: service
    )
    runner = _runner(extra={"allow_from": ["user-1"]})
    runner._is_user_authorized_for_source = lambda _source: True
    denial = "This chat can’t control Group Chats"
    assert denial in await runner._handle_rooms_command(_event("/group"))
    assert denial in await runner._handle_room_command(_event("/group 1 send hello"))
    assert service.sent == []


@pytest.mark.asyncio
async def test_home_dm_controls_rooms_without_duplicate_admin_list(
    tmp_path, monkeypatch
):
    db, _, _ = _seed_rooms(tmp_path)
    service = _FakeService(db)
    monkeypatch.setattr(
        "gateway.hosted_room_messaging.current_room_backend", lambda: service
    )
    monkeypatch.setenv("SIGNAL_ALLOWED_USERS", "user-1")
    runner = _runner(extra={})
    runner._is_user_authorized_for_source = lambda _source: True
    runner.config.platforms[Platform.SIGNAL].home_channel = HomeChannel(
        platform=Platform.SIGNAL,
        chat_id="chat-signal",
        name="Home",
    )

    listing = await runner._handle_rooms_command(_event("/group list"))
    assert listing.startswith("👥 **Group Chats**")
    result = await runner._handle_room_command(
        _event("/group 1 send hello", message_id="home-send-1")
    )
    assert result.startswith("Queued in Release room")
    assert service.sent[0]["payload"]["text"] == "hello"


@pytest.mark.asyncio
async def test_explicit_home_group_allows_only_the_selecting_operator(
    tmp_path, monkeypatch
):
    db, _, _ = _seed_rooms(tmp_path)
    service = _FakeService(db)
    monkeypatch.setattr(
        "gateway.hosted_room_messaging.current_room_backend", lambda: service
    )
    monkeypatch.setenv("SIGNAL_ALLOWED_USERS", "user-1")
    runner = _runner(extra={"allow_from": ["user-1"]})
    runner._is_user_authorized_for_source = lambda _source: True
    runner.config.platforms[Platform.SIGNAL].home_channel = HomeChannel(
        platform=Platform.SIGNAL,
        chat_id="chat-signal",
        name="Home team",
        user_id="user-1",
    )
    event = _event(
        "/group 1 send hello from home",
        message_id="home-group-send-1",
        chat_type="group",
        is_one_to_one=False,
    )

    result = await runner._handle_room_command(event)

    assert result.startswith("Queued in Release room")
    assert service.sent[0]["payload"]["text"] == "hello from home"


@pytest.mark.asyncio
async def test_non_home_dm_still_requires_explicit_admin(tmp_path, monkeypatch):
    db, _, _ = _seed_rooms(tmp_path)
    service = _FakeService(db)
    monkeypatch.setattr(
        "gateway.hosted_room_messaging.current_room_backend", lambda: service
    )
    runner = _runner(extra={})
    runner.config.platforms[Platform.SIGNAL].home_channel = HomeChannel(
        platform=Platform.SIGNAL,
        chat_id="different-chat",
        name="Home",
    )

    result = await runner._handle_room_command(
        _event("/group 1 send hello", message_id="other-send-1")
    )
    assert result.startswith("This chat can’t control Group Chats")
    assert service.sent == []


@pytest.mark.asyncio
async def test_shared_home_dm_does_not_auto_promote_an_allowed_user(
    tmp_path, monkeypatch
):
    db, _, _ = _seed_rooms(tmp_path)
    service = _FakeService(db)
    monkeypatch.setattr(
        "gateway.hosted_room_messaging.current_room_backend", lambda: service
    )
    runner = _runner(extra={"allow_from": ["user-1", "user-2"]})
    runner._is_user_authorized_for_source = lambda _source: True
    runner.config.platforms[Platform.SIGNAL].home_channel = HomeChannel(
        platform=Platform.SIGNAL,
        chat_id="chat-signal",
        name="Home",
    )

    result = await runner._handle_room_command(
        _event("/group 1 send hello", message_id="shared-send-1")
    )
    assert result.startswith("This chat can’t control Group Chats")
    assert service.sent == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("allowed_users", "allow_all"),
    [
        ("user-1,user-2", ""),
        ("user-1", "true"),
    ],
)
async def test_builtin_platform_grants_fail_closed_without_registry(
    tmp_path,
    monkeypatch,
    allowed_users,
    allow_all,
):
    db, _, _ = _seed_rooms(tmp_path)
    service = _FakeService(db)
    monkeypatch.setattr(
        "gateway.hosted_room_messaging.current_room_backend", lambda: service
    )
    monkeypatch.setenv("SIGNAL_ALLOWED_USERS", allowed_users)
    monkeypatch.setenv("SIGNAL_ALLOW_ALL_USERS", allow_all)
    runner = _runner(extra={})
    runner._is_user_authorized_for_source = lambda _source: True
    runner.config.platforms[Platform.SIGNAL].home_channel = HomeChannel(
        platform=Platform.SIGNAL,
        chat_id="chat-signal",
        name="Home",
    )

    result = await runner._handle_room_command(
        _event("/group 1 send hello", message_id="grant-send-1")
    )
    assert result.startswith("This chat can’t control Group Chats")
    assert service.sent == []


@pytest.mark.asyncio
async def test_routed_profile_uses_transport_principals_for_owner_census(
    tmp_path, monkeypatch
):
    from contextlib import contextmanager

    from gateway import authz_mixin, run as gateway_run

    db, _, _ = _seed_rooms(tmp_path)
    service = _FakeService(db)
    monkeypatch.setattr(
        "gateway.hosted_room_messaging.current_room_backend", lambda: service
    )
    active_scope = {"name": "routed"}

    @contextmanager
    def fake_profile_scope(path):
        previous = active_scope["name"]
        active_scope["name"] = Path(path).name
        try:
            yield
        finally:
            active_scope["name"] = previous

    def fake_auth_env(name, default=""):
        if name == "SIGNAL_ALLOWED_USERS":
            return "user-1,user-2" if active_scope["name"] == "transport" else "user-1"
        return default

    monkeypatch.setattr(gateway_run, "_profile_runtime_scope", fake_profile_scope)
    monkeypatch.setattr(authz_mixin, "_auth_env", fake_auth_env)
    runner = _runner(extra={})
    runner._is_user_authorized_for_source = lambda _source: True
    runner.config.platforms[Platform.SIGNAL].home_channel = HomeChannel(
        platform=Platform.SIGNAL,
        chat_id="chat-signal",
        name="Home",
    )
    event = _event("/group 1 send hello", message_id="routed-send-1")
    event.source.profile = "worker"
    event.source._authorization_profile_home = Path("/transport")

    result = await runner._handle_room_command(event)
    assert result.startswith("This chat can’t control Group Chats")
    assert service.sent == []


@pytest.mark.asyncio
async def test_secondary_adapter_allowlist_owns_the_owner_census(tmp_path, monkeypatch):
    db, _, _ = _seed_rooms(tmp_path)
    service = _FakeService(db)
    monkeypatch.setattr(
        "gateway.hosted_room_messaging.current_room_backend", lambda: service
    )
    runner = _runner(extra={"allow_from": ["user-1"]})
    runner._is_user_authorized_for_source = lambda _source: True
    runner.config.platforms[Platform.SIGNAL].home_channel = HomeChannel(
        platform=Platform.SIGNAL,
        chat_id="chat-signal",
        name="Home",
    )
    adapter = SimpleNamespace(
        config=PlatformConfig(extra={"allow_from": ["user-1", "user-2"]})
    )
    runner._profile_adapters = {"worker": {Platform.SIGNAL: adapter}}
    runner.adapters = {}
    event = _event("/group 1 send hello", message_id="secondary-send-1")
    event.source.profile = "worker"
    event.source._transport_adapter_ref = lambda: adapter

    result = await runner._handle_room_command(event)
    assert result.startswith("This chat can’t control Group Chats")
    assert service.sent == []


@pytest.mark.asyncio
async def test_registered_adapter_without_visible_allowlist_fails_closed(
    tmp_path, monkeypatch
):
    db, _, _ = _seed_rooms(tmp_path)
    service = _FakeService(db)
    monkeypatch.setattr(
        "gateway.hosted_room_messaging.current_room_backend", lambda: service
    )
    runner = _runner(extra={"allow_from": ["user-1"]})
    runner._is_user_authorized_for_source = lambda _source: True
    runner.config.platforms[Platform.SIGNAL].home_channel = HomeChannel(
        platform=Platform.SIGNAL,
        chat_id="chat-signal",
        name="Home",
    )
    adapter = SimpleNamespace(config=PlatformConfig(extra={}))
    runner._profile_adapters = {"worker": {Platform.SIGNAL: adapter}}
    runner.adapters = {}
    event = _event("/group 1 send hello", message_id="opaque-send-1")
    event.source.profile = "worker"
    event.source._transport_adapter_ref = lambda: adapter

    result = await runner._handle_room_command(event)
    assert result.startswith("This chat can’t control Group Chats")
    assert service.sent == []


@pytest.mark.asyncio
async def test_relayed_home_dm_requires_explicit_admin(tmp_path, monkeypatch):
    db, _, _ = _seed_rooms(tmp_path)
    service = _FakeService(db)
    monkeypatch.setattr(
        "gateway.hosted_room_messaging.current_room_backend", lambda: service
    )
    runner = _runner(extra={"allow_from": ["user-1"]})
    runner._is_user_authorized_for_source = lambda _source: True
    runner.config.platforms[Platform.SIGNAL].home_channel = HomeChannel(
        platform=Platform.SIGNAL,
        chat_id="chat-signal",
        name="Home",
    )
    event = _event("/group 1 send hello", message_id="relay-send-1")
    event.source.delivered_via_upstream_relay = True
    event.metadata = {
        "relay_author_classified": True,
        "relay_edit_classified": True,
    }

    result = await runner._handle_room_command(event)
    assert result.startswith("This chat can’t control Group Chats")
    assert service.sent == []


@pytest.mark.asyncio
async def test_busy_dispatch_runs_room_control_without_touching_main_agent():
    runner = _runner()
    runner._handle_rooms_command = AsyncMock(return_value="room-dispatched")
    event = _event("/group 1 send hello")
    result = await runner._dispatch_busy_slash_command(
        event,
        resolve_command("group"),
        "session-key",
        event.source,
    )
    assert result == "room-dispatched"
    runner._handle_rooms_command.assert_awaited_once_with(event)


@pytest.mark.asyncio
async def test_real_service_persists_server_owned_messaging_actor(
    tmp_path, monkeypatch
):
    service = _TestHostedRoomService(tmp_path / "state.db")
    service.create_room(
        room_id="release-room",
        name="Release room",
        members=[
            {"member_id": "default", "profile": "default", "handle": "hermes"},
            {"member_id": "ops", "profile": "ops", "handle": "ops"},
        ],
    )
    monkeypatch.setattr(
        "gateway.hosted_room_messaging.current_room_backend",
        lambda: MessagingRoomBackend(db_path=service.db_path, service=service),
    )
    event = _event(
        "/group 1 send @ops inspect the release",
        platform=Platform.SIGNAL,
    )
    result = await _runner()._handle_room_command(event)
    assert result == "Queued in Release room. Check: `/group 1`."
    delta = hosted_rooms.read_events(
        service.db_path,
        room_id="release-room",
        since_seq=0,
        limit=20,
    )
    user_event = next(row for row in delta["events"] if row["kind"] == "message.user")
    assert user_event["actor"]["display_name"] == "Display Name via Signal"
    assert user_event["actor"]["id"].startswith("messaging:signal:")
    assert "user-1" not in user_event["actor"]["id"]


def test_cross_process_store_wakes_owner_without_desktop_transport(tmp_path):
    service = _TestHostedRoomService(tmp_path / "state.db")
    service.rpc = _ImmediateRPC()
    service.runtime.rpc = service.rpc
    service.create_room(
        room_id="release-room",
        name="Release room",
        members=[
            {"member_id": "default", "profile": "default", "handle": "hermes"},
            {"member_id": "ops", "profile": "ops", "handle": "ops"},
        ],
    )
    messaging_process = MessagingRoomBackend(db_path=service.db_path)
    first_event = _event(
        "/group 1 send @ops inspect the release",
        platform=Platform.WHATSAPP,
        message_id="message-1",
    )
    second_event = _event(
        "/group 1 send @ops check the notes",
        platform=Platform.WHATSAPP,
        message_id="message-2",
    )
    for event, text in (
        (first_event, "@ops inspect the release"),
        (second_event, "@ops check the notes"),
    ):
        event_id = messaging_event_id(event)
        messaging_process.send(
            room_id="release-room",
            event_id=event_id,
            payload={"text": text, "thread_id": event_id},
            actor=messaging_actor(event, gateway_id="install:test-gateway"),
        )
    service.start()
    try:
        _wait_for(
            lambda: (
                sum(
                    row["kind"] == "message.member"
                    for row in hosted_rooms.read_events(
                        service.db_path,
                        room_id="release-room",
                        since_seq=0,
                        limit=40,
                    )["events"]
                )
                == 2
            )
        )
    finally:
        assert service.stop(timeout=1.0)

    events = hosted_rooms.read_events(
        service.db_path,
        room_id="release-room",
        since_seq=0,
        limit=40,
    )["events"]
    assert [row["kind"] for row in events[:2]] == ["message.user", "message.user"]
    assert sum(row["kind"] == "message.member" for row in events) == 2
    assert events[0]["actor"]["display_name"] == "Display Name via Whatsapp"


def test_cross_process_stop_cancels_durable_queued_work(tmp_path):
    service = _TestHostedRoomService(tmp_path / "state.db")
    service.create_room(
        room_id="release-room",
        name="Release room",
        members=[
            {"member_id": "default", "profile": "default", "handle": "hermes"},
            {"member_id": "ops", "profile": "ops", "handle": "ops"},
        ],
    )
    service.send(
        room_id="release-room",
        event_id="user-1",
        payload={"text": "@ops inspect", "thread_id": "thread-1"},
    )
    messaging_process = MessagingRoomBackend(db_path=service.db_path)
    assert messaging_process.stop_room("release-room", cancel_id="stop-1") == 1
    assert messaging_process.stop_room("release-room", cancel_id="stop-1") == 1
    service.prepare_room(service.bindings()[0])
    assert messaging_process.status("release-room")["working"] is False
    tasks = hosted_room_driver.list_tasks(service.db_path, room_id="release-room")
    assert [task["status"] for task in tasks] == ["cancelled"]


def test_cross_process_stop_cancels_deferred_work(tmp_path):
    service = _TestHostedRoomService(tmp_path / "state.db")
    service.create_room(
        room_id="release-room",
        name="Release room",
        members=[
            {"member_id": "default", "profile": "default", "handle": "hermes"},
            {"member_id": "ops", "profile": "ops", "handle": "ops"},
        ],
    )
    service.send(
        room_id="release-room",
        event_id="user-1",
        payload={"text": "@ops inspect", "thread_id": "thread-1"},
    )
    with sqlite3.connect(service.db_path) as conn:
        conn.execute(
            """UPDATE hosted_room_driver_tasks
                  SET status='deferred', execution_generation=1
                WHERE room_id='release-room'"""
        )
        conn.commit()

    messaging_process = MessagingRoomBackend(db_path=service.db_path)
    assert messaging_process.stop_room("release-room", cancel_id="stop-deferred") == 1
    task = hosted_room_driver.list_tasks(service.db_path, room_id="release-room")[0]
    assert task["status"] == "stopping"
    assert task["cancel_id"] == "stop-deferred"


def test_cross_process_send_is_idempotent_on_transport_redelivery(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        hosted_rooms,
        "local_authority_gateway_id",
        lambda: "install:test-gateway",
    )
    db, room, _ = _seed_rooms(tmp_path)
    messaging_process = MessagingRoomBackend(db_path=db)
    event = _event("/group 1 send hello", platform=Platform.TELEGRAM)
    event_id = messaging_event_id(event)
    payload = {"text": "hello", "thread_id": event_id}
    actor = messaging_actor(event, gateway_id="install:test-gateway")
    first = messaging_process.send(
        room_id=room["room_id"],
        event_id=event_id,
        payload=payload,
        actor=actor,
    )
    second = messaging_process.send(
        room_id=room["room_id"],
        event_id=event_id,
        payload=payload,
        actor=actor,
    )
    assert first["seq"] == second["seq"] == 1
    assert second["idempotent"] is True
    delta = hosted_rooms.read_events(db, room_id=room["room_id"], since_seq=0, limit=20)
    assert len(delta["events"]) == 1


def test_cross_process_stop_is_acknowledged_by_owner_for_running_work(tmp_path):
    service = _TestHostedRoomService(tmp_path / "state.db")
    service.rpc = _HoldingRPC()
    service.runtime.rpc = service.rpc
    service.create_room(
        room_id="release-room",
        name="Release room",
        members=[
            {"member_id": "default", "profile": "default", "handle": "hermes"},
            {"member_id": "ops", "profile": "ops", "handle": "ops"},
        ],
    )
    messaging_process = MessagingRoomBackend(db_path=service.db_path)
    event = _event("/group 1 send inspect")
    messaging_process.send(
        room_id="release-room",
        event_id=messaging_event_id(event),
        payload={"text": "inspect", "thread_id": messaging_event_id(event)},
        actor=messaging_actor(event, gateway_id="install:test-gateway"),
    )
    service.start()
    try:
        _wait_for(
            lambda: any(
                task["status"] == "running"
                for task in hosted_room_driver.list_tasks(
                    service.db_path, room_id="release-room"
                )
            )
        )
        assert messaging_process.stop_room("release-room", cancel_id="stop-1") == 1
        _wait_for(
            lambda: (
                hosted_room_driver.list_tasks(service.db_path, room_id="release-room")[
                    0
                ]["status"]
                == "cancelled"
            )
        )
    finally:
        assert service.stop(timeout=1.0)

    assert len(service.rpc.interrupts) == 1
    tasks = hosted_room_driver.list_tasks(service.db_path, room_id="release-room")
    assert len(tasks) == 1
    assert tasks[0]["status"] == "cancelled"
