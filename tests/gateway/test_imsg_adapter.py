"""Contract tests for the native local-imsg platform plugin."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.config_env import _enable_plugin_platform
from tests.gateway._plugin_adapter_loader import load_plugin_adapter


imsg = load_plugin_adapter("imessage")
ImsgAdapter = imsg.ImsgAdapter


class FakeRpc:
    def __init__(self, history=None):
        self.calls = []
        self.history = history or []
        self.started = False
        self.stopped = False

    async def start(self):
        self.started = True

    async def stop(self):
        self.stopped = True

    async def request(self, method, params=None, timeout=None):
        self.calls.append((method, params or {}))
        if method == "status":
            return {"available": True, "capabilities": {"rpc": True}}
        if method == "watch.subscribe":
            return {"subscription": "sub-1"}
        if method == "watch.unsubscribe":
            return {"ok": True}
        if method == "messages.history":
            return {"messages": self.history}
        if method == "messages.after":
            return {"messages": [], "next_rowid": 0, "has_more": False}
        if method == "send":
            return {"guid": "out-guid", "status": "sent"}
        return {}


class ClosableRpc(FakeRpc):
    def __init__(self):
        super().__init__()
        self.closed = asyncio.Event()

    async def wait_closed(self):
        await self.closed.wait()

    async def stop(self):
        await super().stop()
        self.closed.set()


class SubscribeFailRpc(FakeRpc):
    async def request(self, method, params=None, timeout=None):
        if method == "watch.subscribe":
            raise RuntimeError("subscribe failed")
        return await super().request(method, params, timeout)


class BarrierSubscribeRpc(FakeRpc):
    def __init__(self):
        super().__init__()
        self.subscribe_entered = asyncio.Event()
        self.release_subscribe = asyncio.Event()

    async def request(self, method, params=None, timeout=None):
        if method == "watch.subscribe":
            self.subscribe_entered.set()
            await self.release_subscribe.wait()
        return await super().request(method, params, timeout)

    async def stop(self):
        await super().stop()
        self.release_subscribe.set()


def cfg(**extra):
    return PlatformConfig(
        enabled=True,
        extra={
            "cli_path": "/opt/homebrew/bin/imsg",
            "db_path": "/Users/jarvis/Library/Messages/chat.db",
            "allowed_users": ["+15550001111"],
            "dm_allowed_users": ["+15550001111"],
            "group_allowed_users": ["+15550002222"],
            "allowed_chats": ["42"],
            "require_mention": True,
            "mention_patterns": [r"(?<!\w)@?jarvis\b"],
            "history_limit": 100,
            **extra,
        },
    )


def message(**overrides):
    value = {
        "id": 101,
        "guid": "in-guid",
        "chat_id": 42,
        "chat_guid": "iMessage;+;chat-guid",
        "chat_identifier": "chat-identifier",
        "chat_name": "Family",
        "is_group": False,
        "sender": "+15550001111",
        "is_from_me": False,
        "text": "hello",
        "created_at": "2026-09-16T12:00:00Z",
        "attachments": [],
    }
    value.update(overrides)
    return value


@pytest.fixture
def state_path(monkeypatch, tmp_path):
    path = tmp_path / "imsg-state.json"
    monkeypatch.setattr(imsg, "_state_path", lambda: path)
    return path


@pytest.mark.asyncio
async def test_connect_resumes_checkpoint_and_reports_healthy(state_path):
    state_path.write_text(
        json.dumps({"last_rowid": 88, "seen_guids": []}), encoding="utf-8"
    )
    rpc = FakeRpc()
    adapter = ImsgAdapter(cfg(), rpc_factory=lambda **_: rpc)

    assert await adapter.connect() is True

    assert rpc.started is True
    subscribe = next(
        params for method, params in rpc.calls if method == "watch.subscribe"
    )
    assert subscribe == {
        "attachments": True,
        "include_reactions": False,
        "since_rowid": 88,
    }
    assert adapter.health_status()["status"] == "healthy"
    assert adapter.health_status()["last_rowid"] == 88
    await adapter.disconnect()
    assert ("watch.unsubscribe", {"subscription": "sub-1"}) in rpc.calls
    assert rpc.stopped is True


@pytest.mark.asyncio
async def test_first_connect_persists_tail_baseline_before_subscribing(state_path):
    class BaselineRpc(FakeRpc):
        def __init__(self):
            super().__init__()
            self.pages = iter((
                {"messages": [], "next_rowid": 500, "has_more": True},
                {"messages": [], "next_rowid": 900, "has_more": False},
            ))

        async def request(self, method, params=None, timeout=None):
            if method == "messages.after":
                self.calls.append((method, params or {}))
                return next(self.pages)
            return await super().request(method, params, timeout)

    rpc = BaselineRpc()
    adapter = ImsgAdapter(cfg(), rpc_factory=lambda **_: rpc)

    assert await adapter.connect() is True
    after_calls = [params for method, params in rpc.calls if method == "messages.after"]
    assert [params["since_rowid"] for params in after_calls] == [0, 500]
    subscribe = next(
        params for method, params in rpc.calls if method == "watch.subscribe"
    )
    assert subscribe["since_rowid"] == 900
    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    assert persisted["cursor_initialized"] is True
    assert persisted["last_rowid"] == 900


@pytest.mark.asyncio
async def test_empty_baseline_reconnect_uses_replay_cursor_instead_of_tail(state_path):
    first, second = ClosableRpc(), ClosableRpc()
    clients = iter((first, second))
    adapter = ImsgAdapter(cfg(), rpc_factory=lambda **_: next(clients))

    assert await adapter.connect() is True
    first.closed.set()
    for _ in range(100):
        if adapter._rpc is second:
            break
        await asyncio.sleep(0.02)

    subscribe = next(
        params for method, params in second.calls if method == "watch.subscribe"
    )
    assert subscribe["since_rowid"] == -1
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_supervisor_restarts_failed_bridge_from_durable_checkpoint(state_path):
    first, second = ClosableRpc(), ClosableRpc()
    clients = iter((first, second))
    adapter = ImsgAdapter(cfg(), rpc_factory=lambda **_: next(clients))

    assert await adapter.connect() is True
    first.closed.set()
    await asyncio.sleep(0)
    assert adapter.health_status()["status"] == "unhealthy"
    for _ in range(50):
        if adapter._rpc is second:
            break
        await asyncio.sleep(0.03)

    assert adapter._rpc is second
    assert adapter.health_status()["bridge_restarts"] == 1
    assert any(method == "watch.subscribe" for method, _ in second.calls)
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_watch_overflow_resubscribes_from_terminal_resume_cursor(state_path):
    state_path.write_text(
        json.dumps({"cursor_initialized": True, "last_rowid": 88, "seen_guids": []}),
        encoding="utf-8",
    )
    rpc = FakeRpc()
    adapter = ImsgAdapter(cfg(), rpc_factory=lambda **_: rpc)
    adapter._rpc = rpc
    adapter._subscription = "sub-1"

    await adapter._handle_notification(
        "watch.overflow",
        {
            "subscription": "sub-1",
            "resume_after_rowid": 90,
            "reason": "buffer_limit_exceeded",
            "terminal": True,
        },
    )

    subscribe_calls = [
        params for method, params in rpc.calls if method == "watch.subscribe"
    ]
    assert subscribe_calls[-1]["since_rowid"] == 90
    assert adapter._subscription == "sub-1"
    assert adapter.health_status()["last_rowid"] == 88


@pytest.mark.asyncio
async def test_failed_subscription_stops_candidate_bridge(state_path):
    rpc = SubscribeFailRpc()
    adapter = ImsgAdapter(cfg(), rpc_factory=lambda **_: rpc)

    assert await adapter.connect() is False
    assert rpc.stopped is True


@pytest.mark.asyncio
async def test_disconnect_during_initial_subscribe_stops_unpublished_candidate(
    state_path,
):
    rpc = BarrierSubscribeRpc()
    adapter = ImsgAdapter(cfg(), rpc_factory=lambda **_: rpc)

    connecting = asyncio.create_task(adapter.connect())
    await asyncio.wait_for(rpc.subscribe_entered.wait(), timeout=1)
    await asyncio.wait_for(adapter.disconnect(), timeout=1)

    assert await asyncio.wait_for(connecting, timeout=1) is False
    assert rpc.stopped is True
    assert adapter._rpc is None
    assert adapter.health_status()["status"] == "unhealthy"


@pytest.mark.asyncio
async def test_supervisor_retries_more_than_one_failed_restart(state_path, monkeypatch):
    monkeypatch.setattr(imsg, "_RESTART_BASE_SECONDS", 0.001)
    first, failed, recovered = ClosableRpc(), SubscribeFailRpc(), ClosableRpc()
    clients = iter((first, failed, recovered))
    adapter = ImsgAdapter(cfg(), rpc_factory=lambda **_: next(clients))

    assert await adapter.connect() is True
    first.closed.set()
    for _ in range(100):
        if adapter._rpc is recovered:
            break
        await asyncio.sleep(0.002)

    assert failed.stopped is True
    assert adapter._rpc is recovered
    assert adapter.health_status()["bridge_restarts"] == 1
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_inbound_dm_preserves_thread_and_attachments(state_path, tmp_path):
    attachment = tmp_path / "photo.jpg"
    attachment.write_bytes(b"jpeg")
    rpc = FakeRpc()
    adapter = ImsgAdapter(cfg(), rpc_factory=lambda **_: rpc)
    adapter.handle_message = AsyncMock()

    await adapter._handle_notification(
        "message",
        {
            "message": message(
                attachments=[
                    {
                        "original_path": str(attachment),
                        "mime_type": "image/jpeg",
                        "missing": False,
                    }
                ]
            )
        },
    )

    assert adapter.handle_message.await_args is not None
    event = adapter.handle_message.await_args.args[0]
    assert event.source.chat_id == "42"
    assert event.source.chat_type == "dm"
    assert event.source.user_id == "+15550001111"
    assert event.message_id == "in-guid"
    assert event.media_urls == [str(attachment)]
    assert event.media_types == ["image/jpeg"]
    saved = json.loads(state_path.read_text(encoding="utf-8"))
    assert saved["last_rowid"] == 101
    assert "in-guid" in saved["seen_guids"]


@pytest.mark.asyncio
async def test_self_messages_and_restart_duplicates_are_suppressed(state_path):
    first = ImsgAdapter(cfg(), rpc_factory=lambda **_: FakeRpc())
    first.handle_message = AsyncMock()
    await first._handle_notification(
        "message", {"message": message(is_from_me=True, guid="self")}
    )
    first.handle_message.assert_not_awaited()

    await first._handle_notification("message", {"message": message()})
    first.handle_message.assert_awaited_once()

    restarted = ImsgAdapter(cfg(), rpc_factory=lambda **_: FakeRpc())
    restarted.handle_message = AsyncMock()
    await restarted._handle_notification("message", {"message": message()})
    restarted.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_ordered_consumer_retries_failed_row_before_advancing(
    state_path, monkeypatch
):
    monkeypatch.setattr(imsg, "_INBOUND_RETRY_BASE_SECONDS", 0.001)
    adapter = ImsgAdapter(cfg(), rpc_factory=lambda **_: FakeRpc())
    seen = []

    async def handle(event):
        seen.append(event.message_id)
        if len(seen) == 1:
            raise RuntimeError("transient")

    adapter.handle_message = handle
    consumer = asyncio.create_task(adapter._consume_notifications())
    adapter._on_notification("message", {"message": message(guid="first", id=101)})
    adapter._on_notification("message", {"message": message(guid="second", id=102)})
    await asyncio.wait_for(adapter._notification_queue.join(), timeout=1)
    consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consumer

    assert seen == ["first", "first", "second"]
    assert json.loads(state_path.read_text(encoding="utf-8"))["last_rowid"] == 102


@pytest.mark.asyncio
async def test_rejected_gateway_admission_does_not_advance_checkpoint(state_path):
    adapter = ImsgAdapter(cfg(), rpc_factory=lambda **_: FakeRpc())
    adapter._message_handler = AsyncMock()
    adapter._start_session_processing = MagicMock(return_value=False)

    with pytest.raises(RuntimeError, match="did not accept"):
        await adapter._handle_notification("message", {"message": message()})

    assert not state_path.exists()


@pytest.mark.asyncio
async def test_missing_gateway_handler_does_not_advance_checkpoint(state_path):
    adapter = ImsgAdapter(cfg(), rpc_factory=lambda **_: FakeRpc())

    with pytest.raises(RuntimeError, match="handler is not installed"):
        await adapter._handle_notification("message", {"message": message()})

    assert not state_path.exists()


@pytest.mark.asyncio
async def test_active_base_dispatch_without_acceptance_marker_advances_checkpoint(
    state_path,
):
    adapter = ImsgAdapter(cfg(), rpc_factory=lambda **_: FakeRpc())
    adapter._message_handler = AsyncMock(return_value=None)
    adapter._event_session_key = MagicMock(return_value="active-session")
    adapter._active_sessions["active-session"] = asyncio.Event()

    await adapter._handle_notification("message", {"message": message(text="/help")})

    adapter._message_handler.assert_awaited_once()
    assert json.loads(state_path.read_text(encoding="utf-8"))["last_rowid"] == 101


@pytest.mark.asyncio
async def test_active_queue_debounce_without_acceptance_marker_advances_checkpoint(
    state_path,
):
    adapter = ImsgAdapter(cfg(), rpc_factory=lambda **_: FakeRpc())
    adapter._message_handler = AsyncMock(return_value=None)
    adapter._event_session_key = MagicMock(return_value="active-session")
    adapter._active_sessions["active-session"] = asyncio.Event()
    adapter._busy_text_mode = "queue"
    adapter._busy_text_debounce_seconds = 60
    adapter._busy_text_hard_cap_seconds = 60

    await adapter._handle_notification("message", {"message": message()})

    assert "active-session" in adapter._text_debounce_store()
    assert json.loads(state_path.read_text(encoding="utf-8"))["last_rowid"] == 101
    adapter._discard_text_debounce("active-session")


@pytest.mark.asyncio
async def test_group_policy_requires_allowed_chat_sender_and_mention(state_path):
    history = [
        message(id=i, guid=f"h-{i}", sender=f"member-{i}", text=f"line {i}")
        for i in range(1, 110)
    ]
    rpc = FakeRpc(history=history)
    adapter = ImsgAdapter(cfg(), rpc_factory=lambda **_: rpc)
    adapter._rpc = rpc
    adapter.handle_message = AsyncMock()

    base = dict(is_group=True, sender="+15550002222", text="general chat")
    await adapter._handle_notification(
        "message", {"message": message(guid="no-mention", **base)}
    )
    adapter.handle_message.assert_not_awaited()

    await adapter._handle_notification(
        "message",
        {
            "message": message(
                id=102,
                guid="mentioned",
                is_group=True,
                sender="+15550002222",
                text="Jarvis, help",
            )
        },
    )
    assert adapter.handle_message.await_args is not None
    event = adapter.handle_message.await_args.args[0]
    assert event.source.chat_type == "group"
    assert event.source.chat_id == "42"
    assert event.channel_context is not None
    assert "[member-1] line 1\n" not in event.channel_context
    assert "line 109" in event.channel_context
    history_call = next(
        params for method, params in rpc.calls if method == "messages.history"
    )
    assert history_call == {"chat_id": 42, "limit": 101, "attachments": False}

    adapter.handle_message.reset_mock()
    await adapter._handle_notification(
        "message",
        {
            "message": message(
                id=103,
                guid="wrong-sender",
                is_group=True,
                sender="+15559999999",
                text="Jarvis help",
            )
        },
    )
    await adapter._handle_notification(
        "message",
        {
            "message": message(
                id=104,
                guid="wrong-chat",
                chat_id=99,
                is_group=True,
                sender="+15550002222",
                text="Jarvis help",
            )
        },
    )
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_send_replies_to_exact_chat_and_supports_media(state_path, tmp_path):
    rpc = FakeRpc()
    adapter = ImsgAdapter(cfg(), rpc_factory=lambda **_: rpc)
    adapter._rpc = rpc

    sent = await adapter.send("42", "**done**", reply_to="in-guid")
    assert sent.success is True
    assert sent.message_id == "out-guid"
    assert (
        "send",
        {
            "chat_id": 42,
            "text": "done",
            "service": "auto",
            "transport": "auto",
            "reply_to": "in-guid",
        },
    ) in rpc.calls

    media = tmp_path / "report.pdf"
    media.write_bytes(b"pdf")
    delivered = await adapter.send_media("42", "report", str(media), reply_to="in-guid")
    assert delivered.success is True
    assert (
        "send",
        {
            "chat_id": 42,
            "text": "report",
            "service": "auto",
            "transport": "auto",
            "reply_to": "in-guid",
            "file": str(media),
        },
    ) in rpc.calls


@pytest.mark.asyncio
async def test_standalone_send_delivers_every_attachment(monkeypatch, tmp_path):
    rpc = FakeRpc()
    monkeypatch.setattr(imsg, "ImsgRpcClient", lambda **_: rpc)
    first, second = tmp_path / "one.jpg", tmp_path / "two.jpg"
    first.write_bytes(b"one")
    second.write_bytes(b"two")

    result = await imsg._standalone_send(
        cfg(), "42", "caption", media_files=[str(first), str(second)]
    )

    assert result["success"] is True
    sends = [params for method, params in rpc.calls if method == "send"]
    assert [params["file"] for params in sends] == [str(first), str(second)]
    assert [params["text"] for params in sends] == ["caption", ""]


def test_registration_and_config_defaults(monkeypatch):
    ctx = SimpleNamespace(register_platform=MagicMock())
    imsg.register(ctx)
    kwargs = ctx.register_platform.call_args.kwargs
    assert kwargs["name"] == "imessage"
    assert kwargs["allowed_users_env"] == "IMESSAGE_ALLOWED_USERS"
    assert kwargs["pii_safe"] is True

    adapter = ImsgAdapter(PlatformConfig(enabled=True, extra={}))
    assert adapter.cli_path == "/opt/homebrew/bin/imsg"
    assert adapter.history_limit == 100
    assert adapter.require_mention is True


def test_config_validation_requires_explicit_imessage_opt_in(monkeypatch):
    monkeypatch.setattr(imsg, "check_requirements", lambda: True)

    transient = PlatformConfig(enabled=True, extra={})
    assert imsg.validate_config(transient) is False
    assert imsg.is_connected(transient) is False

    yaml_seed = imsg._apply_yaml_config({}, {"enabled": True})
    configured = PlatformConfig(enabled=True, extra=yaml_seed)
    assert yaml_seed["_configured"] is True
    assert imsg.validate_config(configured) is True
    assert imsg.is_connected(configured) is True


def test_real_plugin_enablement_path_requires_opt_in(monkeypatch):
    for name in (
        "IMESSAGE_ALLOWED_USERS",
        "IMESSAGE_DM_ALLOWED_USERS",
        "IMESSAGE_GROUP_ALLOWED_USERS",
        "IMESSAGE_CLI_PATH",
        "IMESSAGE_DB_PATH",
        "IMESSAGE_HOME_CHANNEL",
        "IMESSAGE_HISTORY_LIMIT",
        "IMESSAGE_REQUIRE_MENTION",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(imsg, "check_requirements", lambda: True)
    entry = SimpleNamespace(
        name="imessage",
        env_enablement_fn=imsg._env_enablement,
        is_connected=imsg.is_connected,
        check_fn=lambda: True,
        ensure_deps_fn=None,
    )

    unconfigured = GatewayConfig()
    _enable_plugin_platform(unconfigured, entry)
    assert Platform("imessage") not in unconfigured.platforms

    monkeypatch.setenv("IMESSAGE_ALLOWED_USERS", "opted-in-user")
    configured = GatewayConfig()
    _enable_plugin_platform(configured, entry)
    platform_cfg = configured.platforms[Platform("imessage")]
    assert platform_cfg.enabled is True
    assert platform_cfg.extra["_configured"] is True
    assert platform_cfg.extra["dm_allowed_users"] == ["opted-in-user"]


def test_yaml_bridge_unions_dm_and_group_allowlists(monkeypatch):
    for name in (
        "IMESSAGE_ALLOWED_USERS",
        "IMESSAGE_DM_ALLOWED_USERS",
        "IMESSAGE_GROUP_ALLOWED_USERS",
        "IMESSAGE_ALLOWED_CHATS",
    ):
        monkeypatch.delenv(name, raising=False)
    seeded = imsg._apply_yaml_config(
        {},
        {
            "extra": {
                "dm_allowed_users": ["dm-user"],
                "group_allowed_users": ["group-user"],
                "allowed_chats": ["42"],
            }
        },
    )
    assert seeded["allowed_users"] == ["dm-user", "group-user"]
    assert seeded["dm_allowed_users"] == ["dm-user"]
    assert seeded["group_allowed_users"] == ["group-user"]
    assert seeded["allowed_chats"] == ["42"]


@pytest.mark.parametrize(
    ("extra", "dm_allowed", "group_allowed"),
    [
        ({"dm_allowed_users": ["dm-user"]}, True, False),
        ({"group_allowed_users": ["group-user"]}, False, True),
        ({"dm_allowed_users": [], "group_allowed_users": []}, False, False),
        ({"allowed_users": []}, False, False),
    ],
)
def test_explicit_allowlist_policy_configures_both_scopes(
    state_path, monkeypatch, extra, dm_allowed, group_allowed
):
    for name in (
        "IMESSAGE_ALLOWED_USERS",
        "IMESSAGE_DM_ALLOWED_USERS",
        "IMESSAGE_GROUP_ALLOWED_USERS",
    ):
        monkeypatch.delenv(name, raising=False)
    adapter = ImsgAdapter(PlatformConfig(enabled=True, extra=extra))

    assert adapter._sender_allowed(message(sender="dm-user")) is dm_allowed
    assert (
        adapter._sender_allowed(message(sender="group-user", is_group=True))
        is group_allowed
    )


def test_yaml_single_scoped_allowlist_seeds_missing_scope_as_deny_all(monkeypatch):
    for name in (
        "IMESSAGE_ALLOWED_USERS",
        "IMESSAGE_DM_ALLOWED_USERS",
        "IMESSAGE_GROUP_ALLOWED_USERS",
    ):
        monkeypatch.delenv(name, raising=False)

    seeded = imsg._apply_yaml_config({}, {"extra": {"dm_allowed_users": ["dm-user"]}})

    assert seeded["dm_allowed_users"] == ["dm-user"]
    assert seeded["group_allowed_users"] == []


def test_env_enablement_requires_opt_in_and_keeps_chat_allowlists_separate(monkeypatch):
    for name in (
        "IMESSAGE_ALLOWED_USERS",
        "IMESSAGE_DM_ALLOWED_USERS",
        "IMESSAGE_GROUP_ALLOWED_USERS",
        "IMESSAGE_ALLOWED_CHATS",
        "IMESSAGE_CLI_PATH",
        "IMESSAGE_DB_PATH",
        "IMESSAGE_HOME_CHANNEL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(imsg, "check_requirements", lambda: True)
    monkeypatch.setattr(
        imsg, "_get_scoped_secret", lambda _name, default=None, **_: default
    )
    assert imsg._env_enablement() is None

    def scoped(name, default=None, **_):
        return {
            "IMESSAGE_CLI_PATH": "/opt/homebrew/bin/imsg",
            "IMESSAGE_DM_ALLOWED_USERS": "dm-only",
        }.get(name, default)

    monkeypatch.setattr(imsg, "_get_scoped_secret", scoped)
    monkeypatch.setenv("IMESSAGE_CLI_PATH", "/opt/homebrew/bin/imsg")
    monkeypatch.setenv("IMESSAGE_DM_ALLOWED_USERS", "dm-only")
    seeded = imsg._env_enablement()
    assert seeded["dm_allowed_users"] == ["dm-only"]
    assert seeded["group_allowed_users"] == []
    adapter = ImsgAdapter(PlatformConfig(enabled=True, extra=seeded))
    assert adapter._sender_allowed(message(sender="dm-only")) is True
    assert adapter._sender_allowed(message(sender="dm-only", is_group=True)) is False


def test_env_enablement_seeds_legacy_policy_and_mention_setting(monkeypatch):
    monkeypatch.setattr(imsg, "check_requirements", lambda: True)
    monkeypatch.delenv("IMESSAGE_DM_ALLOWED_USERS", raising=False)
    monkeypatch.delenv("IMESSAGE_GROUP_ALLOWED_USERS", raising=False)
    monkeypatch.setenv("IMESSAGE_ALLOWED_USERS", "legacy-one,legacy-two")
    monkeypatch.setenv("IMESSAGE_REQUIRE_MENTION", "false")

    seeded = imsg._env_enablement()

    assert seeded["_configured"] is True
    assert seeded["allowed_users"] == ["legacy-one", "legacy-two"]
    assert seeded["dm_allowed_users"] == ["legacy-one", "legacy-two"]
    assert seeded["group_allowed_users"] == ["legacy-one", "legacy-two"]
    assert seeded["require_mention"] is False


def test_env_policy_and_mention_override_yaml(monkeypatch):
    monkeypatch.delenv("IMESSAGE_DM_ALLOWED_USERS", raising=False)
    monkeypatch.delenv("IMESSAGE_GROUP_ALLOWED_USERS", raising=False)
    monkeypatch.setenv("IMESSAGE_ALLOWED_USERS", "env-user")
    monkeypatch.setenv("IMESSAGE_REQUIRE_MENTION", "false")
    seeded = imsg._apply_yaml_config(
        {},
        {
            "extra": {
                "dm_allowed_users": ["yaml-user"],
                "require_mention": True,
            }
        },
    )

    adapter = ImsgAdapter(PlatformConfig(enabled=True, extra=seeded))

    assert adapter.require_mention is False
    assert adapter._sender_allowed(message(sender="env-user")) is True
    assert adapter._sender_allowed(message(sender="yaml-user")) is False


def test_rpc_error_normalizes_full_disk_access_without_leaking_diagnostics():
    error = {
        "message": "Internal error",
        "data": "database permission denied: Full Disk Access required for chat.db",
    }
    assert imsg._rpc_error_message(error) == (
        "imsg cannot access Messages chat.db; grant Full Disk Access to the Hermes gateway launcher"
    )
