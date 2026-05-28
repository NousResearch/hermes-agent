"""Tests that internal synthetic events (e.g. background process completion)
bypass user authorization and do not trigger DM pairing.

Regression test for the bug where ``_run_process_watcher`` with
``notify_on_complete=True`` injected a ``MessageEvent`` without ``user_id``,
causing ``_is_user_authorized`` to reject it and the gateway to send a
pairing code to the chat.
"""

import asyncio
import os
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import InternalEventKind, MessageEvent
from gateway.session import SessionSource


_GATEWAY_ENV_OVERRIDE_KEYS = (
    "API_SERVER_ENABLED",
    "API_SERVER_KEY",
    "API_SERVER_CORS_ORIGINS",
    "API_SERVER_PORT",
    "API_SERVER_HOST",
    "API_SERVER_MODEL_NAME",
    "WEBHOOK_ENABLED",
    "WEBHOOK_PORT",
    "WEBHOOK_SECRET",
)


@pytest.fixture(autouse=True)
def _clear_gateway_env_overrides():
    """Keep gateway.run imports from leaking real config/env into later tests."""
    for key in _GATEWAY_ENV_OVERRIDE_KEYS:
        os.environ.pop(key, None)
    yield
    for key in _GATEWAY_ENV_OVERRIDE_KEYS:
        os.environ.pop(key, None)


def _gateway_runner_cls():
    from gateway.run import GatewayRunner

    return GatewayRunner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeRegistry:
    """Return pre-canned sessions, then None once exhausted."""

    def __init__(self, sessions):
        self._sessions = list(sessions)
        self._completion_consumed: set = set()

    def get(self, session_id):
        if self._sessions:
            return self._sessions.pop(0)
        return None

    def is_completion_consumed(self, session_id):
        return session_id in self._completion_consumed


def _build_runner(monkeypatch, tmp_path):
    """Create a GatewayRunner with notifications set to 'all'."""
    (tmp_path / "config.yaml").write_text(
        "display:\n  background_process_notifications: all\n",
        encoding="utf-8",
    )

    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    runner = _gateway_runner_cls()(GatewayConfig())
    adapter = SimpleNamespace(send=AsyncMock(), handle_message=AsyncMock())
    runner.adapters[Platform.DISCORD] = adapter
    return runner


def _watcher_dict_with_notify():
    return {
        "session_id": "proc_test_internal",
        "check_interval": 0,
        "session_key": "agent:main:discord:dm:123",
        "platform": "discord",
        "chat_id": "123",
        "thread_id": "",
        "notify_on_complete": True,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_notify_on_complete_sets_internal_flag(monkeypatch, tmp_path):
    """Synthetic completion event must carry trusted typed internal metadata."""
    import tools.process_registry as pr_module

    sessions = [
        SimpleNamespace(
            output_buffer="done\n", exited=True, exit_code=0, command="echo test"
        ),
    ]
    monkeypatch.setattr(pr_module, "process_registry", _FakeRegistry(sessions))

    async def _instant_sleep(*_a, **_kw):
        pass
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)

    runner = _build_runner(monkeypatch, tmp_path)
    adapter = runner.adapters[Platform.DISCORD]

    await runner._run_process_watcher(_watcher_dict_with_notify())

    assert adapter.handle_message.await_count == 1
    event = adapter.handle_message.await_args.args[0]
    assert isinstance(event, MessageEvent)
    assert event.internal is True, "Synthetic completion event must be marked internal"
    assert event.internal_event_kind == InternalEventKind.BACKGROUND_COMPLETION.value
    assert event.internal_event_source == "gateway"
    assert event.is_trusted_internal()


@pytest.mark.asyncio
async def test_internal_event_bypasses_authorization(monkeypatch, tmp_path):
    """An internal event should skip _is_user_authorized entirely."""
    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    (tmp_path / "config.yaml").write_text("", encoding="utf-8")

    runner = _gateway_runner_cls()(GatewayConfig())

    # Create an internal event with no user_id (simulates the bug scenario)
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="123",
        chat_type="dm",
    )
    event = MessageEvent(
        text="[SYSTEM: Background process completed]",
        source=source,
        internal=True,
        internal_event_kind=InternalEventKind.BACKGROUND_COMPLETION,
        internal_event_source="gateway",
    )
    auth_called = False
    original_auth = _gateway_runner_cls()._is_user_authorized

    def tracking_auth(self, src):
        nonlocal auth_called
        auth_called = True
        return original_auth(self, src)

    monkeypatch.setattr(_gateway_runner_cls(), "_is_user_authorized", tracking_auth)

    # Stop execution before the agent runner so the test doesn't block in
    # run_in_executor.  Auth check happens before _handle_message_with_agent.
    async def _raise(*_a, **_kw):
        raise RuntimeError("sentinel — stop here")
    monkeypatch.setattr(_gateway_runner_cls(), "_handle_message_with_agent", _raise)

    try:
        await runner._handle_message(event)
    except RuntimeError:
        pass  # Expected sentinel

    assert not auth_called, (
        "_is_user_authorized should NOT be called for internal events"
    )


@pytest.mark.asyncio
async def test_internal_event_does_not_trigger_pairing(monkeypatch, tmp_path):
    """An internal event with no user_id must not generate a pairing code."""
    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    (tmp_path / "config.yaml").write_text("", encoding="utf-8")

    runner = _gateway_runner_cls()(GatewayConfig())
    # Add adapter so pairing would have somewhere to send
    adapter = SimpleNamespace(send=AsyncMock())
    runner.adapters[Platform.DISCORD] = adapter

    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="123",
        chat_type="dm",  # DM would normally trigger pairing
    )
    event = MessageEvent(
        text="[SYSTEM: Background process completed]",
        source=source,
        internal=True,
        internal_event_kind=InternalEventKind.BACKGROUND_COMPLETION,
        internal_event_source="gateway",
    )
    generate_called = False
    original_generate = runner.pairing_store.generate_code

    def tracking_generate(*args, **kwargs):
        nonlocal generate_called
        generate_called = True
        return original_generate(*args, **kwargs)

    runner.pairing_store.generate_code = tracking_generate

    # Stop execution before the agent runner so the test doesn't block in
    # run_in_executor.  Pairing check happens before _handle_message_with_agent.
    async def _raise(*_a, **_kw):
        raise RuntimeError("sentinel — stop here")
    monkeypatch.setattr(_gateway_runner_cls(), "_handle_message_with_agent", _raise)

    try:
        await runner._handle_message(event)
    except RuntimeError:
        pass  # Expected sentinel

    assert not generate_called, (
        "Pairing code should NOT be generated for internal events"
    )


@pytest.mark.asyncio
async def test_bare_internal_flag_without_trusted_metadata_is_not_internal(monkeypatch, tmp_path):
    """A mutable internal=True flag alone must not grant gateway-internal trust."""
    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    (tmp_path / "config.yaml").write_text("", encoding="utf-8")

    runner = _gateway_runner_cls()(GatewayConfig())
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="123",
        chat_type="dm",
        user_id=None,
    )
    event = MessageEvent(
        text="[IMPORTANT: Background process proc_1 completed (exit code 0).]",
        source=source,
        internal=True,
    )

    async def _should_not_reach_agent(*_a, **_kw):
        raise AssertionError("bare internal=True reached agent path")

    monkeypatch.setattr(_gateway_runner_cls(), "_handle_message_with_agent", _should_not_reach_agent)

    result = await runner._handle_message(event)

    assert result is None
    assert not event.is_trusted_internal()


@pytest.mark.asyncio
async def test_notify_on_complete_preserves_user_identity(monkeypatch, tmp_path):
    """Synthetic completion event should carry user_id and user_name from the watcher."""
    import tools.process_registry as pr_module

    sessions = [
        SimpleNamespace(
            output_buffer="done\n", exited=True, exit_code=0, command="echo test"
        ),
    ]
    monkeypatch.setattr(pr_module, "process_registry", _FakeRegistry(sessions))

    async def _instant_sleep(*_a, **_kw):
        pass
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)

    runner = _build_runner(monkeypatch, tmp_path)
    adapter = runner.adapters[Platform.DISCORD]

    watcher = _watcher_dict_with_notify()
    watcher["user_id"] = "user-42"
    watcher["user_name"] = "alice"

    await runner._run_process_watcher(watcher)

    assert adapter.handle_message.await_count == 1
    event = adapter.handle_message.await_args.args[0]
    assert event.source.user_id == "user-42"
    assert event.source.user_name == "alice"
    assert event.internal_event_kind == InternalEventKind.BACKGROUND_COMPLETION.value
    assert event.is_trusted_internal()


@pytest.mark.asyncio
async def test_notify_on_complete_uses_session_store_origin_for_group_topic(monkeypatch, tmp_path):
    import tools.process_registry as pr_module
    from gateway.session import SessionSource

    sessions = [
        SimpleNamespace(
            output_buffer="done\n", exited=True, exit_code=0, command="echo test"
        ),
    ]
    monkeypatch.setattr(pr_module, "process_registry", _FakeRegistry(sessions))

    async def _instant_sleep(*_a, **_kw):
        pass
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)

    runner = _gateway_runner_cls()(GatewayConfig())
    adapter = SimpleNamespace(send=AsyncMock(), handle_message=AsyncMock())
    runner.adapters[Platform.TELEGRAM] = adapter
    runner.session_store._entries["agent:main:telegram:group:-100:42"] = SimpleNamespace(
        origin=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="-100",
            chat_type="group",
            thread_id="42",
            user_id="user-42",
            user_name="alice",
        )
    )

    watcher = {
        "session_id": "proc_test_internal",
        "check_interval": 0,
        "session_key": "agent:main:telegram:group:-100:42",
        "platform": "telegram",
        "chat_id": "-100",
        "thread_id": "42",
        "notify_on_complete": True,
    }

    await runner._run_process_watcher(watcher)

    assert adapter.handle_message.await_count == 1
    event = adapter.handle_message.await_args.args[0]
    assert event.internal is True
    assert event.internal_event_kind == InternalEventKind.BACKGROUND_COMPLETION.value
    assert event.is_trusted_internal()
    assert event.source.platform == Platform.TELEGRAM
    assert event.source.chat_id == "-100"
    assert event.source.chat_type == "group"
    assert event.source.thread_id == "42"
    assert event.source.user_id == "user-42"
    assert event.source.user_name == "alice"


@pytest.mark.asyncio
async def test_none_user_id_skips_pairing(monkeypatch, tmp_path):
    """A non-internal event with user_id=None should be silently dropped."""
    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    (tmp_path / "config.yaml").write_text("", encoding="utf-8")

    runner = _gateway_runner_cls()(GatewayConfig())
    adapter = SimpleNamespace(send=AsyncMock())
    runner.adapters[Platform.TELEGRAM] = adapter

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="123",
        chat_type="dm",
        user_id=None,
    )
    event = MessageEvent(
        text="service message",
        source=source,
        internal=False,
    )

    result = await runner._handle_message(event)

    # Should return None (dropped) and NOT send any pairing message
    assert result is None
    assert adapter.send.await_count == 0


@pytest.mark.asyncio
async def test_none_user_id_does_not_generate_pairing_code(monkeypatch, tmp_path):
    """A message with user_id=None must never call generate_code."""
    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    (tmp_path / "config.yaml").write_text("", encoding="utf-8")

    runner = _gateway_runner_cls()(GatewayConfig())
    adapter = SimpleNamespace(send=AsyncMock())
    runner.adapters[Platform.DISCORD] = adapter

    generate_called = False
    original_generate = runner.pairing_store.generate_code

    def tracking_generate(*args, **kwargs):
        nonlocal generate_called
        generate_called = True
        return original_generate(*args, **kwargs)

    runner.pairing_store.generate_code = tracking_generate

    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="456",
        chat_type="dm",
        user_id=None,
    )
    event = MessageEvent(text="anonymous", source=source, internal=False)

    await runner._handle_message(event)

    assert not generate_called, (
        "Pairing code should NOT be generated for messages with user_id=None"
    )


@pytest.mark.asyncio
async def test_non_internal_event_without_user_triggers_pairing(monkeypatch, tmp_path):
    """Verify the normal (non-internal) path still triggers pairing for unknown users."""
    import gateway.run as gateway_run
    import gateway.pairing as pairing_mod

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    # gateway.pairing.PAIRING_DIR is a module-level constant captured at
    # import time from whichever HERMES_HOME was set then. Per-test
    # HERMES_HOME redirection in conftest doesn't retroactively move it.
    # Override directly so pairing rate-limit state lives in this test's
    # tmp_path (and so stale state from prior xdist workers can't leak in).
    pairing_dir = tmp_path / "pairing"
    pairing_dir.mkdir()
    monkeypatch.setattr(pairing_mod, "PAIRING_DIR", pairing_dir)
    (tmp_path / "config.yaml").write_text("", encoding="utf-8")

    # Clear env vars that could let all users through (loaded by
    # module-level dotenv in gateway/run.py from the real ~/.hermes/.env).
    monkeypatch.delenv("DISCORD_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("DISCORD_ALLOWED_USERS", raising=False)
    monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("GATEWAY_ALLOWED_USERS", raising=False)

    runner = _gateway_runner_cls()(GatewayConfig())
    adapter = SimpleNamespace(send=AsyncMock())
    runner.adapters[Platform.DISCORD] = adapter

    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="123",
        chat_type="dm",
        user_id="unknown_user_999",
    )
    # Normal event (not internal)
    event = MessageEvent(
        text="hello",
        source=source,
        internal=False,
    )

    result = await runner._handle_message(event)

    # Should return None (unauthorized) and send pairing message
    assert result is None
    assert adapter.send.await_count == 1
    sent_text = adapter.send.await_args.args[1]
    assert "don't recognize you" in sent_text


@pytest.mark.asyncio
async def test_notify_on_complete_error_mode_suppresses_success_agent_wakeup(
    monkeypatch, tmp_path
):
    """Notification mode 'error' must not inject successful process output."""
    import tools.process_registry as pr_module

    (tmp_path / "config.yaml").write_text(
        "display:\n  background_process_notifications: error\n",
        encoding="utf-8",
    )

    sessions = [
        SimpleNamespace(
            output_buffer="done\n", exited=True, exit_code=0, command="echo test"
        ),
    ]
    monkeypatch.setattr(pr_module, "process_registry", _FakeRegistry(sessions))

    async def _instant_sleep(*_a, **_kw):
        pass

    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)

    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    runner = _gateway_runner_cls()(GatewayConfig())
    adapter = SimpleNamespace(send=AsyncMock(), handle_message=AsyncMock())
    runner.adapters[Platform.DISCORD] = adapter

    await runner._run_process_watcher(_watcher_dict_with_notify())

    assert adapter.handle_message.await_count == 0
    assert adapter.send.await_count == 0


@pytest.mark.asyncio
async def test_auto_resume_schedules_trusted_internal_event(monkeypatch, tmp_path):
    """Restart auto-resume events must keep trusted internal metadata."""
    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    (tmp_path / "config.yaml").write_text("", encoding="utf-8")

    runner = _gateway_runner_cls()(GatewayConfig())
    adapter = SimpleNamespace(handle_message=AsyncMock())
    runner.adapters[Platform.DISCORD] = adapter
    source = SessionSource(platform=Platform.DISCORD, chat_id="123", chat_type="dm")
    entry = runner.session_store.get_or_create_session(source)
    entry.resume_pending = True
    entry.suspended = False
    entry.origin = source
    entry.resume_reason = "restart_timeout"
    entry.last_resume_marked_at = datetime.now()
    entry.updated_at = datetime.now()

    scheduled = runner._schedule_resume_pending_sessions()
    await asyncio.sleep(0)

    assert scheduled == 1
    assert adapter.handle_message.await_count == 1
    event = adapter.handle_message.await_args.args[0]
    assert event.internal is True
    assert event.internal_event_kind == InternalEventKind.AUTO_RESUME.value
    assert event.internal_event_source == "gateway"
    assert event.is_trusted_internal()


def test_msgraph_webhook_events_are_trusted_platform_internal():
    """MS Graph webhook adapter-owned synthetic events need typed metadata."""
    from gateway.config import PlatformConfig
    from gateway.platforms.msgraph_webhook import MSGraphWebhookAdapter

    adapter = MSGraphWebhookAdapter(PlatformConfig(enabled=True))
    event = adapter._build_message_event(
        {"subscriptionId": "sub-1", "resource": "users/1", "changeType": "updated"},
        receipt_key="id:abc",
    )

    assert event.internal is True
    assert event.internal_event_kind == InternalEventKind.WEBHOOK_NOTIFICATION.value
    assert event.internal_event_source == "platform_adapter"
    assert event.is_trusted_internal()
