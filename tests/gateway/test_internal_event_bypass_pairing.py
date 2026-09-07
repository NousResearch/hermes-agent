"""Tests that internal synthetic events (e.g. background process completion)
bypass user authorization and do not trigger DM pairing.

Regression test for the bug where ``_run_process_watcher`` with
``notify_on_complete=True`` injected a ``MessageEvent`` without ``user_id``,
causing ``_is_user_authorized`` to reject it and the gateway to send a
pairing code to the chat.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from tools.process_registry import ProcessRegistry, ProcessSession


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


def _build_runner(monkeypatch, tmp_path) -> GatewayRunner:
    """Create a GatewayRunner with notifications set to 'all'."""
    (tmp_path / "config.yaml").write_text(
        "display:\n  background_process_notifications: all\n",
        encoding="utf-8",
    )

    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    runner = GatewayRunner(GatewayConfig())
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
async def test_internal_event_bypasses_authorization(monkeypatch, tmp_path):
    """An internal event should skip _is_user_authorized entirely."""
    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    (tmp_path / "config.yaml").write_text("", encoding="utf-8")

    runner = GatewayRunner(GatewayConfig())

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
    )

    # Track if _is_user_authorized is called
    auth_called = False
    original_auth = GatewayRunner._is_user_authorized

    def tracking_auth(self, src):
        nonlocal auth_called
        auth_called = True
        return original_auth(self, src)

    monkeypatch.setattr(GatewayRunner, "_is_user_authorized", tracking_auth)

    # Stop execution before the agent runner so the test doesn't block in
    # run_in_executor.  Auth check happens before _handle_message_with_agent.
    async def _raise(*_a, **_kw):
        raise RuntimeError("sentinel — stop here")
    monkeypatch.setattr(GatewayRunner, "_handle_message_with_agent", _raise)

    try:
        await runner._handle_message(event)
    except RuntimeError:
        pass  # Expected sentinel

    assert not auth_called, (
        "_is_user_authorized should NOT be called for internal events"
    )


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

    runner = GatewayRunner(GatewayConfig())
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
    assert event.source.platform == Platform.TELEGRAM
    assert event.source.chat_id == "-100"
    assert event.source.chat_type == "group"
    assert event.source.thread_id == "42"
    assert event.source.user_id == "user-42"
    assert event.source.user_name == "alice"


@pytest.mark.asyncio
async def test_none_user_id_does_not_generate_pairing_code(monkeypatch, tmp_path):
    """A message with user_id=None must never call generate_code."""
    import gateway.run as gateway_run

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    (tmp_path / "config.yaml").write_text("", encoding="utf-8")

    runner = GatewayRunner(GatewayConfig())
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


