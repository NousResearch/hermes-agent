import asyncio
import os

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner
from gateway.session import SessionContext, SessionSource
from gateway.session_context import (
    get_session_env,
    set_session_vars,
    clear_session_vars,
    set_current_session_id,
    _VAR_MAP,
    _UNSET,
)


@pytest.fixture(autouse=True)
def _reset_contextvars():
    """Reset all session contextvars to _UNSET between tests.

    In production each asyncio.Task gets a fresh context copy where the
    defaults are _UNSET.  In tests all functions share the same thread
    context, so a clear_session_vars() from test A (which sets vars to "")
    would leak into test B.  This fixture ensures each test starts clean.
    """
    yield
    for var in _VAR_MAP.values():
        # Can't use var.reset() without a token; just set back to sentinel.
        var.set(_UNSET)


def test_set_session_env_sets_contextvars(monkeypatch):
    """_set_session_env should populate contextvars, not os.environ."""
    runner = object.__new__(GatewayRunner)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_name="Group",
        chat_type="group",
        user_id="123456",
        user_name="alice",
        thread_id="17585",
    )
    context = SessionContext(source=source, connected_platforms=[], home_channels={})

    monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
    monkeypatch.delenv("HERMES_SESSION_CHAT_ID", raising=False)
    monkeypatch.delenv("HERMES_SESSION_CHAT_NAME", raising=False)
    monkeypatch.delenv("HERMES_SESSION_USER_ID", raising=False)
    monkeypatch.delenv("HERMES_SESSION_USER_NAME", raising=False)
    monkeypatch.delenv("HERMES_SESSION_THREAD_ID", raising=False)
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)

    context.session_id = "session-abc123"
    tokens = runner._set_session_env(context)

    # Values should be readable via get_session_env (contextvar path)
    assert get_session_env("HERMES_SESSION_PLATFORM") == "telegram"
    assert get_session_env("HERMES_SESSION_CHAT_ID") == "-1001"
    assert get_session_env("HERMES_SESSION_CHAT_NAME") == "Group"
    assert get_session_env("HERMES_SESSION_USER_ID") == "123456"
    assert get_session_env("HERMES_SESSION_USER_NAME") == "alice"
    assert get_session_env("HERMES_SESSION_THREAD_ID") == "17585"
    assert get_session_env("HERMES_SESSION_ID") == "session-abc123"

    # os.environ should NOT be touched
    assert os.getenv("HERMES_SESSION_PLATFORM") is None
    assert os.getenv("HERMES_SESSION_THREAD_ID") is None
    assert os.getenv("HERMES_SESSION_ID") is None

    # Clean up
    runner._clear_session_env(tokens)


def test_clear_session_env_restores_previous_state(monkeypatch):
    """_clear_session_env should restore contextvars to their pre-handler values."""
    runner = object.__new__(GatewayRunner)

    monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
    monkeypatch.delenv("HERMES_SESSION_CHAT_ID", raising=False)
    monkeypatch.delenv("HERMES_SESSION_CHAT_NAME", raising=False)
    monkeypatch.delenv("HERMES_SESSION_USER_ID", raising=False)
    monkeypatch.delenv("HERMES_SESSION_USER_NAME", raising=False)
    monkeypatch.delenv("HERMES_SESSION_THREAD_ID", raising=False)

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_name="Group",
        chat_type="group",
        user_id="123456",
        user_name="alice",
        thread_id="17585",
    )
    context = SessionContext(source=source, connected_platforms=[], home_channels={})

    tokens = runner._set_session_env(context)
    assert get_session_env("HERMES_SESSION_PLATFORM") == "telegram"
    assert get_session_env("HERMES_SESSION_USER_ID") == "123456"

    runner._clear_session_env(tokens)

    # After clear, contextvars should return to defaults (empty)
    assert get_session_env("HERMES_SESSION_PLATFORM") == ""
    assert get_session_env("HERMES_SESSION_CHAT_ID") == ""
    assert get_session_env("HERMES_SESSION_CHAT_NAME") == ""
    assert get_session_env("HERMES_SESSION_USER_ID") == ""
    assert get_session_env("HERMES_SESSION_USER_NAME") == ""
    assert get_session_env("HERMES_SESSION_THREAD_ID") == ""
    assert get_session_env("HERMES_SESSION_ID") == ""


def test_get_session_env_falls_back_to_os_environ(monkeypatch):
    """get_session_env should fall back to os.environ when contextvar is unset."""
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "discord")

    # No contextvar set — should read from os.environ
    assert get_session_env("HERMES_SESSION_PLATFORM") == "discord"

    # Now set a contextvar — should prefer it
    tokens = set_session_vars(platform="telegram")
    assert get_session_env("HERMES_SESSION_PLATFORM") == "telegram"

    # After clear — should return "" (explicitly cleared), NOT fall back
    # to os.environ.  This is the fix for #10304: stale os.environ values
    # must not leak through after a gateway session is cleaned up.
    clear_session_vars(tokens)
    assert get_session_env("HERMES_SESSION_PLATFORM") == ""


def test_get_session_env_default_when_nothing_set(monkeypatch):
    """get_session_env returns default when neither contextvar nor env is set."""
    monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)

    assert get_session_env("HERMES_SESSION_PLATFORM") == ""
    assert get_session_env("HERMES_SESSION_PLATFORM", "fallback") == "fallback"


def test_omitted_session_id_keeps_env_fallback_for_acp(monkeypatch):
    """ACP sets session vars for routing but bridges SESSION_ID through env."""
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)

    tokens = set_session_vars(session_key="acp-session-id")
    try:
        monkeypatch.setenv("HERMES_SESSION_ID", "acp-session-id")
        assert get_session_env("HERMES_SESSION_KEY") == "acp-session-id"
        assert get_session_env("HERMES_SESSION_ID") == "acp-session-id"
    finally:
        clear_session_vars(tokens)

    # After an explicit clear, stale env fallback is suppressed inside this
    # context until the next task/test starts with fresh ContextVars.
    assert get_session_env("HERMES_SESSION_ID") == ""


def test_set_session_env_handles_missing_optional_fields():
    """_set_session_env should handle None chat_name and thread_id gracefully."""
    runner = object.__new__(GatewayRunner)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_name=None,
        chat_type="private",
        thread_id=None,
    )
    context = SessionContext(source=source, connected_platforms=[], home_channels={})

    tokens = runner._set_session_env(context)

    assert get_session_env("HERMES_SESSION_PLATFORM") == "telegram"
    assert get_session_env("HERMES_SESSION_CHAT_ID") == "-1001"
    assert get_session_env("HERMES_SESSION_CHAT_NAME") == ""
    assert get_session_env("HERMES_SESSION_THREAD_ID") == ""

    runner._clear_session_env(tokens)


# ---------------------------------------------------------------------------
# SESSION_KEY contextvars tests
# ---------------------------------------------------------------------------


def test_session_key_set_via_contextvars(monkeypatch):
    """set_session_vars should set HERMES_SESSION_KEY via contextvars."""
    monkeypatch.delenv("HERMES_SESSION_KEY", raising=False)

    tokens = set_session_vars(
        platform="telegram",
        chat_id="-1001",
        session_key="tg:-1001:17585",
    )
    assert get_session_env("HERMES_SESSION_KEY") == "tg:-1001:17585"

    clear_session_vars(tokens)
    assert get_session_env("HERMES_SESSION_KEY") == ""


def test_session_key_falls_back_to_os_environ(monkeypatch):
    """get_session_env for SESSION_KEY should fall back to os.environ."""
    monkeypatch.setenv("HERMES_SESSION_KEY", "env-session-123")

    # No contextvar set — should read from os.environ
    assert get_session_env("HERMES_SESSION_KEY") == "env-session-123"

    # Set contextvar — should prefer it
    tokens = set_session_vars(session_key="ctx-session-456")
    assert get_session_env("HERMES_SESSION_KEY") == "ctx-session-456"

    # After clear — should return "" (explicitly cleared), not os.environ (#10304)
    clear_session_vars(tokens)
    assert get_session_env("HERMES_SESSION_KEY") == ""


def test_set_session_env_includes_session_key():
    """_set_session_env should propagate session_key from SessionContext."""
    runner = object.__new__(GatewayRunner)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_name="Group",
        chat_type="group",
        thread_id="17585",
    )
    context = SessionContext(
        source=source,
        connected_platforms=[],
        home_channels={},
        session_key="tg:-1001:17585",
    )

    # Capture baseline value before setting (may be non-empty from another
    # test in the same pytest-xdist worker sharing the context).
    tokens = runner._set_session_env(context)
    assert get_session_env("HERMES_SESSION_KEY") == "tg:-1001:17585"
    runner._clear_session_env(tokens)
    # After clearing, the session key must not retain the value we just set.
    # The exact post-clear value depends on context propagation from other
    # tests, so only check that our value was removed, not what replaced it.
    assert get_session_env("HERMES_SESSION_KEY") != "tg:-1001:17585"


def test_session_key_no_race_condition_with_contextvars(monkeypatch):
    """Prove contextvars isolates SESSION_KEY across concurrent async tasks.

    Two tasks set different session keys. With contextvars each task
    reads back its own value. With os.environ the second task would
    overwrite the first (the old bug).
    """
    monkeypatch.delenv("HERMES_SESSION_KEY", raising=False)

    results = {}

    async def handler(key: str, delay: float):
        tokens = set_session_vars(session_key=key)
        try:
            await asyncio.sleep(delay)
            read_back = get_session_env("HERMES_SESSION_KEY")
            results[key] = read_back
        finally:
            clear_session_vars(tokens)

    async def run():
        task_a = asyncio.create_task(handler("session-A", 0.15))
        await asyncio.sleep(0.05)
        task_b = asyncio.create_task(handler("session-B", 0.05))
        await asyncio.gather(task_a, task_b)

    asyncio.run(run())

    # Both tasks must read back their own session key
    assert results["session-A"] == "session-A", (
        f"Session A got '{results['session-A']}' instead of 'session-A' — race condition!"
    )
    assert results["session-B"] == "session-B", (
        f"Session B got '{results['session-B']}' instead of 'session-B' — race condition!"
    )


@pytest.mark.asyncio
async def test_run_in_executor_with_context_preserves_session_env(monkeypatch):
    """Gateway executor work should inherit session contextvars for tool routing."""
    runner = object.__new__(GatewayRunner)
    monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
    monkeypatch.delenv("HERMES_SESSION_CHAT_ID", raising=False)
    monkeypatch.delenv("HERMES_SESSION_THREAD_ID", raising=False)
    monkeypatch.delenv("HERMES_SESSION_USER_ID", raising=False)

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="2144471399",
        chat_type="dm",
        user_id="123456",
        user_name="alice",
        thread_id=None,
    )
    context = SessionContext(
        source=source,
        connected_platforms=[],
        home_channels={},
        session_key="agent:main:telegram:dm:2144471399",
        session_id="20260602_120000_gatewayabc",
    )

    tokens = runner._set_session_env(context)
    try:
        result = await runner._run_in_executor_with_context(
            lambda: {
                "platform": get_session_env("HERMES_SESSION_PLATFORM"),
                "chat_id": get_session_env("HERMES_SESSION_CHAT_ID"),
                "user_id": get_session_env("HERMES_SESSION_USER_ID"),
                "session_key": get_session_env("HERMES_SESSION_KEY"),
                "session_id": get_session_env("HERMES_SESSION_ID"),
            }
        )
    finally:
        runner._clear_session_env(tokens)

    assert result == {
        "platform": "telegram",
        "chat_id": "2144471399",
        "user_id": "123456",
        "session_key": "agent:main:telegram:dm:2144471399",
        "session_id": "20260602_120000_gatewayabc",
    }


@pytest.mark.asyncio
async def test_repeated_gateway_turns_copy_active_session_id_into_executor(monkeypatch):
    """Cached-agent gateway turns should still expose the active session id."""
    runner = object.__new__(GatewayRunner)
    monkeypatch.setenv("HERMES_SESSION_ID", "stale-global")

    async def run_turn(session_id: str) -> str:
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="2144471399",
            chat_type="dm",
        )
        context = SessionContext(
            source=source,
            connected_platforms=[],
            home_channels={},
            session_key="agent:main:telegram:dm:2144471399",
            session_id=session_id,
        )
        tokens = runner._set_session_env(context)
        try:
            return await runner._run_in_executor_with_context(
                lambda: get_session_env("HERMES_SESSION_ID")
            )
        finally:
            runner._clear_session_env(tokens)

    assert await run_turn("session-one") == "session-one"
    assert await run_turn("session-two") == "session-two"
    assert os.environ["HERMES_SESSION_ID"] == "stale-global"


def test_kanban_worker_metadata_uses_context_session_id(monkeypatch):
    """Kanban tools should read gateway session id from ContextVars, not env."""
    from tools.kanban_tools import _stamp_worker_session_metadata

    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-1")
    monkeypatch.setenv("HERMES_SESSION_ID", "stale-global")
    tokens = set_session_vars(
        platform="telegram",
        chat_id="chat-1",
        session_key="telegram:chat-1",
        session_id="ctx-session-id",
    )
    try:
        stamped = _stamp_worker_session_metadata("task-1", {"existing": True})
    finally:
        clear_session_vars(tokens)

    assert stamped == {"existing": True, "worker_session_id": "ctx-session-id"}


@pytest.mark.asyncio
async def test_run_in_executor_with_context_forwards_args():
    """_run_in_executor_with_context should forward *args to the callable."""
    runner = object.__new__(GatewayRunner)

    def add(a, b):
        return a + b

    result = await runner._run_in_executor_with_context(add, 3, 7)
    assert result == 10


def test_gateway_session_id_contextvar_does_not_clobber_process_env(monkeypatch):
    """Gateway-mode session ID rotation is ContextVar-only, never global env."""
    monkeypatch.setenv("HERMES_SESSION_ID", "stale-global")
    tokens = set_session_vars(
        platform="telegram",
        chat_id="chat-1",
        session_key="telegram:chat-1",
    )
    try:
        set_current_session_id("gateway-session-1")

        assert get_session_env("HERMES_SESSION_ID") == "gateway-session-1"
        assert os.environ["HERMES_SESSION_ID"] == "stale-global"
    finally:
        clear_session_vars(tokens)


def test_cli_session_id_sync_can_explicitly_update_process_env(monkeypatch):
    """Single-session CLI paths still have an explicit env-sync escape hatch."""
    monkeypatch.delenv("_HERMES_GATEWAY", raising=False)
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)

    set_current_session_id("cli-session-1", sync_process_env=True)

    assert get_session_env("HERMES_SESSION_ID") == "cli-session-1"
    assert os.environ["HERMES_SESSION_ID"] == "cli-session-1"


@pytest.mark.asyncio
async def test_run_in_executor_with_context_propagates_exceptions():
    """Exceptions inside the executor should propagate to the caller."""
    runner = object.__new__(GatewayRunner)

    def boom():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        await runner._run_in_executor_with_context(boom)


def test_gateway_run_does_not_write_session_key_to_global_env():
    """Gateway workers must not mirror per-message session keys into os.environ.

    The gateway handles concurrent sessions in one process, so per-message
    HERMES_SESSION_KEY belongs in gateway.session_context ContextVars only.
    CLI/cron/ACP compatibility may still use env fallback outside gateway/run.py.
    """
    from pathlib import Path

    gateway_run = (Path(__file__).parent.parent.parent / "gateway" / "run.py").read_text(
        encoding="utf-8"
    )
    assert 'os.environ["HERMES_SESSION_KEY"]' not in gateway_run
    assert "os.environ['HERMES_SESSION_KEY']" not in gateway_run
    assert 'os.environ["HERMES_SESSION_ID"]' not in gateway_run
    assert "os.environ['HERMES_SESSION_ID']" not in gateway_run
