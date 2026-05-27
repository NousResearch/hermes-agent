import asyncio
import os

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner
from gateway.session import SessionContext, SessionSource
from contextvars import ContextVar

from gateway.session_context import (
    get_session_env,
    get_registered_var_names,
    register_session_context_var,
    set_session_vars,
    clear_session_vars,
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

    tokens = runner._set_session_env(context)

    # Values should be readable via get_session_env (contextvar path)
    assert get_session_env("HERMES_SESSION_PLATFORM") == "telegram"
    assert get_session_env("HERMES_SESSION_CHAT_ID") == "-1001"
    assert get_session_env("HERMES_SESSION_CHAT_NAME") == "Group"
    assert get_session_env("HERMES_SESSION_USER_ID") == "123456"
    assert get_session_env("HERMES_SESSION_USER_NAME") == "alice"
    assert get_session_env("HERMES_SESSION_THREAD_ID") == "17585"

    # os.environ should NOT be touched
    assert os.getenv("HERMES_SESSION_PLATFORM") is None
    assert os.getenv("HERMES_SESSION_THREAD_ID") is None

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
    )

    tokens = runner._set_session_env(context)
    try:
        result = await runner._run_in_executor_with_context(
            lambda: {
                "platform": get_session_env("HERMES_SESSION_PLATFORM"),
                "chat_id": get_session_env("HERMES_SESSION_CHAT_ID"),
                "user_id": get_session_env("HERMES_SESSION_USER_ID"),
                "session_key": get_session_env("HERMES_SESSION_KEY"),
            }
        )
    finally:
        runner._clear_session_env(tokens)

    assert result == {
        "platform": "telegram",
        "chat_id": "2144471399",
        "user_id": "123456",
        "session_key": "agent:main:telegram:dm:2144471399",
    }


@pytest.mark.asyncio
async def test_run_in_executor_with_context_forwards_args():
    """_run_in_executor_with_context should forward *args to the callable."""
    runner = object.__new__(GatewayRunner)

    def add(a, b):
        return a + b

    result = await runner._run_in_executor_with_context(add, 3, 7)
    assert result == 10


@pytest.mark.asyncio
async def test_run_in_executor_with_context_propagates_exceptions():
    """Exceptions inside the executor should propagate to the caller."""
    runner = object.__new__(GatewayRunner)

    def blow_up():
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        await runner._run_in_executor_with_context(blow_up)


# ---------------------------------------------------------------------------
# register_session_context_var — plugin-extensible registry
# ---------------------------------------------------------------------------

def test_register_session_context_var_makes_name_resolvable(monkeypatch):
    """A registered ContextVar should resolve via get_session_env using
    the registered name, even though it isn't a built-in HERMES_SESSION_*
    name."""
    var: ContextVar = ContextVar("CUSTOM_PRINCIPAL", default=_UNSET)
    register_session_context_var("CUSTOM_PRINCIPAL", var)
    monkeypatch.delenv("CUSTOM_PRINCIPAL", raising=False)
    try:
        var.set("alice@example.com")
        assert get_session_env("CUSTOM_PRINCIPAL") == "alice@example.com"
    finally:
        var.set(_UNSET)
        _VAR_MAP.pop("CUSTOM_PRINCIPAL", None)


def test_register_session_context_var_falls_back_to_env(monkeypatch):
    """When the ContextVar is _UNSET, the resolver falls back to os.environ
    (same semantics as built-in vars)."""
    var: ContextVar = ContextVar("CUSTOM_FALLBACK", default=_UNSET)
    register_session_context_var("CUSTOM_FALLBACK", var)
    monkeypatch.setenv("CUSTOM_FALLBACK", "from-env")
    try:
        assert get_session_env("CUSTOM_FALLBACK") == "from-env"
    finally:
        _VAR_MAP.pop("CUSTOM_FALLBACK", None)


def test_register_session_context_var_explicit_empty_does_not_fall_back(
    monkeypatch,
):
    """Setting the ContextVar to "" is an explicit clear — the resolver
    must return "" and NOT fall back to os.environ."""
    var: ContextVar = ContextVar("CUSTOM_EMPTY", default=_UNSET)
    register_session_context_var("CUSTOM_EMPTY", var)
    monkeypatch.setenv("CUSTOM_EMPTY", "from-env-should-not-be-seen")
    try:
        var.set("")
        assert get_session_env("CUSTOM_EMPTY") == ""
    finally:
        var.set(_UNSET)
        _VAR_MAP.pop("CUSTOM_EMPTY", None)


def test_register_session_context_var_rejects_invalid_inputs():
    """Defensive checks — bad inputs raise before corrupting the registry."""
    valid_var: ContextVar = ContextVar("X", default=_UNSET)
    with pytest.raises(ValueError):
        register_session_context_var("", valid_var)
    with pytest.raises(TypeError):
        register_session_context_var("X", "not a contextvar")  # type: ignore[arg-type]


def test_register_session_context_var_is_idempotent_and_last_writer_wins():
    """Re-registering the same name swaps the binding (last writer wins)."""
    a: ContextVar = ContextVar("DUP", default=_UNSET)
    b: ContextVar = ContextVar("DUP", default=_UNSET)
    register_session_context_var("DUP", a)
    register_session_context_var("DUP", b)
    try:
        b.set("from-b")
        a.set("from-a")
        # b is the registered one — resolver should see "from-b".
        assert get_session_env("DUP") == "from-b"
    finally:
        a.set(_UNSET)
        b.set(_UNSET)
        _VAR_MAP.pop("DUP", None)


def test_get_registered_var_names_includes_builtin_and_plugin():
    """The introspection helper lists both built-in and plugin-registered names."""
    var: ContextVar = ContextVar("INTROSPECT_TEST", default=_UNSET)
    register_session_context_var("INTROSPECT_TEST", var)
    try:
        names = get_registered_var_names()
        assert "HERMES_SESSION_USER_ID" in names  # built-in
        assert "INTROSPECT_TEST" in names         # plugin-registered
    finally:
        _VAR_MAP.pop("INTROSPECT_TEST", None)


@pytest.mark.asyncio
async def test_register_session_context_var_asyncio_task_isolation():
    """Custom ContextVars registered via the public API inherit the same
    per-asyncio-task isolation as built-in HERMES_SESSION_* vars."""
    var: ContextVar = ContextVar("ISO_TEST", default=_UNSET)
    register_session_context_var("ISO_TEST", var)

    results: dict[str, str] = {}

    async def handler(label: str, value: str, delay: float):
        token = var.set(value)
        try:
            await asyncio.sleep(delay)
            results[label] = get_session_env("ISO_TEST")
        finally:
            var.reset(token)

    try:
        await asyncio.gather(
            handler("a", "alice", 0.02),
            handler("b", "bob",   0.01),
        )
        # Each task reads its own value back — no bleed.
        assert results == {"a": "alice", "b": "bob"}
    finally:
        _VAR_MAP.pop("ISO_TEST", None)
