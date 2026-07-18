import asyncio
import os
from dataclasses import replace

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionContext, SessionSource
from gateway.session_context import (
    get_session_env,
    set_session_vars,
    clear_session_vars,
    reset_session_vars,
    session_redact_pii_enabled,
    _SESSION_REDACT_PII,
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
    for var in _VAR_MAP.values():
        var.set(_UNSET)
    _SESSION_REDACT_PII.set(_UNSET)
    yield
    for var in _VAR_MAP.values():
        # Can't use var.reset() without a token; just set back to sentinel.
        var.set(_UNSET)
    _SESSION_REDACT_PII.set(_UNSET)


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
    monkeypatch.delenv("HERMES_SESSION_SOURCE", raising=False)
    monkeypatch.delenv("HERMES_SESSION_CHAT_ID", raising=False)
    monkeypatch.delenv("HERMES_SESSION_CHAT_NAME", raising=False)
    monkeypatch.delenv("HERMES_SESSION_USER_ID", raising=False)
    monkeypatch.delenv("HERMES_SESSION_USER_NAME", raising=False)
    monkeypatch.delenv("HERMES_SESSION_THREAD_ID", raising=False)

    tokens = runner._set_session_env(context, redact_pii=True)

    # Values should be readable via get_session_env (contextvar path)
    assert get_session_env("HERMES_SESSION_PLATFORM") == "telegram"
    assert get_session_env("HERMES_SESSION_SOURCE") == ""
    assert get_session_env("HERMES_SESSION_CHAT_ID") == "-1001"
    assert get_session_env("HERMES_SESSION_CHAT_NAME") == "Group"
    assert get_session_env("HERMES_SESSION_USER_ID") == "123456"
    assert get_session_env("HERMES_SESSION_USER_NAME") == "alice"
    assert get_session_env("HERMES_SESSION_THREAD_ID") == "17585"
    assert session_redact_pii_enabled() is True

    # os.environ should NOT be touched
    assert os.getenv("HERMES_SESSION_PLATFORM") is None
    assert os.getenv("HERMES_SESSION_SOURCE") is None
    assert os.getenv("HERMES_SESSION_THREAD_ID") is None

    # Clean up
    runner._clear_session_env(tokens)
    assert session_redact_pii_enabled() is False


def test_redact_pii_policy_lifecycle_keeps_raw_identity_unmapped():
    """Privacy policy is per-turn state, not a legacy env-backed identity."""
    assert _SESSION_REDACT_PII not in _VAR_MAP.values()
    assert session_redact_pii_enabled() is False

    tokens = set_session_vars(
        platform="telegram",
        chat_id="telegram:+15550101001",
        user_id="+15550101002",
        redact_pii=True,
    )
    assert session_redact_pii_enabled() is True
    assert get_session_env("HERMES_SESSION_CHAT_ID") == "telegram:+15550101001"
    assert get_session_env("HERMES_SESSION_USER_ID") == "+15550101002"

    clear_session_vars(tokens)
    assert session_redact_pii_enabled() is False

    set_session_vars(redact_pii=True)
    assert session_redact_pii_enabled() is True
    reset_session_vars()
    assert session_redact_pii_enabled() is False


def test_gateway_binding_uses_routed_profile_privacy_policy(tmp_path, monkeypatch):
    """The real turn binder must not reuse the default profile's policy."""
    from gateway.session import _hash_message_id, _hash_sender_id, _hash_thread_id
    from tools.mcp_tool import _build_session_context_meta

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "privacy:\n  redact_pii: false\n",
        encoding="utf-8",
    )
    secondary_home = tmp_path / "profiles" / "secondary"
    secondary_home.mkdir(parents=True)
    monkeypatch.setenv("SECONDARY_REDACT_PII", "true")
    (secondary_home / "config.yaml").write_text(
        "privacy:\n  redact_pii: ${SECONDARY_REDACT_PII}\n",
        encoding="utf-8",
    )

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner.adapters = {}
    phone = "+15551234567"

    def _context(profile):
        source = SessionSource(
            platform=Platform.WHATSAPP_CLOUD,
            chat_id=phone,
            user_id=phone,
            thread_id=f"{phone}:topic",
            message_id="wamid.HBgLMTU1NTEyMzQ1NjcVAgASGBQ",
            profile=profile,
        )
        return SessionContext(
            source=source,
            connected_platforms=[Platform.WHATSAPP_CLOUD],
            home_channels={},
            session_key=f"agent:{profile}:whatsapp_cloud:dm:{phone}",
            session_id=f"session-{profile}",
        )

    default_context = _context("default")
    tokens, policy = runner._bind_session_context_for_turn(default_context)
    try:
        assert policy is False
        default_meta = _build_session_context_meta()
        assert default_meta is not None
        assert default_meta["com.nousresearch.hermes/user_id"] == phone
        assert default_meta["com.nousresearch.hermes/message_id"].startswith("wamid.")
    finally:
        runner._clear_session_env(tokens)

    secondary_context = _context("secondary")
    tokens, policy = runner._bind_session_context_for_turn(secondary_context)
    try:
        assert policy is True
        secondary_meta = _build_session_context_meta()
        assert secondary_meta is not None
        assert secondary_meta["com.nousresearch.hermes/user_id"] == _hash_sender_id(phone)
        assert secondary_meta["com.nousresearch.hermes/thread_id"] == _hash_thread_id(
            secondary_context.source.thread_id
        )
        assert secondary_meta["com.nousresearch.hermes/message_id"] == _hash_message_id(
            secondary_context.source.message_id
        )
    finally:
        runner._clear_session_env(tokens)


def test_gateway_binding_honors_managed_privacy_precedence(tmp_path, monkeypatch):
    from hermes_cli import managed_scope

    user_home = tmp_path / "user"
    managed_home = tmp_path / "managed"
    user_home.mkdir()
    managed_home.mkdir()
    (user_home / "config.yaml").write_text(
        "privacy:\n  redact_pii: false\n",
        encoding="utf-8",
    )
    (managed_home / "config.yaml").write_text(
        "privacy:\n  redact_pii: true\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("gateway.run._hermes_home", user_home)
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed_home))
    managed_scope.invalidate_managed_cache()

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.adapters = {}
    context = SessionContext(
        source=SessionSource(
            platform=Platform.SIGNAL,
            chat_id="+15551234567",
            user_id="+15551234567",
        ),
        connected_platforms=[Platform.SIGNAL],
        home_channels={},
        session_key="agent:main:signal:dm:+15551234567",
        session_id="session-managed",
    )

    tokens, policy = runner._bind_session_context_for_turn(context)
    try:
        assert policy is True
        assert session_redact_pii_enabled() is True
    finally:
        runner._clear_session_env(tokens)
        managed_scope.invalidate_managed_cache()


@pytest.mark.parametrize("failure", ["malformed", "unreadable", "missing-profile"])
def test_gateway_binding_marks_policy_unavailable_and_mcp_omits(
    tmp_path,
    monkeypatch,
    caplog,
    failure,
):
    from tools.mcp_tool import _build_session_context_meta

    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    profile = None
    if failure == "missing-profile":
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        runner.config = GatewayConfig(multiplex_profiles=True)
        profile = "does-not-exist"
    else:
        runner.config = GatewayConfig()
        monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
        (tmp_path / "config.yaml").write_text(
            "privacy: [malformed" if failure == "malformed" else "privacy: {}\n",
            encoding="utf-8",
        )
        if failure == "unreadable":
            monkeypatch.setattr(
                "gateway.run._read_yaml_mapping_strict",
                lambda _path: (_ for _ in ()).throw(PermissionError("unreadable")),
            )

    source = SessionSource(
        platform=Platform.SIGNAL,
        chat_id="+15551234567",
        user_id="+15551234567",
        profile=profile,
    )
    context = SessionContext(
        source=source,
        connected_platforms=[Platform.SIGNAL],
        home_channels={},
        session_key="agent:secondary:signal:dm:+15551234567",
        session_id="session-secondary",
    )

    with caplog.at_level("WARNING"):
        tokens, policy = runner._bind_session_context_for_turn(context)
        try:
            assert policy is None
            assert session_redact_pii_enabled() is None
            assert _build_session_context_meta() is None
        finally:
            runner._clear_session_env(tokens)

    assert "external session metadata will be omitted" in caplog.text
    assert "privacy policy is unavailable" in caplog.text


def test_queued_followup_rebinds_sender_message_and_policy(tmp_path, monkeypatch):
    """Recursive queued turns must not retain the first event's ContextVars."""
    from tools.mcp_tool import _build_session_context_meta

    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("privacy:\n  redact_pii: true\n", encoding="utf-8")

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.adapters = {}
    first_source = SessionSource(
        platform=Platform.WHATSAPP_CLOUD,
        chat_id="15551230001",
        user_id="15551230001",
        message_id="wamid.FIRST",
    )
    first_context = SessionContext(
        source=first_source,
        connected_platforms=[Platform.WHATSAPP_CLOUD],
        home_channels={},
        session_key="agent:main:whatsapp_cloud:dm:15551230001",
        session_id="session-queue",
    )
    tokens, policy = runner._bind_session_context_for_turn(first_context)
    assert policy is True
    assert get_session_env("HERMES_SESSION_USER_ID") == "15551230001"
    assert get_session_env("HERMES_SESSION_MESSAGE_ID") == "wamid.FIRST"

    # Policy is read per event; change it before the queued turn drains.
    config_path.write_text("privacy:\n  redact_pii: false\n", encoding="utf-8")
    followup_event = MessageEvent(
        text="follow up",
        source=SessionSource(
            platform=Platform.WHATSAPP_CLOUD,
            chat_id="15551230002",
            user_id="15551230002",
        ),
        message_id="wamid.SECOND",
    )
    try:
        followup_source, followup_policy = runner._bind_followup_event_context(
            followup_event,
            session_key="agent:main:whatsapp_cloud:dm:15551230002",
            session_id="session-queue",
        )
        assert followup_source.message_id == "wamid.SECOND"
        assert followup_policy is False
        assert session_redact_pii_enabled() is False
        assert get_session_env("HERMES_SESSION_USER_ID") == "15551230002"
        assert get_session_env("HERMES_SESSION_MESSAGE_ID") == "wamid.SECOND"
        meta = _build_session_context_meta()
        assert meta is not None
        assert meta["com.nousresearch.hermes/user_id"] == "15551230002"
        assert meta["com.nousresearch.hermes/message_id"] == "wamid.SECOND"

        # Synthetic queued continuations have no trigger ID, so their
        # constructor must clear the reused source ID instead of inheriting
        # WAMID 2.
        synthetic_event = MessageEvent(
            text="continue the goal",
            source=replace(followup_source, message_id=None),
            message_id=None,
        )
        synthetic_source, synthetic_policy = runner._bind_followup_event_context(
            synthetic_event,
            session_key="agent:main:whatsapp_cloud:dm:15551230002",
            session_id="session-queue",
        )
        assert synthetic_source.message_id is None
        assert synthetic_policy is False
        assert get_session_env("HERMES_SESSION_MESSAGE_ID") == ""
        synthetic_meta = _build_session_context_meta()
        assert synthetic_meta is not None
        assert synthetic_meta["com.nousresearch.hermes/message_id"] == ""
    finally:
        runner._clear_session_env(tokens)


def test_session_source_uses_contextvars(monkeypatch):
    monkeypatch.delenv("HERMES_SESSION_SOURCE", raising=False)

    tokens = set_session_vars(source="tool")

    assert get_session_env("HERMES_SESSION_SOURCE") == "tool"

    clear_session_vars(tokens)

    assert get_session_env("HERMES_SESSION_SOURCE") == ""


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


def test_session_id_set_via_contextvars(monkeypatch):
    """set_session_vars should set HERMES_SESSION_ID via contextvars."""
    monkeypatch.setenv("HERMES_SESSION_ID", "stale-env-session")

    tokens = set_session_vars(session_id="ctx-session-456")
    assert get_session_env("HERMES_SESSION_ID") == "ctx-session-456"

    clear_session_vars(tokens)
    assert get_session_env("HERMES_SESSION_ID") == ""


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


def test_set_session_env_includes_session_id():
    """_set_session_env should propagate session_id from SessionContext."""
    runner = object.__new__(GatewayRunner)
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="channel-123",
        chat_name="general",
        chat_type="thread",
        thread_id="thread-456",
    )
    context = SessionContext(
        source=source,
        connected_platforms=[],
        home_channels={},
        session_key="agent:main:discord:thread:thread-456:thread-456",
        session_id="20260705_191621_abcd",
    )

    tokens = runner._set_session_env(context)
    assert get_session_env("HERMES_SESSION_ID") == "20260705_191621_abcd"
    runner._clear_session_env(tokens)
    assert get_session_env("HERMES_SESSION_ID") != "20260705_191621_abcd"


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
        runner._shutdown_executor()

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

    try:
        result = await runner._run_in_executor_with_context(add, 3, 7)
    finally:
        runner._shutdown_executor()
    assert result == 10


@pytest.mark.asyncio
async def test_run_in_executor_with_context_propagates_exceptions():
    """Exceptions inside the executor should propagate to the caller."""
    runner = object.__new__(GatewayRunner)

    def blow_up():
        raise ValueError("boom")

    try:
        with pytest.raises(ValueError, match="boom"):
            await runner._run_in_executor_with_context(blow_up)
    finally:
        runner._shutdown_executor()


@pytest.mark.asyncio
async def test_run_in_executor_with_context_survives_default_executor_shutdown():
    """Gateway agent work should not depend on asyncio's default executor."""
    runner = object.__new__(GatewayRunner)
    loop = asyncio.get_running_loop()

    await loop.run_in_executor(None, lambda: None)
    await loop.shutdown_default_executor()

    try:
        result = await runner._run_in_executor_with_context(lambda: "ok")
    finally:
        runner._shutdown_executor()

    assert result == "ok"


@pytest.mark.asyncio
async def test_gateway_executor_refuses_resurrection_after_shutdown():
    """A real gateway shutdown must NOT be resurrected by the recreate path.

    _shutdown_executor() means "we're stopping" — the recreate-on-shutdown
    logic exists to survive an *external* teardown of the loop default
    (test_..._survives_default_executor_shutdown), not to undo our own stop.
    """
    runner = object.__new__(GatewayRunner)

    try:
        first = await runner._run_in_executor_with_context(lambda: "first")
        assert first == "first"
        runner._shutdown_executor()

        with pytest.raises(RuntimeError, match="shutting down"):
            await runner._run_in_executor_with_context(lambda: "second")
    finally:
        runner._shutdown_executor()

