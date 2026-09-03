"""Host-authenticated call context for plugin tool handlers.

A plugin tool handler receives only what the model wrote into the tool
arguments. Anything a plugin must *trust* -- which profile, which gateway
session, which thread, which user -- has to come from the host instead, and
has to stay correct when two gateway sessions run concurrently in one process.
"""

import asyncio

import pytest

from gateway.session_context import clear_session_vars, set_session_vars
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


def _context() -> PluginContext:
    return PluginContext(
        PluginManifest(name="wake-plugin", key="wake-plugin", source="user"),
        PluginManager(),
    )


@pytest.fixture
def bound_session():
    tokens = set_session_vars(
        platform="telegram",
        source="gateway",
        chat_id="42",
        chat_type="dm",
        thread_id="7",
        user_id="u-1",
        user_name="tester",
        scope_id="guild-9",
        session_key="agent:main:telegram:dm:42",
        session_id="session-42",
        message_id="m-3",
        profile="agent-management",
    )
    try:
        yield
    finally:
        clear_session_vars(tokens)


def test_call_context_exposes_host_bound_session_identity(bound_session):
    ctx = _context().call_context

    assert ctx["session_key"] == "agent:main:telegram:dm:42"
    assert ctx["session_id"] == "session-42"
    assert ctx["platform"] == "telegram"
    assert ctx["chat_id"] == "42"
    assert ctx["chat_type"] == "dm"
    assert ctx["thread_id"] == "7"
    assert ctx["user_id"] == "u-1"
    assert ctx["user_name"] == "tester"
    assert ctx["scope_id"] == "guild-9"
    assert ctx["message_id"] == "m-3"
    # Profile is session identity, not process identity.
    assert ctx["profile"] == "agent-management"


def test_bound_profile_wins_over_the_process_profile(
    monkeypatch, tmp_path, bound_session
):
    """A multiplexed gateway serves several profiles from one process.

    Reading the process profile while a session is bound would hand a plugin the
    right chat and the wrong policy domain.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    ctx = _context()

    assert ctx.call_context["profile"] == "agent-management"
    assert ctx.call_context["profile"] != ctx.profile_name


def test_call_context_falls_back_to_the_process_profile_when_unbound(
    monkeypatch, tmp_path
):
    """Compatibility path: plain CLI, cron, and tests bind no session."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_SESSION_PROFILE", raising=False)
    assert _context().call_context["profile"] == _context().profile_name


def test_call_context_is_empty_when_no_session_is_bound(monkeypatch):
    for name in (
        "HERMES_SESSION_KEY",
        "HERMES_SESSION_ID",
        "HERMES_SESSION_PLATFORM",
        "HERMES_SESSION_THREAD_ID",
    ):
        monkeypatch.delenv(name, raising=False)

    ctx = _context().call_context
    assert ctx["session_key"] == ""
    assert ctx["session_id"] == ""


def test_call_context_carries_no_operator_approval_receipt(bound_session):
    """Documents the upstream gap: no host-authenticated approval identity.

    Callers that need "a human approved this specific action" cannot get it
    from the plugin call context, so any such action must fail closed rather
    than trust a model-supplied assertion.
    """
    ctx = _context().call_context

    assert "operator_approval" not in ctx
    assert "approved_by" not in ctx
    assert not any(key.startswith("approval") for key in ctx)


def test_call_context_is_read_only(bound_session):
    ctx = _context()
    snapshot = ctx.call_context
    snapshot["session_key"] = "agent:main:telegram:dm:999"

    assert ctx.call_context["session_key"] == "agent:main:telegram:dm:42"


@pytest.mark.asyncio
async def test_call_context_is_task_local_across_concurrent_sessions():
    """Two concurrent gateway turns must not read each other's identity."""
    ctx = _context()
    seen: dict[str, str] = {}
    first_bound = asyncio.Event()

    async def _turn(name: str, session_key: str, wait_for=None, release=None):
        tokens = set_session_vars(session_key=session_key, session_id=name)
        try:
            if release is not None:
                release.set()
            if wait_for is not None:
                await wait_for.wait()
            await asyncio.sleep(0)
            seen[name] = ctx.call_context["session_key"]
        finally:
            clear_session_vars(tokens)

    await asyncio.gather(
        _turn("a", "agent:main:telegram:dm:1", wait_for=first_bound),
        _turn("b", "agent:main:telegram:dm:2", release=first_bound),
    )

    assert seen == {
        "a": "agent:main:telegram:dm:1",
        "b": "agent:main:telegram:dm:2",
    }


@pytest.mark.asyncio
async def test_call_context_profile_is_task_local_across_concurrent_sessions():
    """Two profiles multiplexed in one process must not read each other's.

    The failure this pins is a qualified-identity break at the policy boundary:
    a plugin trusting ``call_context`` gets the right session key and the wrong
    profile, so it authorizes against the wrong domain.
    """
    ctx = _context()
    seen: dict[str, tuple[str, str]] = {}
    first_bound = asyncio.Event()

    async def _turn(name, *, profile, session_key, wait_for=None, release=None):
        tokens = set_session_vars(
            session_key=session_key, session_id=name, profile=profile
        )
        try:
            if release is not None:
                release.set()
            if wait_for is not None:
                await wait_for.wait()
            await asyncio.sleep(0)
            snapshot = ctx.call_context
            seen[name] = (snapshot["profile"], snapshot["session_key"])
        finally:
            clear_session_vars(tokens)

    await asyncio.gather(
        _turn(
            "a",
            profile="agent-management",
            session_key="agent:main:telegram:dm:1",
            wait_for=first_bound,
        ),
        _turn(
            "b",
            profile="research",
            session_key="agent:main:telegram:dm:2",
            release=first_bound,
        ),
    )

    assert seen == {
        "a": ("agent-management", "agent:main:telegram:dm:1"),
        "b": ("research", "agent:main:telegram:dm:2"),
    }
