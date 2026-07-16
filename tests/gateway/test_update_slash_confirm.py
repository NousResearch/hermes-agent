"""Tests for the /update confirmation prompt.

/update ALWAYS routes through the slash-confirm primitive — there is no
opt-out and no "Always Approve" button.  It pulls new code and restarts
the gateway (interrupting every active session on the host), so it's a
rare, high-stakes action where an accidental invoke must never fire
instantly and a permanent one-tap opt-out would be a footgun.

Prompt renders Approve Once / Cancel only (``allow_always=False``).
These tests isolate the confirm routing: ``_execute_update`` (the
detached spawn) is mocked out; its real mechanics live in
tests/gateway/test_update_command.py.
"""

from __future__ import annotations

import itertools
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource, build_session_key


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str = "/update") -> MessageEvent:
    return MessageEvent(text=text, source=_make_source(), message_id="m1")


def _make_runner():
    """Bare GatewayRunner with just enough wiring for the confirm path.

    Mirrors tests/gateway/test_destructive_slash_confirm.py::_make_runner.
    """
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    # Capture kwargs so we can assert allow_always=False is forwarded.
    adapter.send_slash_confirm = AsyncMock(return_value=None)
    runner.adapters = {Platform.TELEGRAM: adapter}

    runner._slash_confirm_counter = itertools.count(1)
    runner._session_key_for_source = lambda src: build_session_key(src)
    runner._thread_metadata_for_source = lambda *a, **kw: None
    runner._reply_anchor_for_event = lambda _e: None
    runner._adapter_for_source = lambda src: adapter
    # Mock the detached spawn — we only test confirm routing here.
    runner._execute_update = AsyncMock(return_value="⚕ Starting Hermes update…")
    return runner


# ---------------------------------------------------------------------------
# /update always prompts — never spawns without explicit approval
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_always_prompts_without_spawning(monkeypatch):
    """/update returns a text prompt (button fallback) and does NOT spawn
    until the user approves."""
    monkeypatch.setenv("HERMES_MANAGED", "")
    runner = _make_runner()

    result = await runner._handle_update_command(_make_event())

    runner._execute_update.assert_not_awaited()
    assert isinstance(result, str)
    assert "Confirm /update" in result
    assert "Approve Once" in result
    assert "Cancel" in result


@pytest.mark.asyncio
async def test_prompt_has_no_always_option(monkeypatch):
    """The /update prompt must NOT offer 'Always Approve' (no opt-out)."""
    monkeypatch.setenv("HERMES_MANAGED", "")
    runner = _make_runner()

    result = await runner._handle_update_command(_make_event())

    assert "Always Approve" not in result
    assert "/always" not in result


@pytest.mark.asyncio
async def test_forwards_allow_always_false_to_adapter(monkeypatch):
    """_request_slash_confirm must forward allow_always=False so the adapter
    suppresses the middle button."""
    monkeypatch.setenv("HERMES_MANAGED", "")
    runner = _make_runner()

    await runner._handle_update_command(_make_event())

    adapter = runner.adapters[Platform.TELEGRAM]
    adapter.send_slash_confirm.assert_awaited_once()
    assert adapter.send_slash_confirm.await_args.kwargs["allow_always"] is False


@pytest.mark.asyncio
async def test_default_allow_always_not_passed_to_legacy_adapter(monkeypatch):
    """When allow_always=True (the default), the kwarg must NOT be sent.

    Legacy platform adapters don't have the allow_always parameter in their
    signature. Passing it unconditionally would raise TypeError and silently
    fall back to text — losing native confirmation buttons. The fix gates
    the kwarg so legacy adapters see only the params they already accept.
    """
    monkeypatch.setenv("HERMES_MANAGED", "")
    runner = _make_runner()

    # Simulate a caller that uses the default allow_always=True (e.g. /reload-mcp)
    event = _make_event("/reload-mcp")
    await runner._request_slash_confirm(
        event=event,
        command="reload-mcp",
        title="/reload-mcp",
        message="Confirm reload",
        handler=lambda choice: None,
        # allow_always defaults to True — explicitly omitted here
    )

    adapter = runner.adapters[Platform.TELEGRAM]
    adapter.send_slash_confirm.assert_awaited_once()
    # The default-True case must NOT include allow_always in kwargs,
    # so a legacy adapter without that parameter won't TypeError.
    assert "allow_always" not in adapter.send_slash_confirm.await_args.kwargs


@pytest.mark.asyncio
async def test_registers_pending_confirm(monkeypatch):
    """A pending slash-confirm entry is registered for the session."""
    from tools import slash_confirm as _slash_confirm_mod

    monkeypatch.setenv("HERMES_MANAGED", "")
    runner = _make_runner()
    session_key = build_session_key(_make_source())
    _slash_confirm_mod.clear(session_key)

    await runner._handle_update_command(_make_event())

    pending = _slash_confirm_mod.get_pending(session_key)
    assert pending is not None
    assert pending["command"] == "update"
    _slash_confirm_mod.clear(session_key)


@pytest.mark.asyncio
async def test_resolve_once_spawns_update(monkeypatch):
    """Resolving 'once' spawns the update and returns its output."""
    from tools import slash_confirm as _slash_confirm_mod

    monkeypatch.setenv("HERMES_MANAGED", "")
    runner = _make_runner()
    session_key = build_session_key(_make_source())
    _slash_confirm_mod.clear(session_key)

    await runner._handle_update_command(_make_event())
    pending = _slash_confirm_mod.get_pending(session_key)
    assert pending is not None

    resolved = await _slash_confirm_mod.resolve(
        session_key, pending["confirm_id"], "once",
    )

    runner._execute_update.assert_awaited_once()
    assert "Starting Hermes update" in resolved
    assert _slash_confirm_mod.get_pending(session_key) is None


@pytest.mark.asyncio
async def test_resolve_cancel_does_not_spawn(monkeypatch):
    """Resolving 'cancel' must NOT spawn the update."""
    from tools import slash_confirm as _slash_confirm_mod

    monkeypatch.setenv("HERMES_MANAGED", "")
    runner = _make_runner()
    session_key = build_session_key(_make_source())
    _slash_confirm_mod.clear(session_key)

    await runner._handle_update_command(_make_event())
    pending = _slash_confirm_mod.get_pending(session_key)
    assert pending is not None

    resolved = await _slash_confirm_mod.resolve(
        session_key, pending["confirm_id"], "cancel",
    )

    runner._execute_update.assert_not_awaited()
    assert resolved is not None
    assert "cancelled" in resolved.lower()


@pytest.mark.asyncio
async def test_resolve_always_spawns_once_and_never_persists(monkeypatch):
    """Defense-in-depth: even if a stray 'always' arrives (e.g. a typed
    /always on a text-fallback platform), /update proceeds exactly once and
    NEVER writes a config opt-out."""
    from tools import slash_confirm as _slash_confirm_mod

    monkeypatch.setenv("HERMES_MANAGED", "")
    runner = _make_runner()
    session_key = build_session_key(_make_source())
    _slash_confirm_mod.clear(session_key)

    # If the handler ever tried to persist, this would record it.
    saved: dict = {}

    def _fake_save(path, value):
        saved[path] = value
        return True

    import cli as cli_mod
    monkeypatch.setattr(cli_mod, "save_config_value", _fake_save)

    await runner._handle_update_command(_make_event())
    pending = _slash_confirm_mod.get_pending(session_key)
    assert pending is not None

    resolved = await _slash_confirm_mod.resolve(
        session_key, pending["confirm_id"], "always",
    )

    # Proceeds once…
    runner._execute_update.assert_awaited_once()
    assert "Starting Hermes update" in resolved
    # …but persists NOTHING (no opt-out path exists for /update).
    assert saved == {}


# ---------------------------------------------------------------------------
# Pre-flight validation fires BEFORE the prompt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_managed_install_blocks_before_prompt(monkeypatch):
    """A managed install is rejected before any prompt is shown."""
    monkeypatch.setenv("HERMES_MANAGED", "homebrew")
    runner = _make_runner()

    result = await runner._handle_update_command(_make_event())

    assert "managed by Homebrew" in result
    runner._execute_update.assert_not_awaited()
    runner.adapters[Platform.TELEGRAM].send_slash_confirm.assert_not_awaited()

