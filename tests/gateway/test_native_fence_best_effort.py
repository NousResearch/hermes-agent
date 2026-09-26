"""Native-turn boundary fences on the gateway are best-effort.

Every caller of ``_fence_native_gateway_key`` / ``_fence_native_gateway_session`` /
``_fence_all_native_gateway_sessions`` (``/stop``, ``/new``, ``/resume``, ``stop()``) runs
mandatory interrupt/reset/switch/shutdown cleanup right after the fence. A failing
session-store read or identity lookup inside the fence must therefore be logged and
swallowed — never propagated into the caller — while cancellation still propagates.
"""
import asyncio
from types import SimpleNamespace

import pytest

from gateway.run import GatewayRunner


class _SyncStore:
    """Minimal synchronous SessionStore double behind the real AsyncSessionStore facade."""

    def __init__(self, entries=None, exc=None, current=None):
        self.entries, self.exc, self.current = list(entries or []), exc, current

    def list_sessions(self, *args, **kwargs):
        if self.exc is not None:
            raise self.exc
        return list(self.entries)

    def get_or_create_session(self, *args, **kwargs):
        return self.current


def _runner(store, *, view=None):
    runner = object.__new__(GatewayRunner)
    runner.session_store = store  # async_session_store wraps this in the real facade
    if view is not None:
        runner._native_gateway_view = view
    runner._session_key_for_source = lambda source: "telegram:dm:123"
    return runner


@pytest.mark.asyncio
async def test_fence_key_swallows_store_failure():
    runner = _runner(_SyncStore(exc=RuntimeError("native fence boom")))
    await GatewayRunner._fence_native_gateway_key(runner, "telegram:dm:123", "user_stop")


@pytest.mark.asyncio
async def test_fence_session_swallows_view_failure():
    entry = SimpleNamespace(session_key="telegram:dm:123", session_id="s1", origin=object())

    async def _boom_view(_entry):
        raise RuntimeError("identity lookup boom")

    runner = _runner(_SyncStore(entries=[entry]), view=_boom_view)
    await GatewayRunner._fence_native_gateway_session(runner, object(), "new_session")


@pytest.mark.asyncio
async def test_fence_all_continues_past_a_bad_route(monkeypatch):
    """One failing route is logged and skipped; the remaining routes are still fenced."""
    import hermes_cli.native_turn_sources as nts

    fenced = []
    monkeypatch.setattr(nts, "fence_native_turn_sources", lambda view, reason: fenced.append((view, reason)))
    bad = SimpleNamespace(session_key="k-bad", session_id="s-bad", origin=object())
    good = SimpleNamespace(session_key="k-good", session_id="s-good", origin=object())

    async def _view(entry):
        if entry is bad:
            raise RuntimeError("identity lookup boom")
        return "view-good"

    runner = _runner(_SyncStore(entries=[bad, good]), view=_view)
    await GatewayRunner._fence_all_native_gateway_sessions(runner, "user_exit")
    assert fenced == [("view-good", "user_exit")]


@pytest.mark.asyncio
async def test_fence_all_swallows_listing_failure():
    runner = _runner(_SyncStore(exc=RuntimeError("native fence boom")))
    await GatewayRunner._fence_all_native_gateway_sessions(runner, "user_exit")


@pytest.mark.asyncio
async def test_fence_propagates_cancellation():
    """``except Exception`` must not swallow cancellation (a BaseException)."""
    entry = SimpleNamespace(session_key="telegram:dm:123", session_id="s1", origin=object())

    async def _cancel_view(_entry):
        raise asyncio.CancelledError()

    runner = _runner(_SyncStore(entries=[entry]), view=_cancel_view)
    with pytest.raises(asyncio.CancelledError):
        await GatewayRunner._fence_native_gateway_key(runner, "telegram:dm:123", "user_stop")
    with pytest.raises(asyncio.CancelledError):
        await GatewayRunner._fence_all_native_gateway_sessions(runner, "user_exit")


@pytest.mark.asyncio
async def test_stop_command_still_interrupts_when_fence_store_fails():
    """/stop must honour the interrupt even when the native fence's store read fails."""
    from gateway.config import Platform
    from gateway.platforms.event import MessageEvent
    from gateway.session import SessionSource

    source = SessionSource(platform=Platform.TELEGRAM, chat_id="123", chat_type="dm", user_id="u1")
    key = "telegram:dm:123"
    stopped = []
    store = _SyncStore(exc=RuntimeError("native fence boom"), current=SimpleNamespace(session_key=key))
    runner = _runner(store)
    runner._running_agents = {key: object()}

    async def _interrupt(k, _source, **kwargs):
        stopped.append((k, kwargs.get("invalidation_reason")))

    runner._interrupt_and_clear_session = _interrupt
    await GatewayRunner._handle_stop_command(runner, MessageEvent(text="/stop", source=source))
    assert stopped == [(key, "stop_command_handler")]
