"""Session-finalize plugin hooks must not block the gateway event loop.

A plugin ``on_session_finalize`` hook doing heavy synchronous work (e.g. an
observability plugin serializing a multi-day session's trace export) used to
run inline on the event loop from three call sites:

  * ``GatewayRunner._finalize_shutdown_agents`` (shutdown drain)
  * the session-expiry watcher
  * the ``/new`` session-reset handler

On a wedged/slow hook the whole loop froze — adapter heartbeats stopped and
systemd SIGKILLed the process mid-shutdown. All three sites now dispatch
through ``GatewayRunner._finalize_session_off_loop``, which runs
``hermes_cli.lifecycle.finalize_session`` in the gateway executor under a
bounded ``asyncio.wait_for``.
"""

import asyncio
import threading
import time

from gateway.run import GatewayRunner


def _make_runner():
    runner = object.__new__(GatewayRunner)
    return runner


def test_finalize_off_loop_invokes_lifecycle(monkeypatch):
    """The helper reaches the real lifecycle entry point with the kwargs."""
    calls = []

    def _fake_finalize(**kwargs):
        calls.append(kwargs)
        return []

    import hermes_cli.lifecycle as lifecycle

    monkeypatch.setattr(lifecycle, "finalize_session", _fake_finalize)

    runner = _make_runner()
    asyncio.run(
        runner._finalize_session_off_loop(
            session_id="s-123",
            platform="gateway",
            reason="shutdown",
            old_session_id="s-123",
        )
    )

    assert len(calls) == 1
    assert calls[0]["session_id"] == "s-123"
    assert calls[0]["reason"] == "shutdown"
    assert calls[0]["old_session_id"] == "s-123"


def test_finalize_off_loop_keeps_loop_alive_and_bounds_wedged_hook(monkeypatch):
    """A hook that blocks past the budget cannot freeze the event loop.

    The loop must keep servicing other callbacks while the hook runs, and
    the await must return once the budget expires even though the hook
    thread is still blocked.
    """
    release = threading.Event()

    def _wedged_finalize(**kwargs):
        # Simulates a multi-minute trace export.
        release.wait(timeout=30)

    import hermes_cli.lifecycle as lifecycle

    monkeypatch.setattr(lifecycle, "finalize_session", _wedged_finalize)

    runner = _make_runner()
    monkeypatch.setattr(GatewayRunner, "_FINALIZE_TIMEOUT_S", 0.5, raising=False)

    loop_ticks = []

    async def _ticker():
        while True:
            loop_ticks.append(time.monotonic())
            await asyncio.sleep(0.05)

    async def _scenario():
        ticker = asyncio.ensure_future(_ticker())
        started = time.monotonic()
        try:
            await runner._finalize_session_off_loop(
                session_id="s-wedged", platform="gateway", reason="shutdown"
            )
        finally:
            elapsed = time.monotonic() - started
            ticker.cancel()
        return elapsed

    elapsed = asyncio.run(_scenario())
    release.set()

    # Returned promptly at the budget, not after the 30s hook.
    assert elapsed < 5.0
    # The loop stayed live while the hook was blocked off-loop.
    assert len(loop_ticks) >= 3


def test_finalize_off_loop_swallows_hook_exceptions(monkeypatch):
    """A raising hook is contained — callers proceed with shutdown."""

    def _raising_finalize(**kwargs):
        raise RuntimeError("exporter blew up")

    import hermes_cli.lifecycle as lifecycle

    monkeypatch.setattr(lifecycle, "finalize_session", _raising_finalize)

    runner = _make_runner()
    # Must not raise.
    asyncio.run(
        runner._finalize_session_off_loop(
            session_id="s-err", platform="gateway", reason="shutdown"
        )
    )


def test_shutdown_finalize_path_uses_off_loop_dispatch(monkeypatch):
    """_finalize_shutdown_agents routes finalize through the bounded helper."""
    seen = []

    async def _fake_off_loop(self, **kwargs):
        seen.append(kwargs)

    monkeypatch.setattr(
        GatewayRunner, "_finalize_session_off_loop", _fake_off_loop
    )

    async def _fake_cleanup(self, agent, *, context=""):
        return None

    monkeypatch.setattr(
        GatewayRunner, "_cleanup_agent_resources_off_loop", _fake_cleanup
    )

    class _Agent:
        session_id = "s-shutdown"
        _session_messages = None

    runner = _make_runner()
    asyncio.run(runner._finalize_shutdown_agents({"k": _Agent()}))

    assert len(seen) == 1
    assert seen[0]["session_id"] == "s-shutdown"
    assert seen[0]["reason"] == "shutdown"
