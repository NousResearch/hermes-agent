"""Regression tests: the gateway heartbeat poller must follow session rotation.

``/heartbeat`` registers a watch as ``quick_key -> (source, session_id)``, and
before this fix the poller kept polling that captured ``session_id`` forever.

A context compression rotates the route onto a fresh session and carries the
heartbeat across with ``migrate_heartbeat_to_session``, which marks the parent
row ``cleared``. The poller, still holding the parent id, read that cleared row,
concluded the user had removed the heartbeat, and dropped the watch — so the
heartbeat went silent for the rest of the gateway process. Nothing surfaced it:
``/heartbeat status`` resolves the *live* session, so it kept reporting the
heartbeat as active with a ticking countdown.

The poller now re-resolves the route's current session id every tick through
``_live_heartbeat_session_id`` (``SessionStore.peek_session_id``, the same
mapping ``advance_compression_session`` repairs on rotation).
"""
import asyncio
import time
from types import SimpleNamespace

import pytest

from hermes_cli.heartbeat import (
    HeartbeatState,
    migrate_heartbeat_to_session,
    save_heartbeat,
)


def _make_runner(routes, running=()):
    """Bare GatewayRunner with just the collaborators the poller touches.

    ``routes`` is the persisted session-key → session-id mapping that
    ``SessionStore.peek_session_id`` reads; mutating it mid-test is exactly what
    a compression rotation does to the real store.
    """
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.session_store = SimpleNamespace(
        peek_session_id=lambda key: routes.get(key)
    )
    # _running_agents is a mapping-backed session field view, not a plain set.
    runner._running_agents = {key: object() for key in running}
    runner._adapter_for_source = lambda source: object()
    runner._background_tasks = set()

    enqueued = []
    runner._enqueue_fifo = lambda key, event, adapter: enqueued.append((key, event))
    return runner, enqueued


async def _let_poller_tick(runner, ticks=6):
    """Yield to the poll loop long enough for a few iterations."""
    for _ in range(ticks):
        await asyncio.sleep(0)
        await asyncio.sleep(0.01)
    task = getattr(runner, "_heartbeat_poll_task", None)
    if task is not None:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


def _due_state(prompt="Check the deploy", interval=600):
    """A heartbeat whose next tick is already due."""
    return HeartbeatState(
        prompt=prompt,
        interval_seconds=interval,
        created_at=time.time() - (interval + 100),
    )


@pytest.mark.asyncio
async def test_poller_follows_compression_rotation(monkeypatch):
    """The heartbeat keeps firing after the route rotates onto a new session."""
    monkeypatch.setattr("hermes_cli.heartbeat.POLL_SECONDS", 0.01)

    parent, child = "rot-parent-sid", "rot-child-sid"
    save_heartbeat(parent, _due_state())

    routes = {"route-rot": parent}
    runner, enqueued = _make_runner(routes)
    runner._register_heartbeat_watch("route-rot", "source-obj", parent)

    # Compression rotates the route and carries the heartbeat to the child.
    assert migrate_heartbeat_to_session(parent, child) is True
    routes["route-rot"] = child

    await _let_poller_tick(runner)

    assert enqueued, "heartbeat never fired after the session rotated"
    key, event = enqueued[0]
    assert key == "route-rot"
    assert "Check the deploy" in event.text

    # The watch survived and now caches the child id, so the next tick starts
    # from the live session instead of re-resolving from a stale one.
    assert runner._heartbeat_watch["route-rot"] == ("source-obj", child)


@pytest.mark.asyncio
async def test_poller_drops_watch_when_heartbeat_is_gone(monkeypatch):
    """A route with no heartbeat on its live session still unregisters."""
    monkeypatch.setattr("hermes_cli.heartbeat.POLL_SECONDS", 0.01)

    routes = {"route-gone": "gone-sid"}
    runner, enqueued = _make_runner(routes)
    runner._register_heartbeat_watch("route-gone", "source-obj", "gone-sid")

    await _let_poller_tick(runner)

    assert enqueued == []
    assert "route-gone" not in runner._heartbeat_watch


@pytest.mark.asyncio
async def test_busy_route_coalesces_instead_of_firing(monkeypatch):
    """An in-flight turn defers the tick — unchanged by the rotation fix."""
    monkeypatch.setattr("hermes_cli.heartbeat.POLL_SECONDS", 0.01)

    save_heartbeat("busy-sid", _due_state())
    routes = {"route-busy": "busy-sid"}
    runner, enqueued = _make_runner(routes, running=("route-busy",))
    runner._register_heartbeat_watch("route-busy", "source-obj", "busy-sid")

    await _let_poller_tick(runner)

    assert enqueued == []
    # Deferred, not dropped.
    assert "route-busy" in runner._heartbeat_watch


def test_live_session_id_prefers_the_route_mapping():
    runner, _ = _make_runner({"route-a": "live-sid"})
    assert runner._live_heartbeat_session_id("route-a", "stale-sid") == "live-sid"


def test_live_session_id_falls_back_for_unknown_route():
    """An unmapped route keeps the captured id rather than resolving to None."""
    runner, _ = _make_runner({})
    assert runner._live_heartbeat_session_id("route-missing", "stale-sid") == "stale-sid"


def test_live_session_id_falls_back_when_the_store_raises():
    """A failing lookup must not take the poller down."""
    from gateway.run import GatewayRunner

    def _boom(_key):
        raise RuntimeError("store unavailable")

    runner = object.__new__(GatewayRunner)
    runner.session_store = SimpleNamespace(peek_session_id=_boom)
    assert runner._live_heartbeat_session_id("route-x", "stale-sid") == "stale-sid"
