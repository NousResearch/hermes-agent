"""The gateway heartbeat poller must survive a compression session rotation.

``/heartbeat`` registers a watch keyed by the session id that was current when
the command ran. Context compression re-homes the heartbeat onto the
continuation session and clears the parent row, so that captured id reads
empty the first time the session rotates — and the poller used to take that as
"the user cleared it" and unregister the watch. The heartbeat then never fired
again, silently, in exactly the long-running sessions it exists for.
"""

from __future__ import annotations

import asyncio
import types

import pytest

from gateway.run import GatewayRunner


class _StubSessionDB:
    """Minimal AsyncSessionDB stand-in exposing the compression chain."""

    def __init__(self, tips: dict[str, str]):
        self._tips = tips
        self.lookups: list[str] = []

    async def get_compression_tip(self, session_id: str) -> str:
        self.lookups.append(session_id)
        return self._tips.get(session_id, session_id)


def _make_runner(watch: dict, tips: dict[str, str]):
    runner = types.SimpleNamespace()
    runner._heartbeat_watch = watch
    runner._running_agents = {}
    runner._session_db = _StubSessionDB(tips)
    runner._background_tasks = set()
    runner._heartbeat_poll_task = None
    runner.enqueued: list = []
    runner._adapter_for_source = lambda source: object()
    runner._enqueue_fifo = lambda qk, ev, ad: runner.enqueued.append((qk, ev))

    async def _tip(session_id: str) -> str:
        return await runner._session_db.get_compression_tip(session_id)

    runner._heartbeat_session_tip = _tip
    runner._start_heartbeat_poller = types.MethodType(
        GatewayRunner._start_heartbeat_poller, runner,
    )
    return runner


async def _run_one_poll(runner) -> None:
    runner._start_heartbeat_poller()
    task = runner._heartbeat_poll_task
    try:
        await asyncio.sleep(0.15)
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


@pytest.fixture()
def hb(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    from hermes_cli import heartbeat as module

    monkeypatch.setattr(module, "POLL_SECONDS", 0.01)
    return module


@pytest.mark.asyncio
async def test_poller_follows_the_compression_rotation(hb):
    """A rotated session rebinds the watch instead of dropping the heartbeat."""
    hb.HeartbeatManager("sess-parent").set("Check the deployment", 600)
    assert hb.migrate_heartbeat_to_session("sess-parent", "sess-child") is True

    source = object()
    watch = {"chat:1": (source, "sess-parent")}
    runner = _make_runner(watch, {"sess-parent": "sess-child"})

    await _run_one_poll(runner)

    assert "chat:1" in watch, "live heartbeat was unregistered after rotation"
    assert watch["chat:1"] == (source, "sess-child")


@pytest.mark.asyncio
async def test_poller_unregisters_a_genuinely_cleared_heartbeat(hb):
    """/heartbeat clear on a session that never rotated still unregisters.

    Pins the boundary: following the chain must not keep dead watches alive.
    """
    mgr = hb.HeartbeatManager("sess-solo")
    mgr.set("Check the deployment", 600)
    mgr.clear()

    watch = {"chat:1": (object(), "sess-solo")}
    runner = _make_runner(watch, {})

    await _run_one_poll(runner)

    assert watch == {}


@pytest.mark.asyncio
async def test_poller_unregisters_when_the_tip_has_no_heartbeat(hb):
    """Rotation plus a real clear still unregisters — the tip is authoritative."""
    hb.HeartbeatManager("sess-parent").set("Check the deployment", 600)
    hb.migrate_heartbeat_to_session("sess-parent", "sess-child")
    hb.HeartbeatManager("sess-child").clear()

    watch = {"chat:1": (object(), "sess-parent")}
    runner = _make_runner(watch, {"sess-parent": "sess-child"})

    await _run_one_poll(runner)

    assert watch == {}


class _ExplodingSessionDB:
    async def get_compression_tip(self, session_id: str) -> str:
        raise RuntimeError("db down")


@pytest.mark.asyncio
async def test_session_tip_resolves_and_degrades_to_the_input():
    """The tip helper follows the chain and never raises into the poll loop."""
    runner = types.SimpleNamespace()
    resolve = types.MethodType(GatewayRunner._heartbeat_session_tip, runner)

    runner._session_db = _StubSessionDB({"sess-parent": "sess-child"})
    assert await resolve("sess-parent") == "sess-child"
    assert await resolve("sess-solo") == "sess-solo"

    runner._session_db = None
    assert await resolve("sess-parent") == "sess-parent"

    runner._session_db = _ExplodingSessionDB()
    assert await resolve("sess-parent") == "sess-parent"
