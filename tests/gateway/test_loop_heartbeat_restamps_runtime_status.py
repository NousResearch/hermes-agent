"""The loop heartbeat must also re-stamp ``gateway_state.json`` (#*).

``run.py`` writes that file on lifecycle transitions and at in-process turn boundaries, and nowhere
else. An idle gateway therefore stops touching it -- and so does a busy one whose work is spawning
kanban *workers*, since those are separate processes rather than gateway turns. Its ``updated_at`` then
ages without bound while the gateway is alive, and every consumer that reads recency off that field
concludes the opposite. Measured against hermes-webui, which uses it as the cross-container liveness
signal with a 120s window: ``gateway_stale_running_state`` reported against a gateway whose
``updated_at`` was 2h16m old, with ``active_agents: 0`` and pid 1 = ``hermes gateway run``.

The heartbeat loop is the right owner because it already ticks well inside that window and already ages
when the loop freezes, so both files keep the same liveness meaning.
"""

import asyncio
from pathlib import Path
from unittest.mock import patch

from gateway.shutdown_watchdog import loop_heartbeat_forever


def _run_one_tick(**kwargs):
    """Drive the loop for a single write, then cancel it."""
    async def main():
        task = asyncio.create_task(loop_heartbeat_forever(interval_s=1.0, **kwargs))
        for _ in range(200):                      # the first write is immediate; give it a moment
            await asyncio.sleep(0.01)
            if calls["heartbeat"]:
                break
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    calls = {"heartbeat": 0, "runtime": 0}

    def _heartbeat(**_kw):
        calls["heartbeat"] += 1

    def _runtime(*_a, **_kw):
        calls["runtime"] += 1

    with patch("gateway.shutdown_watchdog.write_loop_heartbeat", side_effect=_heartbeat), \
         patch("gateway.status.write_runtime_status", side_effect=_runtime):
        asyncio.run(main())
    return calls


def test_heartbeat_tick_also_restamps_runtime_status():
    calls = _run_one_tick()
    assert calls["heartbeat"] >= 1
    assert calls["runtime"] >= 1, "gateway_state.json was not re-stamped by the heartbeat tick"


def test_home_override_skips_the_runtime_status_write(tmp_path: Path):
    """The runtime-status path comes from the ambient HERMES_HOME and takes no ``home`` argument.

    A caller that redirects ``home`` (only tests do) must not have its heartbeat write into one place
    and its runtime status into another -- the developer's real Hermes home.
    """
    calls = _run_one_tick(home=tmp_path)
    assert calls["heartbeat"] >= 1
    assert calls["runtime"] == 0, "a home override still wrote to the ambient runtime status path"


def test_a_failing_runtime_status_write_does_not_kill_the_heartbeat():
    """Liveness reporting must never be able to stop liveness reporting."""
    calls = {"heartbeat": 0}

    async def main():
        task = asyncio.create_task(loop_heartbeat_forever(interval_s=0.05))
        for _ in range(200):
            await asyncio.sleep(0.01)
            if calls["heartbeat"] >= 2:
                break
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    def _heartbeat(**_kw):
        calls["heartbeat"] += 1

    with patch("gateway.shutdown_watchdog.write_loop_heartbeat", side_effect=_heartbeat), \
         patch("gateway.status.write_runtime_status", side_effect=RuntimeError("disk full")):
        asyncio.run(main())

    assert calls["heartbeat"] >= 2, "the heartbeat stopped after a runtime-status write failed"
