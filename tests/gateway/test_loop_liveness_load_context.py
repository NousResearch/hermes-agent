"""The loop-liveness CRITICAL must name host LOAD, not just the missed probes.

2026-09-20 incident: a kanban worker's runaway busy-loops drove the host to load
average 538 on 32 cores. The resident gateway's loop-liveness watchdog fired
twice with "Gateway event loop missed 3 consecutive liveness probes; exiting
with code 75" — a line that is true and useless. It describes the SYMPTOM
(the loop did not answer) and omits the one number that names the CAUSE, so the
first 55 minutes of the incident were spent looking for a deadlock in the
gateway rather than at ``uptime``.

Attaching ``load1=`` and ``ncpu=`` to that same line costs one ``os.getloadavg``
call on a path that is already hard-exiting, and makes the next occurrence
self-diagnosing from the log alone.
"""

from __future__ import annotations

import asyncio
import re
import threading
from unittest.mock import MagicMock, patch

from gateway.shutdown_watchdog import start_loop_liveness_watchdog


def test_missed_probe_critical_carries_load1_and_ncpu():
    """The CRITICAL emitted before hard-exit names host load and cpu count."""
    loop = MagicMock(spec=asyncio.AbstractEventLoop)
    # Never schedules the probe callback -> the probe always times out.
    loop.call_soon_threadsafe.side_effect = lambda callback: None
    exited = threading.Event()
    messages: list[str] = []

    def _record(msg, *args, **_kwargs):
        try:
            messages.append(msg % args if args else str(msg))
        except Exception:
            messages.append(str(msg))

    with (
        patch("gateway.shutdown_watchdog.logger.critical", side_effect=_record),
        patch("gateway.shutdown_watchdog.faulthandler.dump_traceback"),
        patch(
            "gateway.shutdown_watchdog.os._exit",
            side_effect=lambda _code: exited.set(),
        ),
    ):
        handle = start_loop_liveness_watchdog(
            loop, probe_interval=0.01, probe_timeout=0.01, max_strikes=1
        )
        assert handle is not None
        assert exited.wait(timeout=10.0), "watchdog never reached the exit path"
        handle.stop()

    joined = "\n".join(messages)
    assert "missed" in joined and "liveness probes" in joined
    assert re.search(r"load1=[0-9]+\.[0-9]+", joined), (
        "loop-liveness CRITICAL does not report the host 1-min load average; "
        f"got: {joined!r}"
    )
    assert re.search(r"ncpu=[0-9]+", joined), (
        f"loop-liveness CRITICAL does not report ncpu; got: {joined!r}"
    )


def test_load_snapshot_degrades_gracefully_without_getloadavg():
    """A platform without ``os.getloadavg`` must not break the exit path."""
    from gateway import shutdown_watchdog as sw

    with patch.object(sw.os, "getloadavg", side_effect=OSError("unsupported")):
        text = sw._host_load_suffix()
    assert isinstance(text, str)
    assert "load1=" in text  # still emitted, as 'unknown'
