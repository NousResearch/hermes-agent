"""Tests for the planned-stop marker watcher thread (gateway/run.py).

The watcher is the Windows-fallback path for the v0.13.0 session-resume
feature — on Windows ``asyncio.add_signal_handler`` raises
NotImplementedError, so the SIGTERM signal handler never runs and the
shutdown drain (which writes ``resume_pending=True``) is skipped. The
watcher closes this gap by polling for the planned-stop marker file
and translating its existence into the same shutdown-handler call a
real SIGTERM would have produced.

See issue #33778 for the original Windows session-loss bug report.
"""

import asyncio
import json
import os
import threading
import time
from unittest.mock import MagicMock


from gateway.run import _run_planned_stop_watcher
from gateway import status as status_mod


def _write_self_marker(marker, *, stale: bool = False):
    """Write a planned-stop marker that targets the CURRENT process.

    The watcher only fires for markers naming our PID + start_time (the
    fix for issue #34597), so tests that expect a fire must write a
    self-targeting marker. Pass ``stale=True`` to backdate ``written_at``
    past the TTL.
    """
    written_at = "2000-01-01T00:00:00+00:00" if stale else status_mod._utc_now_iso()
    record = {
        "target_pid": os.getpid(),
        "target_start_time": status_mod._get_process_start_time(os.getpid()),
        "stopper_pid": os.getpid(),
        "written_at": written_at,
    }
    marker.write_text(json.dumps(record), encoding="utf-8")


class _FakeRunner:
    """Stand-in for GatewayRunner — only exposes the two flags the watcher reads."""

    def __init__(self, *, running: bool = True, draining: bool = False):
        self._running = running
        self._draining = draining


def _make_loop_capturing_calls():
    """Build a fake asyncio loop whose call_soon_threadsafe records its args."""
    loop = MagicMock(spec=asyncio.AbstractEventLoop)
    loop._captured = []

    def fake_call_soon_threadsafe(fn, *args):
        loop._captured.append((fn, args))

    loop.call_soon_threadsafe = fake_call_soon_threadsafe
    return loop


def test_watcher_fires_shutdown_when_marker_appears(tmp_path, monkeypatch):
    """When a marker targeting THIS process exists, fire the shutdown handler."""
    marker = tmp_path / ".gateway-planned-stop.json"

    # Patch the marker-path resolver so the watcher polls our temp location.
    monkeypatch.setattr(status_mod, "_get_planned_stop_marker_path", lambda: marker)

    runner = _FakeRunner(running=True, draining=False)
    loop = _make_loop_capturing_calls()
    shutdown_handler = MagicMock(name="shutdown_signal_handler")
    stop_event = threading.Event()

    # Drop a self-targeting marker before the thread starts.
    _write_self_marker(marker)

    watcher = threading.Thread(
        target=_run_planned_stop_watcher,
        args=(stop_event, runner, loop, shutdown_handler),
        kwargs={"poll_interval": 0.05},
        daemon=True,
    )
    watcher.start()
    watcher.join(timeout=10.0)

    assert not watcher.is_alive(), "Watcher should exit after firing"
    assert len(loop._captured) == 1, (
        f"Expected exactly one shutdown invocation, got {loop._captured}"
    )
    fn, args = loop._captured[0]
    assert fn is shutdown_handler
    # The handler must be called with signal=None (planned stop sentinel).
    assert args == (None,)


def test_watcher_tolerates_marker_path_resolution_errors(tmp_path, monkeypatch, caplog):
    """If _get_planned_stop_marker_path() raises, the watcher logs and continues."""
    from gateway import status as status_mod

    call_count = [0]
    def explode():
        call_count[0] += 1
        # First call (the one outside the loop, at thread start) is fine —
        # but subsequent .exists() calls on a corrupt Path could explode.
        if call_count[0] == 1:
            return tmp_path / "nonexistent"
        raise OSError("filesystem failed")

    monkeypatch.setattr(status_mod, "_get_planned_stop_marker_path", explode)

    runner = _FakeRunner(running=True, draining=False)
    loop = _make_loop_capturing_calls()
    stop_event = threading.Event()

    watcher = threading.Thread(
        target=_run_planned_stop_watcher,
        args=(stop_event, runner, loop, MagicMock()),
        kwargs={"poll_interval": 0.05},
        daemon=True,
    )
    watcher.start()
    time.sleep(0.2)
    stop_event.set()
    watcher.join(timeout=10.0)

    assert not watcher.is_alive(), "Watcher should still honour stop_event after errors"
    # No shutdown fired because the marker never reported existence.
    assert loop._captured == []


# ---------------------------------------------------------------------------
# Regression coverage for issue #34597:
# A marker left behind by a PREVIOUS gateway instance (different PID, or
# past its TTL) must NOT crash the freshly booted gateway. The watcher
# only fires when the marker targets the current process, and self-heals
# by cleaning up stale/malformed markers.
# ---------------------------------------------------------------------------


def test_watcher_does_not_fire_for_foreign_pid_marker(tmp_path, monkeypatch):
    """A marker naming a DIFFERENT process must not trigger our shutdown.

    This is the core #34597 regression: a stale marker from a prior
    gateway instance was firing the handler, driving the new gateway into
    a false "Received UNKNOWN" shutdown and a watchdog crash loop.
    """
    marker = tmp_path / ".gateway-planned-stop.json"
    # Foreign PID + a start_time that cannot match ours, freshly written
    # so the TTL does NOT remove it — the watcher must still decline.
    record = {
        "target_pid": os.getpid() + 1,
        "target_start_time": -1,
        "stopper_pid": os.getpid() + 1,
        "written_at": status_mod._utc_now_iso(),
    }
    marker.write_text(json.dumps(record), encoding="utf-8")

    monkeypatch.setattr(status_mod, "_get_planned_stop_marker_path", lambda: marker)

    runner = _FakeRunner(running=True, draining=False)
    loop = _make_loop_capturing_calls()
    shutdown_handler = MagicMock(name="shutdown_signal_handler")
    stop_event = threading.Event()

    watcher = threading.Thread(
        target=_run_planned_stop_watcher,
        args=(stop_event, runner, loop, shutdown_handler),
        kwargs={"poll_interval": 0.05},
        daemon=True,
    )
    watcher.start()
    time.sleep(0.2)  # several poll cycles
    stop_event.set()
    watcher.join(timeout=10.0)

    assert not watcher.is_alive()
    assert loop._captured == [], (
        f"Watcher fired on a foreign-PID marker (#34597 regression): {loop._captured}"
    )
    shutdown_handler.assert_not_called()
    # Foreign (but live) marker is left in place — it may still belong to
    # the process it names.
    assert marker.exists()


def test_planned_stop_marker_targets_self_probe_is_non_destructive(tmp_path, monkeypatch):
    """The probe returns True for a self-marker WITHOUT unlinking it.

    The shutdown handler performs the authoritative consume on its own
    thread, so the watcher's probe must leave a matching marker intact.
    """
    marker = tmp_path / ".gateway-planned-stop.json"
    _write_self_marker(marker)
    monkeypatch.setattr(status_mod, "_get_planned_stop_marker_path", lambda: marker)

    assert status_mod.planned_stop_marker_targets_self() is True
    assert marker.exists(), "Probe must not consume a matching marker"
    # Idempotent: still True on a second call.
    assert status_mod.planned_stop_marker_targets_self() is True


