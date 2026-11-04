"""The loop-liveness dump keeps the event-loop thread's stack past faulthandler's 100-thread cap.

``faulthandler.dump_traceback(all_threads=True)`` writes at most 100 threads, newest first, so the
loop thread (started before every worker) is the one it drops in a busy gateway. Both copies of
the dump (file and stderr, the stderr copy is the only one a journald/launchd host keeps when the
disk is wedged) must carry the loop thread's own stack ahead of the capped section.
"""

from __future__ import annotations

import asyncio
import io
import threading
from unittest.mock import patch

from gateway.shutdown_watchdog import get_shutdown_watchdog_dump_path, start_loop_liveness_watchdog

_FILLER_THREADS = 110


def _wedge_loop_behind_fillers(tmp_path, monkeypatch, secret=None):
    """Run a real loop on a NON-main thread, wedge it, start 110 filler threads, fire the watchdog.

    Returns (file dump text, stderr text)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    state: dict[str, object] = {"secret": secret}
    frozen, release, fired, fillers_up = (threading.Event() for _ in range(4))
    stop_fillers = threading.Event()

    def wedged_loop_callback_marker() -> None:  # the frame the dump has to show
        held_credential = state.get("secret")  # a live local: its VALUE must never reach the dump
        frozen.set()
        fillers_up.wait(timeout=10.0)
        release.wait(timeout=20.0)
        del held_credential

    def drive_loop() -> None:
        loop = asyncio.new_event_loop()
        state["loop"] = loop
        loop.call_soon(lambda: start_loop_liveness_watchdog(
            loop, probe_interval=0.05, probe_timeout=0.05, max_strikes=1))
        loop.call_soon(wedged_loop_callback_marker)
        loop.run_forever()
        loop.close()

    fillers = [threading.Thread(target=stop_fillers.wait, args=(20.0,), daemon=True, name=f"filler-{i}")
               for i in range(_FILLER_THREADS)]
    loop_thread = threading.Thread(target=drive_loop, name="test-gateway-loop", daemon=True)
    stderr = io.StringIO()
    with patch("gateway.shutdown_watchdog._hard_exit", side_effect=lambda code: fired.set()), \
            patch("gateway.shutdown_watchdog._mark_exited_quietly"), \
            patch("gateway.shutdown_watchdog.sys.stderr", stderr), \
            patch("gateway.shutdown_watchdog.faulthandler.dump_traceback",
                  side_effect=lambda file=None, all_threads=True: (file or stderr).write(
                      "Thread 0x1 (most recent call first):\n...\n")):
        loop_thread.start()
        assert frozen.wait(timeout=5.0), "loop thread never reached its wedged callback"
        for t in fillers:
            t.start()  # started AFTER the loop: newest-first truncation drops the loop thread
        fillers_up.set()
        try:
            assert fired.wait(timeout=10.0), "watchdog never fired"
        finally:
            release.set()
            state["loop"].call_soon_threadsafe(state["loop"].stop)
            loop_thread.join(timeout=5.0)
            stop_fillers.set()
            for t in fillers:
                t.join(timeout=5.0)
    assert threading.active_count() < _FILLER_THREADS
    dump = get_shutdown_watchdog_dump_path(tmp_path).read_text(encoding="utf-8-sig")
    return dump, stderr.getvalue()


def test_file_dump_leads_with_the_wedged_loop_thread_past_100_threads(tmp_path, monkeypatch):
    dump, _ = _wedge_loop_behind_fillers(tmp_path, monkeypatch)
    leading = dump.split("--- faulthandler dump (all threads) ---")[0]
    assert "wedged_loop_callback_marker" in leading, "loop thread stack missing from the file dump"
    assert "test-gateway-loop" in leading


def test_stderr_dump_carries_the_loop_thread_stack_too(tmp_path, monkeypatch):
    _, stderr = _wedge_loop_behind_fillers(tmp_path, monkeypatch)
    assert "wedged_loop_callback_marker" in stderr, (
        "the stderr copy (journald/launchd) must carry the loop thread's stack, not only the capped dump")
    assert stderr.index("wedged_loop_callback_marker") < stderr.index("most recent call first")


def test_loop_stack_section_carries_no_frame_locals(tmp_path, monkeypatch):
    """The section is a code-location stack (file, line, function, source line), never values."""
    secret = "-".join(["sk", "test", "SHOULD", "NOT", "APPEAR"])  # built at runtime: not in any source line
    dump, stderr = _wedge_loop_behind_fillers(tmp_path, monkeypatch, secret=secret)
    assert "wedged_loop_callback_marker" in dump  # the frame holding the secret IS in the dump
    assert secret not in dump and secret not in stderr
