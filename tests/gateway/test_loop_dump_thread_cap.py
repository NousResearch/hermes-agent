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
    assert stderr.index("wedged_loop_callback_marker") < stderr.index("Python thread stacks (innermost frame last)")


def test_loop_stack_section_carries_no_frame_locals(tmp_path, monkeypatch):
    """The section is a code-location stack (file, line, function), never values."""
    secret = "-".join(["sk", "test", "SHOULD", "NOT", "APPEAR"])  # built at runtime: not in any source line
    dump, stderr = _wedge_loop_behind_fillers(tmp_path, monkeypatch, secret=secret)
    assert "wedged_loop_callback_marker" in dump  # the frame holding the secret IS in the dump
    assert secret not in dump and secret not in stderr


def test_stderr_stacks_are_capped_and_read_no_source(monkeypatch):
    """Both stack sections open no source file on the kill path; the all-thread one keeps
    faulthandler's thread cap."""
    import gateway.shutdown_watchdog as watchdog

    monkeypatch.setattr(watchdog, "_STDERR_STACK_THREADS", 2)
    release = threading.Event()
    extra = [threading.Thread(target=release.wait, daemon=True) for _ in range(3)]
    for thread in extra:
        thread.start()
    source_reads = []
    try:
        with monkeypatch.context() as patched:  # undone before the asserts: pytest reads source
            patched.setattr("linecache.getline", lambda *args, **_: source_reads.append(args) or "")
            patched.setattr("linecache.updatecache", lambda *args, **_: source_reads.append(args) or [])
            text = watchdog._format_thread_stacks()
            loop_section = watchdog._format_loop_thread_section(threading.get_ident())
    finally:
        release.set()
    assert source_reads == [], "a stack renderer read a source file on the kill path"
    assert "test_stderr_stacks_are_capped_and_read_no_source" in loop_section
    assert text.count("\nThread 0x") == 2
    assert "more threads not shown" in text
    assert 'File "' in text and ", line " in text
