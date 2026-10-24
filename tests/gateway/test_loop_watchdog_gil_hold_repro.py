"""REPRO + ARCHITECTURAL GUARD: a GIL-holding wedge defeats any IN-PROCESS watchdog.

This is the root cause of the 2026-10-23 gateway wedge escaping the built-in safety net.

``start_loop_liveness_watchdog`` probes the event loop from a thread inside the SAME
process. A wedge whose root cause is a C-extension call that HOLDS the GIL starves that
thread exactly as hard as it starves the event loop, so the strike never lands,
``max_strikes`` is never reached, and the gateway stays wedged. That is exactly what the
incident showed: the loop was frozen for 27 minutes and there is no watchdog strike
anywhere in the journal.

This test documents WHY the in-process watchdog cannot be the fix, and therefore WHY the
backstop has to live in another PROCESS (systemd's ``WatchdogSec``, whose timer runs in the
systemd manager and never touches the gateway's GIL).

THE POINT OF THIS FILE
----------------------
1. ``test_gil_holding_call_actually_holds_the_gil`` proves the fixture really holds the GIL.
2. ``test_in_process_watchdog_thread_is_starved_by_a_gil_hold`` proves the watchdog thread
   cannot run while the GIL is held.
3. ``test_in_process_watchdog_cannot_escalate_a_gil_holding_wedge`` records the consequence:
   the in-process watchdog does NOT fire. This is a DOCUMENTED LIMITATION, not a bug to be
   "fixed" in-process -- the fix is the external systemd watchdog
   (``gateway.systemd_watchdog_seconds``), covered by
   ``tests/hermes_cli/test_systemd_watchdog_unit.py`` and
   ``tests/gateway/test_systemd_notify.py``.

HOW THE GIL IS ACTUALLY HELD HERE
----------------------------------
``time.sleep()`` RELEASES the GIL, so a blocker built on it does NOT reproduce the failure
(an earlier revision of this file used ``time.sleep`` and the watchdog fired anyway, which
is why the old "RED" assertion failed). A real C-extension call that keeps the GIL is
reproduced with a ``ctypes.PYFUNCTYPE`` binding: unlike ``CFUNCTYPE``, ``PYFUNCTYPE`` does
NOT release the GIL around the call, so the calling thread holds it for the whole call.
"""

from __future__ import annotations

import asyncio
import ctypes
import threading
import time

from gateway.shutdown_watchdog import start_loop_liveness_watchdog


def _gil_holding_call(seconds: float) -> None:
    """Hold the GIL for *seconds* using a ctypes binding that does not release it.

    ``PYFUNCTYPE`` keeps the GIL held across the call (the default ``CFUNCTYPE`` releases
    it). ``libc.usleep`` then blocks the calling thread in the kernel with the GIL still
    held, which is the closest stdlib stand-in for a blocking C extension.
    """
    libc = ctypes.CDLL("libc.so.6")
    usleep = ctypes.PYFUNCTYPE(ctypes.c_int, ctypes.c_uint)(("usleep", libc))
    usleep(int(seconds * 1_000_000))


def test_gil_holding_call_actually_holds_the_gil():
    """Sanity check on the fixture: the helper must starve other Python threads.

    Without this, the tests below could pass for the wrong reason (e.g. if a future ctypes
    change started releasing the GIL and the blocker stopped being a GIL-hold at all).
    """
    ticks: list[float] = []
    stop = threading.Event()

    def _ticker():
        while not stop.wait(timeout=0.05):
            ticks.append(time.monotonic())

    t = threading.Thread(target=_ticker, daemon=True)
    t.start()
    time.sleep(0.3)
    before = len(ticks)
    _gil_holding_call(2.0)
    after = len(ticks)
    stop.set()
    t.join(timeout=2)

    # During a 2s GIL hold a 50ms ticker can manage at most a couple of wake-ups; without a
    # GIL hold it would land roughly 40 ticks in the same window.
    assert (after - before) < 15, (
        f"ticker advanced {after - before} times during a 2s GIL hold -- the fixture is not "
        f"actually holding the GIL"
    )


def test_in_process_watchdog_thread_is_starved_by_a_gil_hold():
    """The watchdog's own thread cannot run while the GIL is held."""
    observed: list[float] = []
    stop = threading.Event()

    def _ticker():
        while not stop.is_set():
            observed.append(time.monotonic())
            time.sleep(0.1)

    t = threading.Thread(target=_ticker, daemon=True)
    t.start()
    time.sleep(0.4)
    before = len(observed)
    _gil_holding_call(2.0)
    starved = len(observed) - before
    stop.set()
    t.join(timeout=2)

    # ~20 wake-ups expected in 2s without a GIL hold; a handful at most under one.
    assert starved < 8, (
        f"watchdog-style thread advanced {starved} times during a 2s GIL hold -- the fixture "
        f"is not actually starving the thread"
    )


def test_in_process_watchdog_cannot_escalate_a_gil_holding_wedge():
    """DOCUMENTED LIMITATION: the in-process watchdog does not fire on a GIL-holding wedge.

    This is the whole reason the gateway has an EXTERNAL backstop
    (``gateway.systemd_watchdog_seconds`` -> systemd ``WatchdogSec``). The systemd timer
    lives in the systemd manager process and keeps ticking no matter what the gateway's
    interpreter is doing, so it is the only backstop that survives this failure mode.

    If this test ever starts failing because the in-process watchdog DID fire, that is
    good news -- but it must not be the ONLY protection, because the in-process watchdog
    also has to cope with wedges that release the GIL and re-block it repeatedly.
    """
    loop = asyncio.new_event_loop()
    release = threading.Event()
    blocker_started = threading.Event()

    def _block():
        blocker_started.set()
        _gil_holding_call(3.0)
        if not release.is_set():
            loop.call_soon(_block)

    def _runner():
        asyncio.set_event_loop(loop)
        loop.call_soon(_block)
        loop.run_forever()

    t = threading.Thread(target=_runner, daemon=True, name="blocked-loop")
    t.start()
    assert blocker_started.wait(timeout=2.0)

    exited = threading.Event()
    exit_codes: list[int] = []

    import gateway.shutdown_watchdog as sw

    real_exit, real_mark = sw.os._exit, sw._mark_exited_quietly
    sw.os._exit = lambda code: (exit_codes.append(code), exited.set())
    sw._mark_exited_quietly = lambda *a, **k: None
    handle = None
    try:
        handle = start_loop_liveness_watchdog(
            loop, probe_interval=0.5, probe_timeout=0.5, max_strikes=2
        )
        assert handle is not None
        fired = exited.wait(timeout=10.0)
    finally:
        # Stop the watchdog BEFORE restoring os._exit: a watchdog thread that is still
        # live would fire the REAL os._exit(75) later and kill the pytest process.
        if handle is not None:
            handle.stop()
            handle.join(timeout=2.0)
        release.set()
        time.sleep(0.2)
        try:
            loop.call_soon_threadsafe(loop.stop)
        except RuntimeError:
            pass
        sw.os._exit, sw._mark_exited_quietly = real_exit, real_mark

    assert not fired, (
        "the in-process watchdog fired against a GIL-holding wedge. The external systemd "
        "watchdog is still required (it is the only one that survives a GIL hold), so do not "
        "remove that layer -- but note this behaviour changed."
    )
    assert exit_codes == []
