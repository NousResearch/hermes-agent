"""Stream liveness is measured on a monotonic clock, not the wall clock.

Every value the streaming monitor subtracts is an ELAPSED DURATION: the heartbeat
interval, the managed-local load-notice gate, and the stale-stream deadline that
cancels a live request. Measuring those against ``time.time()`` puts an NTP step (or
a VM restored from a snapshot, or a laptop resuming from sleep) straight into the
arithmetic: a step forward reads as a silent stream and kills a healthy request, and
a step backward reads as a fresh chunk, disabling the detector for the rest of the
turn — exactly when the watchdog is wanted.

Contract:
- A wall-clock jump of any size does not trip the stale deadline.
- A REAL stall is still detected, whatever the wall clock is doing.
"""

from types import SimpleNamespace

from agent import chat_completion_helpers as h

_STALE_TIMEOUT = 180.0


def _run_monitor(monkeypatch, *, wall_offset, polls, step=0.3):
    """Run ``_monitor_loop`` with a monotonic clock advancing ``step`` per poll and a wall
    clock offset by ``wall_offset``. Returns the list of stale-kill elapsed values."""
    call = h._StreamingCall.__new__(h._StreamingCall)
    call.agent = SimpleNamespace(
        base_url="https://example.com",  # non-local: skips the load-notice probe
        _interrupt_requested=False,
        _emit_wait_notice=lambda text: None,
        _touch_activity=lambda text: None,
    )
    call.api_kwargs = {"model": "test-model"}

    mono = [1000.0]
    wall_reads = []

    def _wall():
        # Offset applies to the first reading only; after that the wall clock tracks the
        # monotonic one, so an absolute stamp stays self-consistent.
        wall_reads.append(1)
        return 500_000.0 + wall_offset + (mono[0] - 1000.0) * (0 if len(wall_reads) == 1 else 1)

    monkeypatch.setattr(h.time, "monotonic", lambda: mono[0])
    monkeypatch.setattr(h.time, "time", _wall)

    # Stamped the way production stamps it, from whichever clock production uses.
    call.last_chunk_time = {"t": h.time.monotonic()}
    call._stream_stale_timeout = _STALE_TIMEOUT

    class Done:
        def __init__(self):
            self.n = 0

        def is_set(self):
            return self.n >= polls

        def wait(self, timeout):
            self.n += 1
            mono[0] += step

    call._call_done = Done()

    killed: list[float] = []
    call._kill_stale_stream = killed.append
    call._abort_for_interrupt = lambda elapsed: None
    call._monitor_loop()
    return killed


def test_forward_wall_clock_step_does_not_kill_a_healthy_stream(monkeypatch):
    """+1h on the first wall reading while only ~2s of real time passes.

    With a wall-clock epoch this reads as an hour of silence and cancels a stream
    that is delivering chunks fine.
    """
    killed = _run_monitor(monkeypatch, wall_offset=3600.0, polls=6)
    assert killed == [], f"healthy stream killed by a wall-clock step: {killed}"


def test_real_stall_is_still_detected_despite_a_backward_step(monkeypatch):
    """The other half: the clock change must not blunt the watchdog.

    Wall clock an hour behind, but 200s of genuinely monotonic silence crosses the
    180s deadline and must still cancel.
    """
    killed = _run_monitor(monkeypatch, wall_offset=-3600.0, polls=700, step=0.3)
    assert killed, "a 200s stall was not detected"
    assert killed[0] >= _STALE_TIMEOUT
