"""A slow fanout drain-thread spawn must not stall the session event lock.

``SessionEvents.publish`` calls ``FanoutTransport.write`` while holding the session's
``event_stream.lock``, and ``write`` started the peer's drain thread with
``threading.Thread(...).start()`` — which blocks until the new thread is actually running —
under the fanout membership lock and inside the session lock. ``session.events.since`` takes
the same RLock, so a slow ``Thread.start()`` froze the gateway event loop in ``since`` and
with it every ticket reply, until the loop-liveness watchdog dumped all threads and exited
75 (#148517).
"""
import threading

from gateway.session_events import SessionEvents
from tests.waiting import wait_until
from tui_gateway.transport import FanoutTransport


class _RecordingTransport:
    def __init__(self):
        self.frames = []

    def write(self, obj):
        self.frames.append(obj)
        return True

    def close(self):
        return None


def test_slow_drain_thread_start_does_not_stall_session_events(monkeypatch):
    start_entered = threading.Event()
    release_start = threading.Event()
    real_start = threading.Thread.start

    def slow_start(self, *args, **kwargs):
        # Only a fanout drain spawn parks; the test's own helper threads start normally.
        if getattr(getattr(self, '_target', None), '__func__', None) is FanoutTransport._drain:
            start_entered.set()
            release_start.wait(10)
        return real_start(self, *args, **kwargs)

    monkeypatch.setattr(threading.Thread, 'start', slow_start)
    stream = SessionEvents()
    transport = _RecordingTransport()
    stream.fanout.attach(transport)

    publisher = threading.Thread(target=stream.publish, args=('s', {'text': 'x'}), daemon=True)
    publisher.start()
    # On the regression the publisher parks inside Thread.start() while holding the session
    # lock; with a non-blocking spawn the gate stays shut and the probes must pass at once.
    start_entered.wait(1.0)

    since_done = threading.Event()
    contains_done = threading.Event()

    def _probe():
        stream.since(stream.epoch, 0)
        since_done.set()
        stream.fanout.contains(transport)
        contains_done.set()

    prober = threading.Thread(target=_probe, daemon=True)
    prober.start()
    assert since_done.wait(3.0), (
        'session.events.since stalled behind a slow fanout drain-thread start (#148517)')
    assert contains_done.wait(3.0), (
        'fanout membership blocked behind the drain-thread start (#148517)')

    release_start.set()
    publisher.join(10)
    prober.join(10)
    wait_until(lambda: bool(transport.frames), timeout=10,
               what='the fanout drain delivered the published frame')
