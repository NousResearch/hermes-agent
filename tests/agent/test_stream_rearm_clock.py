"""Recovery must rearm the monotonic stale deadline without mixing clock origins."""

from types import SimpleNamespace

import pytest

from agent import chat_completion_helpers as helpers
from agent import chat_completion_stream_monitor as monitor
from agent import stream_max_lifetime as lifetime


@pytest.mark.parametrize("kill", ["stale", "lifetime"])
def test_stale_detection_survives_a_rearmed_kill(monkeypatch, kill):
    clock = SimpleNamespace(mono=1000.0, wall=500_000.0)
    readings = SimpleNamespace(monotonic=lambda: clock.mono, time=lambda: clock.wall)
    for module in (helpers, monitor, lifetime):
        monkeypatch.setattr(module, "time", readings)
    monkeypatch.setattr(monitor, "resolve_stream_max_lifetime", lambda stale: 0.0)
    agent = SimpleNamespace(
        base_url="https://example.com", _interrupt_requested=False,
        _emit_diagnostic_wait=lambda text: None, _emit_wait_notice=lambda text: None,
        _touch_activity=lambda text: None, _buffer_diagnostic_status=lambda text: None,
    )
    call = helpers._StreamingCall(agent, {"model": "test"}, None)
    call._stream_stale_timeout = 180.0
    call._start_stream_attempt()
    call._lifetime = lifetime.StreamLifetime(180.0)
    call._lifetime.exceeded(1)
    clock.wall += 181.0
    call._call_done = _Polls(clock)
    if kill == "stale":
        call._rearm_or_abandon_after_kill(181.0)
        # The lifetime cap keeps its own origin and gets a fresh window as well.
        assert call._lifetime.exceeded(1) is None
    else:
        call._check_stream_lifetime()
        assert call._lifetime_kill == (1, 181.0)
    killed = []
    call._kill_stale_stream = killed.append
    call._monitor_loop()
    assert killed == [181.0], "rearming with a wall-clock epoch disabled the stale detector"


class _Polls:
    def __init__(self, clock):
        self.clock = clock
        self.count = 0

    def is_set(self):
        return self.count == 1

    def wait(self, timeout):
        self.clock.mono += 181.0
        self.clock.wall -= 3600.0
        self.count += 1
