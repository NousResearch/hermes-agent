"""Provider-silence clocks exclude host suspend from watchdog policy."""
from types import SimpleNamespace

import pytest

from agent import chat_completion_helpers as helpers


class _Clock:
    def __init__(self):
        self.wall = 1000.0
        self.monotonic = 1000.0

    def time(self):
        return self.wall

    def monotonic_time(self):
        return self.monotonic

    def awake(self, seconds):
        self.wall += seconds
        self.monotonic += seconds

    def suspend(self, seconds):
        self.wall += seconds


class _Liveness:
    """Small clock-backed double for the shared provider-silence contract."""

    def __init__(self, clock):
        self.clock = clock
        self.touch()

    def touch(self):
        self.wall_stamp = self.clock.time()
        self.monotonic_stamp = self.clock.monotonic_time()
        return self.wall_stamp

    def silence(self):
        awake = self.clock.monotonic_time() - self.monotonic_stamp
        wall = self.clock.time() - self.wall_stamp
        return awake, max(0.0, wall - awake)


def _provider_liveness(clock):
    try:
        from agent.stream_liveness import StreamLiveness
    except ModuleNotFoundError as exc:
        if exc.name != "agent.stream_liveness":
            raise
        return _Liveness(clock)
    liveness = StreamLiveness()
    liveness.touch()
    return liveness


def _install_clock(monkeypatch, clock):
    monkeypatch.setattr(helpers.time, "time", clock.time)
    monkeypatch.setattr(helpers.time, "monotonic", clock.monotonic_time)


def _stream_call(clock, *, timeout, wait, agent_overrides=None):
    call = helpers._StreamingCall.__new__(helpers._StreamingCall)
    call.agent = SimpleNamespace(
        base_url="https://provider.example",
        _interrupt_requested=False,
        _touch_activity=lambda _message: None,
        _emit_wait_notice=lambda _message: None,
        _buffer_diagnostic_status=lambda _message: None,
        _emit_diagnostic_wait=lambda _message: None,
        _consecutive_stale_streams=0,
        **(agent_overrides or {}),
    )
    call.api_kwargs = {"model": "test-model", "input": "hello"}
    call.stream_liveness = _provider_liveness(clock)
    call.last_chunk_time = {"t": call.stream_liveness.touch()}
    call._stream_stale_timeout = timeout
    call._call_done = wait
    call._mon = None
    call.clients = SimpleNamespace(diag=None)
    call._attempt_stream_response = None
    call._request_cancelled = {"value": False}
    call.deltas_were_sent = {"yes": False}
    call.stream_attempt_lock = __import__("threading").Lock()
    call.stream_attempt_state = {"current": 0, "cancelled": set(), "discarded_chunks": 0, "discarded_bytes": 0}
    return call


class _DoneAfterOnePoll:
    def __init__(self, clock, advance):
        self.clock = clock
        self.advance = advance
        self.waited = False

    def is_set(self):
        return self.waited

    def wait(self, timeout):
        self.advance(self.clock, timeout)
        self.waited = True


def test_stream_resume_reports_awake_silence_and_does_not_kill(monkeypatch):
    clock = _Clock()
    _install_clock(monkeypatch, clock)
    notices = []
    killed = []
    done = _DoneAfterOnePoll(
        clock,
        lambda c, _timeout: (c.awake(60.1), c.suspend(6.5 * 60 * 60)),
    )
    call = _stream_call(clock, timeout=600.0, wait=done)
    call.agent._emit_wait_notice = notices.append
    call._kill_stale_stream = lambda *args: killed.append(args)

    call._monitor_loop()

    assert killed == [], "host suspend alone must not trip the stale-stream deadline"
    assert notices
    assert "60s awake (23400s host suspend)" in notices[0]
    assert "23460s" not in notices[0], "wall elapsed must not be presented as provider silence"


def test_genuine_stream_stall_is_killed_and_reports_suspend_separately(monkeypatch, caplog):
    clock = _Clock()
    _install_clock(monkeypatch, clock)
    diagnostics = []
    statuses = []
    done = _DoneAfterOnePoll(
        clock,
        lambda c, _timeout: (c.awake(10.1), c.suspend(6.5 * 60 * 60)),
    )
    call = _stream_call(clock, timeout=10.0, wait=done)
    call.agent._buffer_diagnostic_status = diagnostics.append
    call.agent._emit_diagnostic_wait = statuses.append
    call._cancel_current_stream_attempt = lambda _reason: None
    call._shutdown_stale_attempt_socket = lambda _response: None
    call.clients.close_once = lambda _reason: None

    call._monitor_loop()

    report = "10s awake (23400s host suspend)"
    assert any(report in text for text in diagnostics)
    assert any(report in text for text in statuses)
    assert "23410s" not in " ".join(diagnostics + statuses)
    assert report in caplog.text


def test_nonstream_resume_rearms_watchdog_until_awake_deadline(monkeypatch):
    clock = _Clock()
    _install_clock(monkeypatch, clock)
    aborts = []
    timers = []

    class Timer:
        def __init__(self, interval, function):
            self.interval = interval
            self.function = function
            self.daemon = False
            self.name = ""
            timers.append(self)

        def start(self):
            pass

    monkeypatch.setattr(helpers.threading, "Timer", Timer)
    request = helpers._InlineRequest.__new__(helpers._InlineRequest)
    request.agent = SimpleNamespace(_touch_activity=lambda _message: None)
    request.api_kwargs = {"model": "test-model", "input": "hello"}
    request.stale_timeout = 10.0
    request.call_start = clock.time()
    request.liveness = _provider_liveness(clock)
    request.done = False
    request.cancelled = False
    request.stale = False
    request.lock = __import__("threading").Lock()
    clock.awake(0.1)
    clock.suspend(6.5 * 60 * 60)
    request.abort = lambda reason: (aborts.append(reason), True)[1]

    request._on_stale()

    assert aborts == [], "a timer waking from suspend must not abort before awake silence reaches threshold"
    assert timers and timers[-1].interval == pytest.approx(9.9)


def test_nonstream_stall_report_uses_awake_elapsed_and_suspend(monkeypatch, caplog):
    clock = _Clock()
    _install_clock(monkeypatch, clock)
    diagnostics = []
    activity = []
    request = helpers._InlineRequest.__new__(helpers._InlineRequest)
    request.agent = SimpleNamespace(
        _buffer_diagnostic_status=diagnostics.append,
        _touch_activity=activity.append,
        _consecutive_stale_streams=0,
    )
    request.api_kwargs = {"model": "test-model", "input": "hello"}
    request.stale_timeout = 10.0
    request.call_start = clock.time()
    request.liveness = _provider_liveness(clock)
    clock.awake(10.1)
    clock.suspend(6.5 * 60 * 60)
    request.abort = lambda _reason: True

    request._on_stale()

    report = "10s awake (23400s host suspend)"
    assert diagnostics and report in diagnostics[0]
    assert report in caplog.text
    assert activity and report in activity[0]
    assert "23410s" not in " ".join(diagnostics + activity)


def test_bedrock_stall_waits_for_awake_threshold_and_reports_suspend(monkeypatch):
    clock = _Clock()
    _install_clock(monkeypatch, clock)
    joins = []
    diagnostics = []

    class Worker:
        def __init__(self, *, target, daemon):
            self.target = target

        def start(self):
            pass

        def is_alive(self):
            return True

        def join(self, timeout=None):
            if not joins:
                clock.suspend(6.5 * 60 * 60)
            clock.awake(timeout)
            joins.append(timeout)

    monkeypatch.setattr(helpers.threading, "Thread", Worker)
    stream = helpers._BedrockStream.__new__(helpers._BedrockStream)
    stream.agent = SimpleNamespace(
        _interrupt_requested=False,
        _buffer_diagnostic_status=diagnostics.append,
        _consecutive_stale_streams=0,
    )
    stream.api_kwargs = {"modelId": "bedrock-test-model"}
    stream.started_liveness = _provider_liveness(clock)
    stream.event_liveness = _provider_liveness(clock)
    stream.started_at = clock.time()
    stream.last_event = clock.time()
    stream.result = {"response": None, "error": None}
    stream.first_delta_fired = False
    stream.response_started = False
    stream.stale_timeout = 1.0
    stream.region = "us-east-1"

    with pytest.raises(TimeoutError) as excinfo:
        stream._poll()

    assert len(joins) >= 4, "Bedrock must wait for awake silence to reach its threshold"
    report = "1s awake (23400s host suspend)"
    assert report in str(excinfo.value)
    assert diagnostics and report in diagnostics[0]
    assert "23401s" not in str(excinfo.value)
