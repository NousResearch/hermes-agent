"""Wait status describes request-local silence, never active generation."""

import logging
from types import SimpleNamespace
import threading

import pytest

from agent import chat_completion_helpers as h
from agent.chat_completion_nonstream import _NonStreamRequest
from agent.chat_completion_wait_notice import WaitNoticeState

WAIT_NOTICE_LOGGER = "agent.chat_completion_wait_notice"


def _waits(caplog):
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith("provider-wait")]


def _request(**wd_overrides):
    request = _NonStreamRequest.__new__(_NonStreamRequest)
    notices, touches = [], []
    request.agent = SimpleNamespace(
        _emit_wait_notice=notices.append,
        _touch_activity=touches.append,
        _buffer_diagnostic_status=lambda *a, **k: None,
        _emit_diagnostic_wait=lambda *a, **k: None,
        _interrupt_requested=False,
        _codex_stream_last_event_ts=9999.0,  # Another request cannot hide this one's stall.
    )
    request.api_kwargs = {"model": "test-model"}
    request.call_start = 1000.0
    wd_kwargs = dict(
        codex=True, stale_timeout=600.0, ttfb_enabled=True, ttfb_timeout=120.0,
        idle_enabled=True, idle_timeout=180.0, idle_requires_progress=False, progress_timeout=0.0,
        est_tokens=0,
    )
    wd_kwargs.update(wd_overrides)
    request.wd = SimpleNamespace(**wd_kwargs)
    request.codex_watchdog_state = SimpleNamespace(
        lock=threading.Lock(), last_event_ts=None, last_progress_ts=None,
        retry_started_ts=None,
    )
    request.wait_notice_started_ts = None
    request.wait_notice = WaitNoticeState()
    request.result = {"error": None, "response": None}
    request.clients = SimpleNamespace(close_once=lambda reason: None)
    request.codex_token = None
    request.codex_retired = False
    return request, notices, touches


@pytest.mark.parametrize(
    "event,progress,retry,expected",
    [
        (59.0, 59.0, None, None),  # Reasoning/text/tool arguments still arriving.
        (59.0, None, None, None),  # Lifecycle traffic is not transport silence either.
        (0.0, 0.0, None, "provider stream active; 60s without stream events"),
        (0.0, None, None, "provider stream active; 60s without stream events"),
        (1.0, None, None, None),  # 59 seconds of silence is still quiet.
        (None, None, None, "60s waiting for the first provider event"),
        (10.0, 10.0, 59.0, None),  # Internal reconnect gets a fresh first-event wait.
        (0.0, 0.0, 0.0, "60s waiting for the first provider event after reconnect"),
        (0.0, 0.0, 1.0, None),
    ],
)
def test_wait_notice_tracks_current_attempt_silence(event, progress, retry, expected):
    request, notices, touches = _request()
    state = request.codex_watchdog_state
    state.last_event_ts = None if event is None else request.call_start + event
    state.last_progress_ts = None if progress is None else request.call_start + progress
    state.retry_started_ts = None if retry is None else request.call_start + retry
    request._emit_wait_notice(30.0)
    request._emit_wait_notice(59.0)
    assert notices == []
    assert touches
    if event is None and retry is None:
        assert "receiving" not in touches[-1]
    request._emit_wait_notice(60.0)
    if expected is None:
        assert notices == []
        assert touches, "Quiet heartbeat must survive suppressing a visible warning"
    else:
        assert len(notices) == 1
        assert expected in notices[0]
        assert "auto-reconnect:" in notices[0]


def test_resumed_events_clear_only_this_requests_wait_notice(monkeypatch):
    request, notices, _ = _request()
    sentinel = object()
    ticks = [0]

    class Worker:
        def __init__(self, **kwargs):
            pass

        def start(self):
            pass

        def is_alive(self):
            return ticks[0] < 204

        def join(self, timeout):
            ticks[0] += 1
            if ticks[0] >= 201:
                request.codex_watchdog_state.last_event_ts = 1000.0 + ticks[0] * 0.3
            if ticks[0] == 204:
                request.result["response"] = sentinel

    monkeypatch.setattr(h.threading, "Thread", Worker)
    monkeypatch.setattr(h.time, "time", lambda: 1000.0 + ticks[0] * 0.3)
    assert request.run() is sentinel
    assert len(notices) == 2
    assert "waiting for the first provider event" in notices[0]
    # Nonempty thinking.delta payloads enter TUI reasoning history.
    assert notices[1] == ""


def _drive_with_fake_worker(request, *, alive_until, resolve_at=None, tick_seconds=1.0):
    """Run the real poll loop with a fake worker thread: ``ticks`` advances the
    mocked clock by ``tick_seconds`` per ``join()`` call, mirroring run()'s
    ``while t.is_alive(): t.join(timeout=0.3)``. ``resolve_at`` sets a response
    on the request so a kill path finds nothing to abort twice."""
    ticks = [0]

    class Worker:
        def __init__(self, **kwargs):
            pass

        def start(self):
            pass

        def is_alive(self):
            return ticks[0] < alive_until

        def join(self, timeout):
            ticks[0] += 1
            if resolve_at is not None and ticks[0] >= resolve_at:
                request.result["response"] = object()

    import agent.chat_completion_nonstream as nonstream_mod

    patched_module = pytest.MonkeyPatch()
    patched_module.setattr(nonstream_mod.h.threading, "Thread", Worker)
    patched_module.setattr(nonstream_mod.h.time, "time", lambda: 1000.0 + ticks[0] * tick_seconds)
    try:
        return request.run()
    finally:
        patched_module.undo()


def test_success_after_wait_logs_end_with_outcome_resumed(caplog):
    """The provider answers right after the last notice, with no further poll
    that would hit the in-loop 'activity resumed' clear — the end record must
    still close, not stay open forever (#129229)."""
    caplog.set_level(logging.INFO, logger=WAIT_NOTICE_LOGGER)
    request, notices, _ = _request(stale_timeout=600.0, ttfb_enabled=False, idle_enabled=False)
    # Response resolves at tick 210 (~70s), after the tick-200 heartbeat (~60s)
    # that starts the notice, and before any watchdog threshold fires.
    result = _drive_with_fake_worker(request, alive_until=210, resolve_at=210)
    assert result is not None

    waits = _waits(caplog)
    assert len(waits) == 2, waits
    assert "provider-wait start" in waits[0] and "phase=first_event" in waits[0]
    assert "provider-wait end" in waits[1]
    assert "outcome=resumed" in waits[1]


def test_idle_kill_logs_end_with_outcome_idle_kill(caplog):
    caplog.set_level(logging.INFO, logger=WAIT_NOTICE_LOGGER)
    request, _, _ = _request(
        codex=False,  # keep the preset last_event_ts below: run() only resets it when wd.codex
        ttfb_enabled=False, idle_enabled=True, idle_timeout=150.0, stale_timeout=600.0, progress_timeout=0.0)
    request.codex_watchdog_state.last_event_ts = request.call_start  # stream opened, then went quiet

    with pytest.raises(TimeoutError):
        _drive_with_fake_worker(request, alive_until=10_000)

    waits = _waits(caplog)
    assert len(waits) == 2, waits
    assert "provider-wait start" in waits[0] and "phase=post_event" in waits[0]
    assert "provider-wait end" in waits[1] and "outcome=idle_kill" in waits[1]


def test_stale_kill_logs_end_with_outcome_stale_kill(caplog):
    caplog.set_level(logging.INFO, logger=WAIT_NOTICE_LOGGER)
    # Past the tick-100 heartbeat (so a notice actually starts) but before the
    # next one at tick-200.
    request, _, _ = _request(ttfb_enabled=False, idle_enabled=False, stale_timeout=150.0)

    with pytest.raises(TimeoutError):
        _drive_with_fake_worker(request, alive_until=10_000)

    waits = _waits(caplog)
    assert len(waits) == 2, waits
    assert "provider-wait start" in waits[0] and "phase=first_event" in waits[0]
    assert "provider-wait end" in waits[1] and "outcome=stale_kill" in waits[1]


def test_kill_outcome_is_not_overwritten_by_the_later_resumed_reset(caplog):
    """run() unconditionally resets(outcome='resumed') after the poll loop; a
    kill branch above must have already closed the record so that call is a
    no-op, never relabeling a kill as a normal resume."""
    caplog.set_level(logging.INFO, logger=WAIT_NOTICE_LOGGER)
    request, _, _ = _request(ttfb_enabled=False, idle_enabled=False, stale_timeout=150.0)

    with pytest.raises(TimeoutError):
        _drive_with_fake_worker(request, alive_until=10_000)

    waits = _waits(caplog)
    end_lines = [w for w in waits if w.startswith("provider-wait end")]
    assert len(end_lines) == 1, end_lines
    assert "outcome=stale_kill" in end_lines[0]
    assert "outcome=resumed" not in end_lines[0]
