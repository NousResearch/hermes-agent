"""The long-wait status line is one neutral notice per silence, not a 30s warning drumbeat (#92550).

Both builders (Codex non-stream request, chat-completions stream monitor) name the wait
phase and the watchdog that would reconnect; they rewrite the line only on a phase change
or when that deadline is near.
"""
import threading
from types import SimpleNamespace

from agent import chat_completion_helpers as h
from agent.chat_completion_nonstream import _NonStreamRequest
from agent.chat_completion_wait_notice import WaitNoticeState

WARNING_COPY = ("provider may be slow", "no response yet")


def _nonstream_request(ttfb_timeout=300.0):
    request = _NonStreamRequest.__new__(_NonStreamRequest)
    notices, touches = [], []
    request.agent = SimpleNamespace(_emit_wait_notice=notices.append, _touch_activity=touches.append,
                                    _interrupt_requested=False)
    request.api_kwargs = {"model": "test-model"}
    request.call_start = 1000.0
    request.wd = SimpleNamespace(codex=True, stale_timeout=600.0, ttfb_enabled=True, ttfb_timeout=ttfb_timeout,
                                 idle_enabled=True, idle_timeout=180.0, idle_requires_progress=False,
                                 progress_timeout=0.0)
    request.codex_watchdog_state = SimpleNamespace(lock=threading.Lock(), last_event_ts=None,
                                                   last_progress_ts=None, retry_started_ts=None)
    request.wait_notice_started_ts = None
    request.wait_notice = WaitNoticeState()
    request.result = {"error": None, "response": None}
    return request, notices, touches


def test_nonstream_notice_once_per_phase_then_near_deadline_only():
    request, notices, touches = _nonstream_request(ttfb_timeout=300.0)
    for heartbeat in range(1, 10):  # 30s .. 270s of silence, no event ever
        request._emit_wait_notice(30.0 * heartbeat)
    # 60s: first notice; 90..270s: liveness touches only, until the TTFB deadline is within 15s.
    shown = [n for n in notices if n]
    assert len(shown) == 1 and "test-model" in shown[0]
    assert touches, "gateway liveness heartbeat survives the quiet heartbeats"
    request._emit_wait_notice(290.0)
    assert len([n for n in notices if n]) == 2  # one update as the TTFB deadline nears
    # A first event moves the wait into a new phase: clear, then one post-event notice naming stream idle.
    request.codex_watchdog_state.last_event_ts = 1295.0
    request._emit_wait_notice(300.0, heartbeat=False)
    assert notices[-1] == ""
    request._emit_wait_notice(360.0)
    request._emit_wait_notice(390.0)
    assert len([n for n in notices[notices.index("") + 1:] if n]) == 1
    assert not any(copy in n for n in notices for copy in WARNING_COPY)


def test_stream_monitor_notice_does_not_repeat_every_heartbeat():
    call = h._StreamingCall.__new__(h._StreamingCall)
    notices, touches = [], []
    call.agent = SimpleNamespace(_emit_wait_notice=notices.append, _touch_activity=touches.append)
    call.api_kwargs = {"model": "test-model"}
    call._stream_stale_timeout = 180.0
    call.clients = SimpleNamespace(diag={"first_chunk_at": None})
    call._mon = SimpleNamespace(last_heartbeat=1000.0, wait_notice_started_ts=None, wait_notice=WaitNoticeState())
    for secs in (30, 60, 90, 120, 150):
        call._mon.last_heartbeat = 1000.0 + secs
        call._heartbeat(secs)
    assert len(notices) == 1 and "test-model" in notices[0]
    assert len(touches) >= 3  # 30s (pre-threshold) + the suppressed 90/120/150s heartbeats
    call._heartbeat(170)  # within 15s of the stale kill: one "still waiting" update
    assert len(notices) == 2
    # Output that stopped after chunks arrived is a different phase, worded as such.
    call._mon.wait_notice.reset()
    call.clients.diag["first_chunk_at"] = 1005.0
    call._heartbeat(60)
    assert len(notices) == 3 and notices[-1] != notices[0]
    assert not any(copy in n for n in notices for copy in WARNING_COPY)
