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


def test_stream_monitor_phase_split_stale_timeout(monkeypatch):
    """Phase 1 (before first_chunk_at) uses the full prefill stale budget; Phase 2 (after first chunk)
    tightens to the inter-event timeout (default 90s for non-reasoning models, floored by
    get_reasoning_stale_timeout_floor for cloud reasoning models like gemini-4-argon)."""
    call = h._StreamingCall.__new__(h._StreamingCall)
    call.agent = SimpleNamespace(
        provider="openai",
        model="gpt-4o",
        base_url=None,
        _stream_stale_timeout=None,
        _stream_inter_event_timeout=None,
    )
    call.api_kwargs = {"model": "gpt-4o", "messages": [{"role": "user", "content": "x" * 600_000}]}
    diag: dict[str, float | None] = {"first_chunk_at": None}
    call.clients = SimpleNamespace(diag=diag)  # type: ignore[assignment]
    call._resolve_stale_timeout()

    assert (call._stream_stale_timeout or 0.0) >= 300.0
    assert call._stream_post_chunk_stale_timeout == 90.0
    assert call._effective_stale_timeout() == call._stream_stale_timeout

    # Once first chunk arrives, effective timeout drops to 90s inter-event budget on non-reasoning models.
    diag["first_chunk_at"] = 1001.0
    assert call._effective_stale_timeout() == 90.0

    # Cloud reasoning models (gemini-4-argon) preserve their reasoning floor (600s) in Phase 2
    # so server-buffered tool_calls and mid-turn re-prefill hops are not killed at 90s.
    rcall = h._StreamingCall.__new__(h._StreamingCall)
    rcall.agent = SimpleNamespace(
        provider="vertex",
        model="google/gemini-4-argon",
        base_url=None,
        _stream_stale_timeout=None,
        _stream_inter_event_timeout=None,
    )
    rcall.api_kwargs = {"model": "google/gemini-4-argon", "messages": [{"role": "user", "content": "hi"}]}
    rdiag: dict[str, float | None] = {"first_chunk_at": 1001.0}
    rcall.clients = SimpleNamespace(diag=rdiag)  # type: ignore[assignment]
    rcall._resolve_stale_timeout()
    assert rcall._stream_stale_timeout == 600.0
    assert rcall._stream_post_chunk_stale_timeout == 600.0
    assert rcall._effective_stale_timeout() == 600.0

    # While probe holds _stream_stale_timeout at inf, effective timeout stays inf.
    saved = call._stream_stale_timeout
    call._stream_stale_timeout = float("inf")
    assert call._effective_stale_timeout() == float("inf")
    call._stream_stale_timeout = saved

    # Env override HERMES_STREAM_INTER_EVENT_TIMEOUT is honored on all models and capped at Phase 1 budget.
    monkeypatch.setenv("HERMES_STREAM_INTER_EVENT_TIMEOUT", "45")
    call._resolve_stale_timeout()
    rcall._resolve_stale_timeout()
    assert call._effective_stale_timeout() == 45.0
    assert rcall._effective_stale_timeout() == 45.0


def test_open_chat_stream_syncs_monitor_heartbeat_and_read_timeout_lead(monkeypatch):
    """_open_chat_stream syncs _mon.last_heartbeat with last_chunk_time['t'] so client-init
    latency does not shift the first wait notice from 60s to 87s, and _stream_timeouts gives
    the stale detector a 15s lead over the raw httpx read timeout."""
    call = h._StreamingCall.__new__(h._StreamingCall)
    call.agent = SimpleNamespace(
        provider="vertex",
        model="google/gemini-4-argon",
        base_url="https://aiplatform.googleapis.com/v1/projects/p/locations/global/endpoints/openapi",
        _stream_stale_timeout=None,
        _stream_inter_event_timeout=None,
        _stream_options_unsupported=False,
        _create_request_openai_client=lambda **_kw: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_k: "stream"))
        ),
        _touch_activity=lambda _msg: None,
    )
    call.api_kwargs = {"model": "google/gemini-4-argon", "messages": [{"role": "user", "content": "hi"}]}
    call.clients = SimpleNamespace(diag={"first_chunk_at": None}, set_client=lambda c: c)  # type: ignore[assignment]
    call.last_chunk_time = {"t": 1000.0}
    call._mon = SimpleNamespace(last_heartbeat=1000.0)
    call._resolve_stale_timeout()

    _write, read, _conn = call._stream_timeouts()
    assert call._stream_stale_timeout == 600.0
    assert read == 615.0

    monkeypatch.setattr(h.time, "time", lambda: 1003.0)
    monkeypatch.setattr(h, "bypass_chat_sdk_request_transform", lambda kw, _c: kw)
    assert call._open_chat_stream({"model": "google/gemini-4-argon"}) == "stream"
    assert call.last_chunk_time["t"] == 1003.0
    assert call._mon.last_heartbeat == 1003.0


