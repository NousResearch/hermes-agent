"""A connected socket is not thinking: lifecycle-only Codex traffic is named as such.

Every parsed SSE frame refreshes transport liveness, so a backend that keeps sending
lifecycle frames while producing no model output silences BOTH event watchdogs (each
re-arms on ``last_event_ts``) and, before this, the wait notice too — the operator saw
an anonymous spinner until the wall-clock stale kill. These tests drive the real poll
loop on a fake clock with fixture Responses frames classified by the production
``_codex_event_has_content`` predicate.
"""

import threading
from types import SimpleNamespace

import pytest

from agent import chat_completion_helpers as h
from agent import codex_runtime as cr
from agent.chat_completion_wait_notice import (
    METADATA_ONLY_PHASES, WaitNoticeState, no_output_notice_secs, wait_notice_text)
from agent.chat_completion_nonstream import _NonStreamRequest

TICK = 0.3  # _NonStreamRequest.run() poll interval
CALL_START = 1000.0

# Frames an openai-codex request emits before (and between) model output.
LIFECYCLE = SimpleNamespace(type="response.in_progress")
STRUCTURAL = SimpleNamespace(type="response.output_item.added", item={"type": "message"})
EMPTY_DELTA = SimpleNamespace(type="response.output_text.delta", delta="")
REASONING_DELTA = SimpleNamespace(type="response.reasoning_summary_text.delta", delta="thought")
TEXT_DELTA = SimpleNamespace(type="response.output_text.delta", delta="token")


def _request(*, idle_requires_progress=True, idle_timeout=180.0, stale_timeout=1200.0):
    """An openai-codex non-stream request: large context, so the idle watchdog only
    arms once model progress begins (``_resolve_nonstream_watchdogs``)."""
    request = _NonStreamRequest.__new__(_NonStreamRequest)
    notices, touches = [], []
    request.agent = SimpleNamespace(
        _emit_wait_notice=notices.append, _touch_activity=touches.append,
        _interrupt_requested=False)
    request.api_kwargs = {"model": "gpt-5.5-codex"}
    request.call_start = CALL_START
    request.wd = SimpleNamespace(
        codex=True, stale_timeout=stale_timeout, ttfb_enabled=True, ttfb_timeout=300.0,
        idle_enabled=True, idle_timeout=idle_timeout,
        idle_requires_progress=idle_requires_progress, est_tokens=250_000)
    request.codex_watchdog_state = SimpleNamespace(
        lock=threading.Lock(), last_event_ts=None, last_progress_ts=None, retry_started_ts=None)
    request.wait_notice_started_ts = None
    request.wait_notice = WaitNoticeState()
    request.result = {"error": None, "response": None}
    return request, notices, touches


def _drive(request, monkeypatch, *, frames, until):
    """Run the production poll loop on a fake clock, delivering ``(elapsed, event)``
    frames through the production watchdog marker."""
    tick = [0]
    pending = sorted(frames, key=lambda f: f[0])

    def _now():
        return CALL_START + tick[0] * TICK

    class Worker:
        def __init__(self, **kwargs):
            pass

        def start(self):
            pass

        def is_alive(self):
            return _now() - CALL_START < until

        def join(self, timeout=None):
            tick[0] += 1
            while pending and pending[0][0] <= _now() - CALL_START:
                cr._mark_codex_watchdog_event(request.codex_watchdog_state, pending.pop(0)[1], _now())

    monkeypatch.setattr(h.threading, "Thread", Worker)
    monkeypatch.setattr(h.time, "time", _now)
    request.run()


def _metadata_every(seconds, until, *, start=0.0):
    frames = [LIFECYCLE, STRUCTURAL, EMPTY_DELTA]
    return [(at, frames[i % len(frames)])
            for i, at in enumerate(_ticks(start, until, seconds))]


def _ticks(start, until, step):
    at = start
    while at < until:
        yield at
        at += step


def test_lifecycle_frames_alone_are_not_called_progress(monkeypatch):
    """The gap: 10 minutes of lifecycle frames, no model output, no notice at all."""
    request, notices, touches = _request()
    _drive(request, monkeypatch, frames=_metadata_every(10.0, 600.0), until=600.0)

    assert touches, "the quiet gateway heartbeat keeps running"
    shown = [n for n in notices if n]
    assert shown, "an operator must be told the provider sent no model output"
    assert "no model output yet" in shown[0]
    # Neither event watchdog can fire while frames keep arriving, so naming one would
    # promise a reconnect that never comes; the wall-clock stale kill is the real deadline.
    assert "wall-clock stale watchdog" in shown[0]
    assert "stream idle" not in shown[0] and "TTFB" not in shown[0]
    # Desktop's providerWaitText matcher only forwards explained wait frames.
    assert shown[0].startswith("⏳ waiting on gpt-5.5-codex — ")
    assert len(shown) == 1, "one notice per silence, not a heartbeat drumbeat"


def test_a_later_lifecycle_frame_does_not_clear_the_no_output_notice(monkeypatch):
    """Clearing the line reads as recovery; the frames that clear it must be the
    ones whose absence was reported."""
    request, notices, _ = _request()
    _drive(request, monkeypatch, frames=_metadata_every(10.0, 600.0), until=600.0)

    assert "" not in notices


def test_model_output_clears_the_no_output_notice(monkeypatch):
    """Resumed content is real progress: the notice comes down promptly."""
    request, notices, _ = _request()
    frames = _metadata_every(10.0, 400.0) + [(300.0, REASONING_DELTA)]
    _drive(request, monkeypatch, frames=frames, until=400.0)

    assert [n for n in notices if n], "the metadata-only wait was reported"
    assert notices[-1] == "", "the first real delta takes the notice down"


def test_output_that_stops_while_frames_continue_is_named_as_a_pause(monkeypatch):
    """Progress then lifecycle-only: the event-idle watchdog re-arms on every frame,
    so this wait is also invisible today."""
    request, notices, _ = _request()
    frames = [(1.0, TEXT_DELTA), (2.0, REASONING_DELTA)] + _metadata_every(10.0, 600.0, start=10.0)
    _drive(request, monkeypatch, frames=frames, until=600.0)

    shown = [n for n in notices if n]
    assert shown, "output that stopped is still a wait worth naming"
    assert "since the last model output" in shown[0]
    # Progress armed the event-idle watchdog, so it — not the far-off stale kill — is
    # what reconnects if the frames stop. The named deadline is the snapshot's nearest.
    assert "stream idle watchdog" in shown[0]


def test_named_deadline_is_the_nearest_armed_watchdog_not_the_optimistic_one(monkeypatch):
    """Naming the stale kill because lifecycle frames are 'expected to continue' would
    hide an armed watchdog 12 minutes nearer."""
    request, notices, _ = _request()
    state = request.codex_watchdog_state
    cr._mark_codex_watchdog_event(state, TEXT_DELTA, CALL_START + 1.0)
    cr._mark_codex_watchdog_event(state, LIFECYCLE, CALL_START + 299.0)
    request._emit_wait_notice(300.0)

    from agent import chat_completion_wait_notice as wn

    wd = request.wd
    armed = wn.codex_watchdog_deadline(
        stale_timeout=wd.stale_timeout, ttfb_enabled=wd.ttfb_enabled, ttfb_timeout=wd.ttfb_timeout,
        last_event_ts=state.last_event_ts, last_progress_ts=state.last_progress_ts,
        retry_started_ts=state.retry_started_ts, call_start=request.call_start,
        idle_enabled=wd.idle_enabled, idle_timeout=wd.idle_timeout,
        idle_requires_progress=wd.idle_requires_progress, elapsed=300.0)
    assert armed == ("stream idle", 179.0)
    assert [n for n in notices if n] and f"{armed[0]} watchdog in {int(armed[1])}s" in notices[-1]


def test_silent_stream_still_reports_the_first_event_wait(monkeypatch):
    """No frame at all keeps the existing transport-silence wording and watchdog."""
    request, notices, _ = _request()
    _drive(request, monkeypatch, frames=[], until=200.0)

    shown = [n for n in notices if n]
    assert shown and "waiting for the first provider event" in shown[0]
    assert "TTFB watchdog" in shown[0]


def test_frames_that_stop_outrank_the_metadata_wording(monkeypatch):
    """Once frames stop, transport silence is the more specific fact to report."""
    request, notices, _ = _request()
    _drive(request, monkeypatch, frames=_metadata_every(10.0, 30.0), until=400.0)

    shown = [n for n in notices if n]
    assert shown and "provider stream active;" in shown[0]
    assert "without stream events" in shown[0]


def test_reconnect_wording_still_owns_the_post_retry_wait(monkeypatch):
    """A real reconnect keeps its own no-event phase: the metadata wording must not
    take it over, and the notice clears the way it always did."""
    request, notices, _ = _request()
    state = request.codex_watchdog_state
    tick = [0]

    def _now():
        return CALL_START + tick[0] * TICK

    class Worker:
        def __init__(self, **kwargs):
            pass

        def start(self):
            pass

        def is_alive(self):
            return _now() - CALL_START < 400.0

        def join(self, timeout=None):
            tick[0] += 1
            elapsed = _now() - CALL_START
            if elapsed < 100.0 and int(elapsed) % 10 == 0:
                cr._mark_codex_watchdog_event(state, LIFECYCLE, _now())
            elif 100.0 <= elapsed < 300.0 and state.retry_started_ts is None:
                with state.lock:  # what codex_runtime does when the retry loop reconnects
                    state.retry_started_ts = _now()
            elif elapsed >= 300.0:
                cr._mark_codex_watchdog_event(state, TEXT_DELTA, _now())

    monkeypatch.setattr(h.threading, "Thread", Worker)
    monkeypatch.setattr(h.time, "time", _now)
    request.run()

    shown = [n for n in notices if n]
    assert any("waiting for the first provider event after reconnect" in n for n in shown)
    assert not any(phrase in n for n in shown for phrase in ("lifecycle events", "model output"))
    assert notices[-1] == "", "the delta after the reconnect takes the notice down"


def test_stop_during_a_metadata_only_wait_still_interrupts(monkeypatch):
    """Explicit cancellation stays immediate and is not swallowed by the new phase."""
    request, notices, _ = _request()
    tick, aborted = [0], []
    request._abort_request = lambda reason: aborted.append(reason)
    request.clients = SimpleNamespace(close_once=lambda reason: None)

    def _now():
        return CALL_START + tick[0] * TICK

    class Worker:
        def __init__(self, **kwargs):
            pass

        def start(self):
            pass

        def is_alive(self):
            return True

        def join(self, timeout=None):
            tick[0] += 1
            elapsed = _now() - CALL_START
            if int(elapsed) % 10 == 0:
                cr._mark_codex_watchdog_event(request.codex_watchdog_state, LIFECYCLE, _now())
            if elapsed > 240.0:
                request.agent._interrupt_requested = True

    monkeypatch.setattr(h.threading, "Thread", Worker)
    monkeypatch.setattr(h.time, "time", _now)
    monkeypatch.setattr(h, "_record_interrupted_provider_wait", lambda *a, **k: False)
    monkeypatch.setattr(h, "_join_worker_for_relay_teardown", lambda *a, **k: None)
    with pytest.raises(InterruptedError):
        request.run()
    assert aborted == ["interrupt_abort"]
    assert any("no model output yet" in n for n in notices), "the wait was explained before Stop"


@pytest.mark.parametrize(
    "idle_enabled,idle_timeout,expected",
    [
        (True, 300.0, 300.0),   # high-effort silence floor: 5 minutes of grace
        (True, 180.0, 180.0),   # >100k-token context default
        (True, 12.0, 60.0),     # small-context default stays above the notice threshold
        (True, float("inf"), 60.0),
        (False, 0.0, 60.0),     # operator disabled the idle watchdog; the line is not a kill
    ],
)
def test_no_output_threshold_reuses_the_event_idle_knob(idle_enabled, idle_timeout, expected):
    assert no_output_notice_secs(idle_enabled=idle_enabled, idle_timeout=idle_timeout) == expected


def test_high_effort_floor_delays_the_notice_instead_of_narrating_a_healthy_wait(monkeypatch):
    """Reasoning at high+ raises the idle default to the 300s silence floor, so a slow
    healthy request is not narrated at 180s."""
    request, notices, _ = _request(idle_timeout=300.0)
    _drive(request, monkeypatch, frames=_metadata_every(10.0, 290.0), until=290.0)
    assert [n for n in notices if n] == []

    request, notices, _ = _request(idle_timeout=300.0)
    _drive(request, monkeypatch, frames=_metadata_every(10.0, 360.0), until=360.0)
    assert [n for n in notices if n], "past the floor the wait is named"


def test_status_line_never_carries_model_content(monkeypatch):
    """The status line reports timing, never what the model said."""
    request, notices, touches = _request()
    frames = [(1.0, TEXT_DELTA), (2.0, REASONING_DELTA)] + _metadata_every(10.0, 600.0, start=10.0)
    _drive(request, monkeypatch, frames=frames, until=600.0)

    assert not any("token" in line or "thought" in line for line in notices + touches)


def test_metadata_phases_pass_the_desktop_wait_filter():
    """Desktop's providerWaitText only forwards explained wait frames; pinned here as a
    plain string contract (as tests/hermes_cli/test_load_progress.py does) so the two
    sides cannot drift silently."""
    import re

    accept = (r"^(?:⏳|⚠|↻|⚙)\s*(?:(?:still\s+)?waiting on|loading|processing prompt"
              r"|no (?:output|response)|model returned)")
    for phase in sorted(METADATA_ONLY_PHASES):
        for watchdog in (None, ("wall-clock stale", 900.0), ("wall-clock stale", 9.0)):
            text = wait_notice_text("gpt-5.5-codex", 300.0, phase, watchdog)
            assert re.match(accept, text, re.IGNORECASE), text


@pytest.mark.parametrize("event", [LIFECYCLE, STRUCTURAL, EMPTY_DELTA])
def test_marker_keeps_liveness_and_progress_apart(event):
    state = SimpleNamespace(lock=threading.Lock(), last_event_ts=None,
                            last_progress_ts=None, retry_started_ts=None)
    cr._mark_codex_watchdog_event(state, event, 1000.0)
    assert state.last_event_ts == 1000.0 and state.last_progress_ts is None
    cr._mark_codex_watchdog_event(state, TEXT_DELTA, 1001.0)
    assert state.last_event_ts == 1001.0 and state.last_progress_ts == 1001.0


def test_marker_starts_a_fresh_progress_phase_after_a_reconnect():
    state = SimpleNamespace(lock=threading.Lock(), last_event_ts=900.0,
                            last_progress_ts=900.0, retry_started_ts=1000.0)
    cr._mark_codex_watchdog_event(state, LIFECYCLE, 1005.0)
    assert state.retry_started_ts is None
    assert state.last_event_ts == 1005.0 and state.last_progress_ts is None


def test_retry_notice_does_not_claim_prior_attempt_elapsed_was_lifecycle_traffic():
    request, notices, _ = _request()
    state = request.codex_watchdog_state
    assert state is not None
    cr._mark_codex_watchdog_event(state, TEXT_DELTA, CALL_START + 10.0)
    state.retry_started_ts = CALL_START + 290.0
    cr._mark_codex_watchdog_event(state, LIFECYCLE, CALL_START + 299.0)
    request._emit_wait_notice(300.0)
    assert "300s total API-call elapsed" in notices[-1]
    assert "no model output yet in this attempt" in notices[-1]
    assert "300s of lifecycle events" not in notices[-1]
