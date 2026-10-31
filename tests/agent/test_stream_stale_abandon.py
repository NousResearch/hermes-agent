"""A streaming worker that stale kills cannot unwind must not stall the turn forever.

Incident: the monitor killed the same Anthropic stream attempt every stale window for
hours. The worker never unwound (parked outside the socket read, so the socket
shutdown could not wake it), ``_count_stale_attempt`` counts each attempt once so the
stale streak stayed at 1, ``_check_stale_giveup`` only runs before a call, and only
/stop ended the turn. Now the call abandons the worker after ``_STALE_ABANDON_KILLS``
kills of the same attempt and raises a retryable ``TimeoutError``, the way the /stop path
already leaves a worker behind. An inline call (cron, delegated children) cannot be
abandoned, so it stops reporting activity and the owner's inactivity watchdog ends it.
"""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest

from agent import chat_completion_helpers as helpers
from agent import chat_completion_stream_monitor as monitor

_ABANDONED = "request abandoned"


def _anthropic_agent():
    from run_agent import AIAgent

    agent = AIAgent(api_key="test-key", base_url="https://example.com/v1", model="claude-opus-4-7",
                    quiet_mode=True, skip_context_files=True, skip_memory=True)
    agent.api_mode = "anthropic_messages"
    agent._anthropic_client = MagicMock()
    agent._anthropic_api_key = "test-anthropic-key"
    agent._create_request_anthropic_client = lambda *a, **k: agent._anthropic_client
    agent._consecutive_stale_streams = 0
    return agent


def _stream_of(events_factory):
    def _open(*_args, **_kwargs):
        cm, raw = MagicMock(), MagicMock()
        raw.response = None
        raw.__iter__ = MagicMock(return_value=events_factory())
        cm.__enter__ = MagicMock(return_value=raw)
        cm.__exit__ = MagicMock(return_value=False)
        return cm
    return _open


def _call_in_thread(agent):
    outcome: dict = {}

    def _run():
        try:
            outcome["value"] = agent._interruptible_streaming_api_call({})  # no model: no opus stale floor
        except BaseException as exc:  # recorded for the assertions
            outcome["error"] = exc

    caller = threading.Thread(target=_run, name="stale-abandon-caller", daemon=True)
    caller.start()
    return caller, outcome


def _log_count(caplog, needle):
    return sum(needle in r.getMessage() for r in caplog.records)


@pytest.fixture()
def wedged(monkeypatch):
    """An Anthropic stream whose worker blocks on an Event that neither the client abort
    nor the socket shutdown can set: a worker parked outside the socket read."""
    monkeypatch.setenv("HERMES_STREAM_STALE_TIMEOUT", "0.3")
    monkeypatch.setenv("HERMES_STREAM_STALE_GIVEUP", "5")
    agent = _anthropic_agent()
    agent._abort_request_anthropic_client = lambda client, reason=None: None
    release = threading.Event()

    def _events():
        release.wait(timeout=30.0)
        yield SimpleNamespace(type="message_stop")

    agent._anthropic_client.messages.stream.side_effect = _stream_of(_events)
    yield agent
    release.set()  # let the abandoned daemon worker finish


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_wedged_worker_is_abandoned_with_a_retryable_timeout(wedged, caplog):
    with caplog.at_level("WARNING"):
        caller, outcome = _call_in_thread(wedged)
        caller.join(timeout=10.0)

    assert not caller.is_alive(), "the call never ended: the same wedged attempt was re-killed forever"
    assert isinstance(outcome.get("error"), TimeoutError), outcome
    assert _ABANDONED in str(outcome["error"])
    assert _log_count(caplog, "Stream stale for") == 3  # _STALE_ABANDON_KILLS
    assert _log_count(caplog, "abandoning the request") == 1
    assert wedged._consecutive_stale_streams == 1, "one wedged attempt must still count once"


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_worker_that_unwinds_after_a_kill_still_retries(monkeypatch, caplog):
    """The common path: the kill unblocks the read, the worker retries with a fresh
    attempt, and no attempt is killed twice, so nothing is abandoned.

    Relies on ``tests/agent/conftest.py`` zeroing ``jittered_backoff``: the attempt id moves
    only when the worker STARTS its next attempt, so with this 0.3 s stale window kills
    landing in a real 1-4 s retry backoff would count as the same attempt. In production
    the stale window (180 s or more) dwarfs that backoff."""
    monkeypatch.setenv("HERMES_STREAM_STALE_TIMEOUT", "0.3")
    monkeypatch.setenv("HERMES_STREAM_STALE_GIVEUP", "5")
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "2")
    agent = _anthropic_agent()
    unblock = threading.Event()
    agent._abort_request_anthropic_client = lambda client, reason=None: unblock.set()
    opened = []

    def _events():
        opened.append(1)
        unblock.wait(timeout=10.0)
        unblock.clear()
        raise httpx.RemoteProtocolError("peer closed connection without sending complete message body")
        yield  # pragma: no cover - generator marker

    agent._anthropic_client.messages.stream.side_effect = _stream_of(_events)
    with caplog.at_level("WARNING"):
        caller, outcome = _call_in_thread(agent)
        caller.join(timeout=15.0)
    unblock.set()

    assert not caller.is_alive(), "the recoverable call never ended"
    assert _ABANDONED not in str(outcome.get("error")), outcome
    assert len(opened) == 3, f"each kill should lead to a fresh attempt (1 + 2 retries), saw {len(opened)}"
    assert _log_count(caplog, "abandoning the request") == 0


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_abandon_limit_of_one_abandons_on_the_first_kill(wedged, monkeypatch, caplog):
    monkeypatch.setattr(monitor, "_STALE_ABANDON_KILLS", 1, raising=False)
    with caplog.at_level("WARNING"):
        caller, outcome = _call_in_thread(wedged)
        caller.join(timeout=10.0)

    assert not caller.is_alive()
    assert isinstance(outcome.get("error"), TimeoutError), outcome
    assert _log_count(caplog, "Stream stale for") == 1, "a limit of one must not wait a second window"
    assert _log_count(caplog, "abandoning the request") == 1


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_abandon_after_invisible_deltas_retries_instead_of_an_empty_stub(monkeypatch):
    """Whitespace-only text reached no consumer before the worker wedged: nothing was
    delivered, so the abandon must surface as a retryable error, not an empty partial stub."""
    monkeypatch.setenv("HERMES_STREAM_STALE_TIMEOUT", "0.3")
    monkeypatch.setenv("HERMES_STREAM_STALE_GIVEUP", "5")
    agent = _anthropic_agent()
    agent._abort_request_anthropic_client = lambda client, reason=None: None
    release = threading.Event()

    def _events():
        yield SimpleNamespace(type="content_block_start", index=0,
                              content_block=SimpleNamespace(type="text", text=""))
        yield SimpleNamespace(type="content_block_delta", index=0,
                              delta=SimpleNamespace(type="text_delta", text="  \n"))
        release.wait(timeout=30.0)
        yield SimpleNamespace(type="message_stop")

    agent._anthropic_client.messages.stream.side_effect = _stream_of(_events)
    try:
        caller, outcome = _call_in_thread(agent)
        caller.join(timeout=10.0)
    finally:
        release.set()

    assert not caller.is_alive()
    assert "value" not in outcome, f"an empty partial stub was returned: {outcome.get('value')!r}"
    assert isinstance(outcome.get("error"), TimeoutError), outcome


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_wedged_inline_call_stops_reporting_activity(wedged, monkeypatch, caplog):
    """Cron turns and delegated children run the request inline, so there is no worker to
    abandon. Past the kill limit the monitor must stop touching activity (kills and
    heartbeat), so the cron / delegate inactivity watchdog can end the job."""
    monkeypatch.setattr(helpers, "should_use_direct_api_call", lambda agent: True)
    monkeypatch.setattr(monitor, "_STALE_ABANDON_KILLS", 3, raising=False)
    touches, calls = [], []
    real_touch = wedged._touch_activity
    monkeypatch.setattr(wedged, "_touch_activity", lambda desc, **kw: (touches.append(desc), real_touch(desc, **kw)))
    real_rearm = helpers._StreamingCall._rearm_or_abandon_after_kill

    held_from = []  # len(touches) right after the kill that started the hold

    def _rearm(self, elapsed):
        calls.append(self)
        real_rearm(self, elapsed)
        if self._liveness_withheld_at is not None and not held_from:
            held_from.append(len(touches))

    monkeypatch.setattr(helpers._StreamingCall, "_rearm_or_abandon_after_kill", _rearm)
    with caplog.at_level("WARNING"):
        caller, _outcome = _call_in_thread(wedged)
        caller.join(timeout=2.5)  # about 8 stale windows: well past the limit
        kills = _log_count(caplog, "Stream stale for")
        stale_touches = [t for t in touches if t.startswith("stale stream detected")]
        after_hold = touches[held_from[0]:] if held_from else None
        call = calls[-1]
        before = len(touches)
        call._heartbeat(10)
        call._heartbeat(90)
        heartbeat_touches = len(touches) - before

    assert caller.is_alive(), "an inline call runs on the caller's thread and is not abandoned"
    assert kills > 3
    assert len(stale_touches) == 2, f"kills past the limit still reported activity: {stale_touches}"
    # Every activity path counts (the reconnect notice touches activity too), not only the kill's own touch.
    assert after_hold == [], f"activity was still reported during the hold: {after_hold}"
    assert heartbeat_touches == 0, "the heartbeat kept reporting a wedged inline call as active"
    assert _log_count(caplog, "no longer reporting it as active") == 1
    assert _log_count(caplog, "abandoning the request") == 0
    # A real chunk lifts the hold.
    call.last_chunk_time["t"] = call._liveness_withheld_at + 1
    call._heartbeat(10)
    assert len(touches) > before


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_abandon_after_visible_text_returns_a_stub_classified_as_the_abandon(monkeypatch):
    """Visible text already reached the user: the abandon ends in the partial-delivery stub,
    and the stub is classified from the abandon error, not from whatever the worker left."""
    monkeypatch.setenv("HERMES_STREAM_STALE_TIMEOUT", "0.3")
    monkeypatch.setenv("HERMES_STREAM_STALE_GIVEUP", "5")
    agent = _anthropic_agent()
    agent._abort_request_anthropic_client = lambda client, reason=None: None
    agent._current_streamed_assistant_text = "Hello there"
    monkeypatch.setattr(helpers._StreamingCall, "_visible_text_delivered", lambda self: True)
    classified = []
    from agent import error_classifier

    real_classify = error_classifier.classify_api_error
    monkeypatch.setattr(error_classifier, "classify_api_error",
                        lambda error, **kw: (classified.append(error), real_classify(error, **kw))[1])
    release = threading.Event()

    def _events():
        yield SimpleNamespace(type="content_block_start", index=0,
                              content_block=SimpleNamespace(type="text", text=""))
        yield SimpleNamespace(type="content_block_delta", index=0,
                              delta=SimpleNamespace(type="text_delta", text="Hello there"))
        release.wait(timeout=30.0)
        yield SimpleNamespace(type="message_stop")

    agent._anthropic_client.messages.stream.side_effect = _stream_of(_events)
    try:
        caller, outcome = _call_in_thread(agent)
        caller.join(timeout=10.0)
    finally:
        release.set()

    assert not caller.is_alive()
    assert "value" in outcome, outcome
    assert classified and isinstance(classified[-1], TimeoutError) and _ABANDONED in str(classified[-1])


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_abandon_joins_the_worker_for_relay_teardown_like_stop(wedged, monkeypatch):
    """The in-turn retry opens a new Relay scope, so the abandon gets the same bounded
    join as /stop before raising (#81521)."""
    joined = []
    monkeypatch.setattr(helpers, "_join_worker_for_relay_teardown",
                        lambda worker, *, label: joined.append((worker, label)))
    caller, outcome = _call_in_thread(wedged)
    caller.join(timeout=10.0)

    assert isinstance(outcome.get("error"), TimeoutError), outcome
    assert len(joined) == 1 and joined[0][0] is not None and joined[0][1] == "Abandoned streaming"
