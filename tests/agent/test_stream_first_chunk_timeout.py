"""Pre-first-chunk stale deadline vs post-first-chunk stale timeout.

A Gemini request that streams thought summaries emits its first chunk early,
so a connection silent for 90s is dead — waiting out the context-scaled stale
timeout (600s+) before reconnecting just stalls the turn. Once the first chunk
has landed, long silences are normal thinking and the full stale timeout applies.
"""
import threading
import time
from types import SimpleNamespace

import pytest

import run_agent
from agent import chat_completion_helpers as helpers

_VERTEX_BASE = "https://aiplatform.googleapis.com/v1beta1/projects/p/locations/global/publishers/google"
_THOUGHTS = {"extra_body": {"thinking_config": {"includeThoughts": True, "thinkingBudget": 4096}}}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("HERMES_STREAM_FIRST_CHUNK_TIMEOUT", "HERMES_STREAM_STALE_TIMEOUT"):
        monkeypatch.delenv(var, raising=False)


def _agent(base_url=_VERTEX_BASE, provider="vertex", model="google/gemini-3-pro-preview"):
    return run_agent.AIAgent(
        api_key="test-key", base_url=base_url, model=model, provider=provider,
        quiet_mode=True, skip_context_files=True, skip_memory=True, enabled_toolsets=[], max_iterations=1,
    )


def _call(agent, extra=None):
    kwargs = {"model": agent.model, "messages": [{"role": "user", "content": "hi"}], **(extra or {})}
    return helpers._StreamingCall(agent, kwargs, None)


@pytest.mark.parametrize("extra", [
    _THOUGHTS,
    {"extra_body": {"google": {"thinking_config": {"include_thoughts": True}}}},
    {"extra_body": {"extra_body": {"google": {"thinking_config": {"include_thoughts": True}}}}},
])
def test_gemini_thought_stream_gets_short_first_chunk_deadline(extra):
    call = _call(_agent(), extra)
    call._resolve_stale_timeout()
    assert call._first_chunk_stale_timeout == min(90.0, call._stream_stale_timeout)


@pytest.mark.parametrize("extra", [None, {"extra_body": {"thinking_config": {"includeThoughts": False}}}])
def test_without_thought_streaming_first_chunk_deadline_equals_stale(extra):
    call = _call(_agent(), extra)
    call._resolve_stale_timeout()
    assert call._first_chunk_stale_timeout == call._stream_stale_timeout


def test_operator_stale_timeout_env_disables_implicit_first_chunk_deadline(monkeypatch):
    monkeypatch.setenv("HERMES_STREAM_STALE_TIMEOUT", "400")
    call = _call(_agent(), _THOUGHTS)
    call._resolve_stale_timeout()
    assert call._first_chunk_stale_timeout == call._stream_stale_timeout


def test_local_endpoint_keeps_stale_timeout_before_first_chunk():
    call = _call(_agent(base_url="http://127.0.0.1:1/v1", provider="custom", model="m"), _THOUGHTS)
    call._resolve_stale_timeout()
    assert call._first_chunk_stale_timeout == call._stream_stale_timeout


def test_first_chunk_env_override_is_capped_by_stale_timeout(monkeypatch):
    monkeypatch.setenv("HERMES_STREAM_FIRST_CHUNK_TIMEOUT", "45")
    call = _call(_agent())
    call._resolve_stale_timeout()
    assert call._first_chunk_stale_timeout == 45.0
    monkeypatch.setenv("HERMES_STREAM_FIRST_CHUNK_TIMEOUT", "99999")
    call._resolve_stale_timeout()
    assert call._first_chunk_stale_timeout == call._stream_stale_timeout


def _run_monitor_once(call, silent_for: float) -> list:
    kills = []

    def _kill(elapsed, threshold=None):
        kills.append(threshold)
        call._call_done.set()

    call._kill_stale_stream = _kill
    call.last_chunk_time["t"] = time.time() - silent_for
    call.agent._interrupt_requested = False
    call._call_done = threading.Event()
    timer = threading.Timer(1.5, call._call_done.set)
    timer.start()
    try:
        call._monitor_loop()
    finally:
        timer.cancel()
    return kills


def test_monitor_kills_at_first_chunk_deadline_before_any_chunk():
    call = _call(_agent())
    call._stream_stale_timeout, call._first_chunk_stale_timeout = 600.0, 90.0
    call.clients = SimpleNamespace(diag={"first_chunk_at": None})
    assert _run_monitor_once(call, silent_for=100.0) == [90.0]


def test_monitor_uses_full_stale_timeout_after_first_chunk():
    call = _call(_agent())
    call._stream_stale_timeout, call._first_chunk_stale_timeout = 600.0, 90.0
    call.clients = SimpleNamespace(diag={"first_chunk_at": time.time() - 200})
    assert _run_monitor_once(call, silent_for=100.0) == []
    assert _run_monitor_once(call, silent_for=601.0) == [600.0]


def test_stale_warning_logs_active_threshold(caplog):
    call = _call(_agent())
    call._stream_stale_timeout = 600.0
    call.clients = SimpleNamespace(close_once=lambda *_: None)
    call._cancel_current_stream_attempt = lambda *_: None
    call._shutdown_stale_attempt_socket = lambda *_: None
    with caplog.at_level("WARNING", logger=helpers.logger.name):
        call._kill_stale_stream(95.0, threshold=90.0)
    assert "threshold 90s" in caplog.text
