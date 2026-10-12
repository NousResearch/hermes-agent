"""A chat-completions stream's socket read timeout never outlasts its stale budget.

The stale kill can only ``shutdown()`` the worker's socket (a stranger-thread ``close()`` is ruled
out, #29507), and on Windows ``shutdown()`` never wakes a blocked ``recv``. The worker then sits
until the read timeout expires, so whenever the read timeout was longer than the stale budget
(local endpoints: 1800s vs 900s; a shorter ``providers.<id>.stale_timeout_seconds`` vs the 120s
default read) the "reconnecting" retry was held back by the difference. The live recovery
timing is covered by ``test_stream_stale_kill_unwedge.py``.
"""
import pytest

import run_agent
from agent import chat_completion_helpers as helpers


def _call(base_url: str, stale: float):
    agent = run_agent.AIAgent(
        api_key="test-key", base_url=base_url, model="m", provider="custom",
        quiet_mode=True, skip_context_files=True, skip_memory=True, enabled_toolsets=[], max_iterations=1,
    )
    call = helpers._StreamingCall(agent, {"model": "m", "messages": [{"role": "user", "content": "hi"}]}, None)
    call._stream_stale_timeout = stale
    return call


@pytest.mark.parametrize(("base_url", "stale", "request_timeout", "expected_read"), [
    ("http://127.0.0.1:11434/v1", 900.0, None, 900.0),  # local: read was raised to the 1800s API timeout
    ("https://api.example.com/v1", 30.0, None, 30.0),  # shorter explicit stale vs the 120s default read
    ("https://api.example.com/v1", 60.0, 600.0, 60.0),  # per-provider request_timeout_seconds
    ("https://api.example.com/v1", 300.0, None, 300.0),  # reasoning patience still raises the read
], ids=["local", "short-stale", "request-timeout", "long-stale"])
def test_stream_read_timeout_expires_within_the_stale_budget(
        monkeypatch, base_url, stale, request_timeout, expected_read):
    monkeypatch.delenv("HERMES_STREAM_READ_TIMEOUT", raising=False)
    monkeypatch.delenv("HERMES_API_TIMEOUT", raising=False)
    monkeypatch.setattr(helpers, "get_provider_request_timeout", lambda *_a, **_k: request_timeout)

    _write, read, _connect = _call(base_url, stale)._stream_timeouts()

    assert read == expected_read


def test_disarmed_stale_detector_keeps_the_configured_read_timeout(monkeypatch):
    monkeypatch.setenv("HERMES_STREAM_READ_TIMEOUT", "75")
    monkeypatch.setattr(helpers, "get_provider_request_timeout", lambda *_a, **_k: None)

    _write, read, _connect = _call("https://api.example.com/v1", float("inf"))._stream_timeouts()

    assert read == 75.0
