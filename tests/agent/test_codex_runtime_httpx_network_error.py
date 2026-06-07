"""Regression tests for codex_responses stream retry on httpx.NetworkError.

Before this fix: when Copilot's Responses API closed the upstream socket
mid-stream (after partial SSE delivery), httpx wrapped the underlying
[Errno 32] Broken pipe into ``httpx.ReadError``. The retry handler at
codex_runtime.py:478-493 (connect-time) and :519-528 (mid-iteration)
caught BrokenPipeError / OSError / ConnectionError but NOT the
httpx.NetworkError hierarchy.

``httpx.ReadError.__mro__`` is::

    ReadError -> NetworkError -> TransportError -> RequestError ->
    HTTPError -> Exception

Critically, it is NOT a subclass of ``OSError``, ``ConnectionError``,
``BrokenPipeError``, or any stdlib network-error type. So an EPIPE that
httpx wrapped escaped the retry path entirely and bubbled out as::

    API call failed after 3 retries: [Errno 32] Broken pipe

After this fix: ``_httpx.NetworkError`` is added to BOTH retry tuples,
covering ``ReadError`` / ``WriteError`` / ``CloseError`` / etc.

Empirically reproduced on api.githubcopilot.com gpt-5.5 with
reasoning_effort=medium-or-higher on prompts that take >120s of
server-side thinking before the first chunk lands.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import httpx
import pytest

# Stub optional heavy imports so run_agent imports cleanly in isolation
sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())


def _make_codex_agent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    (tmp_path / "config.yaml").write_text("{}\n", encoding="utf-8")
    from run_agent import AIAgent

    agent = AIAgent(
        model="gpt-5.5",
        provider="openai-codex",
        api_key="sk-dummy",
        base_url="https://chatgpt.com/backend-api/codex",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        platform="cli",
    )
    agent.api_mode = "codex_responses"
    monkeypatch.setattr(agent, "_emit_status", lambda *a, **k: None)
    return agent


def _fake_terminal_response():
    return SimpleNamespace(
        status="completed",
        incomplete_details=None,
        error=None,
        usage=None,
        id="resp_test",
        output=[],
    )


def test_httpx_readerror_is_subclass_of_network_error():
    """Documents the hierarchy that motivated this fix."""
    assert issubclass(httpx.ReadError, httpx.NetworkError)
    assert issubclass(httpx.WriteError, httpx.NetworkError)
    assert issubclass(httpx.CloseError, httpx.NetworkError)
    # The critical observation: httpx.ReadError is NOT an OSError.
    # The old retry tuple's OSError catch did not cover it.
    assert not issubclass(httpx.ReadError, OSError)
    assert not issubclass(httpx.ReadError, ConnectionError)


def test_httpx_readerror_during_connect_is_retried(tmp_path, monkeypatch):
    """httpx.ReadError raised by responses.create() before any chunk
    lands is now caught by the stream-retry handler."""
    from agent import codex_runtime

    monkeypatch.setenv("HERMES_CODEX_STREAM_RETRIES", "3")
    agent = _make_codex_agent(tmp_path, monkeypatch)

    connect_count = {"n": 0}

    class _FakeResponses:
        def create(self, **kwargs):
            connect_count["n"] += 1
            if connect_count["n"] == 1:
                # Mimic httpx wrapping an EPIPE: ReadError carries the
                # underlying socket exception's message.
                raise httpx.ReadError("[Errno 32] Broken pipe")
            return _fake_terminal_response()

    class _FakeClient:
        def __init__(self):
            self.responses = _FakeResponses()

    monkeypatch.setattr(
        agent, "_ensure_primary_openai_client", lambda reason=None: _FakeClient()
    )

    result = codex_runtime.run_codex_stream(agent, {"model": "gpt-5.5", "input": "hi"})
    assert connect_count["n"] == 2, (
        f"expected 2 connect attempts (1 ReadError + 1 success), got {connect_count['n']}"
    )
    assert result.status == "completed"


def test_httpx_writeerror_during_connect_is_retried(tmp_path, monkeypatch):
    """Same fix covers httpx.WriteError (less common, fires when the
    request body can't be sent because the socket dropped)."""
    from agent import codex_runtime

    monkeypatch.setenv("HERMES_CODEX_STREAM_RETRIES", "3")
    agent = _make_codex_agent(tmp_path, monkeypatch)

    call_count = {"n": 0}

    class _FakeResponses:
        def create(self, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise httpx.WriteError("send failed")
            return _fake_terminal_response()

    class _FakeClient:
        def __init__(self):
            self.responses = _FakeResponses()

    monkeypatch.setattr(
        agent, "_ensure_primary_openai_client", lambda reason=None: _FakeClient()
    )

    result = codex_runtime.run_codex_stream(agent, {"model": "gpt-5.5", "input": "hi"})
    assert call_count["n"] == 2
    assert result.status == "completed"


def test_httpx_closeerror_during_connect_is_retried(tmp_path, monkeypatch):
    """httpx.CloseError (raised when peer closes during/after send) is
    also covered by the NetworkError catch."""
    from agent import codex_runtime

    monkeypatch.setenv("HERMES_CODEX_STREAM_RETRIES", "3")
    agent = _make_codex_agent(tmp_path, monkeypatch)

    call_count = {"n": 0}

    class _FakeResponses:
        def create(self, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise httpx.CloseError("connection closed")
            return _fake_terminal_response()

    class _FakeClient:
        def __init__(self):
            self.responses = _FakeResponses()

    monkeypatch.setattr(
        agent, "_ensure_primary_openai_client", lambda reason=None: _FakeClient()
    )

    result = codex_runtime.run_codex_stream(agent, {"model": "gpt-5.5", "input": "hi"})
    assert call_count["n"] == 2
    assert result.status == "completed"


def test_httpx_readerror_exhausts_retries_then_propagates(tmp_path, monkeypatch):
    """If httpx.ReadError fires on every attempt, the final raise IS
    httpx.ReadError — not silently turned into a success or a different
    exception type."""
    from agent import codex_runtime

    monkeypatch.setenv("HERMES_CODEX_STREAM_RETRIES", "2")
    agent = _make_codex_agent(tmp_path, monkeypatch)

    call_count = {"n": 0}

    class _AlwaysReadError:
        def create(self, **kwargs):
            call_count["n"] += 1
            raise httpx.ReadError("[Errno 32] Broken pipe")

    class _FakeClient:
        def __init__(self):
            self.responses = _AlwaysReadError()

    monkeypatch.setattr(
        agent, "_ensure_primary_openai_client", lambda reason=None: _FakeClient()
    )

    with pytest.raises(httpx.ReadError):
        codex_runtime.run_codex_stream(agent, {"model": "gpt-5.5", "input": "hi"})

    # max_stream_retries=2 → 3 total attempts
    assert call_count["n"] == 3
