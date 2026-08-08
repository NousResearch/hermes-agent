"""Regression tests: Codex stream watchdog counts ANY inbound bytes as activity.

The Codex TTFB / stream-idle watchdogs read
``agent._codex_stream_last_event_ts``. Historically that timestamp was only
refreshed per *parsed SDK event*. Some Responses-API backends (observed: xAI
Grok with high reasoning effort) emit SSE comment lines (``: keepalive``)
every ~15s during long thinking phases. Per the SSE spec those lines are
comments, and the OpenAI SDK's ``SSEDecoder`` drops them — so a connection
actively streaming keepalives looked dead to the watchdog and was killed
mid-think (operators worked around it with large
HERMES_CODEX_EVENT_STALE_TIMEOUT_SECONDS values).

``codex_runtime._wrap_codex_stream_byte_activity`` wraps the underlying
httpx response's ``iter_raw`` so every raw byte chunk — including comment
lines the SDK swallows — refreshes the watchdog timestamp. The watchdog
comment in ``chat_completion_helpers`` always documented keepalives as
activity; this makes the implementation match the intent.
"""

from __future__ import annotations

import sys
import time
import types
from types import SimpleNamespace

import httpx
import pytest

# Stub optional heavy imports so run_agent imports cleanly in isolation.
sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())


def _sse_response(chunks, delay: float = 0.0) -> httpx.Response:
    """A real streaming httpx.Response yielding ``chunks`` (optionally spaced
    by ``delay`` seconds) over the byte path the OpenAI SDK consumes."""

    class _ByteStream(httpx.SyncByteStream):
        def __iter__(self):
            for chunk in chunks:
                if delay:
                    time.sleep(delay)
                yield chunk

        def close(self):
            pass

    return httpx.Response(
        200,
        headers={"Content-Type": "text/event-stream"},
        stream=_ByteStream(),
        request=httpx.Request("POST", "https://api.x.ai/v1/responses"),
    )


def _sdk_stream(response: httpx.Response):
    """The real OpenAI SDK Stream over ``response`` — the same object
    ``responses.create(stream=True)`` hands to ``run_codex_stream``."""
    import openai
    from openai._streaming import Stream

    client = openai.OpenAI(api_key="sk-dummy")
    return Stream(cast_to=object, response=response, client=client)


def test_byte_activity_wrapper_counts_keepalive_comment_bytes():
    """Keepalive comment chunks refresh the watchdog timestamp even though the
    SDK's SSE decoder yields ZERO events for them."""
    from agent.codex_runtime import _wrap_codex_stream_byte_activity

    agent = SimpleNamespace(_codex_stream_last_event_ts=None)
    body = b": keepalive\n\n: keepalive\n\n"
    stream = _sdk_stream(_sse_response([body]))

    _wrap_codex_stream_byte_activity(agent, stream)

    events = list(stream)  # SDK surfaces nothing — comments are dropped
    assert events == []
    assert agent._codex_stream_last_event_ts is not None


def test_byte_activity_wrapper_updates_timestamp_per_chunk():
    """Each raw chunk advances the timestamp — liveness is tracked at byte
    cadence, not event cadence."""
    from agent.codex_runtime import _wrap_codex_stream_byte_activity

    agent = SimpleNamespace(_codex_stream_last_event_ts=None)
    response = _sse_response([b": keepalive\n\n", b": keepalive\n\n"])
    _wrap_codex_stream_byte_activity(agent, SimpleNamespace(response=response))

    iterator = response.iter_bytes()
    next(iterator)
    first_ts = agent._codex_stream_last_event_ts
    assert first_ts is not None
    time.sleep(0.01)
    next(iterator)
    assert agent._codex_stream_last_event_ts >= first_ts


def test_byte_activity_wrapper_passes_bytes_through_unchanged():
    """The wrapper is observational only — the byte stream is identical."""
    from agent.codex_runtime import _wrap_codex_stream_byte_activity

    agent = SimpleNamespace(_codex_stream_last_event_ts=None)
    chunks = [
        b": keepalive\n\n",
        b'data: {"type":"response.output_text.delta","delta":"hi"}\n\n',
        b": keepalive\n\n",
    ]
    response = _sse_response(chunks)
    _wrap_codex_stream_byte_activity(agent, SimpleNamespace(response=response))

    assert b"".join(response.iter_bytes()) == b"".join(chunks)


def test_byte_activity_wrapper_noop_without_httpx_response():
    """Fake/stub streams (plain iterables used across the test suite) have no
    ``.response`` — the wrapper must no-op, never raise."""
    from agent.codex_runtime import _wrap_codex_stream_byte_activity

    agent = SimpleNamespace(_codex_stream_last_event_ts=None)
    _wrap_codex_stream_byte_activity(agent, SimpleNamespace())  # no .response
    _wrap_codex_stream_byte_activity(agent, object())
    assert agent._codex_stream_last_event_ts is None


def test_byte_activity_wrapper_is_idempotent():
    """A second wrap of the same response must not chain another layer."""
    from agent.codex_runtime import _wrap_codex_stream_byte_activity

    agent = SimpleNamespace(_codex_stream_last_event_ts=None)
    response = _sse_response([b": keepalive\n\n"])
    stream = SimpleNamespace(response=response)
    _wrap_codex_stream_byte_activity(agent, stream)
    wrapped_once = response.iter_raw
    _wrap_codex_stream_byte_activity(agent, stream)
    assert response.iter_raw is wrapped_once


def _make_codex_agent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    (tmp_path / "config.yaml").write_text("{}\n", encoding="utf-8")
    from run_agent import AIAgent

    agent = AIAgent(
        model="grok-4.5",
        provider="xai-oauth",
        api_key="sk-dummy",
        base_url="https://api.x.ai/v1",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        platform="cli",
    )
    agent.api_mode = "codex_responses"
    monkeypatch.setattr(agent, "_emit_status", lambda *a, **k: None)
    # Keep the wall-clock stale timeout high so any early kill is unambiguously
    # the TTFB / idle watchdog path, not the stale-call path.
    monkeypatch.setattr(
        agent, "_compute_non_stream_stale_timeout", lambda *a, **k: 60.0
    )
    return agent


def test_keepalive_comment_bytes_prevent_watchdog_kill(tmp_path, monkeypatch):
    """E2E: a stream that emits ONLY ``: keepalive`` comments — zero SDK
    events — for several times the TTFB cutoff must NOT be killed by the
    watchdog. Before the fix this died with a TimeoutError via
    ``codex_ttfb_kill``; now the stream runs to its natural end and surfaces
    its own error (no terminal response), never a watchdog timeout."""
    from agent import chat_completion_helpers as h
    from agent import codex_runtime

    agent = _make_codex_agent(tmp_path, monkeypatch)
    monkeypatch.setenv("HERMES_CODEX_TTFB_TIMEOUT_SECONDS", "0.5")
    monkeypatch.setenv("HERMES_CODEX_EVENT_STALE_TIMEOUT_SECONDS", "0.7")

    closes: list = []
    dummy_client = SimpleNamespace()
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **k: dummy_client)
    monkeypatch.setattr(agent, "_buffer_status", lambda *a, **k: None)
    monkeypatch.setattr(
        agent, "_abort_request_openai_client",
        lambda c, reason=None: closes.append(reason),
    )
    monkeypatch.setattr(
        agent, "_close_request_openai_client",
        lambda c, reason=None: closes.append(reason),
    )

    # Six keepalive-only chunks at 0.4s intervals ≈ 2.4s of byte activity with
    # ZERO parseable SSE events — well past the 0.5s TTFB cutoff — then EOF.
    keepalive_body = [b": keepalive\n\n"] * 6
    sdk_stream = _sdk_stream(_sse_response(keepalive_body, delay=0.4))
    fake_stream_client = SimpleNamespace(
        responses=SimpleNamespace(create=lambda **kw: sdk_stream)
    )

    def _real_stream(api_kwargs, client=None, on_first_delta=None):
        return codex_runtime.run_codex_stream(
            agent, api_kwargs, client=fake_stream_client
        )

    monkeypatch.setattr(agent, "_run_codex_stream", _real_stream)

    start = time.monotonic()
    with pytest.raises(RuntimeError, match="terminal response"):
        h.interruptible_api_call(agent, {"model": "grok-4.5", "input": "hi"})
    elapsed = time.monotonic() - start

    # The stream ran to its natural end (~2.4s), far past the 0.5s TTFB
    # cutoff — the watchdog never fired. Loose lower bound only; sleeps make
    # the stream strictly longer, never shorter.
    assert elapsed >= 1.5
    assert "codex_ttfb_kill" not in closes
    assert "codex_stream_idle_kill" not in closes
    assert agent._codex_stream_last_event_ts is not None
