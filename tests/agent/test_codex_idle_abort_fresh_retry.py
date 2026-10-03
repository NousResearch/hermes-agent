"""Idle-abort must not inner-retry an aborted Codex client.

Observed field shape: parsed SSE activity, then ~180s with no events, then
reconnect, then ``httpx.ReadError`` / ``Errno 32 Broken pipe``, repeating on
the outer retry cadence. This file pins the client-side mechanism without
claiming that every Broken pipe in logs is reuse.

The inner ``run_codex_stream`` retry currently calls
``active_client.responses.create`` again on the same client whose sockets
the stranger-thread idle abort already shut down. That is a demonstrated
reuse path. Outer retry after idle abort must still build a fresh client.
"""

from __future__ import annotations

import errno
import sys
import types
from types import SimpleNamespace

import httpx
import pytest

sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())


def _make_agent(tmp_path, monkeypatch, *, provider="openai-codex"):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    (tmp_path / "config.yaml").write_text("{}\n", encoding="utf-8")

    from run_agent import AIAgent

    base_url = (
        "https://chatgpt.com/backend-api/codex"
        if provider == "openai-codex"
        else "https://api.x.ai/v1"
    )
    agent = AIAgent(
        model="gpt-5.5" if provider == "openai-codex" else "grok-4.3",
        provider=provider,
        api_key="test-key",
        base_url=base_url,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        platform="cli",
    )
    agent.api_mode = "codex_responses"
    monkeypatch.setattr(agent, "_emit_status", lambda *args, **kwargs: None)
    monkeypatch.setattr(agent, "_buffer_status", lambda *args, **kwargs: None)
    monkeypatch.setattr(agent, "_emit_wait_notice", lambda *args, **kwargs: None)
    return agent


class _EventStream:
    def __init__(self, events):
        self._events = events

    def __iter__(self):
        yield from self._events

    def close(self):
        pass


def _broken_pipe():
    return ConnectionError(errno.EPIPE, "Broken pipe")


def test_run_codex_stream_does_not_inner_retry_aborted_client(tmp_path, monkeypatch):
    """After the idle abort poisons the request client, mid-stream EPIPE
    must not open a second physical request on that same client."""
    from agent.codex_runtime import run_codex_stream

    agent = _make_agent(tmp_path, monkeypatch)
    creates = []

    class Responses:
        def create(self, **kwargs):
            creates.append(kwargs)

            def events():
                yield SimpleNamespace(type="response.in_progress")
                raise _broken_pipe()

            return _EventStream(events())

    client = SimpleNamespace(responses=Responses())
    agent._request_client_cache = {
        "client": client,
        "kwargs": {},
        "poisoned": True,
        "in_use": True,
    }

    with pytest.raises((ConnectionError, httpx.ReadError, OSError)):
        run_codex_stream(agent, {"model": agent.model, "input": "hi"}, client=client)

    assert len(creates) == 1, (
        f"aborted client was inner-retried {len(creates)} times; "
        "Broken pipe after idle abort is the reuse path, not a fresh transport"
    )


def test_run_codex_stream_still_retries_connect_blip_on_healthy_client(
    tmp_path, monkeypatch
):
    """A connect-time blip on a client that was not aborted may still retry."""
    from agent.codex_runtime import run_codex_stream

    agent = _make_agent(tmp_path, monkeypatch)
    creates = {"count": 0}

    class Responses:
        def create(self, **kwargs):
            creates["count"] += 1
            if creates["count"] == 1:
                raise httpx.ConnectError("blip")
            return _EventStream(
                [
                    SimpleNamespace(
                        type="response.completed",
                        response=SimpleNamespace(
                            status="completed", id="resp-ok", usage=None
                        ),
                    )
                ]
            )

    client = SimpleNamespace(responses=Responses())
    agent._request_client_cache = {
        "client": client,
        "kwargs": {},
        "poisoned": False,
        "in_use": True,
    }

    response = run_codex_stream(
        agent, {"model": agent.model, "input": "hi"}, client=client
    )
    assert response.status == "completed"
    assert creates["count"] == 2


def test_idle_kill_surfaces_timeout_not_broken_pipe(tmp_path, monkeypatch):
    """Stranger-thread idle abort unblocks the worker with EPIPE; the call
    must still raise TimeoutError so the outer loop reconnects instead of
    classifying our own abort as a provider ReadError."""
    from agent import chat_completion_helpers as helpers

    agent = _make_agent(tmp_path, monkeypatch)
    monkeypatch.setattr(
        agent, "_compute_non_stream_stale_timeout", lambda _kwargs: 30.0
    )
    monkeypatch.setenv("HERMES_CODEX_TTFB_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("HERMES_CODEX_EVENT_STALE_TIMEOUT_SECONDS", "0.35")
    monkeypatch.setenv("HERMES_CODEX_HARD_TIMEOUT_SECONDS", "30")

    closes = []
    dummy_client = SimpleNamespace()

    def fake_stream(api_kwargs, client=None, on_first_delta=None):
        agent._codex_stream_last_event_ts = helpers.time.time()
        deadline = helpers.time.time() + 5
        while helpers.time.time() < deadline:
            helpers.time.sleep(0.05)
        raise _broken_pipe()

    monkeypatch.setattr(
        agent, "_create_request_openai_client", lambda **kwargs: dummy_client
    )
    monkeypatch.setattr(
        agent,
        "_abort_request_openai_client",
        lambda request_client, reason=None: closes.append(reason),
    )
    monkeypatch.setattr(
        agent,
        "_close_request_openai_client",
        lambda request_client, reason=None: closes.append(reason),
    )
    monkeypatch.setattr(agent, "_run_codex_stream", fake_stream)

    with pytest.raises(TimeoutError, match="after first byte"):
        helpers.interruptible_api_call(agent, {"model": agent.model, "input": "hi"})

    assert "codex_stream_idle_kill" in closes
