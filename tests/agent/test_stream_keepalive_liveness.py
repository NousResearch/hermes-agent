"""SSE comment keepalives are Codex watchdog activity (#78818).

xAI Grok and other Responses backends send ``: keepalive`` comment lines during long
thinking; the OpenAI SDK drops them before they become events, so the TTFB / idle
watchdogs saw a dead stream and killed it mid-think.
"""

import time
from types import SimpleNamespace

import httpx
import pytest

import run_agent


def test_keepalive_comments_count_as_codex_watchdog_activity(tmp_path, monkeypatch):
    """Codex Responses wire (contributed in #78818): keepalive comments, zero SDK events,
    for several TTFB windows must not trip the TTFB / stream-idle watchdogs."""
    import openai
    from openai._streaming import Stream

    from agent import chat_completion_helpers as h
    from agent import codex_runtime

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_CODEX_TTFB_TIMEOUT_SECONDS", "0.5")
    monkeypatch.setenv("HERMES_CODEX_EVENT_STALE_TIMEOUT_SECONDS", "0.7")
    agent = run_agent.AIAgent(model="grok-4.5", provider="xai-oauth", api_key="sk-dummy", base_url="https://api.x.ai/v1",
                              quiet_mode=True, skip_context_files=True, skip_memory=True, platform="cli")
    agent.api_mode = "codex_responses"
    monkeypatch.setattr(agent, "_compute_non_stream_stale_timeout", lambda *a, **k: 60.0)
    closes: list = []
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **k: SimpleNamespace())
    monkeypatch.setattr(agent, "_abort_request_openai_client", lambda c, reason=None: closes.append(reason))
    monkeypatch.setattr(agent, "_close_request_openai_client", lambda c, reason=None: closes.append(reason))

    class _Keepalives(httpx.SyncByteStream):
        def __iter__(self):
            for _ in range(6):  # ~2.4s of bytes, no parseable event
                time.sleep(0.4)
                yield b": keepalive\n\n"

    response = httpx.Response(200, headers={"Content-Type": "text/event-stream"}, stream=_Keepalives(),
                              request=httpx.Request("POST", "https://api.x.ai/v1/responses"))
    sdk_stream = Stream(cast_to=object, response=response, client=openai.OpenAI(api_key="sk-dummy"))
    stream_client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kw: sdk_stream))
    monkeypatch.setattr(agent, "_run_codex_stream", lambda api_kwargs, client=None, on_first_delta=None:
                        codex_runtime.run_codex_stream(agent, api_kwargs, client=stream_client))

    with pytest.raises(RuntimeError, match="terminal response"):  # the stream's own end, not a watchdog kill
        h.interruptible_api_call(agent, {"model": "grok-4.5", "input": "hi"})

    assert "codex_ttfb_kill" not in closes and "codex_stream_idle_kill" not in closes
