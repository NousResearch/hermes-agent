"""A stream retry never replays a request built for a route the agent has since left.

Regression for #112121: ``_StreamingCall`` captures ``api_kwargs`` once, but every stream
(re)open builds its client from the LIVE agent. When ``/model`` (``switch_model``) re-pointed
the agent while attempt 1 was stalled, attempt 2 sent the OLD model slug to the NEW provider's
base_url (404, then a rate-limit hold). The streamer now hands the transient error back to the
turn loop, which rebuilds the request for the current route on its own next attempt.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest


def _make_agent():
    from run_agent import AIAgent

    agent = AIAgent(
        api_key="test-key", base_url="https://openrouter.ai/api/v1", model="deepseek/deepseek-v4-flash",
        provider="openrouter", quiet_mode=True, skip_context_files=True, skip_memory=True,
        enabled_toolsets=[], max_iterations=1,
    )
    agent.api_mode = "chat_completions"
    return agent


def _ok_stream(model):
    chunks = [
        SimpleNamespace(choices=[SimpleNamespace(index=0, delta=SimpleNamespace(
            content="ok", tool_calls=None, reasoning_content=None, reasoning=None), finish_reason=None)],
            model=model, usage=None),
        SimpleNamespace(choices=[SimpleNamespace(index=0, delta=SimpleNamespace(
            content=None, tool_calls=None, reasoning_content=None, reasoning=None), finish_reason="stop")],
            model=model, usage=None),
    ]
    stream = MagicMock()
    stream.__iter__ = MagicMock(return_value=iter(chunks))
    stream.response = MagicMock(headers={})
    return stream


def _run(agent, on_create):
    """Drive the streaming call with a request-local client whose ``create`` is ``on_create``."""
    client = MagicMock()
    client.chat.completions.create.side_effect = on_create
    with patch("run_agent.AIAgent._create_request_openai_client", return_value=client), \
            patch("run_agent.AIAgent._close_request_openai_client"):
        return agent._interruptible_streaming_api_call(
            {"model": "deepseek/deepseek-v4-flash", "messages": [{"role": "user", "content": "hi"}]})


def test_stream_retry_does_not_replay_stale_route_after_switch_model():
    """After a mid-request switch_model the streamer sends nothing more and surfaces the
    transient error (the turn loop rebuilds for the new route); on base it re-sent the
    deepseek slug to the moonshot base_url."""
    agent = _make_agent()
    sent = []

    def switch_then_drop(**kwargs):
        sent.append((kwargs["model"], agent.base_url))
        with patch("agent.model_metadata.get_model_context_length", return_value=128000):
            agent.switch_model("kimi-k2.6", "kimi-coding", api_key="k",
                               base_url="https://api.moonshot.ai/v1", api_mode="chat_completions")
        raise httpx.ReadError("stale stream killed")

    with pytest.raises(httpx.ReadError):
        _run(agent, switch_then_drop)
    assert sent == [("deepseek/deepseek-v4-flash", "https://openrouter.ai/api/v1")]


def test_stream_retry_still_reconnects_in_place_when_route_unchanged():
    """Control: without a switch the transient drop is retried inside the streamer."""
    agent = _make_agent()
    sent = []

    def drop_once_then_succeed(**kwargs):
        sent.append(kwargs["model"])
        if len(sent) == 1:
            raise httpx.ReadError("stale stream killed")
        return _ok_stream(kwargs["model"])

    response = _run(agent, drop_once_then_succeed)
    assert response is not None
    assert sent == ["deepseek/deepseek-v4-flash", "deepseek/deepseek-v4-flash"]
