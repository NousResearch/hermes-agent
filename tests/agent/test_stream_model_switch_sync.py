"""A mid-turn /model switch must not send the old model name to the new provider.

Regression for #112121: the streamer captures ``api_kwargs`` once at
construction and replays it on every stream (re)open, while the request client
is rebuilt per attempt from the agent's live runtime. After ``switch_model``
the replayed request still named the old model, so the new provider's base_url
404'd on a foreign model name. Every stream open now re-reads
``agent.model``; these tests pin that contract for the chat_completions open
path and for the shared sync helper the anthropic_messages open path uses.
"""
from types import SimpleNamespace

import pytest

import run_agent
from agent import chat_completion_helpers as helpers


def _agent():
    return run_agent.AIAgent(
        api_key="test-key", base_url="http://127.0.0.1:1/v1", model="m", provider="custom",
        quiet_mode=True, skip_context_files=True, skip_memory=True, enabled_toolsets=[], max_iterations=1,
    )


class _RecordingCompletions:
    """Stand-in for ``client.chat.completions``: records the kwargs, returns a dummy."""

    def __init__(self):
        self.sent = None

    def create(self, **kwargs):
        self.sent = dict(kwargs)
        return SimpleNamespace(dummy_stream=True)


def _call(agent, model):
    return helpers._StreamingCall(
        agent, {"model": model, "messages": [{"role": "user", "content": "hi"}]}, None)


def test_open_chat_stream_sends_current_model_after_switch(monkeypatch):
    """The wire model follows the agent's live model, not the captured kwargs.

    Fails on base: ``_open_chat_stream`` sent the construction-time model
    (the stale one after ``switch_model``) to the freshly built client.
    """
    agent = _agent()
    agent.model = "kimi-k2.6"
    call = _call(agent, "deepseek/deepseek-v4-flash")
    completions = _RecordingCompletions()
    monkeypatch.setattr(
        agent, "_create_request_openai_client",
        lambda **kwargs: SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )
    call._open_chat_stream(dict(call.api_kwargs))
    assert completions.sent is not None
    assert completions.sent["model"] == "kimi-k2.6"


def test_open_chat_stream_keeps_model_when_unchanged(monkeypatch):
    """No switch: the request passes through with its model untouched."""
    agent = _agent()
    agent.model = "kimi-k2.6"
    call = _call(agent, "kimi-k2.6")
    completions = _RecordingCompletions()
    monkeypatch.setattr(
        agent, "_create_request_openai_client",
        lambda **kwargs: SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )
    call._open_chat_stream(dict(call.api_kwargs))
    assert completions.sent["model"] == "kimi-k2.6"


def test_model_synced_kwargs_returns_same_object_when_matching():
    agent = _agent()
    agent.model = "kimi-k2.6"
    call = _call(agent, "kimi-k2.6")
    kwargs = {"model": "kimi-k2.6", "stream": True}
    assert call._model_synced_kwargs(kwargs) is kwargs


def test_model_synced_kwargs_rekeys_stale_model_without_mutating_input():
    agent = _agent()
    agent.model = "kimi-k2.6"
    call = _call(agent, "deepseek/deepseek-v4-flash")
    kwargs = {"model": "deepseek/deepseek-v4-flash", "stream": True, "timeout": 30}
    synced = call._model_synced_kwargs(kwargs)
    assert synced["model"] == "kimi-k2.6"
    assert synced["stream"] is True and synced["timeout"] == 30
    assert kwargs["model"] == "deepseek/deepseek-v4-flash", "input must not be mutated"


def test_model_synced_kwargs_leaves_model_less_kwargs_alone():
    agent = _agent()
    agent.model = "kimi-k2.6"
    call = _call(agent, "kimi-k2.6")
    kwargs = {"messages": []}
    assert call._model_synced_kwargs(kwargs) is kwargs
