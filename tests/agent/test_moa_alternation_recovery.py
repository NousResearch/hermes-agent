"""MoA aggregator: strict-alternation destinations get their adjacent same-role messages merged
reactively and per destination (#112358, last atom).

The guidance is deliberately a separate trailing ``user`` message so the prefix stays cache-stable;
a chat template that 400s on ``user(task), user(guidance)`` must get ONE merged retry, be remembered
for the session, and never change the bytes sent to destinations that accepted the split shape.
"""

from types import SimpleNamespace

import pytest

from agent import moa_loop
from agent.error_classifier import FailoverReason, classify_api_error
import itertools

ALTERNATION_MSG = "Conversation roles must alternate user/assistant/user/assistant/..."


class _AlternationRejected(Exception):
    status_code = 400

    def __init__(self):
        super().__init__(f"Error code: 400 - {{'error': {{'message': '{ALTERNATION_MSG}', 'type': 'invalid_request_error'}}}}")
        self.response = SimpleNamespace(status_code=400, headers={})
        self.body = {"message": ALTERNATION_MSG, "type": "invalid_request_error"}


def _strict_destination(calls, strict_models):
    """``call_llm`` double: 400s like a strict chat template when adjacent non-system messages share a role."""
    def call_llm(**kw):
        calls.append(kw)
        msgs = [m for m in kw["messages"] if m["role"] != "system"]
        if kw["model"] in strict_models and any(a["role"] == b["role"] for a, b in itertools.pairwise(msgs)):
            raise _AlternationRejected()
        return SimpleNamespace(choices=[])
    return call_llm


@pytest.fixture
def facade(monkeypatch):
    monkeypatch.setattr(
        moa_loop, "_slot_runtime",
        lambda slot: {"provider": "custom", "model": slot["model"], "base_url": "http://strict.local/v1",
                      "api_mode": "chat_completions"},
    )
    f = moa_loop.MoAChatCompletions("default", agent=None)
    f._pending_trace = None
    return f


def _send(facade, model, messages, guidance="private observations"):
    anchor = 2
    prepared = facade.rebase_prepared_request(
        {"guidance": guidance, "guidance_anchor": anchor,
         "guidance_prefix_key": moa_loop._hash_messages(messages[:anchor]),
         "aggregator": {"provider": "custom", "model": model}, "aggregator_temperature": None},
        messages,
    )
    return facade._call_prepared_aggregator(prepared, {"tools": None})


def _history():
    return [
        {"role": "system", "content": "sys"}, {"role": "user", "content": "task"},
        {"role": "assistant", "content": "checking", "reasoning_content": "preserve reasoning",
         "tool_calls": [{"id": "lookup-1", "type": "function",
                         "function": {"name": "lookup", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "lookup-1", "content": "result"},
    ]


def test_assistant_observation_retry_preserves_actions_and_remembers_destination(monkeypatch, facade):
    classified = classify_api_error(_AlternationRejected(), provider="custom", model="strict-model")
    assert classified.reason is FailoverReason.role_alternation
    calls = []
    monkeypatch.setattr(moa_loop, "call_llm", _strict_destination(calls, {"strict-model"}))
    history = _history()
    _send(facade, "strict-model", history)
    assert len(calls) == 2
    assert [m["role"] for m in calls[0]["messages"]] == ["system", "user", "assistant", "assistant", "tool"]
    merged = calls[1]["messages"]
    assert [m["role"] for m in merged] == ["system", "user", "assistant", "tool"]
    assert merged[2]["tool_calls"] == history[2]["tool_calls"]
    assert merged[2]["reasoning_content"] == history[2]["reasoning_content"]
    assert merged[2]["content"] == "private observations\n\nchecking"
    assert merged[-1] == history[-1]
    assert history[2]["content"] == "checking"
    calls.clear()
    _send(facade, "strict-model", history)
    assert len(calls) == 1
    assert calls[0]["messages"] == merged


def test_lenient_destination_keeps_anchored_advice_after_strict_retry(monkeypatch, facade):
    calls = []
    monkeypatch.setattr(moa_loop, "call_llm", _strict_destination(calls, {"strict-model"}))
    history = _history()
    _send(facade, "strict-model", history)
    calls.clear()
    _send(facade, "lenient-model", history)
    assert len(calls) == 1
    assert calls[0]["messages"] == [*history[:2], {"role": "assistant", "content": "private observations"}, *history[2:]]


def test_merge_does_not_drop_existing_assistant_actions():
    from agent.moa_alternation import merge_same_role_messages
    history = _history()
    actions = [history[2], {"role": "assistant", "content": "another action"}]
    assert merge_same_role_messages(actions) is actions
    users = [{"role": "user", "content": "task"}, {"role": "user", "content": "steering"}]
    assert merge_same_role_messages(users) == [{"role": "user", "content": "task\n\nsteering"}]


def test_strict_retry_accepts_tool_call_with_null_content(monkeypatch, facade):
    calls = []
    monkeypatch.setattr(moa_loop, "call_llm", _strict_destination(calls, {"strict-model"}))
    history = _history()
    history[2]["content"] = None
    _send(facade, "strict-model", history)
    assert len(calls) == 2
    assert calls[-1]["messages"][2]["content"] == "private observations"
    assert calls[-1]["messages"][2]["tool_calls"] == history[2]["tool_calls"]
    assert calls[-1]["messages"][2]["reasoning_content"] == history[2]["reasoning_content"]
    assert calls[-1]["messages"][-1]["tool_call_id"] == "lookup-1"
    assert history[2]["content"] is None
