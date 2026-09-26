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
        if kw["model"] in strict_models and any(a["role"] == b["role"] for a, b in zip(msgs, msgs[1:])):
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


def _send(facade, model, messages, guidance="[Mixture of Agents reference context]\nadvice"):
    prepared = facade.rebase_prepared_request(
        {"guidance": guidance, "aggregator": {"provider": "custom", "model": model}, "aggregator_temperature": None},
        messages,
    )
    return facade._call_prepared_aggregator(prepared, {"tools": None})


def test_alternation_400_is_classified_and_retried_once_merged_then_remembered(monkeypatch, facade):
    classified = classify_api_error(_AlternationRejected(), provider="custom", model="strict-model")
    assert classified.reason is FailoverReason.role_alternation

    calls = []
    monkeypatch.setattr(moa_loop, "call_llm", _strict_destination(calls, {"strict-model"}))
    task = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]

    _send(facade, "strict-model", task)  # iteration 1 of turn 1: split shape rejected → one merged retry
    assert [[m["role"] for m in c["messages"]] for c in calls] == [["system", "user", "user"], ["system", "user"]]
    assert calls[-1]["messages"][-1]["content"] == "task\n\n[Mixture of Agents reference context]\nadvice"

    del calls[:]
    _send(facade, "strict-model", [*task, {"role": "assistant", "content": "answer"}, {"role": "user", "content": "task2"}])
    # Remembered destination: iteration 1 of the next turn is pre-merged, no 400 paid again.
    assert [[m["role"] for m in c["messages"]] for c in calls] == [["system", "user", "assistant", "user"]]
    assert calls[0]["messages"][-1]["content"].startswith("task2\n\n[Mixture of Agents reference context]")


def test_destinations_that_accept_the_split_shape_keep_byte_identical_requests(monkeypatch, facade):
    calls = []
    monkeypatch.setattr(moa_loop, "call_llm", _strict_destination(calls, {"strict-model"}))
    task = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
    guidance = "[Mixture of Agents reference context]\nadvice"

    _send(facade, "strict-model", task, guidance)  # teaches the facade about the strict destination
    del calls[:]
    _send(facade, "lenient-model", task, guidance)

    # The lenient destination on the SAME facade still gets the split, cache-stable shape in one request.
    assert len(calls) == 1
    assert calls[0]["messages"] == [*task, {"role": "user", "content": guidance}]
