"""Regression test: the acting MoA aggregator gets the tool->user role policy.

The outer agent runtime under a MoA preset is ``provider="moa"`` with the
preset name as its model, so the main loop's destination gate cannot tell that
the *resolved* aggregator is a Mistral-family endpoint. The prepared
aggregator transcript carries the acting turn's tool results, so it can hold
the exact ``tool`` -> ``user`` shape Mistral rejects with HTTP 400
``Unexpected role 'user' after role 'tool'``.

The projection therefore has to happen where the real destination is known —
after ``_slot_runtime()`` resolves the aggregator's provider/model/base_url —
and must leave the prepared state canonical.

Refs #20154.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest


def _response(content="synthesis"):
    message = SimpleNamespace(content=content, tool_calls=[])
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], usage=None, model="fake")


def _prepared():
    return {
        "messages": [
            {"role": "user", "content": "find the config"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "t1", "type": "function",
                             "function": {"name": "search_files",
                                          "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "t1", "content": "found 3 files"},
            {"role": "user", "content": "actually, check the other dir"},
        ],
        "guidance": None,
        "aggregator": {"provider": "nvidia",
                       "model": "mistralai/mistral-small-4-119b-2603"},
        "aggregator_temperature": None,
    }


def _tool_then_user_indexes(messages):
    return [
        idx for idx in range(len(messages) - 1)
        if messages[idx].get("role") == "tool"
        and messages[idx + 1].get("role") == "user"
    ]


@pytest.fixture
def captured_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "agent.moa_loop.call_llm",
        lambda **kwargs: calls.append(kwargs) or _response(),
    )
    return calls


def _run(monkeypatch, captured_calls, runtime):
    from agent import moa_loop

    monkeypatch.setattr(moa_loop, "_slot_runtime", lambda slot: dict(runtime))
    completions = moa_loop.MoAChatCompletions.__new__(moa_loop.MoAChatCompletions)
    completions._pending_trace = None
    prepared = _prepared()
    canonical = copy.deepcopy(prepared)

    completions._call_prepared_aggregator(prepared, {})

    assert prepared == canonical, "prepared state must stay canonical"
    return captured_calls[0]["messages"]


def test_mistral_aggregator_receives_no_tool_then_user(monkeypatch, captured_calls):
    wire = _run(monkeypatch, captured_calls, {
        "provider": "nvidia",
        "model": "mistralai/mistral-small-4-119b-2603",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "api_mode": "chat_completions",
    })

    assert _tool_then_user_indexes(wire) == []
    assert [m["role"] for m in wire] == [
        "user", "assistant", "tool", "assistant", "user",
    ]
    # Wire-shaped: the MoA path rebuilds a request-local copy per call, so the
    # bridge carries no reversal scaffolding for a sweeper to strip.
    assert not any(
        key.startswith("_") for msg in wire for key in msg if isinstance(key, str)
    )


def test_lenient_aggregator_payload_is_unchanged(monkeypatch, captured_calls):
    wire = _run(monkeypatch, captured_calls, {
        "provider": "openai",
        "model": "gpt-5.5",
        "base_url": "https://api.openai.com/v1",
        "api_mode": "chat_completions",
    })

    assert [m["role"] for m in wire] == ["user", "assistant", "tool", "user"]
    assert _tool_then_user_indexes(wire) == [2]
