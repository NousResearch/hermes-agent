"""Bulky historical user text must not make its first summary structurally impossible."""

import copy
import json
from itertools import pairwise

import pytest
import httpx
import openai

import agent.context_compressor as module


def compressor(**kwargs):
    return module.ContextCompressor(model="test/local", config_context_length=65_536, quiet_mode=True, **kwargs)


def historical():
    return {"role": "user", "content": "Remember QA_TOKEN=CEDAR_409.\n" + "ordinary timber measurement\n" * 7000}


@pytest.mark.parametrize("system", [False, True])
@pytest.mark.parametrize("turns", [1, 2, 3])
@pytest.mark.parametrize("protect_first", [0, 1, 3, 99])
def test_large_historical_head_is_summarized_without_rewriting_the_current_user(monkeypatch, system, turns, protect_first):
    requests = []

    def summarize(**kwargs):
        requests.append(kwargs)
        return {"choices": [{"message": {"content": "## Facts\nQA_TOKEN=CEDAR_409."}, "finish_reason": "stop"}]}

    monkeypatch.setattr(module, "call_llm", summarize)
    messages = ([{"role": "system", "content": "Keep this system prompt byte-identical."}] if system else [])
    for _ in range(turns):
        messages.extend([historical(), {"role": "assistant", "content": "SAVED"}])
    active = {"role": "user", "content": "What is QA_TOKEN?"}
    messages.append(active)
    original = copy.deepcopy(messages)
    cc = compressor(protect_first_n=protect_first)

    result = cc.compress(messages, current_tokens=60_000)

    assert requests, "exercise the actual summary dispatch rather than a structural no-op"
    assert "QA_TOKEN=CEDAR_409" in json.dumps(requests[0]["messages"])
    assert "QA_TOKEN=CEDAR_409" in json.dumps(result)
    assert result[-1]["content"] == active["content"]
    assert messages == original
    assert sum(len(str(m.get("content", ""))) for m in result) < sum(len(str(m.get("content", ""))) for m in original)
    if system:
        expected_system = copy.deepcopy(original[0])
        # These already-eligible legacy windows retain their established system note.
        # The new pressure recovery windows preserve the original system bytes.
        if protect_first == 0 and turns >= 2:
            expected_system["content"] += "\n\n" + cc._COMPRESSION_NOTE
        assert result[0] == expected_system
    visible_roles = [m["role"] for m in result if m["role"] in {"user", "assistant"}]
    assert all(a != b for a, b in pairwise(visible_roles))


def test_small_first_compaction_keeps_the_existing_protected_head(monkeypatch):
    calls = []
    monkeypatch.setattr(module, "call_llm", lambda **kwargs: calls.append(kwargs))
    messages = [{"role": "user", "content": "Remember a short fact."},
                {"role": "assistant", "content": "SAVED"}, {"role": "user", "content": "Recall it."}]
    cc = compressor()
    assert cc._effective_protect_first_n(messages) == 3
    assert cc.compress(messages, current_tokens=100) == messages
    assert not calls


def test_active_tool_transaction_is_carried_verbatim(monkeypatch):
    monkeypatch.setattr(module, "call_llm", lambda **kwargs: {
        "choices": [{"message": {"content": "## Facts\nQA_TOKEN=CEDAR_409."}, "finish_reason": "stop"}],
    })
    active = [
        {"role": "user", "content": "Read both files, then answer."},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "one", "type": "function", "function": {"name": "read_file", "arguments": '{"path":"one"}'}},
            {"id": "two", "type": "function", "function": {"name": "read_file", "arguments": '{"path":"two"}'}},
        ]},
        {"role": "tool", "tool_call_id": "one", "content": "Unread first result\n" * 3000},
        {"role": "tool", "tool_call_id": "two", "content": "Unread second result\n" * 3000},
    ]
    for tool in active[2:]:
        tool["content"] = [{"type": "text", "text": tool["content"]},
                           {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGk="}}]
        tool["api_content"] = copy.deepcopy(tool["content"])
    result = compressor().compress([historical(), {"role": "assistant", "content": "SAVED"}] + active,
                                   current_tokens=60_000)
    tail = result[-len(active):]
    for actual, expected in zip(tail, active):
        assert all(actual.get(key) == value for key, value in expected.items())


def test_an_active_giant_user_without_history_is_never_rewritten(monkeypatch):
    calls = []
    monkeypatch.setattr(module, "call_llm", lambda **kwargs: calls.append(kwargs))
    messages = [historical()]
    assert compressor().compress(messages, current_tokens=65_536) == messages
    assert not calls


@pytest.mark.parametrize("protect_first", [0, 1, 3, 99])
@pytest.mark.parametrize("minimum_users", [1, 2])
def test_pressure_recovery_keeps_the_configured_user_tail(monkeypatch, protect_first, minimum_users):
    monkeypatch.setattr(module, "call_llm", lambda **kwargs: {
        "choices": [{"message": {"content": "## Facts\nQA_TOKEN=CEDAR_409."}, "finish_reason": "stop"}],
    })
    historical_turns = []
    for index in range(2):
        row = historical()
        row["content"] += f"Historical turn {index}."
        historical_turns.extend([row, {"role": "assistant", "content": "SAVED"}])
    active = {"role": "user", "content": "Recall the token from both turns."}
    messages = historical_turns + [active]
    users = [row["content"] for row in messages if row["role"] == "user"]
    cc = compressor(protect_first_n=protect_first, min_tail_user_messages=minimum_users)
    result = cc.compress(messages, current_tokens=60_000)
    for content in users[-minimum_users:]:
        assert any(row.get("content") == content for row in result)
    assert result[-1]["content"] == active["content"]


def test_missing_summary_does_not_replace_historical_user_text(monkeypatch):
    monkeypatch.setattr(module, "call_llm", lambda **kwargs: {
        "choices": [{"message": {"content": ""}, "finish_reason": "stop"}],
    })
    messages = [historical(), {"role": "assistant", "content": "SAVED"}, {"role": "user", "content": "Recall QA_TOKEN."}]
    original = copy.deepcopy(messages)
    cc = compressor()
    result = cc.compress(messages, current_tokens=60_000)
    assert result == original
    assert messages == original
    assert cc.compression_count == 0


@pytest.mark.parametrize("failure", ["refusal", "length", "timeout", "overload"])
def test_failed_summary_preserves_history_and_rehydrated_provenance(monkeypatch, failure):
    def summarize(**kwargs):
        if failure == "timeout":
            raise TimeoutError("summary timed out")
        if failure == "overload":
            response = httpx.Response(503, request=httpx.Request("POST", "http://127.0.0.1/completions"))
            raise openai.APIStatusError("upstream overloaded", response=response, body={"error": "overloaded"})
        return {"choices": [{
            "message": {"content": "## Facts\nThis is incomplete.",
                        **({"refusal": "I cannot summarize this conversation."} if failure == "refusal" else {})},
            "finish_reason": "length" if failure == "length" else "stop",
        }]}

    monkeypatch.setattr(module, "call_llm", summarize)
    messages = [historical(), {"role": "assistant", "content": "SAVED"}, {"role": "user", "content": "Recall QA_TOKEN."}]
    original = copy.deepcopy(messages)
    cc = compressor()
    cc._previous_summary = "Earlier verified context."
    cc._summary_has_user_turn = False
    for _ in range(3 if failure == "overload" else 1):
        assert cc.compress(messages, current_tokens=60_000, force=True) == original
        assert messages == original
        assert cc._previous_summary == "Earlier verified context."
        assert cc._summary_has_user_turn is False
        assert cc.compression_count == 0


def test_legacy_pruning_cannot_upgrade_an_already_demoted_active_transaction_to_verbatim():
    cc = compressor(protect_first_n=0)
    cc.tail_token_budget = 4000
    messages = [
        {"role": "user", "content": "Earlier question."},
        {"role": "assistant", "content": "past detail " * 1600},
        {"role": "user", "content": "Read two files."},
    ]
    for identity in ("one", "two"):
        messages.extend([
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": identity, "type": "function",
                "function": {"name": "read_file", "arguments": json.dumps({"path": identity})},
            }]},
            {"role": "tool", "tool_call_id": identity, "content": "output data " * 10000},
        ])
    original = copy.deepcopy(messages)
    initial = cc._window_bounds(messages, 60_000)
    assert initial.protect_active is False
    plan = cc._plan_compaction_window(messages, 60_000)
    assert plan.pruned_count > 0
    assert plan.messages[-1]["content"] != messages[-1]["content"]
    # A full historical assistant anchor fits the established soft ceiling. It must
    # not be combined with the separate head allowance to invent a recovery need.
    assert cc._window_bounds(plan.messages, 60_000).protect_active is False
    assert plan.protect_active is False
    assert (plan.start, plan.end) == cc._compress_window(plan.messages)
    assert messages == original
