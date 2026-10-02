"""Provider-minted parallel tool-call ids are normalized once, at mint time (#130363)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.chat_completion_helpers import _assistant_tool_call_dict
from run_agent import AIAgent


def _tool_call(raw_id, n, **extra):
    return SimpleNamespace(
        id=raw_id, type="function",
        function=SimpleNamespace(name="read_file", arguments='{"path": "%s"}' % n), **extra,
    )


def _agent():
    # Real AIAgent methods (duplicate repair, assistant serialization), no init side effects.
    agent = AIAgent.__new__(AIAgent)
    agent.provider, agent.model, agent.session_id, agent.tools = "nous", "m", "s1", []
    agent.valid_tool_names = {"read_file"}
    agent.log_prefix = ""
    agent._invalid_tool_retries = agent._invalid_json_retries = 0
    return agent


def _mint(tool_calls):
    """Run the real mint path and return the rows that would be stored and replayed."""
    from agent.turn_tool_validation import validate_tool_calls

    agent = _agent()
    verdict = validate_tool_calls(
        agent, SimpleNamespace(content="", tool_calls=tool_calls), "tool_calls",
        messages=[], conversation_history=[], api_call_count=1, effective_task_id="t1",
    )
    assert verdict.action == "ok"
    stored = [_assistant_tool_call_dict(agent, tc, i) for i, tc in enumerate(tool_calls)]
    # Tool results take their id from the same object, so call and result agree.
    assert [row["id"] for row in stored] == [AIAgent._get_tool_call_id_static(tc) for tc in tool_calls]
    return stored


@pytest.fixture(autouse=True)
def _no_metrics(monkeypatch):
    import hermes_cli.observability.shared_metrics_model as metrics

    monkeypatch.setattr(metrics, "record_tool_call_quality", lambda *a, **k: None)


def test_parallel_provider_ids_are_rewritten_stably():
    raw = ["chatcmpl-tool-aaa", "chatcmpl-tool-bbb"]
    ids = [row["id"] for row in _mint([_tool_call(i, n) for n, i in enumerate(raw)])]

    assert all(i.startswith("call_") for i in ids)
    assert len(set(ids)) == 2
    # Byte-stable across a re-mint of the same turn.
    assert [row["id"] for row in _mint([_tool_call(i, n) for n, i in enumerate(raw)])] == ids


@pytest.mark.parametrize("raw", [
    ["chatcmpl-tool-aaa"],
    ["chatcmpl-tool-aaa", "call_1"],
    ["call_0", "call_1"],
])
def test_single_mixed_and_ordinary_batches_are_untouched(raw):
    assert [row["id"] for row in _mint([_tool_call(i, n) for n, i in enumerate(raw)])] == raw


@pytest.mark.parametrize("keys", [("a", "b"), ("a", "a")], ids=["distinct", "duplicate"])
def test_response_item_half_survives_when_only_id_is_composite(keys):
    calls = [
        _tool_call(f"chatcmpl-tool-{k}|fc_original_{n}", n, call_id=f"chatcmpl-tool-{k}")
        for n, k in enumerate(keys)
    ]
    stored = _mint(calls)

    assert [row["response_item_id"] for row in stored] == ["fc_original_0", "fc_original_1"]
    assert all(row["id"].startswith("call_") for row in stored)
    assert len({row["id"] for row in stored}) == 2


def test_unencodable_provider_ids_do_not_crash_and_stay_distinct():
    raw = ["chatcmpl-tool-\ud800", "chatcmpl-tool-?"]
    ids = [row["id"] for row in _mint([_tool_call(i, n) for n, i in enumerate(raw)])]

    assert all(i.startswith("call_") for i in ids)
    assert len(set(ids)) == 2
