"""Behavior of the conflicting tool-call-ID guard."""

import json
from types import SimpleNamespace

from agent.tool_call_identity import _conflicting_calls, reject_conflicting_tool_call_ids


def _call(call_id, name="read_file", args=None):
    return SimpleNamespace(
        id=call_id,
        call_id=None,
        function=SimpleNamespace(name=name, arguments=json.dumps(args or {})),
    )


class _FakeAgent:
    def __init__(self):
        self.persisted = 0
        self.cleaned = 0

    def _build_assistant_message(self, message, finish_reason):
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": c.id, "function": {"name": c.function.name, "arguments": c.function.arguments}}
                for c in message.tool_calls
            ],
        }

    def _cleanup_task_resources(self, task_id):
        self.cleaned += 1

    def _persist_session(self, messages, history):
        self.persisted += 1


def _reject(agent, calls, messages=None):
    return reject_conflicting_tool_call_ids(
        agent,
        SimpleNamespace(tool_calls=calls, content=""),
        "tool_calls",
        messages=[] if messages is None else messages,
        conversation_history=[],
        api_call_count=1,
        effective_task_id="t",
    )


def test_same_id_same_payload_is_not_a_conflict():
    unique, conflicts = _conflicting_calls([_call("a", args={"p": 1}), _call("a", args={"p": 1})])
    assert conflicts == set()
    assert len(unique) == 1


def test_same_id_different_payload_conflicts():
    _, conflicts = _conflicting_calls([_call("a", args={"p": 1}), _call("a", args={})])
    assert conflicts == {"a"}


def test_argument_key_order_does_not_create_a_conflict():
    first = _call("a", args={"x": 1, "y": 2})
    second = _call("a", args={"y": 2, "x": 1})
    assert _conflicting_calls([first, second])[1] == set()


def test_distinct_response_items_sharing_a_pairing_half_are_not_a_conflict():
    """#130363: provider-minted parallel ids share the pairing half; the response-item
    half disambiguates them, so they must reach ``uniquify_tool_call_ids``, not be
    rejected as one identity."""
    calls = [
        SimpleNamespace(
            id=f"chatcmpl-tool-a|fc_{n}", call_id="chatcmpl-tool-a",
            function=SimpleNamespace(name="read_file", arguments=json.dumps({"path": str(n)})),
        )
        for n in range(2)
    ]
    unique, conflicts = _conflicting_calls(calls)
    assert conflicts == set()
    assert len(unique) == 2


def test_clean_batch_passes_and_resets_retry_counter():
    agent = _FakeAgent()
    agent._invalid_tool_identity_retries = 2
    rejected, result = _reject(agent, [_call("a"), _call("b")])
    assert (rejected, result) == (False, None)
    assert agent._invalid_tool_identity_retries == 0


def test_conflicting_batch_is_answered_not_executed_and_retried():
    agent = _FakeAgent()
    messages = []
    rejected, result = _reject(agent, [_call("a", args={"p": 1}), _call("a")], messages)
    assert rejected is True and result is None
    assert [m["role"] for m in messages] == ["assistant", "tool"]
    assert messages[1]["tool_call_id"] == "a"
    assert "Conflicting tool call IDs" in messages[1]["content"]


def test_third_consecutive_conflict_ends_the_turn_without_executing():
    agent = _FakeAgent()
    messages = []
    results = [_reject(agent, [_call("a", args={"p": 1}), _call("a")], messages) for _ in range(3)]
    assert [r[0] for r in results] == [True, True, True]
    assert results[0][1] is None and results[1][1] is None
    final = results[2][1]
    assert final["completed"] is False and final["partial"] is True
    assert "conflicting tool call IDs" in final["final_response"]
    assert agent.persisted == 1 and agent.cleaned == 1
