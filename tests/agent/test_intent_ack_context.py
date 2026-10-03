"""Contextual clarification recovery preserves current intent and durable history."""

from copy import deepcopy
import json

import pytest

from tests.agent import test_clarification_ack_stop as base
from tests.agent.test_clarification_ack_stop import harness  # noqa: F401


EDIT = "Please implement the panel highlight in the repository."
PROCEED = "That approach sounds reasonable; please proceed."
HISTORY = [
    {"role": "user", "content": EDIT},
    {"role": "assistant", "content": "I can implement the panel highlight in the repository. Shall I proceed?"},
]
REVOKED = HISTORY + [
    {"role": "user", "content": "Do not implement it. Explain the options instead."},
    {"role": "assistant", "content": "I can explain the options without edits. Shall I proceed?"},
]
COMPLETE = [
    {"role": "user", "content": EDIT},
    {"role": "assistant", "content": "The requested task is complete. No changes remain."},
]


def _run(harness, request, answer, *, history=None, repeats=1, execute_before=False):
    agent = harness.agent
    agent.clarify_callback = lambda *_args: answer
    responses = [base._response(name="clarify", arguments={"questions": [{"question": base.QUESTION}]})]
    read = base._response(name="read_file", arguments={"path": str(harness.fixture)}, call_id="work")
    if execute_before:
        responses.append(read)
    responses += [base._response(base.ACK)] * repeats
    # Include a real harmless tool even for negative cases: a mistaken recovery
    # becomes an observable duplicate/extraneous read, not a mock exhaustion.
    responses += [read, base._response("The supplied fixture was read.")]
    responses += [base._response("Unexpected extra request.")]
    script = iter(responses)
    sent = []

    def provider(kwargs):
        sent.append(deepcopy(kwargs))
        return next(script)

    agent._interruptible_api_call = provider
    result = agent.run_conversation(request, conversation_history=deepcopy(history or []))
    durable = harness.db.get_messages_as_conversation(agent.session_id)
    assert result.get("error") is None
    assert result["api_calls"] == len(sent)
    assert any(m.get("tool_call_id") == "clarification" for m in durable)
    for old, new in zip(sent, sent[1:]):
        assert old["instructions"] == new["instructions"]
        assert old["tools"] == new["tools"]
        assert new["input"][:len(old["input"])] == old["input"]
    for rows in [result["messages"], durable]:
        pending = set()
        for i, row in enumerate(rows):
            if i:
                assert row["role"] != rows[i - 1]["role"] or row["role"] == "tool"
            if row["role"] == "assistant":
                assert not pending
                pending.update(c["id"] for c in row.get("tool_calls", []))
            elif row["role"] == "tool":
                assert row["tool_call_id"] in pending
                pending.remove(row["tool_call_id"])
        assert not pending
        assert rows[-1]["role"] == "assistant"
        assert rows[-1]["content"] == result["final_response"]
        # Supplied prior history is already persisted by its owning session;
        # this fresh DB contains only the new turn, not a duplicate import.
        prior_users = [] if rows is durable else [
            m["content"] for m in (history or []) if m["role"] == "user"
        ]
        assert [m["content"] for m in rows if m["role"] == "user"] == prior_users + [request]
    return result, durable, sent


@pytest.mark.parametrize("user_text,answer,history,should_read", [
    pytest.param(EDIT, base.CORRECTION, None, True, id="explicit-edit-control"),
    pytest.param(base.USER, base.CORRECTION, None, True, id="proceed-with-implementing"),
    pytest.param(PROCEED, base.CORRECTION, HISTORY, True, id="conversational-proceed-live-task"),
    pytest.param(EDIT, "No, route highlighting, not viewport tracking.", None, True, id="negative-correction-not-decline"),
    pytest.param(EDIT, "No, do not implement anything.", None, False, id="actual-decline"),
    pytest.param(EDIT, "No.", None, False, id="bare-negative"),
    pytest.param(EDIT, "No, do not implement it, not even the highlight.", None, False, id="contrast-with-decline"),
    pytest.param(EDIT, "No, implement nothing, not any changes.", None, False, id="contrast-with-no-action"),
    pytest.param(EDIT, "No, route highlighting, not now.", None, False, id="contrast-with-deferral"),
    pytest.param(EDIT, "Wait for my approval.", None, False, id="approval-wait"),
    pytest.param(PROCEED, base.CORRECTION, REVOKED, False, id="old-authorization-revoked"),
    pytest.param(PROCEED, base.CORRECTION, COMPLETE, False, id="completed-history-not-new-task"),
])
def test_contextual_authorization_is_not_first_word_matching(harness, user_text, answer, history, should_read):
    result, durable, sent = _run(harness, user_text, answer, history=history)
    reads = [m for m in durable if m.get("tool_call_id") == "work"]
    assert bool(reads) is should_read, (result["completed"], len(sent), result["final_response"])
    if should_read:
        assert len(reads) == 1
        assert "selection-panel-fixture" in reads[0]["content"]


@pytest.mark.parametrize("execute_before", [False, True], ids=["unchanged-retry-cap", "no-repeated-execution"])
def test_identical_requests_are_bounded_and_close_the_durable_tool_round(harness, execute_before):
    result, durable, sent = _run(
        harness, EDIT, base.CORRECTION,
        repeats=1 if execute_before else 3, execute_before=execute_before,
    )
    assert result["final_response"] == base.ACK
    reads = [m for m in durable if m.get("tool_call_id") == "work"]
    if execute_before:
        assert len(sent) == 3
        assert len(reads) == 1
    else:
        assert len(sent) == 4
        assert sent[1] == sent[2] == sent[3]
        assert not reads
        assert durable[-2]["role"] == "tool"
        assert durable[-2]["tool_call_id"] == "clarification"
