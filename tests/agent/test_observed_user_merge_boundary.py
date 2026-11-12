"""#154180: a passive (observed) group row must never weld with an addressed user turn.

The restore-time alternation repair (``get_messages_as_conversation(repair_alternation=True)`` →
``_merge_consecutive_users``) joined adjacent ``user`` rows without looking at their ``observed``
flag. The merged row kept ``observed=True``, so gateway replay then folded the ADDRESSED text into
the passive observation string: the addressed turn lost its own UID and position, and the
following tool-call turn was absorbed into the answer that preceded it.

The repair's own contract (``gateway.run._build_gateway_agent_history``) is "Observed context stays
out of ``conversation_history`` so consecutive-user repair can't merge it in", so the boundary has
to survive the merge pass for the extraction to see it.
"""
from __future__ import annotations


def _merge(rows):
    from agent.agent_runtime_helpers import _merge_consecutive_users

    return _merge_consecutive_users(rows)


def _stored_transcript() -> list[dict]:
    return [
        {"role": "user", "content": "Previous addressed turn", "message_uid": "previous"},
        {"role": "assistant", "content": "Previous answer", "message_uid": "answer"},
        {"role": "user", "content": "Passive observation", "observed": True, "message_uid": "observed"},
        {"role": "user", "content": "Addressed tool question", "message_uid": "addressed"},
        {"role": "assistant", "content": None, "message_uid": "assistant-call", "tool_calls": [
            {"id": "call-demo", "type": "function", "function": {"name": "lookup", "arguments": "{}"}},
        ]},
        {"role": "tool", "content": "Demo result", "tool_call_id": "call-demo", "message_uid": "result"},
        {"role": "assistant", "content": "Next answer", "message_uid": "next"},
    ]


def test_merge_does_not_weld_an_observed_row_with_an_addressed_turn():
    observed = {"role": "user", "content": "Passive observation", "observed": True, "message_uid": "observed"}
    addressed = {"role": "user", "content": "Addressed tool question", "message_uid": "addressed"}

    merged, repairs = _merge([observed, addressed, {"role": "assistant", "content": "ok"}])

    assert repairs == 0
    assert [row.get("message_uid") for row in merged] == ["observed", "addressed", None]
    assert merged[0]["content"] == "Passive observation"
    assert merged[0]["observed"] is True
    assert merged[1]["content"] == "Addressed tool question"
    assert "observed" not in merged[1]


def test_merge_does_not_weld_an_addressed_turn_with_an_observed_row():
    addressed = {"role": "user", "content": "Addressed tool question", "message_uid": "addressed"}
    observed = {"role": "user", "content": "Later chatter", "observed": True, "message_uid": "observed"}

    merged, repairs = _merge([addressed, observed, {"role": "assistant", "content": "ok"}])

    assert repairs == 0
    assert [row.get("message_uid") for row in merged] == ["addressed", "observed", None]
    assert merged[0].get("observed") in (None, False)
    assert merged[1]["content"] == "Later chatter"


def test_merge_still_joins_rows_of_the_same_class():
    observed_a = {"role": "user", "content": "one", "observed": True}
    observed_b = {"role": "user", "content": "two", "observed": True}
    addressed_a = {"role": "user", "content": "three"}
    addressed_b = {"role": "user", "content": "four"}
    # Rows written before the flag existed carry no key at all; they are addressed, not passive.
    legacy = {"role": "user", "content": "five"}

    merged_observed, observed_repairs = _merge([observed_a, observed_b, {"role": "assistant", "content": "ok"}])
    merged_addressed, addressed_repairs = _merge([addressed_a, addressed_b, legacy, {"role": "assistant", "content": "ok"}])

    assert observed_repairs == 1
    assert merged_observed[0]["content"] == "one\n\ntwo"
    assert merged_observed[0]["observed"] is True
    assert addressed_repairs == 2
    assert merged_addressed[0]["content"] == "three\n\nfour\n\nfive"


def test_gateway_replay_keeps_the_addressed_turn_after_the_restore_repair():
    """The issue's reproducer: an offline fixture, no Telegram client and no model provider."""
    from agent.agent_runtime_helpers import repair_message_sequence
    from gateway.run import _build_gateway_agent_history

    stored = _stored_transcript()
    repair_message_sequence(None, stored)
    history, observed = _build_gateway_agent_history(
        stored, channel_prompt="observed Telegram group context", inject_timestamps=False,
    )
    repair_message_sequence(None, history)

    assert any(row.get("message_uid") == "addressed" for row in history)
    assert "Addressed tool question" not in (observed or "")
    assert observed == "Passive observation"
    previous = next(row for row in history if row.get("message_uid") == "answer")
    assert "assistant-call" not in previous.get("_absorbed_message_uids", [])


def test_replay_order_keeps_the_addressed_turn_between_the_answer_and_the_tool_turn():
    from agent.agent_runtime_helpers import repair_message_sequence
    from gateway.run import _build_gateway_agent_history

    stored = _stored_transcript()
    repair_message_sequence(None, stored)
    history, _observed = _build_gateway_agent_history(
        stored, channel_prompt="observed Telegram group context", inject_timestamps=False,
    )
    repair_message_sequence(None, history)

    uids = [row.get("message_uid") for row in history]
    assert uids == ["previous", "answer", "addressed", "assistant-call", "result", "next"]
