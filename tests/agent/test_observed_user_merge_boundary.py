"""#154180: a passive (observed) group row must never weld with an addressed user turn.

Restore-time alternation repair (``_merge_consecutive_users``) joined adjacent ``user`` rows
regardless of ``observed``; the merged row kept ``observed=True``, so gateway replay folded the
addressed text into the passive observation string and the following tool-call turn was absorbed
into the previous answer.
"""
from __future__ import annotations


def test_gateway_replay_keeps_the_addressed_turn_after_the_restore_repair():
    from agent.agent_runtime_helpers import repair_message_sequence
    from gateway.run import _build_gateway_agent_history

    stored = [
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
    repair_message_sequence(None, stored)
    history, observed = _build_gateway_agent_history(
        stored, channel_prompt="observed Telegram group context", inject_timestamps=False,
    )
    repair_message_sequence(None, history)

    assert observed == "Passive observation"
    assert [row.get("message_uid") for row in history] == [
        "previous", "answer", "addressed", "assistant-call", "result", "next"]


def test_rows_of_the_same_class_still_merge():
    from agent.agent_runtime_helpers import _merge_consecutive_users

    tail = {"role": "assistant", "content": "ok"}
    merged, repairs = _merge_consecutive_users([
        {"role": "user", "content": "one", "observed": True},
        {"role": "user", "content": "two", "observed": True}, dict(tail)])
    assert repairs == 1 and merged[0]["content"] == "one\n\ntwo" and merged[0]["observed"] is True
    # A row written before the flag existed has no key: it is addressed and merges with addressed rows.
    merged, repairs = _merge_consecutive_users([
        {"role": "user", "content": "three"}, {"role": "user", "content": "four", "observed": False},
        dict(tail)])
    assert repairs == 1 and merged[0]["content"] == "three\n\nfour"
