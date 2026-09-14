"""Unit coverage for the shared pre-persist gate (agent/pre_persist_gate.py).

This is the fix for the persistence-ordering finding in the pre-delivery gate review: an
unvalidated assistant answer could be durably flushed to SessionDB
(agent._persist_session, called from agent/turn_finalizer.py's finalize_turn) with no way
for the gateway's evaluator verdict to withhold it — gateway/run_turn.py's
_run_agent_apply_pre_delivery_gate runs strictly after persistence, and (by explicit
design) never mutates delivery for "shadow" mode. "strict" mode closes that gap by wiring
agent._pre_persist_gate; this module is where that verdict is applied before the durable
write.
"""
from __future__ import annotations

from types import SimpleNamespace

from agent.pre_persist_gate import PRE_PERSIST_WITHHELD_TEXT, apply_pre_persist_gate


def _agent(gate=None):
    return SimpleNamespace(_pre_persist_gate=gate)


def test_no_gate_is_a_pure_noop():
    agent = _agent(gate=None)
    target = {"content": "hello"}
    result = apply_pre_persist_gate(agent, "hello", target_msg=target)
    assert result == "hello"
    assert target["content"] == "hello"  # untouched


def test_blank_or_non_string_response_is_a_noop_even_with_a_gate():
    calls = []
    agent = _agent(gate=lambda final_text: calls.append(final_text) or {"decision": "passed"})
    assert apply_pre_persist_gate(agent, "", target_msg={"content": ""}) == ""
    assert apply_pre_persist_gate(agent, None, target_msg={"content": None}) is None
    assert calls == []  # gate never invoked for empty/non-string text


def test_passed_decision_leaves_content_untouched():
    agent = _agent(gate=lambda final_text: {"decision": "passed"})
    target = {"content": "answer"}
    result = apply_pre_persist_gate(agent, "answer", target_msg=target)
    assert result == "answer"
    assert target["content"] == "answer"


def test_blocked_decision_replaces_content_with_safe_placeholder_not_raw_text():
    agent = _agent(gate=lambda final_text: {"decision": "blocked"})
    target = {"content": "unsafe raw answer"}
    result = apply_pre_persist_gate(agent, "unsafe raw answer", target_msg=target)
    assert result == PRE_PERSIST_WITHHELD_TEXT
    assert target["content"] == PRE_PERSIST_WITHHELD_TEXT
    assert "unsafe raw answer" not in target["content"]


def test_inconclusive_decision_is_also_withheld_fail_closed():
    """gateway/evaluator_shadow.py returns "inconclusive" for a returncode mismatch,
    malformed JSON, or evaluator crash — all of those must withhold, not deliver."""
    agent = _agent(gate=lambda final_text: {"decision": "inconclusive"})
    target = {"content": "unsafe raw answer"}
    result = apply_pre_persist_gate(agent, "unsafe raw answer", target_msg=target)
    assert result == PRE_PERSIST_WITHHELD_TEXT
    assert target["content"] == PRE_PERSIST_WITHHELD_TEXT


def test_gate_exception_is_a_safe_stop_not_a_passthrough():
    def exploding_gate(final_text):
        raise RuntimeError("evaluator subprocess crashed")

    agent = _agent(gate=exploding_gate)
    target = {"content": "unsafe raw answer"}
    result = apply_pre_persist_gate(agent, "unsafe raw answer", target_msg=target)
    assert result == PRE_PERSIST_WITHHELD_TEXT
    assert target["content"] == PRE_PERSIST_WITHHELD_TEXT


def test_malformed_gate_return_value_is_also_a_safe_stop():
    """A gate returning something other than a dict with "decision" must not be
    mistaken for a passing verdict."""
    agent = _agent(gate=lambda final_text: None)
    result = apply_pre_persist_gate(agent, "unsafe raw answer", target_msg={"content": "unsafe raw answer"})
    assert result == PRE_PERSIST_WITHHELD_TEXT


def test_final_text_is_forwarded_to_the_gate():
    seen = []
    agent = _agent(gate=lambda final_text: seen.append(final_text) or {"decision": "passed"})
    apply_pre_persist_gate(agent, "hello", target_msg={"content": "hello"})
    assert seen == ["hello"]


def test_works_without_a_target_msg():
    """finalize_turn passes target_msg=None when messages is empty; must not raise."""
    agent = _agent(gate=lambda final_text: {"decision": "blocked"})
    result = apply_pre_persist_gate(agent, "unsafe", target_msg=None)
    assert result == PRE_PERSIST_WITHHELD_TEXT
