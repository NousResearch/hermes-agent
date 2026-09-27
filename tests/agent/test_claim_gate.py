"""Regression tests for the outgoing-claim evidence gate (#124657).

Contract: a final message asserting an outcome with NO tool result in the turn
gets a visible unverified caveat; any tool result, absent claim language, a
negated outcome word, or an ``off`` gate leaves the message untouched."""

from __future__ import annotations

from agent.claim_gate import (
    _CLAIM_GATE_CAVEAT,
    _contains_outcome_claim,
    _turn_has_tool_evidence,
    apply_claim_gate,
    claim_gate_mode,
)


def test_outcome_claim_without_tool_evidence_is_annotated():
    response = apply_claim_gate("Deployed the fix, all tests pass now.", [])
    assert "unverified" in response
    assert response.startswith("Deployed the fix")


def test_outcome_claim_with_tool_result_is_untouched():
    messages = [
        {"role": "assistant", "content": "running checks"},
        {"role": "tool", "content": "exit 0; 39 passed"},
    ]
    response = "Deployed the fix, all tests pass now."
    assert apply_claim_gate(response, messages) == response


def test_no_claim_language_is_untouched():
    response = "Here is a summary of the repo layout."
    assert apply_claim_gate(response, []) == response


def test_negated_outcome_word_is_not_a_claim():
    assert not _contains_outcome_claim("This is not verified yet.")
    assert not _contains_outcome_claim("that path was never tested")


def test_tool_message_with_non_string_content_counts_as_evidence():
    messages = [{"role": "tool", "content": [{"type": "text", "text": "ok"}]}]
    assert _turn_has_tool_evidence(messages)


def test_off_mode_disables_the_gate():
    config = {"agent": {"claim_gate": "off"}}
    assert apply_claim_gate("Verified.", [], config) == "Verified."


def test_bool_config_maps_to_mode():
    assert claim_gate_mode({"agent": {"claim_gate": True}}) == "annotate"
    assert claim_gate_mode({"agent": {"claim_gate": False}}) == "off"


def test_unrecognized_value_falls_back_to_annotate():
    assert claim_gate_mode({"agent": {"claim_gate": "yolo"}}) == "annotate"


def test_default_config_resolves_to_annotate():
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    mode = claim_gate_mode({"agent": DEFAULT_CONFIG["agent"]})
    assert mode == "annotate"
    # The shipped default must be a mode apply_claim_gate actually honors.
    assert apply_claim_gate("Verified.", [], {"agent": DEFAULT_CONFIG["agent"]}) == "Verified." + _CLAIM_GATE_CAVEAT
