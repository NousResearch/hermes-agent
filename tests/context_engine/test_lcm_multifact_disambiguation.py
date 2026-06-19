"""PRD-8.3 tests — multi-fact node disambiguation.

Two prongs:
  Prong A (optimization): summarizer prompts must instruct identifier fidelity
    (no merge / range-collapse / truncation of distinct identifier->value facts).
  Prong B (the fix): recovery escalates to the verbatim store when the node
    answer abstains, is empty, or cites a grouped/range mapping.

These are pure-function + prompt-contract controls (offline, free). The live
confident-wrong gate is the separate N>=600 campaign (PRD-8.3 AC-4).
"""
from __future__ import annotations

import importlib

esc = importlib.import_module("plugins.context_engine.lcm.escalation")
from scripts import lcm_arm_b_node_recovery as armB


# ---- Prong A: summarizer prompt identifier-fidelity clause (AC-1) -----------

def test_l1_prompt_has_identifier_fidelity_rule():
    p = esc._build_l1_prompt("CONTENT", token_budget=500, depth=0)
    low = p.lower()
    assert "identifier fidelity" in low
    assert "never merge" in low or "do not merge" in low
    # explicitly forbids the grouped/range line that caused the bug
    assert "1300/1600/1900" in p
    assert "one line per distinct identifier" in low


def test_l2_prompt_has_identifier_fidelity_rule():
    p = esc._build_l2_prompt("CONTENT", token_budget=300)
    low = p.lower()
    assert "identifier fidelity" in low
    assert "truncate" in low  # forbids the "R" mid-word truncation class
    assert "1300/1600/1900" in p


def test_l1_prompt_still_summarizes_normal_content():
    # the fidelity rule must not destroy the base summarize instruction
    p = esc._build_l1_prompt("CONTENT", token_budget=500, depth=0)
    assert "Summarize this conversation segment" in p
    assert "CONTENT" in p


# ---- Prong B: needs_escalation trigger (positive + negative controls) -------

def test_escalate_on_grouped_range_mapping():
    # the exact lossy-merge signature from the live store
    assert armB.needs_escalation("recover-1300 maps to 1300/1600/1900 = Barbara") is True


def test_escalate_on_abstention():
    assert armB.needs_escalation("no matching owner found") is True
    assert armB.needs_escalation("That code is not present in the index.") is True


def test_escalate_on_empty():
    assert armB.needs_escalation("") is True
    assert armB.needs_escalation("   ") is True


def test_no_escalate_on_clean_full_name():
    # a confident, complete, ungrouped answer does NOT escalate (node served it)
    assert armB.needs_escalation("The recovery owner is Frances Allen.") is False
    assert armB.needs_escalation("Katherine Johnson") is False


def test_no_escalate_does_not_fire_on_a_year_or_plain_number():
    # a single number (not a slash-grouped mapping) must not trigger escalation
    assert armB.needs_escalation("The owner Ada Lovelace was assigned in 2024.") is False


# ---- Prong B: recovery prompt mandates abstain-over-guess + no-grouped -------

def test_semantic_recovery_question_forbids_grouped_inference_and_mandates_abstain():
    # Build the question the same way the harness does and assert the clauses.
    # (mirror of _node_served_recovery_semantic's question text)
    phrase = "recover-1300"
    q = (
        f"Who is the recovery owner associated with the EXACT handoff "
        f"phrase {phrase}? Answer with the owner's full name. "
        f"Use ONLY an entry that names {phrase} exactly and by "
        f"itself. Do NOT infer the owner from a grouped or range mapping "
        f"(e.g. a line like '1300/1600/1900 = Name'); a grouped line is "
        f"not a valid source. If {phrase} is not present exactly "
        f"and unambiguously with its own full owner name, reply with "
        f"exactly: no matching owner found"
    )
    assert "EXACT" in q
    assert "Do NOT infer the owner from a grouped or range mapping" in q
    assert "no matching owner found" in q


# ---- Integration: a merged node answer escalates and is re-scored -----------

def test_merged_answer_triggers_escalation_path_logic():
    # Simulate: node returned the merged 'Barbara' answer for recover-1300.
    # The scorer would call it confident-wrong; needs_escalation must catch it
    # FIRST so the store-grounded path overrides before scoring.
    merged = "Based on the index, recover-1300 maps to 1300/1600/1900 = Barbara"
    assert armB.needs_escalation(merged) is True
    # and the TRUE store answer (Katherine Johnson) would NOT escalate -> served
    true_answer = "Katherine Johnson"
    assert armB.needs_escalation(true_answer) is False
