#!/usr/bin/env python3
"""
Deterministic spec-kit decision router.

Same shape as decision-hud-cards' card_type_selector.py and the other
gate skills (design-pattern-gate, test-strategy-gate): a rule fires only
if ALL its preconditions match the answers given. No fuzzy scoring, no
LLM judgment call on ROUTING (the LLM still answers the yes/no
discriminant questions from real evidence, same as every other gate in
this family).

Problem this solves: /speckit-clarify, /speckit-plan, and /speckit-tasks
each surface open questions ([NEEDS CLARIFICATION] markers, tech-stack
choices, task-priority calls). Left to free reasoning, an agent either
silently assumes an answer (hides a real decision) or escalates
everything through a flat text `clarify()` (decision fatigue, and loses
richer card shapes Decision HUD already supports). This gate makes the
escalation-target choice itself auditable.

Three possible outcomes, mirroring ears-sensibility-gate's split:
  - "resolve_silently"   -- evidence-answerable from the repo itself; cite
                            the evidence, do not escalate at all.
  - "decision_hud"        -- genuine owner-policy/taste call, and it's fine
                            for it to wait (spec work is not mid-turn
                            blocking). Hand off to card-type-gate
                            (card_type_selector.py) for the actual shape;
                            never default to mcq_context.
  - "clarify_now"        -- must resolve in THIS turn to keep moving
                            (e.g. blocks every subsequent spec-kit step).
                            Use the `clarify` tool directly.

Two entry points, same contract as the other gates:
  - interactive(): human CLI use.
  - non-interactive: `python3 spec_decision_gate.py --answers a.json`
    -> JSON verdict. This is the mode an agent should call.
"""

import argparse
import json
import sys
from dataclasses import dataclass


@dataclass
class Rule:
    outcome: str
    requires: dict
    rationale: str


QUESTIONS = {
    "stage": (
        "Which spec-kit step surfaced this open question? "
        "(clarify / plan / tasks / constitution)"
    ),
    "is_evidence_answerable": (
        "Can this be answered by reading the repo itself (existing code, "
        "prior spec.md/plan.md decisions, an AGENTS.md convention) rather "
        "than by a human preference or unstated business constraint?"
    ),
    "blocks_every_next_step": (
        "Does leaving this open block EVERY subsequent spec-kit command "
        "from producing anything useful (not just one task or one section)?"
    ),
    "is_reversible_default_available": (
        "Is there a safe, clearly-reversible default that would let work "
        "continue now, with the real answer only refining it later?"
    ),
}

RULES = [
    Rule(
        "resolve_silently",
        {"is_evidence_answerable": True},
        "Repo evidence already answers this; cite it in spec.md/plan.md "
        "and move on without escalating.",
    ),
    Rule(
        "clarify_now",
        {"is_evidence_answerable": False, "blocks_every_next_step": True},
        "No repo evidence, and leaving it open stalls every later step -- "
        "use `clarify` so the current turn doesn't produce dead-end work.",
    ),
    Rule(
        "decision_hud",
        {
            "is_evidence_answerable": False,
            "blocks_every_next_step": False,
        },
        "Genuine owner-policy/taste call that can wait for async "
        "resolution -- push to Decision HUD. Do NOT default the "
        "card_type to mcq_context: run card-type-gate's "
        "card_type_selector.py on this same question to pick the real "
        "shape (scalar_slider for a tradeoff range, spider_compare for "
        "multi-axis tech-stack choices, pairwise_duel/sequence_order for "
        "task prioritization, etc.), and only fall back to plain "
        "MCQ+free-text when that gate legitimately returns no_match.",
    ),
]


def evaluate(answers: dict) -> dict:
    missing = [q for q in ("is_evidence_answerable",) if q not in answers]
    if missing:
        return {
            "status": "incomplete",
            "open_questions": [QUESTIONS[q] for q in missing],
        }

    matches = []
    for rule in RULES:
        if all(answers.get(k) == v for k, v in rule.requires.items()):
            matches.append(rule)

    if not matches:
        # blocks_every_next_step unanswered but evidence-answerable is False
        if "blocks_every_next_step" not in answers:
            return {
                "status": "incomplete",
                "open_questions": [QUESTIONS["blocks_every_next_step"]],
            }
        return {
            "status": "no_match",
            "matches": [],
            "note": "Falls through to decision_hud with plain MCQ+free-text "
            "as the safety net -- this is a valid, correct outcome, not a "
            "bug to route around.",
        }

    return {
        "status": "resolved",
        "matches": [[r.outcome, r.rationale] for r in matches],
    }


def interactive():
    print("Spec Decision Gate -- answer each question (y/n):\n")
    answers = {}
    answers["stage"] = input(f"{QUESTIONS['stage']}\n> ").strip()
    for key in ("is_evidence_answerable", "blocks_every_next_step"):
        raw = input(f"{QUESTIONS[key]} (y/n)\n> ").strip().lower()
        answers[key] = raw in ("y", "yes", "true")
    result = evaluate(answers)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questions", action="store_true", help="Print the question bank as JSON."
    )
    parser.add_argument(
        "--answers", type=str, help="Path to a JSON file of answers."
    )
    args = parser.parse_args()

    if args.questions:
        print(json.dumps(QUESTIONS, indent=2))
        return

    if args.answers:
        with open(args.answers) as f:
            answers = json.load(f)
        print(json.dumps(evaluate(answers), indent=2))
        return

    interactive()


if __name__ == "__main__":
    sys.exit(main())
