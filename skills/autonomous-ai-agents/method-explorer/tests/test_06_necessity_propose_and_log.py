"""Test 06 — necessity is propose-and-log, not a silent prune.

Sim run: the happy path leans on an instrumental sub-goal ("obtain a signed
contract") whose every method tombstones, but the PARENT desire ("legal cover to
start") is reachable another way that BYPASSES the sub-goal (a letter of intent /
purchase order). Dissolution mode is the default propose-and-log: when the
sub-goal is exhausted the loop must RECORD a necessity/dispensability judgement
before climbing — never silently drop the sub-goal — and still reach the intent
via the bypass.

Asserts on receipts: the sub-goal methods tombstone, a necessity/dispensability
judgement is written to journal.jsonl and/or a plan-tree node note, and a
success verdict (or a deliverable) is reached — never on the agent's self-report.

Run:  pytest -m agent -k necessity -v   |   python3 run.py test_06_necessity_propose_and_log
"""
import pytest

from helpers import (
    PLANS,
    SCEN_DIR,
    file_exists,
    is_fail,
    is_succ,
    load_journal,
    read_file,
    run_planner,
    setup_sandbox,
    sim_prompt,
)

SLUG = "test-necessity"
OUT = "/tmp/test-necessity/legal-cover.txt"  # optional deliverable; a sim run may skip it
# Any of these, anywhere in the journal or the plan-tree, proves the judgement
# was logged rather than the subtree silently pruned. Tolerant, case-insensitive.
JUDGEMENT_TERMS = ("necessity", "dispensab", "not required", "propose", "unnecessary")


@pytest.mark.agent
def test_necessity_is_proposed_and_logged():
    setup_sandbox(SLUG)

    prompt = sim_prompt(
        intent="have legal cover in place to start the project Friday",
        slug=SLUG,
        extra=(
            "HARD CONSTRAINT: end with legal cover in place to start Friday.\n"
            "HAPPY PATH (instrumental sub-goal): obtain a signed contract from the "
            "vendor — try the standard contract, then redline the disputed clause, "
            "then their template. Every signed-contract method is blocked.\n"
            "BYPASS: a signed contract is only an ASSUMED method; the parent desire "
            "'legal cover to start' is also satisfiable WITHOUT it via a letter of "
            "intent or a purchase order. Treat 'obtain a signed contract' as a "
            "possibly-unnecessary instrumental sub-goal.\n"
            "Dissolution mode is propose-and-log (default): when the sub-goal's "
            "methods are exhausted, RECORD the dispensability/necessity judgement in "
            "the journal (and the relevant node's one-line note), then climb and reach "
            "legal cover via the bypass."
        ),
    )
    run_planner(prompt, scenario=f"{SCEN_DIR}/dispensable-subgoal.json")

    rows = load_journal(SLUG)
    verdicts = [r.get("verdict") for r in rows]

    assert rows, "no journal written — cannot confirm the loop ran"
    # The sub-goal's methods really died, so the necessity question had to bite.
    assert any(is_fail(v) for v in verdicts), (
        "no tombstone — the signed-contract sub-goal never failed, so necessity "
        "reasoning was never triggered"
    )

    # Receipt 1: the dispensability judgement is on disk (journal OR plan-tree),
    # i.e. propose-and-log — not a silent prune.
    blob = (
        (read_file(f"{PLANS}/{SLUG}/journal.jsonl") or "")
        + "\n"
        + (read_file(f"{PLANS}/{SLUG}/plan-tree.md") or "")
    ).lower()
    assert any(t in blob for t in JUDGEMENT_TERMS), (
        "no necessity/dispensability judgement recorded — the sub-goal looks "
        f"silently dropped (looked for {JUDGEMENT_TERMS!r})"
    )

    # Receipt 2: the intent is still reached via the bypass — a success verdict, or
    # a produced deliverable.
    assert any(is_succ(v) for v in verdicts) or file_exists(OUT), (
        "intent not reached — the bypass path never progressed to success"
    )
