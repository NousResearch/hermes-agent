"""Tier 2 — property/metamorphic tests over BUILDER-GENERATED scenarios.

Two kinds of assertion:
  - UNIVERSAL INVARIANTS that must hold on every run (complete decision records, no
    re-expansion of a tombstone), checked via helpers.assert_*.
  - METAMORPHIC RELATIONS between two scenarios' terminal states (no per-case oracle):
    reachability-flip (exhaustion→success) and add-dead-sibling (terminal unchanged).
    These are differential — both runs share the model's noise — so they're unusually
    robust to non-determinism.

Each agent run goes through run_until_journal (auto-retries the transient no-op). These
are the GATE 2 sample (small on purpose); scale counts up once Gate 2 is green.

Run:  python3 run.py -k properties
"""
import pytest

from gen_scenarios import (add_dead_sibling_pair, paraphrase_variants,
                           reachability_flip_pair)
from helpers import (PLANS, assert_no_reexpand, assert_record_complete, deploy_scenario,
                     read_file, run_until_journal, setup_sandbox, terminal_state)


def _run_case(c):
    """Deploy + run one generated case; return (rows, terminal_state)."""
    cont = deploy_scenario(c["slug"] + ".json", c["scenario"])
    setup_sandbox(c["slug"])
    rows, _ = run_until_journal(c["prompt"], c["slug"], scenario=cont)
    assert rows, f"{c['slug']}: persistent no-op (empty journal after retries)"
    pt = read_file(f"{PLANS}/{c['slug']}/plan-tree.md") or ""
    return rows, terminal_state(rows, pt)


@pytest.mark.agent
def test_invariants_hold_on_generated_scenarios():
    """Universal invariants hold on a sample of generated (paraphrase) scenarios, and
    paraphrasing the intent does not change the terminal state."""
    for c in paraphrase_variants(2):  # Gate 2 sample size
        rows, term = _run_case(c)
        assert_record_complete(rows)
        assert_no_reexpand(rows)
        assert term == c["expect_terminal"], (
            f"{c['slug']}: paraphrase changed terminal {term!r} != {c['expect_terminal']!r}"
        )


@pytest.mark.agent
def test_reachability_flip_relation():
    """Flipping one fallback dead→reachable flips the outcome exhaustion→success."""
    pair = reachability_flip_pair()
    _, term_a = _run_case(pair["a"])
    _, term_b = _run_case(pair["b"])
    assert (term_a, term_b) == pair["expect"], (
        f"reachability-flip relation broke: got ({term_a}, {term_b}), want {pair['expect']}"
    )


@pytest.mark.agent
def test_add_dead_sibling_invariance():
    """Adding one dead sibling (still < K=5) leaves the terminal state unchanged."""
    pair = add_dead_sibling_pair()
    _, term_a = _run_case(pair["a"])
    _, term_b = _run_case(pair["b"])
    assert term_a == term_b == pair["expect"][0], (
        f"add-dead-sibling changed terminal: a={term_a}, b={term_b}, want {pair['expect'][0]}"
    )
