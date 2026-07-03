"""Test 17 — sim-mode Locality label: host-down is LOCAL-transient, not STRUCTURAL.

The observed mislabel (3/3 in demos pre-fix): in Simulation Mode a retry always
re-tombstones by scenario rule, so the model read "retry can't succeed" as
standing-ness and labeled a network-down STRUCTURAL. The skill now says: in sim,
answer the time-dependence test from the declared reason's SEMANTICS, not from
retry behavior.

Assertion is NEGATIVE (robust to wording noise): "structural" must be absent from
the network-down ✝ receipts and those cycles' journal fields. The positive wording
(LOCAL/transient) is logged, not asserted — that slot is the stochastic part.

Run:  python3 run.py test_17_sim_locality_label
"""
import pytest

from helpers import PLANS, deploy_scenario, is_fail, read_file, run_until_journal, setup_sandbox
from scenario_builder import Method, build_prompt, build_scenario, canonical_backtrack_methods

SLUG = "test-sim-locality"
EXH_SLUG = "test-sim-locality-exh"
TRACE_SLUGS = [SLUG, EXH_SLUG]  # for `run.py --show`
METHODS = canonical_backtrack_methods()  # alfa/bravo down (network), charlie->delta succeed
NETWORK = ("alfa", "bravo")


@pytest.mark.agent
def test_network_down_not_labeled_structural_in_sim():
    setup_sandbox(SLUG)
    scen = build_scenario("obtain the data", METHODS,
                          notes="locality-label regression: network-down is LOCAL-transient")
    cont = deploy_scenario("sim-locality.json", scen)
    prompt = build_prompt(
        "obtain the data", METHODS, SLUG,
        meanings={"alfa": "primary network source", "bravo": "mirror network source",
                  "charlie": "local cache", "delta": "verify"},
        extra=("Diagnose each failure per the skill's Key Question #3 (Locality) and put "
               "the classification in the ✝ node's receipt."))
    rows, _ = run_until_journal(prompt, SLUG, scenario=cont)
    assert rows, "persistent no-op (empty journal after retries)"

    tree = read_file(f"{PLANS}/{SLUG}/plan-tree.md")
    # The network-down cycles in the journal...
    net_rows = [r for r in rows
                if is_fail(r.get("verdict")) and
                any(t in str(r.get("chosen", "")).lower() for t in NETWORK)]
    assert net_rows, "no network-down failure cycles found — scenario didn't drive the path"
    # ...and the network ✝ node lines in the tree.
    net_lines = [ln for ln in tree.splitlines()
                 if "✝" in ln and any(t in ln.lower() for t in NETWORK)]

    offenders = []
    for r in net_rows:
        blob = (str(r.get("evidence", "")) + " " + str(r.get("next", ""))).lower()
        if "structural" in blob:
            offenders.append(f"journal {r.get('node')}: {blob[:100]}")
    for ln in net_lines:
        if "structural" in ln.lower():
            offenders.append(f"tree: {ln.strip()[:100]}")
    assert not offenders, (
        "network-down mislabeled STRUCTURAL in sim (should be LOCAL-transient per the "
        "reason's semantics):\n  " + "\n  ".join(offenders))

    # Positive wording: log, don't assert (the label slot's wording is stochastic).
    positive = [ln.strip()[:110] for ln in net_lines
                if "local" in ln.lower() or "transient" in ln.lower()]
    print(f"  [info] network ✝ receipts with LOCAL/transient wording: "
          f"{len(positive)}/{len(net_lines)}"
          + (f" e.g. {positive[0]}" if positive else ""))


@pytest.mark.agent
def test_sole_source_down_not_structural_at_exhaustion():
    """The second face of the mislabel: with ZERO siblings left, the model conflates
    'nothing left to try' (EXHAUSTION-STOP's job) with 'the blocker is standing'
    (Locality's job) — a transient outage stays LOCAL-transient even at exhaustion."""
    setup_sandbox(EXH_SLUG)
    methods = [Method("alfa", "tombstone",
                      reason="the only source, and it is unreachable right now")]
    scen = build_scenario("obtain the data from the only source", methods,
                          notes="locality-at-exhaustion: sole source down is LOCAL-transient")
    cont = deploy_scenario("sim-locality-exh.json", scen)
    prompt = build_prompt(
        "obtain the data from the only source", methods, EXH_SLUG,
        meanings={"alfa": "the sole source"},
        extra=("There are no alternative methods. Diagnose the failure per Key Question "
               "#3 (Locality) and put the classification in the ✝ node's receipt; then "
               "terminate per the stop conditions."))
    rows, _ = run_until_journal(prompt, EXH_SLUG, scenario=cont)
    assert rows, "persistent no-op (empty journal after retries)"

    tree = read_file(f"{PLANS}/{EXH_SLUG}/plan-tree.md")
    dead = [ln for ln in tree.splitlines() if "✝" in ln and "alfa" in ln.lower()]
    assert dead, "no ✝ alfa node in the plan-tree"
    offenders = [ln.strip()[:110] for ln in dead if "structural" in ln.lower()]
    assert not offenders, (
        "sole-source outage mislabeled STRUCTURAL at exhaustion (ran-out-of-siblings is "
        "the terminal state's fact, not the blocker's): " + "; ".join(offenders))
    positive = [ln for ln in dead if "local" in ln.lower() or "transient" in ln.lower()]
    print(f"  [info] sole-source ✝ receipt LOCAL/transient wording: {len(positive)}/{len(dead)}")
