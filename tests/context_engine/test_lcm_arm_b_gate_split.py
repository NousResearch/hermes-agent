"""PRD-8.2 gate-split tests for Arm B.

The original single gate scored "no condensation node ever formed" identically
to "model had a node but failed to recover" and counted transient infra errors
(503/429/timeout) as permanent recovery misses. These tests pin classify_trial
into its five buckets and prove compute_split_verdict gates recovery-correctness
separately from condensation-reliability, with infra errors excluded.
"""
from __future__ import annotations

from scripts import lcm_arm_b_node_recovery as armB


def _t(**kw):
    base = dict(idx=0, sentinel="x", session_id="s", depth1_node_id=1,
                sentinel_in_node=True, node_served_answer="", correct=False,
                confident_wrong=False, leaves=0, condensed=1, notes="")
    base.update(kw)
    return base


# ---- classify_trial: one bucket per outcome ---------------------------------

def test_classify_recovered():
    assert armB.classify_trial(_t(correct=True)) == armB.OUTCOME_RECOVERED


def test_classify_confident_wrong_takes_priority():
    # even with an infra-looking answer, confident_wrong is the hard outcome
    assert armB.classify_trial(_t(confident_wrong=True, node_served_answer="503")) \
        == armB.OUTCOME_CONFIDENT_WRONG


def test_classify_recovery_miss_node_formed_but_failed():
    # node formed, fact preserved, not correct, no infra error = REAL recovery gap
    assert armB.classify_trial(
        _t(correct=False, condensed=1, sentinel_in_node=True,
           node_served_answer="no matching owner found in node")
    ) == armB.OUTCOME_RECOVERY_MISS


def test_classify_no_condensation():
    # no depth>=1 node ever formed = condensation-trigger gap, NOT a recovery gap
    assert armB.classify_trial(
        _t(correct=False, condensed=0, depth1_node_id=None,
           sentinel_in_node=False, node_served_answer="")
    ) == armB.OUTCOME_NO_CONDENSATION


def test_classify_infra_error_503():
    assert armB.classify_trial(
        _t(correct=False, condensed=1,
           node_served_answer="[recovery error: Error code: 503 - {'error': 'no eligible sub'}]")
    ) == armB.OUTCOME_INFRA_ERROR


def test_classify_infra_error_429_and_timeout():
    assert armB.classify_trial(
        _t(correct=False, node_served_answer="[recovery error: 429 rate limited]")
    ) == armB.OUTCOME_INFRA_ERROR
    assert armB.classify_trial(
        _t(correct=False, node_served_answer="[recovery error: connection timed out]")
    ) == armB.OUTCOME_INFRA_ERROR


def test_infra_error_does_not_swallow_a_real_miss():
    # a plain failed recovery whose answer is NOT an infra error must stay a real miss
    assert armB.classify_trial(
        _t(correct=False, condensed=1, node_served_answer="the owner is unclear")
    ) == armB.OUTCOME_RECOVERY_MISS


# ---- compute_split_verdict --------------------------------------------------

def test_split_verdict_passes_recovery_despite_condensation_misses():
    # 166 recovered, 0 real recovery_miss, 14 no_condensation = recovery gate PASS
    trials = (
        [_t(correct=True) for _ in range(166)]
        + [_t(correct=False, condensed=0, depth1_node_id=None, sentinel_in_node=False)
           for _ in range(14)]
    )
    v = armB.compute_split_verdict(trials, gate_eligible=True)
    assert v["buckets"][armB.OUTCOME_RECOVERED] == 166
    assert v["buckets"][armB.OUTCOME_NO_CONDENSATION] == 14
    assert v["recovery_eligible_n"] == 166          # condensation misses excluded
    assert v["recovery_rate"] == 1.0
    assert v["recovery_gate_pass"] is True
    # recovery gate is UNDERPOWERED because eligible < 180, but it is NOT a FAIL
    assert v["recovery_gate"] in ("PASS", "PASS-UNDERPOWERED")
    assert v["recovery_gate"] != "FAIL"


def test_split_verdict_infra_error_excluded_from_recovery_denominator():
    trials = (
        [_t(correct=True) for _ in range(100)]
        + [_t(correct=False, node_served_answer="[recovery error: 503 no eligible sub]")]
    )
    v = armB.compute_split_verdict(trials)
    assert v["infra_error"] == 1
    assert v["recovery_eligible_n"] == 100          # the 503 is NOT in the denominator
    assert v["recovery_rate"] == 1.0


def test_split_verdict_real_recovery_miss_fails_the_gate():
    # genuine recovery gaps (node formed, fact preserved, not recovered) DO fail
    trials = (
        [_t(correct=True) for _ in range(150)]
        + [_t(correct=False, condensed=1, sentinel_in_node=True,
              node_served_answer="no matching") for _ in range(30)]
    )
    v = armB.compute_split_verdict(trials)
    assert v["recovery_miss"] == 30
    assert v["recovery_rate"] < 0.95
    assert v["recovery_gate"] == "FAIL"


def test_split_verdict_confident_wrong_fails_even_if_rate_high():
    trials = [_t(correct=True) for _ in range(199)] + [_t(confident_wrong=True)]
    v = armB.compute_split_verdict(trials, gate_eligible=True)
    assert v["confident_wrong"] == 1
    assert v["recovery_gate"] == "FAIL"
