"""PRD-8.1 probe-isolation amendment tests.

Covers:
  - Arm A make_fixtures(probe_kind=...) isolation (C1 / AC-1).
  - Arm B confident-wrong detector positive AND negative control (C2 / AC-3).

The detector control is the single highest fake-green vector flagged in the
Opus review: a detector that never fires trivially yields confident_wrong==0.
These tests prove it fires on a real fabrication and stays silent on a correct
recovery and on an honest abstention.
"""
from __future__ import annotations

import pytest

from scripts import lcm_live_recovery as armA
from scripts import lcm_arm_b_node_recovery as armB


# ---- Arm A: probe-kind isolation (C1 / AC-1) --------------------------------

def test_probe_kind_exact_yields_only_exact_fixtures():
    fx = armA.make_fixtures(60, seed=1729, probe_kind="exact")
    assert len(fx) == 60
    assert {f.arm for f in fx} == {"exact"}


def test_probe_kind_semantic_yields_only_semantic_fixtures():
    fx = armA.make_fixtures(60, seed=1729, probe_kind="semantic")
    assert {f.arm for f in fx} == {"semantic"}


def test_probe_kind_mixed_is_legacy_interleave():
    fx = armA.make_fixtures(30, seed=1729, probe_kind="mixed")
    kinds = {f.arm for f in fx}
    assert kinds == {"exact", "semantic"}
    # legacy ratio: 1/3 semantic
    n_sem = sum(1 for f in fx if f.arm == "semantic")
    assert n_sem == 10


def test_probe_kind_rejects_unknown():
    with pytest.raises(ValueError):
        armA.make_fixtures(10, seed=1, probe_kind="bogus")


# ---- Arm B: confident-wrong detector controls (C2 / AC-3) -------------------

OWNERS = ["Ada Lovelace", "Grace Hopper", "Frances Allen", "Karen Sparck Jones"]


def test_detector_negative_control_correct_recovery_is_not_confident_wrong():
    # target recovered verbatim -> correct, not confident_wrong
    ans = "Frances Allen is the recovery owner associated with handoff phrase recover-4003."
    correct, cw = armB.score_semantic_recovery(ans, "Frances Allen", OWNERS)
    assert correct is True
    assert cw is False


def test_detector_correct_affirmative_recovery_does_not_fire_freestanding():
    # Opus CB-1: the free-standing matcher must NOT fire on a CORRECT affirmative
    # recovery — same matcher, single discriminator (asserted owner != target).
    # "The recovery owner is Frances Allen" asserts the TARGET -> cw must be False,
    # or the detector would tank recall on passing trials.
    ans = "The recovery owner is Frances Allen."
    correct, cw = armB.score_semantic_recovery(ans, "Frances Allen", OWNERS)
    assert correct is True
    assert cw is False


def test_detector_positive_control_fabricated_other_owner_is_confident_wrong():
    # the REAL idx-173 failure (K=4 name-collision path): asked for Frances
    # Allen's phrase, model confidently asserted a DIFFERENT co-located KNOWN owner.
    ans = "Karen Sparck Jones is the recovery owner for recover-4003."
    correct, cw = armB.score_semantic_recovery(ans, "Frances Allen", OWNERS)
    assert correct is False
    assert cw is True


def test_detector_positive_control_freestanding_fabrication_k1_threat_model():
    # Opus GI-3: the K=1 threat model. At K=1 there are NO sibling owners in the
    # node, so the only failure is INVENTING a free-standing wrong owner that is
    # NOT in the known pool. The detector must catch this, not just name-collisions.
    ans = "The recovery owner is Brian Kernighan."  # not in OWNERS
    assert "Brian Kernighan" not in OWNERS
    correct, cw = armB.score_semantic_recovery(ans, "Frances Allen", OWNERS)
    assert correct is False
    assert cw is True, "free-standing fabricated owner (absent from node) must fire confident_wrong"


def test_detector_freestanding_fabrication_other_phrasing():
    ans = "**Linus Torvalds** is the recovery owner associated with handoff phrase recover-4003."
    correct, cw = armB.score_semantic_recovery(ans, "Frances Allen", OWNERS)
    assert correct is False
    assert cw is True


def test_detector_honest_abstention_is_not_confident_wrong():
    # the REAL idx-12 / idx-13 / idx-174 behavior: model said it couldn't find it.
    for ans in (
        "The recovery owner associated with handoff phrase recover-0300 is not found in the provided context.",
        "The recovery owner associated with handoff phrase recover-0301 is not present in the provided context.",
        "recover-4302 is not present in the provided context. The mappings cover 1200-4003.",
    ):
        correct, cw = armB.score_semantic_recovery(ans, "Ada Lovelace", OWNERS)
        assert correct is False
        assert cw is False, f"abstention wrongly flagged confident_wrong: {ans!r}"


def test_detector_empty_answer_is_not_confident_wrong():
    correct, cw = armB.score_semantic_recovery("", "Ada Lovelace", OWNERS)
    assert correct is False
    assert cw is False
