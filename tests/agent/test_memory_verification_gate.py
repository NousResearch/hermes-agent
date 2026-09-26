"""#112102 — writer-isolated verification gate for memory consolidation.

Session-end extraction wrote straight to the store: whichever branch happened to
hold both the extractor and its judgement decided what became durable knowledge,
so a bad session filed a falsehood as fact and it was re-derived from there. The
fix is structural, not prompt-level (RSIAgent: the consolidation verifier must be
isolated from the writer's private reasoning), so these tests pin structure:

* unapproved candidates are NOT persisted (①),
* approved candidates pass through with the writer's own fields intact (②),
* the verifier's view is an allowlist projection: writer-private fields
  (reasoning / transcript / rationale) cannot reach it (③),
* the gate is OFF by default, so existing writes are untouched (④),
* every failure mode (no verifier, raised error, junk verdict) fails closed.
"""

import pytest

from agent.memory_verification import (
    CANDIDATE_FIELDS,
    CandidateView,
    ConsolidationVerdict,
    consolidation_verification_enabled,
    verify_candidates,
)

ON = {"memory": {"verify_consolidation": True}}


def _cand(content, **extra):
    candidate = {"content": content, "target": "memory", "category": "project"}
    candidate.update(extra)
    return candidate


def _deny(view, evidence):
    return ConsolidationVerdict(approved=False, reason="no evidence")


def _approve(view, evidence):
    return ConsolidationVerdict(approved=True)


# ---------------------------------------------------------------------------
# ① Unapproved candidates never become durable memory
# ---------------------------------------------------------------------------


def test_unapproved_candidate_is_dropped_with_reason():
    candidate = _cand("the project uses PostgreSQL")
    approved, dropped = verify_candidates([candidate], _deny, config=ON)
    assert approved == []
    assert [c for c, _ in dropped] == [candidate]
    assert dropped[0][1] == "no evidence"


def test_only_the_approved_subset_survives():
    keep, lose = _cand("verified fact"), _cand("unverified claim")
    approved, dropped = verify_candidates(
        [keep, lose], lambda view, evidence: view.content == "verified fact", config=ON)
    assert approved == [keep]
    assert [c for c, _ in dropped] == [lose]


# ---------------------------------------------------------------------------
# ② Approved candidates pass through unchanged
# ---------------------------------------------------------------------------


def test_approved_candidate_is_returned_verbatim():
    """The writer keeps fields the verifier never saw — the gate filters, it does
    not rewrite what gets stored."""
    candidate = _cand("the project uses PostgreSQL", source_message_id=7)
    approved, dropped = verify_candidates([candidate], _approve, config=ON)
    assert approved == [candidate]
    assert dropped == []
    assert approved[0] is candidate


def test_verdict_accepts_a_plain_bool_and_forwards_evidence():
    seen = []

    def verifier(view, evidence):
        seen.append((view, evidence))
        return True

    approved, _ = verify_candidates(
        [_cand("the project uses PostgreSQL")], verifier, evidence=["psql --version -> 16.2"], config=ON)
    assert len(approved) == 1
    assert seen[0][1] == ("psql --version -> 16.2",)


# ---------------------------------------------------------------------------
# ③ Isolation — the verifier cannot see the writer's private reasoning
# ---------------------------------------------------------------------------


def test_verifier_receives_only_the_allowlisted_projection():
    seen = []

    def spy(view, evidence):
        seen.append(view)
        return True

    candidate = _cand(
        "the project uses PostgreSQL",
        reasoning="I inferred this from the error trace",
        transcript=[{"role": "user", "content": "guess what I run"}],
        rationale="writer's private chain of thought",
        messages=["leak me"],
    )
    approved, _ = verify_candidates([candidate], spy, config=ON)
    assert len(approved) == 1
    assert seen == [CandidateView(content="the project uses PostgreSQL", target="memory", category="project")]
    assert isinstance(seen[0], CandidateView) and seen[0].__dataclass_fields__.keys() == set(CANDIDATE_FIELDS)
    # Nothing writer-private reached the verifier at all — not as an attribute, not in the payload.
    for leak in ("error trace", "guess what I run", "private chain of thought", "leak me"):
        assert leak not in repr(seen)
    assert not hasattr(seen[0], "reasoning") and not hasattr(seen[0], "transcript")


def test_projection_normalizes_missing_or_non_string_fields():
    """A candidate with no target/category (or junk types) still projects to a
    view — the verifier gets a stable shape, not a KeyError or a raw None."""
    seen = []
    approved, _ = verify_candidates(
        [{"content": "bare"}], lambda view, evidence: seen.append(view) or True, config=ON)
    assert len(approved) == 1
    assert seen[0] == CandidateView(content="bare", target="memory", category="general")


# ---------------------------------------------------------------------------
# ④ Default configuration — existing writes unchanged (protection face)
# ---------------------------------------------------------------------------


def test_default_config_approves_everything_and_never_calls_the_verifier():
    calls = []
    candidates = [_cand("one"), _cand("two")]
    approved, dropped = verify_candidates(candidates, lambda *a: calls.append(a) or False, config={})
    assert approved == candidates
    assert dropped == []
    assert calls == []


def test_config_without_the_key_is_off():
    assert consolidation_verification_enabled({}) is False
    assert consolidation_verification_enabled({"memory": {}}) is False
    assert consolidation_verification_enabled({"memory": {"verify_consolidation": False}}) is False


@pytest.mark.parametrize("value", [True, "true", "True", "1", "yes", "on"])
def test_truthy_values_enable(value):
    assert consolidation_verification_enabled({"memory": {"verify_consolidation": value}}) is True


@pytest.mark.parametrize("value", [False, "false", "0", "no", "off", None, ""])
def test_falsey_values_leave_the_gate_off(value):
    assert consolidation_verification_enabled({"memory": {"verify_consolidation": value}}) is False


def test_explicit_enabled_argument_overrides_config():
    candidate = _cand("x")
    assert verify_candidates([candidate], None, config={}, enabled=True)[0] == []
    assert verify_candidates([candidate], _deny, config=ON, enabled=False)[0] == [candidate]


# ---------------------------------------------------------------------------
# Fail closed — every non-approval drops the candidate
# ---------------------------------------------------------------------------


def test_enabled_without_a_verifier_drops_everything():
    candidate = _cand("x")
    approved, dropped = verify_candidates([candidate], None, config=ON)
    assert approved == []
    assert dropped == [(candidate, "no consolidation verifier configured")]


def test_a_raising_verifier_fails_closed():
    def boom(view, evidence):
        raise RuntimeError("verifier offline")

    approved, dropped = verify_candidates([_cand("x")], boom, config=ON)
    assert approved == []
    assert "verifier offline" in dropped[0][1]


def test_an_unrecognized_verdict_fails_closed():
    approved, dropped = verify_candidates([_cand("x")], lambda view, evidence: "sure", config=ON)
    assert approved == []
    assert "unrecognized verdict" in dropped[0][1]


def test_empty_candidate_list_is_a_no_op():
    assert verify_candidates([], _approve, config=ON) == ([], [])
