"""P11-S1: media-orchestration contracts and state machine.

Strict TDD — these tests are written to fail first (RED), then drive the
minimal implementation under content_engine/p11/.

Scope: contracts + state machine only. No P10 import, no provider/network,
no persistence, no publish/schedule/queue.
"""
import hashlib

import pytest

import p11
from p11 import (
    P10Invocation,
    MediaJob,
    MediaAttempt,
    MediaAsset,
    MediaReview,
    CandidateRevision,
    SourceLabel,
    MediaState,
    InvalidTransition,
    transition,
    legal_transitions,
    SOURCE_LABELS,
    PERSISTABLE_MODELS,
    SENSITIVE_FIELDS,
)


# ---------------------------------------------------------------------------
# Source label integrity
# ---------------------------------------------------------------------------

def test_source_label_set_is_exact_and_honest():
    """No silent source-strengthening: the allowed set is fixed and honest."""
    expected = {
        "manual_queue", "manual", "topic_bank_static",
        "research-paper", "research_digest",
        "internal_activity", "verified_visual",
    }
    assert set(SOURCE_LABELS) == expected


def test_source_label_enum_values_match_strings():
    """Each enum member's value equals its string name (no aliasing/inflation)."""
    for label in SOURCE_LABELS:
        member = SourceLabel(label)
        assert member.value == label


def test_source_label_rejects_unknown():
    with pytest.raises(ValueError):
        SourceLabel("ai_generated")


def test_source_label_rejects_inflated_synonyms():
    """Common inflation synonyms must not silently map to a stronger label."""
    for bad in ("ai", "auto", "generated", "llm", "model", ""):
        with pytest.raises((ValueError, KeyError)):
            SourceLabel(bad)


# ---------------------------------------------------------------------------
# Transient P10Invocation vs persistable models
# ---------------------------------------------------------------------------

def test_p10_invocation_carries_prompt_and_references_in_memory():
    inv = P10Invocation(
        prompt="a scene with warm light",
        references=["/private/refs/x.png", "/private/refs/y.png"],
        p10_error="backend timeout",
        sidecar={"prompt": "full prompt text", "refs": ["/raw/path"]},
    )
    # Transient model MAY hold sensitive data.
    assert inv.prompt == "a scene with warm light"
    assert inv.references == ["/private/refs/x.png", "/private/refs/y.png"]
    assert inv.p10_error == "backend timeout"
    assert inv.sidecar["prompt"] == "full prompt text"


def test_p10_invocation_is_not_in_persistable_set():
    assert P10Invocation not in PERSISTABLE_MODELS


def test_persistable_models_do_not_expose_sensitive_fields():
    """No persistable model may carry prompt, raw reference path, raw P10
    error, or a copied full sidecar."""
    for model_cls in PERSISTABLE_MODELS:
        fields = set(model_cls.model_fields)
        forbidden = set(SENSITIVE_FIELDS)
        leaked = fields & forbidden
        assert not leaked, (
            f"{model_cls.__name__} leaks sensitive fields: {leaked}"
        )


def test_media_job_has_no_prompt_or_raw_error():
    job = MediaJob(
        id="job-1",
        source_bundle_id="sb-1",
        claim_set_id="cs-1",
        brand="coachos",
        brief_id="b-1",
        style_preset="editorial",
        aspect="landscape",
        review_status="unreviewed",
        source_label="manual_queue",
    )
    assert "prompt" not in type(job).model_fields
    assert "p10_error" not in type(job).model_fields
    assert "references" not in type(job).model_fields
    assert "sidecar" not in type(job).model_fields


def test_media_attempt_has_no_prompt_or_raw_error():
    att = MediaAttempt(
        id="att-1",
        job_id="job-1",
        revision=1,
        backend="codex",
    )
    assert "prompt" not in type(att).model_fields
    assert "p10_error" not in type(att).model_fields
    assert "references" not in type(att).model_fields
    assert "sidecar" not in type(att).model_fields


def test_media_asset_has_no_prompt_or_raw_reference_path():
    asset = MediaAsset(
        id="asset-1",
        attempt_id="att-1",
        kind="image",
        content_hash="sha256:abc",
    )
    assert "prompt" not in type(asset).model_fields
    assert "references" not in type(asset).model_fields
    assert "sidecar" not in type(asset).model_fields
    assert "raw_reference_path" not in type(asset).model_fields


def test_media_review_has_no_prompt_or_sidecar():
    rev = MediaReview(
        id="rev-1",
        job_id="job-1",
        decision="approved_for_handoff",
    )
    assert "prompt" not in type(rev).model_fields
    assert "sidecar" not in type(rev).model_fields
    assert "references" not in type(rev).model_fields


# ---------------------------------------------------------------------------
# Immutable candidate revision + opaque source/claim refs
# ---------------------------------------------------------------------------

def test_candidate_revision_is_immutable():
    cr = CandidateRevision(
        revision=1,
        source_bundle_id="sb-1",
        claim_set_id="cs-1",
        source_ref="opaque-src-001",
        claim_ref="opaque-claim-002",
        brand="coachos",
        brief_id="b-1",
        style_preset="editorial",
        aspect="landscape",
        source_label="research-paper",
    )
    with pytest.raises(Exception):
        cr.revision = 2  # frozen
    with pytest.raises(Exception):
        cr.source_ref = "tampered"


def test_candidate_revision_uses_opaque_refs_not_raw_content():
    cr = CandidateRevision(
        revision=1,
        source_bundle_id="sb-1",
        claim_set_id="cs-1",
        source_ref="opaque-src-001",
        claim_ref="opaque-claim-002",
        brand="coachos",
        brief_id="b-1",
        style_preset="editorial",
        aspect="landscape",
        source_label="research-paper",
    )
    # Must NOT carry prompt or raw reference data.
    assert "prompt" not in type(cr).model_fields
    assert "references" not in type(cr).model_fields
    assert "sidecar" not in type(cr).model_fields
    # Refs are opaque strings, not raw paths/content.
    assert isinstance(cr.source_ref, str)
    assert isinstance(cr.claim_ref, str)


def test_candidate_revision_rejects_inflated_source_label():
    with pytest.raises(Exception):
        CandidateRevision(
            revision=1,
            source_bundle_id="sb-1",
            claim_set_id="cs-1",
            source_ref="s",
            claim_ref="c",
            brand="coachos",
            brief_id="b-1",
            style_preset="editorial",
            aspect="landscape",
            source_label="ai_generated",
        )


def test_candidate_revision_is_not_persistable_directly_but_job_carries_mapping():
    """The P10 mapping (source bundle ID / claim set ID, brand, brief_id,
    style_preset, aspect, review_status) lives on MediaJob, not on a raw
    P10 object."""
    job = MediaJob(
        id="job-1",
        source_bundle_id="sb-1",
        claim_set_id="cs-1",
        brand="coachos",
        brief_id="b-1",
        style_preset="editorial",
        aspect="landscape",
        review_status="unreviewed",
        source_label="manual_queue",
    )
    assert job.source_bundle_id == "sb-1"
    assert job.claim_set_id == "cs-1"
    assert job.brand == "coachos"
    assert job.brief_id == "b-1"
    assert job.style_preset == "editorial"
    assert job.aspect == "landscape"
    assert job.review_status == "unreviewed"


# ---------------------------------------------------------------------------
# State machine - legal happy path
# ---------------------------------------------------------------------------

def test_state_machine_happy_path():
    """candidate_received -> eligible -> media_requested -> media_staged
    -> review_pending -> approved_for_handoff"""
    states = [
        MediaState.CANDIDATE_RECEIVED,
        MediaState.ELIGIBLE,
        MediaState.MEDIA_REQUESTED,
        MediaState.MEDIA_STAGED,
        MediaState.REVIEW_PENDING,
        MediaState.APPROVED_FOR_HANDOFF,
    ]
    current = MediaState.CANDIDATE_RECEIVED
    for nxt in states[1:]:
        current = transition(current, nxt)
    assert current == MediaState.APPROVED_FOR_HANDOFF


def test_state_machine_labels_match_contract():
    expected = {
        "candidate_received", "eligible", "media_requested",
        "media_staged", "review_pending", "approved_for_handoff",
    }
    actual = {s.value for s in MediaState}
    assert expected <= actual


# ---------------------------------------------------------------------------
# State machine - block / failure / retry / reject / stale / supersede / expire
# ---------------------------------------------------------------------------

def test_block_transition():
    # blocking from a pre-approval state must land in BLOCKED
    current = transition(MediaState.ELIGIBLE, MediaState.BLOCKED)
    assert current == MediaState.BLOCKED


def test_failure_transition_from_media_requested():
    current = transition(MediaState.MEDIA_REQUESTED, MediaState.FAILED)
    assert current == MediaState.FAILED


def test_retry_from_failed_back_to_media_requested():
    current = transition(MediaState.FAILED, MediaState.MEDIA_REQUESTED)
    assert current == MediaState.MEDIA_REQUESTED


def test_reject_transition():
    current = transition(MediaState.REVIEW_PENDING, MediaState.REJECTED)
    assert current == MediaState.REJECTED


def test_stale_transition():
    current = transition(MediaState.ELIGIBLE, MediaState.STALE)
    assert current == MediaState.STALE


def test_supersede_transition():
    current = transition(MediaState.CANDIDATE_RECEIVED, MediaState.SUPERSEDED)
    assert current == MediaState.SUPERSEDED


def test_expire_transition():
    current = transition(MediaState.MEDIA_STAGED, MediaState.EXPIRED)
    assert current == MediaState.EXPIRED


# ---------------------------------------------------------------------------
# State machine - illegal transitions rejected
# ---------------------------------------------------------------------------

def test_illegal_skip_transition_rejected():
    """Cannot skip eligible -> media_staged directly."""
    with pytest.raises(InvalidTransition):
        transition(MediaState.ELIGIBLE, MediaState.MEDIA_STAGED)


def test_illegal_transition_from_terminal_approved():
    """approved_for_handoff is terminal for the media path - no further
    media state transition allowed."""
    with pytest.raises(InvalidTransition):
        transition(MediaState.APPROVED_FOR_HANDOFF, MediaState.MEDIA_REQUESTED)


def test_illegal_transition_from_rejected():
    with pytest.raises(InvalidTransition):
        transition(MediaState.REJECTED, MediaState.ELIGIBLE)


def test_illegal_transition_to_publish_state():
    """Approval must NEVER produce a publish/schedule/queue state."""
    for forbidden in ("published", "scheduled", "queued"):
        assert not any(
            s.value == forbidden for s in MediaState
        ), f"{forbidden} must not exist as a MediaState"


def test_illegal_reverse_from_approved():
    with pytest.raises(InvalidTransition):
        transition(MediaState.APPROVED_FOR_HANDOFF, MediaState.REVIEW_PENDING)


def test_illegal_block_from_approved():
    with pytest.raises(InvalidTransition):
        transition(MediaState.APPROVED_FOR_HANDOFF, MediaState.BLOCKED)


# ---------------------------------------------------------------------------
# Approval must bind current payload digest / revision
# ---------------------------------------------------------------------------

def test_approval_binds_payload_digest():
    """Transitioning into approved_for_handoff requires the current payload
    digest to be bound - mismatch raises."""
    from p11 import approve
    digest = hashlib.sha256(b"payload-v1").hexdigest()
    job = MediaJob(
        id="job-1",
        source_bundle_id="sb-1",
        claim_set_id="cs-1",
        brand="coachos",
        brief_id="b-1",
        style_preset="editorial",
        aspect="landscape",
        review_status="unreviewed",
        source_label="manual_queue",
        candidate_revision=1,
        payload_digest=digest,
    )
    # happy: supplied digest matches job.payload_digest
    approved = approve(job, current_state=MediaState.REVIEW_PENDING,
                       payload_digest=digest)
    assert approved.state == MediaState.APPROVED_FOR_HANDOFF
    assert approved.bound_digest == digest
    assert approved.bound_digest == job.payload_digest


def test_approval_rejects_digest_mismatch():
    from p11 import approve
    stored = hashlib.sha256(b"payload-a").hexdigest()
    supplied = hashlib.sha256(b"payload-b").hexdigest()
    job = MediaJob(
        id="job-1",
        source_bundle_id="sb-1",
        claim_set_id="cs-1",
        brand="coachos",
        brief_id="b-1",
        style_preset="editorial",
        aspect="landscape",
        review_status="unreviewed",
        source_label="manual_queue",
        candidate_revision=1,
        payload_digest=stored,
    )
    with pytest.raises(InvalidTransition):
        approve(job, current_state=MediaState.REVIEW_PENDING,
                payload_digest=supplied)


def test_approval_rejects_if_not_in_review_pending():
    from p11 import approve
    digest = hashlib.sha256(b"x").hexdigest()
    job = MediaJob(
        id="job-1",
        source_bundle_id="sb-1",
        claim_set_id="cs-1",
        brand="coachos",
        brief_id="b-1",
        style_preset="editorial",
        aspect="landscape",
        review_status="unreviewed",
        source_label="manual_queue",
        candidate_revision=1,
        payload_digest=digest,
    )
    with pytest.raises(InvalidTransition):
        approve(job, current_state=MediaState.MEDIA_STAGED,
                payload_digest=digest)


def test_approval_binds_revision():
    """Approval also binds the candidate revision number from the job."""
    from p11 import approve
    digest = hashlib.sha256(b"payload-v1").hexdigest()
    job = MediaJob(
        id="job-1",
        source_bundle_id="sb-1",
        claim_set_id="cs-1",
        brand="coachos",
        brief_id="b-1",
        style_preset="editorial",
        aspect="landscape",
        review_status="unreviewed",
        source_label="manual_queue",
        candidate_revision=3,
        payload_digest=digest,
    )
    approved = approve(job, current_state=MediaState.REVIEW_PENDING,
                       payload_digest=digest)
    assert approved.bound_revision == 3
    assert approved.bound_revision == job.candidate_revision


# ---------------------------------------------------------------------------
# legal_transitions helper
# ---------------------------------------------------------------------------

def test_legal_transitions_returns_frozenset():
    lt = legal_transitions(MediaState.ELIGIBLE)
    assert isinstance(lt, frozenset)
    assert MediaState.MEDIA_REQUESTED in lt
    assert MediaState.STALE in lt
    assert MediaState.BLOCKED in lt
    assert MediaState.MEDIA_STAGED not in lt  # skip illegal


def test_legal_transitions_terminal_empty():
    lt = legal_transitions(MediaState.APPROVED_FOR_HANDOFF)
    assert lt == frozenset()


def test_legal_transitions_rejected_empty():
    lt = legal_transitions(MediaState.REJECTED)
    assert lt == frozenset()


# ---------------------------------------------------------------------------
# P10 mapping - source bundle ID / claim set ID used, not raw P10 object
# ---------------------------------------------------------------------------

def test_p10_mapping_uses_bundle_and_claim_ids():
    """MediaJob maps to P10 via source_bundle_id and claim_set_id, brand,
    brief_id, style_preset, aspect - NOT via importing P10 or carrying a
    raw P10 result object."""
    job = MediaJob(
        id="job-1",
        source_bundle_id="sb-42",
        claim_set_id="cs-99",
        brand="plenishd",
        brief_id="brief-7",
        style_preset="warm_dark",
        aspect="portrait_4_5",
        review_status="unreviewed",
        source_label="topic_bank_static",
    )
    mapping = job.p10_mapping()
    assert mapping["source_bundle_id"] == "sb-42"
    assert mapping["claim_set_id"] == "cs-99"
    assert mapping["brand"] == "plenishd"
    assert mapping["brief_id"] == "brief-7"
    assert mapping["style_preset"] == "warm_dark"
    assert mapping["aspect"] == "portrait_4_5"
    assert mapping["review_status"] == "unreviewed"
    # No prompt / raw references in the mapping.
    assert "prompt" not in mapping
    assert "references" not in mapping
    assert "sidecar" not in mapping


def test_p10_mapping_does_not_import_p10():
    """The p11 module must not import any P10 module at import time."""
    import importlib
    import sys
    # Remove any cached p10-ish modules and re-import p11 fresh.
    p10_keys = [k for k in sys.modules if "image_gen_wrapper" in k
                or k == "p10" or k.startswith("p10.")]
    assert not p10_keys, f"P10 already imported: {p10_keys}"
    importlib.reload(p11)
    p10_keys = [k for k in sys.modules if "image_gen_wrapper" in k
                or k == "p10" or k.startswith("p10.")]
    assert not p10_keys, f"P10 imported by p11: {p10_keys}"


# ---------------------------------------------------------------------------
# Default review_status is unreviewed
# ---------------------------------------------------------------------------

def test_media_job_defaults_review_status_unreviewed():
    job = MediaJob(
        id="job-1",
        source_bundle_id="sb-1",
        claim_set_id="cs-1",
        brand="coachos",
        brief_id="b-1",
        style_preset="editorial",
        aspect="landscape",
        source_label="manual_queue",
    )
    assert job.review_status == "unreviewed"


def test_media_job_rejects_non_unreviewed_initial():
    """At creation a job must not be auto-approved/inflated."""
    with pytest.raises(Exception):
        MediaJob(
            id="job-1",
            source_bundle_id="sb-1",
            claim_set_id="cs-1",
            brand="coachos",
            brief_id="b-1",
            style_preset="editorial",
            aspect="landscape",
            source_label="manual_queue",
            review_status="approved",
        )


# ---------------------------------------------------------------------------
# P11-S1 approval binding repair: job must persist candidate_revision and
# payload_digest; approve() must derive bindings from the job, not from
# arbitrary caller-supplied revision/expected-digest pairs.
# ---------------------------------------------------------------------------

def test_media_job_persists_candidate_revision_and_payload_digest():
    """MediaJob must carry candidate_revision (int) and payload_digest (str).

    The payload_digest is a hash only — never the raw payload.
    """
    digest = hashlib.sha256(b"payload-v1").hexdigest()
    job = MediaJob(
        id="job-1",
        source_bundle_id="sb-1",
        claim_set_id="cs-1",
        brand="coachos",
        brief_id="b-1",
        style_preset="editorial",
        aspect="landscape",
        source_label="manual_queue",
        candidate_revision=2,
        payload_digest=digest,
    )
    assert job.candidate_revision == 2
    assert job.payload_digest == digest


def test_media_job_candidate_revision_defaults_none():
    """When not set, candidate_revision is None (approval will reject)."""
    job = MediaJob(
        id="job-1",
        source_bundle_id="sb-1",
        claim_set_id="cs-1",
        brand="coachos",
        brief_id="b-1",
        style_preset="editorial",
        aspect="landscape",
        source_label="manual_queue",
    )
    assert job.candidate_revision is None
    assert job.payload_digest is None


def test_approval_derives_bound_digest_from_job_not_caller():
    """approve() must bind job.payload_digest, not an arbitrary expected_digest."""
    from p11 import approve
    digest = hashlib.sha256(b"payload-v1").hexdigest()
    job = MediaJob(
        id="job-1",
        source_bundle_id="sb-1",
        claim_set_id="cs-1",
        brand="coachos",
        brief_id="b-1",
        style_preset="editorial",
        aspect="landscape",
        source_label="manual_queue",
        candidate_revision=1,
        payload_digest=digest,
    )
    approved = approve(job, current_state=MediaState.REVIEW_PENDING,
                       payload_digest=digest)
    assert approved.bound_digest == digest
    assert approved.bound_digest == job.payload_digest


def test_approval_derives_bound_revision_from_job_not_caller():
    """approve() must bind job.candidate_revision, not a caller-supplied revision."""
    from p11 import approve
    digest = hashlib.sha256(b"payload-v1").hexdigest()
    job = MediaJob(
        id="job-1",
        source_bundle_id="sb-1",
        claim_set_id="cs-1",
        brand="coachos",
        brief_id="b-1",
        style_preset="editorial",
        aspect="landscape",
        source_label="manual_queue",
        candidate_revision=5,
        payload_digest=digest,
    )
    approved = approve(job, current_state=MediaState.REVIEW_PENDING,
                       payload_digest=digest)
    assert approved.bound_revision == 5
    assert approved.bound_revision == job.candidate_revision


def test_approval_rejects_when_job_payload_digest_missing():
    """If job.payload_digest is None, approval must be rejected."""
    from p11 import approve, InvalidTransition as _IT
    digest = hashlib.sha256(b"payload-v1").hexdigest()
    job = MediaJob(
        id="job-1",
        source_bundle_id="sb-1",
        claim_set_id="cs-1",
        brand="coachos",
        brief_id="b-1",
        style_preset="editorial",
        aspect="landscape",
        source_label="manual_queue",
        candidate_revision=1,
        # payload_digest deliberately omitted -> None
    )
    with pytest.raises(_IT):
        approve(job, current_state=MediaState.REVIEW_PENDING,
                payload_digest=digest)


def test_approval_rejects_when_job_candidate_revision_missing():
    """If job.candidate_revision is None, approval must be rejected."""
    from p11 import approve, InvalidTransition as _IT
    digest = hashlib.sha256(b"payload-v1").hexdigest()
    job = MediaJob(
        id="job-1",
        source_bundle_id="sb-1",
        claim_set_id="cs-1",
        brand="coachos",
        brief_id="b-1",
        style_preset="editorial",
        aspect="landscape",
        source_label="manual_queue",
        # candidate_revision deliberately omitted -> None
        payload_digest=digest,
    )
    with pytest.raises(_IT):
        approve(job, current_state=MediaState.REVIEW_PENDING,
                payload_digest=digest)


def test_approval_rejects_supplied_digest_mismatch_job():
    """If supplied payload_digest != job.payload_digest, approval must reject."""
    from p11 import approve, InvalidTransition as _IT
    stored = hashlib.sha256(b"payload-v1").hexdigest()
    supplied = hashlib.sha256(b"payload-v2").hexdigest()
    job = MediaJob(
        id="job-1",
        source_bundle_id="sb-1",
        claim_set_id="cs-1",
        brand="coachos",
        brief_id="b-1",
        style_preset="editorial",
        aspect="landscape",
        source_label="manual_queue",
        candidate_revision=1,
        payload_digest=stored,
    )
    with pytest.raises(_IT):
        approve(job, current_state=MediaState.REVIEW_PENDING,
                payload_digest=supplied)
