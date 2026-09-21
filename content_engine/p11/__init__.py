"""P11-S1: private content-media orchestration contracts and state machine.

This module defines the pure contract layer for media orchestration around
the separately-accepted P10 image/video generation wrapper.  It deliberately
imports **nothing** from P10 — the mapping is by opaque identifier
(source_bundle_id / claim_set_id) only.

Design rules enforced here and by the accompanying tests:

* **Transient vs persistable separation.** ``P10Invocation`` is a transient
  in-memory carrier that MAY hold prompt, raw reference paths, raw P10 error
  text and a copied full sidecar.  The persistable models (``MediaJob``,
  ``MediaAttempt``, ``MediaAsset``, ``MediaReview``) MUST NOT expose any of
  those sensitive fields — they carry only opaque hashes/IDs and orchestration
  metadata.

* **Honest source labels.** The allowed source-label set is fixed; there is
  no silent strengthening or default inflation.  Unknown / synonym labels are
  rejected.

* **Immutable candidate revision.** ``CandidateRevision`` is frozen once
  created and carries only opaque source/claim refs (never raw prompt or
  reference content).

* **State machine.** A strict finite state machine governs the media
  lifecycle.  Approval lands in ``approved_for_handoff`` and NEVER produces a
  publish / schedule / queue state.

* **Approval binding.** Approval binds the current payload digest and
  candidate revision; a mismatch is rejected.
"""
from __future__ import annotations

from enum import Enum
from typing import ClassVar, FrozenSet, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Source labels — honest, fixed, no inflation
# ---------------------------------------------------------------------------

class SourceLabel(str, Enum):
    """Allowed honest source labels.  No silent strengthening."""

    MANUAL_QUEUE = "manual_queue"
    MANUAL = "manual"
    TOPIC_BANK_STATIC = "topic_bank_static"
    RESEARCH_PAPER = "research-paper"
    RESEARCH_DIGEST = "research_digest"
    INTERNAL_ACTIVITY = "internal_activity"
    VERIFIED_VISUAL = "verified_visual"

    @classmethod
    def _missing_(cls, value):  # type: ignore[override]
        # Reject unknown values explicitly (no default/inflation mapping).
        raise ValueError(f"Unknown source label: {value!r}")


SOURCE_LABELS = frozenset(s.value for s in SourceLabel)


# ---------------------------------------------------------------------------
# State machine (defined early so models can reference it)
# ---------------------------------------------------------------------------

class MediaState(str, Enum):
    """Finite states for the media orchestration lifecycle.

    The happy path is:
        candidate_received -> eligible -> media_requested -> media_staged
        -> review_pending -> approved_for_handoff

    Terminal / outcome states: approved_for_handoff, rejected, blocked,
    failed, stale, superseded, expired.

    Note: there is deliberately NO published / scheduled / queued state.
    Approval produces ``approved_for_handoff`` only.
    """

    # Happy path
    CANDIDATE_RECEIVED = "candidate_received"
    ELIGIBLE = "eligible"
    MEDIA_REQUESTED = "media_requested"
    MEDIA_STAGED = "media_staged"
    REVIEW_PENDING = "review_pending"
    APPROVED_FOR_HANDOFF = "approved_for_handoff"

    # Outcomes
    BLOCKED = "blocked"
    FAILED = "failed"
    REJECTED = "rejected"
    STALE = "stale"
    SUPERSEDED = "superseded"
    EXPIRED = "expired"


class InvalidTransition(Exception):
    """Raised when an illegal state transition is attempted."""


# Legal transitions table: current_state -> set of allowed next states.
_TRANSITIONS: dict[MediaState, FrozenSet[MediaState]] = {
    MediaState.CANDIDATE_RECEIVED: frozenset({
        MediaState.ELIGIBLE,
        MediaState.SUPERSEDED,
        MediaState.BLOCKED,
        MediaState.STALE,
    }),
    MediaState.ELIGIBLE: frozenset({
        MediaState.MEDIA_REQUESTED,
        MediaState.BLOCKED,
        MediaState.STALE,
        MediaState.SUPERSEDED,
    }),
    MediaState.MEDIA_REQUESTED: frozenset({
        MediaState.MEDIA_STAGED,
        MediaState.FAILED,
        MediaState.BLOCKED,
        MediaState.EXPIRED,
        MediaState.SUPERSEDED,
    }),
    MediaState.MEDIA_STAGED: frozenset({
        MediaState.REVIEW_PENDING,
        MediaState.FAILED,
        MediaState.EXPIRED,
        MediaState.BLOCKED,
        MediaState.SUPERSEDED,
    }),
    MediaState.REVIEW_PENDING: frozenset({
        MediaState.APPROVED_FOR_HANDOFF,
        MediaState.REJECTED,
        MediaState.BLOCKED,
        MediaState.STALE,
        MediaState.SUPERSEDED,
    }),
    MediaState.FAILED: frozenset({
        MediaState.MEDIA_REQUESTED,  # retry
        MediaState.BLOCKED,
        MediaState.SUPERSEDED,
        MediaState.EXPIRED,
    }),
    # Terminal states — no outgoing transitions.
    MediaState.APPROVED_FOR_HANDOFF: frozenset(),
    MediaState.REJECTED: frozenset(),
    MediaState.BLOCKED: frozenset(),
    MediaState.STALE: frozenset(),
    MediaState.SUPERSEDED: frozenset(),
    MediaState.EXPIRED: frozenset(),
}


def legal_transitions(current: MediaState) -> FrozenSet[MediaState]:
    """Return the set of states reachable from *current*."""
    return _TRANSITIONS.get(current, frozenset())


def transition(current: MediaState, target: MediaState) -> MediaState:
    """Attempt a state transition.

    Returns the new state on success, raises ``InvalidTransition`` if the
    transition is not legal.
    """
    allowed = _TRANSITIONS.get(current, frozenset())
    if target not in allowed:
        raise InvalidTransition(
            f"Illegal transition: {current.value} -> {target.value}"
        )
    return target


# ---------------------------------------------------------------------------
# Sensitive fields that must NEVER appear on persistable models
# ---------------------------------------------------------------------------

SENSITIVE_FIELDS = frozenset({
    "prompt",
    "references",
    "p10_error",
    "sidecar",
    "raw_reference_path",
})


# ---------------------------------------------------------------------------
# Transient P10 invocation — in-memory only, MAY carry sensitive data
# ---------------------------------------------------------------------------

class P10Invocation(BaseModel):
    """Transient in-memory carrier for a P10 generation request.

    This is deliberately NOT persistable: it may carry the full prompt, raw
    reference paths, raw P10 error text and a copied full sidecar.  It lives
    only for the duration of a single generation call.
    """

    model_config = ConfigDict(extra="forbid")

    prompt: str = ""
    references: list[str] = Field(default_factory=list)
    p10_error: Optional[str] = None
    sidecar: Optional[dict] = None


# ---------------------------------------------------------------------------
# Persistable models — no sensitive fields
# ---------------------------------------------------------------------------

class _PersistableModel(BaseModel):
    """Base for all persistable P11 models.

    Enforces ``extra='forbid'`` so no sensitive field can sneak in via
    construction.
    """

    model_config = ConfigDict(extra="forbid")


class MediaJob(_PersistableModel):
    """A persistable media job.

    Maps to P10 via opaque identifiers only (``source_bundle_id``,
    ``claim_set_id``) plus brand / brief / style / aspect.  Never carries
    prompt, raw references, P10 error or sidecar.
    """

    id: str
    source_bundle_id: str
    claim_set_id: str
    brand: str
    brief_id: str
    style_preset: str
    aspect: str
    source_label: str
    review_status: str = "unreviewed"
    state: Optional[MediaState] = None
    # Current candidate revision and payload digest persisted on the job.
    # payload_digest is a hash only — never raw payload.
    candidate_revision: Optional[int] = None
    payload_digest: Optional[str] = None
    # Bound outputs set only by approve().
    bound_digest: Optional[str] = None
    bound_revision: Optional[int] = None

    @field_validator("source_label")
    @classmethod
    def _validate_source_label(cls, v: str) -> str:
        # Reject unknown / inflated labels.
        SourceLabel(v)  # raises ValueError if unknown
        return v

    @field_validator("review_status")
    @classmethod
    def _validate_review_status(cls, v: str) -> str:
        if v != "unreviewed":
            raise ValueError(
                f"MediaJob review_status must be 'unreviewed' at creation, "
                f"got {v!r}"
            )
        return v

    def p10_mapping(self) -> dict:
        """Return the opaque P10 mapping dict.

        Contains only source_bundle_id, claim_set_id, brand, brief_id,
        style_preset, aspect, review_status — never prompt/references/sidecar.
        """
        return {
            "source_bundle_id": self.source_bundle_id,
            "claim_set_id": self.claim_set_id,
            "brand": self.brand,
            "brief_id": self.brief_id,
            "style_preset": self.style_preset,
            "aspect": self.aspect,
            "review_status": self.review_status,
        }


class MediaAttempt(_PersistableModel):
    """A persistable generation attempt for a job."""

    id: str
    job_id: str
    revision: int
    backend: str
    error_code: Optional[str] = None  # stable code, never raw error text


class MediaAsset(_PersistableModel):
    """A persistable media asset produced by an attempt."""

    id: str
    attempt_id: str
    kind: str  # "image" | "video"
    content_hash: str  # opaque hash, never raw content


class MediaReview(_PersistableModel):
    """A persistable review record for a job."""

    id: str
    job_id: str
    decision: str


class CandidateRevision(BaseModel):
    """Immutable candidate revision with opaque source/claim refs.

    Frozen on creation.  Carries only opaque identifiers — never raw prompt
    or reference content.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    revision: int
    source_bundle_id: str
    claim_set_id: str
    source_ref: str  # opaque
    claim_ref: str   # opaque
    brand: str
    brief_id: str
    style_preset: str
    aspect: str
    source_label: str

    @field_validator("source_label")
    @classmethod
    def _validate_source_label(cls, v: str) -> str:
        SourceLabel(v)  # raises ValueError if unknown/inflated
        return v


# Register persistable models (excluding transient P10Invocation and the
# immutable CandidateRevision which is a value object, not a job record).
PERSISTABLE_MODELS: tuple[type[BaseModel], ...] = (
    MediaJob,
    MediaAttempt,
    MediaAsset,
    MediaReview,
)


# ---------------------------------------------------------------------------
# Approval — binds payload digest + revision
# ---------------------------------------------------------------------------

def approve(
    job: MediaJob,
    *,
    current_state: MediaState,
    payload_digest: str,
) -> MediaJob:
    """Approve a job for handoff.

    The approval is bound to the job's persisted ``payload_digest`` and
    ``candidate_revision`` — never to arbitrary caller-supplied values.

    Requires:
      * ``current_state`` is ``review_pending``.
      * ``job.payload_digest`` is set (not None).
      * ``job.candidate_revision`` is set (not None).
      * The supplied ``payload_digest`` matches ``job.payload_digest``.

    On success returns a copy of *job* with ``state`` set to
    ``approved_for_handoff``, ``bound_digest`` set to ``job.payload_digest``,
    and ``bound_revision`` set to ``job.candidate_revision``.
    Raises ``InvalidTransition`` otherwise.
    """
    if current_state != MediaState.REVIEW_PENDING:
        raise InvalidTransition(
            f"Approval requires review_pending, got {current_state.value}"
        )
    if job.payload_digest is None:
        raise InvalidTransition(
            "Approval rejected: job has no persisted payload_digest"
        )
    if job.candidate_revision is None:
        raise InvalidTransition(
            "Approval rejected: job has no persisted candidate_revision"
        )
    if payload_digest != job.payload_digest:
        raise InvalidTransition(
            "Approval rejected: payload digest mismatch"
        )
    return job.model_copy(update={
        "state": MediaState.APPROVED_FOR_HANDOFF,
        "bound_digest": job.payload_digest,
        "bound_revision": job.candidate_revision,
    })


__all__ = [
    "SourceLabel",
    "SOURCE_LABELS",
    "SENSITIVE_FIELDS",
    "P10Invocation",
    "MediaJob",
    "MediaAttempt",
    "MediaAsset",
    "MediaReview",
    "CandidateRevision",
    "PERSISTABLE_MODELS",
    "MediaState",
    "InvalidTransition",
    "transition",
    "legal_transitions",
    "approve",
]
