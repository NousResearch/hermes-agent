from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import re
from typing import Optional

from agent.memory_records import (
    MemoryRecord,
    MemoryScope,
    RecordStatus,
    SalienceTier,
    TrustTier,
)


class WriteClass(str, Enum):
    MUST_WRITE = "must_write"
    SHOULD_WRITE = "should_write"
    MAY_WRITE = "may_write"
    DO_NOT_WRITE = "do_not_write"


@dataclass
class WriteDecision:
    write_class: WriteClass
    reason: str
    trust_tier: TrustTier
    salience_tier: SalienceTier
    ambiguity_flag: bool = False


@dataclass
class ConflictDecision:
    winner: MemoryRecord
    loser: MemoryRecord
    loser_status: RecordStatus
    reason: str


_TRUST_PRIORITY = {
    TrustTier.USER_ASSERTED: 0,
    TrustTier.OBSERVED: 1,
    TrustTier.USER_APPROVED: 2,
    TrustTier.INFERRED: 3,
    TrustTier.UNVERIFIED: 4,
}

_TRANSIENT_PHRASES = (
    "let me think",
    "step by step",
    "maybe try",
    "thinking out loud",
    "brainstorm",
    "draft answer",
    "chain of thought",
    "raw reasoning",
)

_UNSAFE_PHRASES = (
    "ignore previous instructions",
    "system prompt",
    "developer prompt",
    "exfiltrate",
)

_RESPONSE_DETAIL_TERMS = ("concise", "brief", "detailed", "verbose", "fuller")
_RESPONSE_OBJECT_TERMS = ("response", "responses", "reply", "replies", "writeup", "writeups")
_SCOPE_REFINEMENT_PATTERN = re.compile(r"\b(?:for|when|during|while|if|unless|except|under)\b\s+[^.]+")


def classify_write_candidate(
    *,
    target: str,
    content: str,
    source_kind: str,
    explicit_remember: bool,
    explicit_correction: bool,
) -> WriteDecision:
    normalized = _normalize_text(content)
    inferred_trust = _trust_tier_for_source_kind(source_kind)

    if explicit_remember or explicit_correction:
        return WriteDecision(
            write_class=WriteClass.MUST_WRITE,
            reason="explicit_operator_signal",
            trust_tier=TrustTier.USER_ASSERTED,
            salience_tier=SalienceTier.HIGH,
        )

    if _looks_do_not_write(normalized=normalized, source_kind=source_kind):
        return WriteDecision(
            write_class=WriteClass.DO_NOT_WRITE,
            reason="ephemeral_or_unsafe",
            trust_tier=inferred_trust,
            salience_tier=SalienceTier.LOW,
        )

    if target == "user" and _looks_like_operator_preference(normalized):
        return WriteDecision(
            write_class=WriteClass.SHOULD_WRITE,
            reason="durable_operator_preference",
            trust_tier=TrustTier.USER_ASSERTED,
            salience_tier=SalienceTier.HIGH,
        )

    if _looks_like_workspace_fact(normalized) and inferred_trust in {
        TrustTier.USER_ASSERTED,
        TrustTier.OBSERVED,
        TrustTier.USER_APPROVED,
    }:
        return WriteDecision(
            write_class=WriteClass.SHOULD_WRITE,
            reason="durable_scoped_fact",
            trust_tier=inferred_trust,
            salience_tier=SalienceTier.MEDIUM,
        )

    return WriteDecision(
        write_class=WriteClass.MAY_WRITE,
        reason="possible_durable_fact_requires_validation",
        trust_tier=inferred_trust,
        salience_tier=SalienceTier.MEDIUM,
        ambiguity_flag=True,
    )


def assign_topic_key(*, target: str, content: str, scope: MemoryScope) -> Optional[str]:
    normalized = _normalize_text(content)
    if not normalized:
        return None

    if target == "user" or scope in {MemoryScope.OPERATOR, MemoryScope.PROFILE}:
        if any(phrase in normalized for phrase in ("british spelling", "uk spelling", "british english")):
            return "preference:spelling"
        if any(term in normalized for term in _RESPONSE_DETAIL_TERMS) and any(
            term in normalized for term in _RESPONSE_OBJECT_TERMS
        ):
            return "preference:response-detail"
        if any(term in normalized for term in ("tone", "cadence", "voice", "style")):
            return "preference:tone"
        if any(term in normalized for term in ("prefer", "default", "always use")):
            subject = normalized
            for prefix in ("user prefers", "i prefer", "prefer", "default to", "always use"):
                subject = subject.replace(prefix, " ")
            return f"preference:{_slugify(subject)}"
        return None

    if scope is MemoryScope.WORKSPACE:
        if any(word in normalized for word in ("deploy", "ship", "release")):
            return "workspace:deploy-command"
        if any(word in normalized for word in ("shell", "bash", "zsh", "fish")):
            return "env:shell"
        if any(word in normalized for word in ("operating system", "ubuntu", "debian", "macos", "linux", "windows")):
            return "env:os"
        if any(word in normalized for word in ("python", "venv", "virtualenv")):
            return "env:python"

    if scope is MemoryScope.SESSION and any(word in normalized for word in ("incident", "task", "ticket")):
        return f"session:{_slugify(normalized)}"

    return None


def transition_freshness(record: MemoryRecord, *, now: str) -> MemoryRecord:
    updated = deepcopy(record)
    now_dt = _parse_timestamp(now)
    review_after_dt = _parse_timestamp(updated.review_after)
    status_changed = False

    if updated.status is RecordStatus.ACTIVE and review_after_dt and now_dt > review_after_dt:
        if not _is_reconfirmed_since(updated, review_after_dt):
            updated.status = RecordStatus.STALE
            status_changed = True
    elif updated.status is RecordStatus.STALE:
        if review_after_dt and (_is_reconfirmed_since(updated, review_after_dt) or _is_reused_since(updated, review_after_dt)):
            updated.status = RecordStatus.ACTIVE
            status_changed = True
        else:
            expires_at_dt = _parse_timestamp(updated.expires_at)
            if expires_at_dt and now_dt > expires_at_dt and not _is_reused_since(updated, expires_at_dt):
                updated.status = RecordStatus.EXPIRED
                status_changed = True
    elif updated.status is RecordStatus.EXPIRED and updated.metadata.get("archive_on_review"):
        updated.status = RecordStatus.ARCHIVED
        status_changed = True

    if status_changed:
        updated.revision += 1

    return updated


def resolve_conflict(old: MemoryRecord, new: MemoryRecord, *, explicit_correction: bool) -> ConflictDecision:
    existing = deepcopy(old)
    incoming = deepcopy(new)

    if existing.scope != incoming.scope or not existing.topic_key or existing.topic_key != incoming.topic_key:
        return ConflictDecision(
            winner=incoming,
            loser=existing,
            loser_status=existing.status,
            reason="coexist_different_scope_or_topic",
        )

    existing_rank = _trust_rank(existing.trust_tier)
    incoming_rank = _trust_rank(incoming.trust_tier)

    if incoming_rank < existing_rank and _is_at_least_as_recent(incoming, existing):
        incoming.supersedes = _bounded_supersedes_target(predecessor=existing, successor=incoming)
        existing.status = RecordStatus.SUPERSEDED
        _append_unique(existing.conflicts_with, incoming.record_id)
        return ConflictDecision(
            winner=incoming,
            loser=existing,
            loser_status=existing.status,
            reason="higher_trust_new_record",
        )

    if existing_rank < incoming_rank:
        incoming.status = RecordStatus.DISPUTED
        _append_unique(existing.conflicts_with, incoming.record_id)
        _append_unique(incoming.conflicts_with, existing.record_id)
        return ConflictDecision(
            winner=existing,
            loser=incoming,
            loser_status=incoming.status,
            reason="existing_record_higher_trust",
        )

    if explicit_correction and _is_at_least_as_recent(incoming, existing):
        incoming.supersedes = _bounded_supersedes_target(predecessor=existing, successor=incoming)
        existing.status = RecordStatus.SUPERSEDED
        _append_unique(existing.conflicts_with, incoming.record_id)
        return ConflictDecision(
            winner=incoming,
            loser=existing,
            loser_status=existing.status,
            reason="newer_explicit_correction",
        )

    if incoming_rank == existing_rank and _is_at_least_as_recent(incoming, existing) and _is_scoped_refinement(
        existing.content, incoming.content
    ):
        incoming.metadata["scope_narrowed"] = True
        incoming.metadata["scope_refinement_of"] = existing.record_id
        scope_qualifier = _extract_scope_qualifier(incoming.content)
        if scope_qualifier:
            incoming.metadata["scope_qualifier"] = scope_qualifier
        return ConflictDecision(
            winner=incoming,
            loser=existing,
            loser_status=existing.status,
            reason="scoped_refinement_keep_both",
        )

    if _is_more_specific(incoming.content, existing.content) and _is_at_least_as_recent(incoming, existing):
        incoming.supersedes = _bounded_supersedes_target(predecessor=existing, successor=incoming)
        existing.status = RecordStatus.SUPERSEDED
        _append_unique(existing.conflicts_with, incoming.record_id)
        return ConflictDecision(
            winner=incoming,
            loser=existing,
            loser_status=existing.status,
            reason="newer_more_specific_refinement",
        )

    existing.status = RecordStatus.DISPUTED
    incoming.status = RecordStatus.DISPUTED
    _append_unique(existing.conflicts_with, incoming.record_id)
    _append_unique(incoming.conflicts_with, existing.record_id)

    winner = incoming if _is_newer(incoming, existing) else existing
    loser = existing if winner is incoming else incoming
    return ConflictDecision(
        winner=winner,
        loser=loser,
        loser_status=loser.status,
        reason="unresolved_conflict_prefer_none",
    )


def _looks_do_not_write(*, normalized: str, source_kind: str) -> bool:
    if any(phrase in normalized for phrase in _TRANSIENT_PHRASES):
        return True
    if any(phrase in normalized for phrase in _UNSAFE_PHRASES):
        return True
    return False


def _looks_like_operator_preference(normalized: str) -> bool:
    if not normalized:
        return False
    preference_terms = (
        "prefer",
        "preference",
        "default",
        "always use",
        "spelling",
        "tone",
        "cadence",
        "response style",
    )
    return any(term in normalized for term in preference_terms)


def _looks_like_workspace_fact(normalized: str) -> bool:
    durable_terms = (
        "repo",
        "repository",
        "workspace",
        "project",
        "deploy",
        "ship",
        "release",
        "shell",
        "bash",
        "zsh",
        "python",
        "venv",
        "convention",
        "workflow",
    )
    return any(term in normalized for term in durable_terms)


def _trust_tier_for_source_kind(source_kind: str) -> TrustTier:
    return {
        "explicit_user_statement": TrustTier.USER_ASSERTED,
        "tool_observation": TrustTier.OBSERVED,
        "provider_sync": TrustTier.OBSERVED,
        "user_approved": TrustTier.USER_APPROVED,
        "transcript_extraction": TrustTier.UNVERIFIED,
        "model_inference": TrustTier.INFERRED,
    }.get(source_kind, TrustTier.UNVERIFIED)


def _trust_rank(trust_tier: TrustTier) -> int:
    return _TRUST_PRIORITY[trust_tier]


def _is_more_specific(new_content: str, old_content: str) -> bool:
    new_tokens = set(re.findall(r"[a-z0-9]+", new_content.lower()))
    old_tokens = set(re.findall(r"[a-z0-9]+", old_content.lower()))
    if not old_tokens:
        return bool(new_tokens)
    return old_tokens < new_tokens


def _is_scoped_refinement(old_content: str, new_content: str) -> bool:
    old_qualifier = _extract_scope_qualifier(old_content)
    new_qualifier = _extract_scope_qualifier(new_content)
    if not new_qualifier:
        return False
    return old_qualifier != new_qualifier


def _extract_scope_qualifier(content: str) -> Optional[str]:
    match = _SCOPE_REFINEMENT_PATTERN.search(_normalize_text(content))
    if match is None:
        return None
    return match.group(0)


def _bounded_supersedes_target(*, predecessor: MemoryRecord, successor: MemoryRecord) -> Optional[str]:
    if predecessor.supersedes == successor.record_id:
        return None
    if predecessor.supersedes and predecessor.supersedes != predecessor.record_id:
        return predecessor.supersedes
    return predecessor.record_id


def _is_at_least_as_recent(new_record: MemoryRecord, old_record: MemoryRecord) -> bool:
    return _timestamp_sort_key(new_record.created_at) >= _timestamp_sort_key(old_record.created_at)


def _is_newer(new_record: MemoryRecord, old_record: MemoryRecord) -> bool:
    return _timestamp_sort_key(new_record.created_at) > _timestamp_sort_key(old_record.created_at)


def _timestamp_sort_key(timestamp: Optional[str]) -> float:
    parsed = _parse_timestamp(timestamp)
    if parsed is None:
        return float("-inf")
    return parsed.timestamp()


def _is_reconfirmed_since(record: MemoryRecord, threshold: datetime) -> bool:
    confirmed_at = _parse_timestamp(record.last_confirmed_at)
    return confirmed_at is not None and confirmed_at >= threshold


def _is_reused_since(record: MemoryRecord, threshold: datetime) -> bool:
    used_at = _parse_timestamp(record.last_used_at)
    return used_at is not None and used_at >= threshold


def _parse_timestamp(timestamp: Optional[str]) -> Optional[datetime]:
    if not timestamp:
        return None
    normalized = timestamp.strip()
    if not normalized:
        return None
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _normalize_text(content: str) -> str:
    return " ".join(content.strip().lower().split())


def _slugify(value: str) -> str:
    words = re.findall(r"[a-z0-9]+", value.lower())
    if not words:
        return "entry"
    return "-".join(words[:4])


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


__all__ = [
    "ConflictDecision",
    "WriteClass",
    "WriteDecision",
    "assign_topic_key",
    "classify_write_candidate",
    "resolve_conflict",
    "transition_freshness",
]
