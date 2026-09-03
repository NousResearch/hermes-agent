"""Transactional, deterministic Kanban handoff for specialist routing."""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, replace
from typing import Optional

from gateway.capability_registry import CapabilityRegistry, CapabilitySignature, RegistryResolution
from gateway.candidate_profile_requests import (
    CandidateProfileRequest,
    CandidateProfileRequests,
    OpaqueEvidenceReference,
    SanitizedTaskEnvelope,
)
from gateway.configured_board import configured_board_db_path
from gateway.specialist_routing import (
    SPECIALIST_PROFILES,
    SpecialistRouteDecision,
    apply_registry_resolution,
    resolve_registry,
)


_ORCHESTRATION_GOAL_MAX_TURNS = 12


@dataclass(frozen=True)
class HandoffSource:
    """Trusted source fields needed to create a task notification route."""

    platform: str
    chat_id: str
    chat_type: str
    user_id: Optional[str]
    message_id: str
    guild_id: Optional[str] = None
    thread_id: Optional[str] = None
    user_id_alt: Optional[str] = None
    notifier_profile: Optional[str] = None
    session_id: Optional[str] = None
    delivery_metadata: Optional[dict] = None


@dataclass(frozen=True)
class HandoffResult:
    ok: bool
    task_id: Optional[str] = None
    created: bool = False
    reason: str = ""
    candidate_request_id: Optional[str] = None
    candidate_status: Optional[str] = None


def _idempotency_key(source: HandoffSource) -> Optional[str]:
    if not source.message_id or not source.platform or not source.chat_id:
        return None
    return "specialist-routing:" + ":".join(
        (source.platform, source.guild_id or "", source.chat_id, source.thread_id or "", source.message_id)
    )


def _body(
    *,
    decision: SpecialistRouteDecision,
    source: HandoffSource,
    request: str,
    router_model: str,
    candidate_request_id: str | None = None,
) -> str:
    payload = {
        "schema": "specialist_routing.v1",
        "request": request[:4_000],
        "profile": decision.profile,
        "confidence": decision.confidence,
        "reason": decision.reason,
        "router_model": router_model or "configured_auxiliary",
        "ingress": {
            "platform": source.platform,
            "guild_id": source.guild_id,
            "chat_id": source.chat_id,
            "thread_id": source.thread_id,
            "message_id": source.message_id,
            "user_id": source.user_id,
        },
    }
    if candidate_request_id:
        payload["candidate_request_id"] = candidate_request_id
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _candidate_source_ref(source_key: str) -> OpaqueEvidenceReference:
    """Return an opaque source reference; candidate storage never gets ingress IDs."""
    return OpaqueEvidenceReference(
        digest=hashlib.sha256(source_key.encode("utf-8")).hexdigest()
    )


def _candidate_fallback(
    *,
    decision: SpecialistRouteDecision,
    signature: CapabilitySignature | None,
    resolution: RegistryResolution | None,
    source_key: str,
    db_path: object,
    candidate_requests: CandidateProfileRequests | None,
) -> tuple[SpecialistRouteDecision, CandidateProfileRequest | None]:
    """Queue a no-match locally and retain the source task's safe known owner."""
    if resolution is None or resolution.status not in {"no_match", "ambiguous"}:
        return decision, None

    fallback = replace(decision, profile="task-orchestrator")
    if signature is None:
        return fallback, None
    try:
        requests = candidate_requests or CandidateProfileRequests(db_path=db_path)
        result = requests.open_or_reuse(
            signature,
            source_key=source_key,
            envelope=SanitizedTaskEnvelope(evidence_refs=(_candidate_source_ref(source_key),)),
        )
    except Exception:
        # A candidate ledger is advisory and local-only. Its unavailability
        # must never prevent the original task's existing triage fallback.
        result = None
    return fallback, result


def _is_candidate_orchestration_fallback(
    decision: SpecialistRouteDecision, resolution: RegistryResolution | None
) -> bool:
    """Allow an explicit missing-scope handoff to reach the inert Task-3 queue.

    The supplied candidate name is never persisted as an assignee: the caller
    immediately replaces it with ``task-orchestrator`` in ``_candidate_fallback``.
    Without a no-match/ambiguous resolution, an unlisted profile remains a
    no-dispatch decision.
    """
    return (
        decision.dispatches
        and decision.profile not in SPECIALIST_PROFILES
        and resolution is not None
        and resolution.status in {"no_match", "ambiguous"}
    )


def create_specialist_handoff(
    *,
    decision: SpecialistRouteDecision,
    source: HandoffSource,
    request: str,
    router_model: str = "",
    board: Optional[str] = None,
    signature: CapabilitySignature | None = None,
    registry: CapabilityRegistry | None = None,
    candidate_requests: CandidateProfileRequests | None = None,
) -> HandoffResult:
    """Create a subscribed, durable triage root for specialist orchestration."""
    effective_decision = decision
    effective_resolution: RegistryResolution | None = None
    if signature is not None and type(registry) is CapabilityRegistry:
        effective_resolution = resolve_registry(signature, registry)
        if not _is_candidate_orchestration_fallback(decision, effective_resolution):
            effective_decision = apply_registry_resolution(
                effective_resolution, fallback=decision
            )
    elif signature is not None:
        # A capability signature without the concrete local registry has no
        # authority to select a profile or create a candidate request. In
        # particular, a caller-supplied duck-typed ``resolve()`` result cannot
        # turn an invented profile into a dispatch target.
        effective_decision = apply_registry_resolution(
            resolve_registry(signature, None), fallback=decision
        )
    elif decision.dispatches and decision.profile not in SPECIALIST_PROFILES:
        effective_decision = SpecialistRouteDecision(
            kind=decision.kind,
            profile=None,
            confidence=decision.confidence,
            reason=decision.reason,
            title=decision.title,
            audit_reason="inactive_profile",
        )
    if not effective_decision.dispatches:
        return HandoffResult(False, reason="non_dispatch_decision")
    if not source.platform or not source.chat_id or not source.message_id:
        return HandoffResult(False, reason="incomplete_source")
    if not isinstance(request, str) or not request.strip():
        return HandoffResult(False, reason="empty_request")
    try:
        from hermes_cli import kanban_db as kb
        from hermes_cli.profiles import profile_exists

        key = _idempotency_key(source)
        db_path = configured_board_db_path(board)
        effective_decision, candidate_result = _candidate_fallback(
            decision=effective_decision,
            signature=signature,
            resolution=effective_resolution,
            source_key=key or "",
            db_path=db_path,
            candidate_requests=candidate_requests,
        )
        if not profile_exists(effective_decision.profile):
            return HandoffResult(False, reason="profile_unavailable")
        conn = kb.connect(db_path=db_path, board=board)
        try:
            existing_id = None
            if key:
                row = conn.execute(
                    "SELECT id FROM tasks WHERE idempotency_key = ? AND status != 'archived' ORDER BY created_at DESC LIMIT 1",
                    (key,),
                ).fetchone()
                existing_id = row["id"] if row else None
            with kb.write_txn(conn):
                task_id = kb.create_task(
                    conn, title=effective_decision.title,
                    body=_body(
                        decision=effective_decision,
                        source=source,
                        request=request,
                        router_model=router_model,
                        candidate_request_id=candidate_result.request_id if candidate_result else None,
                    ),
                    assignee=effective_decision.profile, created_by="specialist-routing",
                    idempotency_key=key, session_id=source.session_id, board=board,
                    triage=True,
                    goal_mode=True,
                    goal_max_turns=_ORCHESTRATION_GOAL_MAX_TURNS,
                )
                kb.add_notify_sub(
                    conn, task_id=task_id, platform=source.platform, chat_id=source.chat_id,
                    chat_type=source.chat_type, thread_id=source.thread_id,
                    user_id=source.user_id, user_id_alt=source.user_id_alt,
                    notifier_profile=source.notifier_profile, delivery_mode="notify",
                    delivery_metadata=source.delivery_metadata, allow_nested=True,
                )
            return HandoffResult(
                True,
                task_id=task_id,
                created=existing_id is None,
                candidate_request_id=candidate_result.request_id if candidate_result else None,
                candidate_status=candidate_result.status if candidate_result else None,
            )
        finally:
            conn.close()
    except Exception as exc:
        return HandoffResult(False, reason=f"handoff_error:{type(exc).__name__}")
