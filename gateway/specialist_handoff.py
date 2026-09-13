"""Transactional, deterministic Kanban handoff for specialist routing."""

from __future__ import annotations

import hashlib
import json
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
_EXAMPLEPROJECT_NAVIGATION_SKILL = "exampleproject-worktree-navigation"


def _required_skills(board: Optional[str]) -> Optional[list[str]]:
    """Pin the project navigation contract on ExampleProject task graphs."""
    if board == "exampleproject-burndown":
        return [_EXAMPLEPROJECT_NAVIGATION_SKILL]
    return None


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
    return OpaqueEvidenceReference(digest=hashlib.sha256(source_key.encode("utf-8")).hexdigest())


def _candidate_fallback(
    *,
    decision: SpecialistRouteDecision,
    signature: CapabilitySignature | None,
    resolution: RegistryResolution | None,
    source_key: str,
    db_path: object,
    candidate_requests: CandidateProfileRequests | None,
    connection: object,
    registry: CapabilityRegistry,
) -> tuple[SpecialistRouteDecision, CandidateProfileRequest | None]:
    """Queue a no-match only when a distinct active orchestrator is available."""
    if resolution is None or resolution.status not in {"no_match", "ambiguous"}:
        return decision, None
    fallback_signature = registry.configured_signature("task-orchestrator")
    if fallback_signature is None:
        return replace(decision, profile=None, audit_reason="inactive_fallback"), None
    fallback_resolution = _resolve_handoff_registry(
        signature=fallback_signature,
        registry=registry,
        connection=connection,
        profile_id="task-orchestrator",
    )
    if fallback_resolution.status != "active_match":
        return replace(decision, profile=None, audit_reason="inactive_fallback"), None
    fallback = replace(decision, profile="task-orchestrator")
    if signature is None:
        return fallback, None
    try:
        requests = candidate_requests or CandidateProfileRequests(db_path=db_path)
        result = requests.open_or_reuse(
            signature,
            source_key=source_key,
            envelope=SanitizedTaskEnvelope(evidence_refs=(_candidate_source_ref(source_key),)),
            profile_id=(
                decision.profile
            ),
            connection=connection,
        )
    except Exception:
        # The candidate ledger is advisory and local-only. Its unavailability
        # must never prevent the existing triage fallback.
        result = None
    return fallback, result


def _resolve_handoff_registry(
    *,
    signature: CapabilitySignature,
    registry: CapabilityRegistry,
    connection: object,
    profile_id: str | None,
) -> RegistryResolution:
    """Resolve the route against the handoff transaction's live connection."""
    return registry.resolve(
        signature,
        profile_id=profile_id,
        connection=connection,
    )


def _is_candidate_orchestration_fallback(
    decision: SpecialistRouteDecision,
    resolution: RegistryResolution | None,
    registry: CapabilityRegistry | None = None,
) -> bool:
    """Allow an explicit missing-scope handoff to reach the inert candidate queue."""
    return (
        decision.dispatches
        and (
            decision.profile not in SPECIALIST_PROFILES
            or (
                registry is not None
                and registry.has_configured_profile(decision.profile or "")
            )
        )
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
    import hermes_cli.kanban_db_connect as _hermes_cli_kanban_db_connect
    import hermes_cli.kanban_db_notify as _hermes_cli_kanban_db_notify

    effective_decision = decision
    effective_resolution: RegistryResolution | None = None
    if signature is not None and type(registry) is CapabilityRegistry:
        effective_resolution = resolve_registry(
            signature, registry, profile_id=decision.profile
        )
        if not _is_candidate_orchestration_fallback(decision, effective_resolution, registry):
            effective_decision = apply_registry_resolution(effective_resolution, fallback=decision)
    elif signature is not None:
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
        conn = _hermes_cli_kanban_db_connect.connect(db_path=db_path, board=board)
        try:
            if key:
                row = conn.execute(
                    "SELECT id FROM tasks WHERE idempotency_key = ? AND status != 'archived' ORDER BY created_at DESC LIMIT 1",
                    (key,),
                ).fetchone()
                if row is not None:
                    # Resolve the source-message idempotency before reopening a
                    # terminal candidate; otherwise the task keeps the old
                    # candidate in its body while the new ledger row is orphaned.
                    return HandoffResult(True, task_id=row["id"], created=False)
            with kb.write_txn(conn):
                candidate_result = None
                if signature is not None and type(registry) is CapabilityRegistry:
                    authoritative_resolution = _resolve_handoff_registry(
                        signature=signature,
                        registry=registry,
                        connection=conn,
                        profile_id=(
                            decision.profile
                            if decision.profile in SPECIALIST_PROFILES
                            else None
                        ),
                    )
                    if authoritative_resolution.status == "active_match":
                        effective_decision = apply_registry_resolution(
                            authoritative_resolution, fallback=decision
                        )
                    elif effective_resolution is not None and effective_resolution.status == "active_match":
                        # A route that was active before the transaction but is
                        # no longer active must not be dispatched using stale
                        # classifier or registry output.
                        effective_decision = apply_registry_resolution(
                            authoritative_resolution
                        )
                    elif _is_candidate_orchestration_fallback(
                        decision, authoritative_resolution, registry
                    ):
                        effective_decision, candidate_result = _candidate_fallback(
                            decision=decision,
                            signature=signature,
                            resolution=authoritative_resolution,
                            source_key=key or "",
                            db_path=db_path,
                            candidate_requests=candidate_requests,
                            connection=conn,
                            registry=registry,
                        )
                    else:
                        effective_decision = apply_registry_resolution(
                            authoritative_resolution, fallback=decision
                        )
                if not effective_decision.dispatches:
                    return HandoffResult(
                        False,
                        reason=effective_decision.audit_reason or "registry_unresolved",
                    )
                selected_profile = effective_decision.profile
                if signature is not None and not profile_exists(selected_profile):
                    return HandoffResult(False, reason="profile_unavailable")
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
                    skills=_required_skills(board),
                )
                _hermes_cli_kanban_db_notify.add_notify_sub(
                    conn, task_id=task_id, platform=source.platform, chat_id=source.chat_id,
                    chat_type=source.chat_type, thread_id=source.thread_id,
                    user_id=source.user_id, user_id_alt=source.user_id_alt,
                    notifier_profile=source.notifier_profile, delivery_mode="notify",
                    delivery_metadata=source.delivery_metadata, allow_nested=True,
                )
            return HandoffResult(
                True,
                task_id=task_id,
                created=True,
                candidate_request_id=candidate_result.request_id if candidate_result else None,
                candidate_status=candidate_result.status if candidate_result else None,
            )
        finally:
            conn.close()
    except Exception as exc:
        return HandoffResult(False, reason=f"handoff_error:{type(exc).__name__}")
