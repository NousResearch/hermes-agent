"""Safe synchronous primitives for autonomous Hermes production-order execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import sqlite3
from typing import Any, Callable

from .production_order_db import ProductionOrder, run_orchestrator_triage_bridge
from .production_order_dispatch import (
    DispatchManifestError,
    ProfileTaskEnvelope,
    _load_production_order,
    apply_accepted_result_action,
    build_dispatch_manifest,
    build_profile_task_envelope,
    ingest_profile_result_packet,
    log_dispatch_event,
    validate_profile_result_packet,
)


Runner = Callable[[dict[str, Any]], Any]
_TERMINAL_OR_PAUSE_STATES = {
    "DONE",
    "BLOCKED_NEEDS_JARREN",
    "SCOPE_CHANGE_REQUIRED",
    "CANCELLED",
}
_DIRECT_ORCHESTRATOR_BRIDGES = {"run_orchestrator_triage_bridge": run_orchestrator_triage_bridge}


@dataclass(frozen=True)
class ProfileInvocationResult:
    production_order_id: str
    target_profile: str
    source_state: str
    dispatch_attempt: int
    idempotency_key: str
    timeout_seconds: int | None
    stdout: str
    stderr: str
    exit_code: int
    duration_ms: int | None
    log_ref: str
    result_channel: Any = None
    runner_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProductionRunResult:
    production_order_id: str
    final_state: str
    final_owner_profile: str
    steps_run: int
    terminal_reason: str
    applied_actions: list[dict[str, Any]]
    errors: list[str]
    blocked_reason: str | None
    done: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def invoke_profile_task(
    envelope: ProfileTaskEnvelope,
    runner: Runner | None = None,
    timeout_seconds: int | None = None,
) -> ProfileInvocationResult:
    """Invoke exactly one profile task through an injected synchronous runner."""
    if not isinstance(envelope, ProfileTaskEnvelope):
        raise TypeError("invoke_profile_task requires exactly one ProfileTaskEnvelope")
    if runner is None:
        raise ValueError("invoke_profile_task requires an injected synchronous runner")

    payload = envelope.to_dict()
    if timeout_seconds is not None:
        payload["timeout_seconds"] = timeout_seconds

    raw = runner(payload)
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("profile runner must return a dict-like result")

    log_ref = str(
        raw.get("log_ref")
        or raw.get("log_refs")
        or f"profile-invocation:{envelope.idempotency_key}"
    )
    reserved = {
        "stdout",
        "stderr",
        "exit_code",
        "duration_ms",
        "log_ref",
        "log_refs",
        "result_channel",
    }
    metadata = {key: value for key, value in raw.items() if key not in reserved}
    return ProfileInvocationResult(
        production_order_id=envelope.production_order_id,
        target_profile=envelope.target_profile,
        source_state=envelope.source_state,
        dispatch_attempt=envelope.dispatch_attempt,
        idempotency_key=envelope.idempotency_key,
        timeout_seconds=timeout_seconds,
        stdout=str(raw.get("stdout", "") or ""),
        stderr=str(raw.get("stderr", "") or ""),
        exit_code=int(raw.get("exit_code", 0) or 0),
        duration_ms=_maybe_int(raw.get("duration_ms")),
        log_ref=log_ref,
        result_channel=raw.get("result_channel"),
        runner_metadata=metadata,
    )


def collect_profile_result_packet(
    invocation_result: ProfileInvocationResult,
    envelope: ProfileTaskEnvelope,
) -> dict[str, Any]:
    """Extract and validate exactly one structured result packet from one invocation."""
    if not isinstance(invocation_result, ProfileInvocationResult):
        raise TypeError("collect_profile_result_packet requires a ProfileInvocationResult")
    if not isinstance(envelope, ProfileTaskEnvelope):
        raise TypeError("collect_profile_result_packet requires a ProfileTaskEnvelope")
    if invocation_result.production_order_id != envelope.production_order_id:
        raise ValueError("invocation result production_order_id does not match the provided envelope")
    if invocation_result.target_profile != envelope.target_profile:
        raise ValueError("invocation result target_profile does not match the provided envelope")
    if invocation_result.source_state != envelope.source_state:
        raise ValueError("invocation result source_state does not match the provided envelope")
    if invocation_result.exit_code != 0:
        raise ValueError(
            f"profile invocation failed with exit_code={invocation_result.exit_code}"
        )

    packet = _extract_single_packet(invocation_result)
    return validate_profile_result_packet(envelope, packet)


def apply_profile_result_return(
    conn: sqlite3.Connection,
    production_order_id: str,
    envelope: ProfileTaskEnvelope,
    result_packet: dict[str, Any],
) -> dict[str, Any]:
    """Ingest one validated profile result packet, then apply its accepted action."""
    if envelope.production_order_id != production_order_id:
        raise ValueError("envelope production_order_id does not match the requested production order")
    ingestion = ingest_profile_result_packet(conn, production_order_id, result_packet)
    if not ingestion.get("accepted"):
        raise ValueError(ingestion.get("error") or "result packet was not accepted")
    applied = apply_accepted_result_action(
        conn,
        production_order_id,
        result_packet=result_packet,
    )
    return {
        "envelope": envelope.to_dict(),
        "ingestion": ingestion,
        "applied": applied,
    }


def run_production_order_autonomously(
    conn: sqlite3.Connection,
    production_order_id: str,
    *,
    runner: Runner | None = None,
    max_steps: int = 20,
    max_retries: int = 1,
    timeout_seconds: int | None = None,
) -> ProductionRunResult:
    """Run the bounded foreground production workflow for one existing order."""
    po = _load_production_order(conn, production_order_id)
    if po.current_state in _TERMINAL_OR_PAUSE_STATES:
        return _result_from_order(
            po,
            steps_run=0,
            terminal_reason="already_terminal_or_paused",
            applied_actions=[],
            errors=[],
        )

    if max_steps <= 0:
        return _result_from_order(
            po,
            steps_run=0,
            terminal_reason="max_steps_exceeded",
            applied_actions=[],
            errors=[],
        )

    steps_run = 0
    errors: list[str] = []
    applied_actions: list[dict[str, Any]] = []
    retry_counts: dict[str, int] = {}

    while steps_run < max_steps:
        po = _load_production_order(conn, production_order_id)
        if po.current_state in _TERMINAL_OR_PAUSE_STATES:
            return _result_from_order(
                po,
                steps_run=steps_run,
                terminal_reason=_terminal_reason_for_state(po.current_state),
                applied_actions=applied_actions,
                errors=errors,
            )

        manifest = build_dispatch_manifest(conn, production_order_id)
        if manifest.target_profile == "orchestrator_os":
            direct_bridge = _DIRECT_ORCHESTRATOR_BRIDGES.get(manifest.bridge_function)
            if direct_bridge is None:
                errors.append(
                    f"unsupported direct orchestrator bridge: {manifest.bridge_function}"
                )
                return _result_from_order(
                    po,
                    steps_run=steps_run,
                    terminal_reason="validation_failed",
                    applied_actions=applied_actions,
                    errors=errors,
                )
            log_dispatch_event(
                conn,
                production_order_id=production_order_id,
                event_type="dispatch_started",
                from_state=po.current_state,
                to_state=None,
                owner_profile=po.current_owner_profile,
                target_profile=manifest.target_profile,
                kanban_card_id=manifest.target_child_card_id,
                packet_id=None,
                result="direct_bridge_started",
                next_action=manifest.bridge_function,
            )
            updated = direct_bridge(conn, production_order_id=production_order_id)
            steps_run += 1
            applied_actions.append(
                {
                    "bridge_function": manifest.bridge_function,
                    "from_state": manifest.current_state,
                    "to_state": updated.current_state,
                    "target_profile": manifest.target_profile,
                    "mode": "direct_bridge",
                }
            )
            log_dispatch_event(
                conn,
                production_order_id=production_order_id,
                event_type="dispatch_completed",
                from_state=manifest.current_state,
                to_state=updated.current_state,
                owner_profile=manifest.current_owner_profile,
                target_profile=manifest.target_profile,
                kanban_card_id=manifest.target_child_card_id,
                packet_id=None,
                result="direct_bridge_applied",
                next_action=updated.current_state,
            )
            continue

        envelope = build_profile_task_envelope(conn, production_order_id)
        log_dispatch_event(
            conn,
            production_order_id=production_order_id,
            event_type="dispatch_started",
            from_state=envelope.source_state,
            to_state=envelope.expected_next_state,
            owner_profile=po.current_owner_profile,
            target_profile=envelope.target_profile,
            kanban_card_id=envelope.child_kanban_card_id,
            packet_id=None,
            result="profile_invocation_started",
            next_action="invoke_profile_task",
        )
        try:
            invocation = invoke_profile_task(
                envelope,
                runner=runner,
                timeout_seconds=timeout_seconds,
            )
        except Exception as exc:
            retry_counts[envelope.idempotency_key] = retry_counts.get(envelope.idempotency_key, 0) + 1
            error = str(exc)
            errors.append(error)
            log_dispatch_event(
                conn,
                production_order_id=production_order_id,
                event_type="dispatch_failed",
                from_state=envelope.source_state,
                to_state=envelope.expected_next_state,
                owner_profile=envelope.target_profile,
                target_profile=envelope.target_profile,
                kanban_card_id=envelope.child_kanban_card_id,
                packet_id=None,
                result="profile_invocation_failed",
                error=error,
                next_action="manual_review_failed_invocation",
            )
            if retry_counts[envelope.idempotency_key] > max_retries:
                return _result_from_order(
                    po,
                    steps_run=steps_run,
                    terminal_reason="retry_limit_exceeded",
                    applied_actions=applied_actions,
                    errors=errors,
                )
            continue

        if invocation.exit_code != 0:
            retry_counts[envelope.idempotency_key] = retry_counts.get(envelope.idempotency_key, 0) + 1
            error = f"profile invocation failed with exit_code={invocation.exit_code}"
            errors.append(error)
            log_dispatch_event(
                conn,
                production_order_id=production_order_id,
                event_type="dispatch_failed",
                from_state=envelope.source_state,
                to_state=envelope.expected_next_state,
                owner_profile=envelope.target_profile,
                target_profile=envelope.target_profile,
                kanban_card_id=envelope.child_kanban_card_id,
                packet_id=None,
                result="profile_invocation_failed",
                error=error,
                next_action="manual_review_failed_invocation",
            )
            if retry_counts[envelope.idempotency_key] > max_retries:
                return _result_from_order(
                    po,
                    steps_run=steps_run,
                    terminal_reason="retry_limit_exceeded",
                    applied_actions=applied_actions,
                    errors=errors,
                )
            continue

        try:
            result_packet = collect_profile_result_packet(invocation, envelope)
            applied = apply_profile_result_return(
                conn,
                production_order_id,
                envelope,
                result_packet,
            )
        except Exception as exc:
            error = str(exc)
            errors.append(error)
            log_dispatch_event(
                conn,
                production_order_id=production_order_id,
                event_type="dispatch_failed",
                from_state=envelope.source_state,
                to_state=envelope.expected_next_state,
                owner_profile=envelope.target_profile,
                target_profile=envelope.target_profile,
                kanban_card_id=envelope.child_kanban_card_id,
                packet_id=None,
                result="profile_result_rejected",
                error=error,
                next_action="manual_review_rejected_packet",
            )
            return _result_from_order(
                po,
                steps_run=steps_run,
                terminal_reason="validation_failed",
                applied_actions=applied_actions,
                errors=errors,
            )

        steps_run += 1
        applied_actions.append(
            {
                "bridge_function": applied["applied"]["bridge_function"],
                "from_state": applied["applied"]["from_state"],
                "to_state": applied["applied"]["to_state"],
                "target_profile": envelope.target_profile,
                "mode": "profile_result",
            }
        )

    po = _load_production_order(conn, production_order_id)
    return _result_from_order(
        po,
        steps_run=steps_run,
        terminal_reason="max_steps_exceeded",
        applied_actions=applied_actions,
        errors=errors,
    )


def _extract_single_packet(invocation_result: ProfileInvocationResult) -> dict[str, Any]:
    candidate = invocation_result.result_channel
    if candidate is None:
        text = invocation_result.stdout.strip()
        if not text:
            raise ValueError("profile invocation returned no structured result packet")
        candidate = _parse_packet_text(text)

    if isinstance(candidate, str):
        candidate = _parse_packet_text(candidate.strip())

    if isinstance(candidate, list):
        if len(candidate) != 1:
            raise ValueError("profile invocation returned multiple competing packets")
        candidate = candidate[0]

    if not isinstance(candidate, dict):
        raise ValueError("profile invocation result packet must be a JSON object")
    return dict(candidate)


def _parse_packet_text(text: str) -> Any:
    if not text:
        raise ValueError("profile invocation returned no structured result packet")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        if any(ch in text for ch in "{}[]"):
            raise ValueError("profile invocation returned malformed JSON") from exc
        raise ValueError("profile invocation returned free-text-only output") from exc


def _maybe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _result_from_order(
    po: ProductionOrder,
    *,
    steps_run: int,
    terminal_reason: str,
    applied_actions: list[dict[str, Any]],
    errors: list[str],
) -> ProductionRunResult:
    blocked_reason = po.current_state if po.current_state in _TERMINAL_OR_PAUSE_STATES and po.current_state != "DONE" else None
    return ProductionRunResult(
        production_order_id=po.production_order_id,
        final_state=po.current_state,
        final_owner_profile=po.current_owner_profile,
        steps_run=steps_run,
        terminal_reason=terminal_reason,
        applied_actions=list(applied_actions),
        errors=list(errors),
        blocked_reason=blocked_reason,
        done=po.current_state == "DONE",
    )


def _terminal_reason_for_state(state: str) -> str:
    return "done" if state == "DONE" else "blocked_or_paused"
