"""Safe synchronous primitives for autonomous Hermes production-order execution."""

from __future__ import annotations

import concurrent.futures
import contextlib
from dataclasses import asdict, dataclass, field
import io
import json
import re
import os
import sqlite3
import time
import traceback
from typing import Any, Callable
from uuid import uuid4

from hermes_cli.config import load_config
from hermes_cli.profiles import normalize_profile_name, resolve_profile_env
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from run_agent import AIAgent

from .production_order_db import (
    ProductionOrder,
    _find_existing_order,
    run_full_bridge,
    run_orchestrator_triage_bridge,
)
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
_APPROVED_ACTION_ENVELOPE_CONTROL_OBJECT_TYPE = "production_workflow_brief"
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


def hermes_profile_runner(
    envelope_payload: dict[str, Any],
    *,
    timeout_seconds: int | None = None,
) -> dict[str, Any]:
    """Run exactly one profile task through an explicit Hermes profile runtime.

    This adapter is opt-in via HERMES_ENABLE_PRODUCTION_PROFILE_RUNTIME=1.
    When disabled, it fails safe with a structured non-zero result instead of
    pretending real profile execution is stable.
    """
    started_at = time.monotonic()
    try:
        envelope = _coerce_profile_task_envelope(envelope_payload)
    except Exception as exc:
        return _adapter_error_result(
            envelope_payload,
            error=str(exc),
            started_at=started_at,
            log_ref="profile-runtime:invalid-envelope",
        )

    if os.getenv("HERMES_ENABLE_PRODUCTION_PROFILE_RUNTIME", "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return _adapter_error_result(
            envelope.to_dict(),
            error=(
                "Real Hermes profile runtime adapter is disabled. "
                "Set HERMES_ENABLE_PRODUCTION_PROFILE_RUNTIME=1 to opt in."
            ),
            started_at=started_at,
            log_ref=f"profile-runtime:{envelope.target_profile}:disabled",
        )

    runtime_session_id = f"production-order-{envelope.production_order_id}-{uuid4().hex[:12]}"
    runtime_log_ref = f"profile-runtime:{envelope.target_profile}:{runtime_session_id}"
    try:
        agent_result = _run_profile_agent_once(
            envelope,
            timeout_seconds=timeout_seconds,
            runtime_session_id=runtime_session_id,
        )
    except Exception as exc:
        return _adapter_error_result(
            envelope.to_dict(),
            error=str(exc),
            started_at=started_at,
            log_ref=runtime_log_ref,
            stderr=traceback.format_exc(),
        )

    duration_ms = int((time.monotonic() - started_at) * 1000)
    final_response = agent_result.get("final_response")
    captured_stdout = str(agent_result.get("stdout", "") or "")
    if not captured_stdout and final_response is not None:
        captured_stdout = _stringify_result_channel(final_response)
    adapter_result = {
        "stdout": captured_stdout,
        "stderr": str(agent_result.get("stderr", "") or ""),
        "exit_code": int(agent_result.get("exit_code", 0) or 0),
        "duration_ms": duration_ms,
        "log_ref": runtime_log_ref,
        "result_channel": final_response,
        "session_id": runtime_session_id,
        "target_profile": envelope.target_profile,
    }
    for key in (
        "resolved_hermes_home",
        "resolved_model_default",
        "resolved_model_provider",
        "resolved_model_base_url",
    ):
        if key in agent_result:
            adapter_result[key] = agent_result[key]
    return adapter_result


def invoke_profile_task(
    envelope: ProfileTaskEnvelope,
    runner: Runner | None = None,
    timeout_seconds: int | None = None,
) -> ProfileInvocationResult:
    """Invoke exactly one profile task through an injected synchronous runner."""
    if not isinstance(envelope, ProfileTaskEnvelope):
        raise TypeError("invoke_profile_task requires exactly one ProfileTaskEnvelope")
    if runner is None:
        runner = lambda payload: hermes_profile_runner(payload, timeout_seconds=timeout_seconds)

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
            metadata = invocation.runner_metadata or {}
            failure_diagnostics = {
                "result": "profile_invocation_failed",
                "error": error,
                "invocation_log_ref": invocation.log_ref,
                "stdout_preview": _bounded_preview(invocation.stdout),
                "stderr_preview": _bounded_preview(invocation.stderr),
                "result_channel_preview": _bounded_preview(
                    json.dumps(invocation.result_channel, ensure_ascii=False)
                    if invocation.result_channel is not None
                    else ""
                ),
                "resolved_hermes_home": metadata.get("resolved_hermes_home"),
                "resolved_model_default": metadata.get("resolved_model_default"),
                "resolved_model_provider": metadata.get("resolved_model_provider"),
                "resolved_model_base_url": metadata.get("resolved_model_base_url"),
            }
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
                result=json.dumps(failure_diagnostics, ensure_ascii=False),
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
            # Record bounded diagnostic previews to help triage malformed profile output.
            try:
                rc_preview_src = None
                if invocation is not None:
                    rc = invocation.result_channel
                    try:
                        rc_preview_src = json.dumps(rc, ensure_ascii=False)
                    except Exception:
                        rc_preview_src = str(rc)
                else:
                    rc_preview_src = ""
            except Exception:
                rc_preview_src = ""

            diagnostics = {
                "result": "profile_result_rejected",
                "parse_error": error,
                "invocation_log_ref": getattr(invocation, "log_ref", None) if invocation is not None else None,
                "stdout_preview": _bounded_preview(getattr(invocation, "stdout", None) if invocation is not None else None),
                "stderr_preview": _bounded_preview(getattr(invocation, "stderr", None) if invocation is not None else None),
                "result_channel_preview": _bounded_preview(rc_preview_src),
            }
            if invocation is not None:
                metadata = invocation.runner_metadata or {}
                diagnostics["resolved_hermes_home"] = metadata.get("resolved_hermes_home")
                diagnostics["resolved_model_default"] = metadata.get("resolved_model_default")
                diagnostics["resolved_model_provider"] = metadata.get("resolved_model_provider")
                diagnostics["resolved_model_base_url"] = metadata.get("resolved_model_base_url")

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
                result=json.dumps(diagnostics, ensure_ascii=False),
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
    stripped = text.strip()
    if not stripped:
        raise ValueError("profile invocation returned no structured result packet")

    # Preferred: entire text is a JSON value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # Next: look for a single fenced code block containing JSON
    fence_re = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S | re.I)
    fence_matches = fence_re.findall(text)
    successes: list[Any] = []
    if fence_matches:
        for match in fence_matches:
            try:
                parsed = json.loads(match)
            except json.JSONDecodeError as exc:
                raise ValueError("profile invocation returned malformed JSON") from exc
            successes.append(parsed)
        if len(successes) == 1:
            return successes[0]
        raise ValueError("profile invocation returned multiple competing packets")

    # Next: defensively extract top-level JSON object substrings (balanced braces)
    objs: list[Any] = []
    depth = 0
    start_idx: int | None = None
    for idx, ch in enumerate(text):
        if ch == '{':
            if depth == 0:
                start_idx = idx
            depth += 1
        elif ch == '}' and depth > 0:
            depth -= 1
            if depth == 0 and start_idx is not None:
                candidate = text[start_idx: idx + 1]
                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError as exc:
                    raise ValueError("profile invocation returned malformed JSON") from exc
                objs.append(parsed)
                start_idx = None

    if len(objs) == 1:
        return objs[0]
    if len(objs) > 1:
        raise ValueError("profile invocation returned multiple competing packets")

    # No JSON objects found; determine if there were any JSON-like characters
    if any(ch in text for ch in "{}[]"):
        raise ValueError("profile invocation returned malformed JSON")
    raise ValueError("profile invocation returned free-text-only output")


def _maybe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _bounded_preview(value: Any, limit: int = 4000) -> str:
    if value is None:
        return ""
    try:
        s = str(value)
    except Exception:
        s = repr(value)
    if len(s) <= limit:
        return s
    # keep start and end context for very long outputs
    head = s[: limit - 64]
    tail = s[-60:]
    return head + "\n...<truncated>...\n" + tail


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


def create_approved_action_envelope(
    *,
    approved_brief: dict[str, Any],
    approved_by: str,
    approved_at: str | int,
    approval_phrase: str,
    priority_lane: str,
    repo_or_workspace: str,
    scope: list[str] | tuple[str, ...] | str,
    out_of_scope: list[str] | tuple[str, ...] | str,
    acceptance_criteria: list[str] | tuple[str, ...] | str,
    approval_boundaries: list[str] | tuple[str, ...] | str,
    stop_conditions: list[str] | tuple[str, ...] | str,
    source_truth: list[str] | tuple[str, ...] | str,
    silence_protocol: dict[str, Any] | str,
    idempotency_key: str,
    approved_action_envelope_id: str | None = None,
    control_object_type: str = _APPROVED_ACTION_ENVELOPE_CONTROL_OBJECT_TYPE,
) -> dict[str, Any]:
    envelope = {
        "approved_action_envelope_id": approved_action_envelope_id or f"AAE-{uuid4().hex[:12]}",
        "control_object_type": control_object_type,
        "approved_brief": dict(approved_brief),
        "approved_by": approved_by,
        "approved_at": approved_at,
        "approval_phrase": approval_phrase,
        "priority_lane": priority_lane,
        "repo_or_workspace": repo_or_workspace,
        "scope": _normalize_required_list(scope, field_name="scope"),
        "out_of_scope": _normalize_required_list(out_of_scope, field_name="out_of_scope"),
        "acceptance_criteria": _normalize_required_list(
            acceptance_criteria,
            field_name="acceptance_criteria",
        ),
        "approval_boundaries": _normalize_required_list(
            approval_boundaries,
            field_name="approval_boundaries",
        ),
        "stop_conditions": _normalize_required_list(stop_conditions, field_name="stop_conditions"),
        "source_truth": _normalize_required_list(source_truth, field_name="source_truth"),
        "silence_protocol": _normalize_silence_protocol(silence_protocol),
        "idempotency_key": str(idempotency_key or "").strip(),
    }
    return validate_approved_action_envelope(envelope)


def validate_approved_action_envelope(envelope: Any) -> dict[str, Any]:
    if not isinstance(envelope, dict):
        raise ValueError("Approved Action Envelope must be a JSON object")
    for raw_chat_key in ("raw_chat", "chat_history", "conversation"):
        if raw_chat_key in envelope and envelope.get(raw_chat_key):
            raise ValueError("Approved Action Envelope rejects raw chat input")

    required_fields = (
        "approved_action_envelope_id",
        "control_object_type",
        "approved_brief",
        "approved_by",
        "approved_at",
        "approval_phrase",
        "priority_lane",
        "repo_or_workspace",
        "scope",
        "out_of_scope",
        "acceptance_criteria",
        "approval_boundaries",
        "stop_conditions",
        "source_truth",
        "silence_protocol",
        "idempotency_key",
    )
    missing = [field for field in required_fields if field not in envelope]
    if missing:
        raise ValueError(f"Approved Action Envelope missing required fields: {', '.join(missing)}")
    if envelope.get("control_object_type") != _APPROVED_ACTION_ENVELOPE_CONTROL_OBJECT_TYPE:
        raise ValueError("Approved Action Envelope control_object_type must equal production_workflow_brief")

    approved_brief = envelope.get("approved_brief")
    if not isinstance(approved_brief, dict):
        raise ValueError("Approved Action Envelope approved_brief must be a structured object, not raw chat")

    normalized = dict(envelope)
    normalized["approved_action_envelope_id"] = str(normalized["approved_action_envelope_id"] or "").strip()
    normalized["approved_by"] = str(normalized["approved_by"] or "").strip()
    normalized["approval_phrase"] = str(normalized["approval_phrase"] or "").strip()
    normalized["priority_lane"] = str(normalized["priority_lane"] or "").strip()
    normalized["repo_or_workspace"] = str(normalized["repo_or_workspace"] or "").strip()
    normalized["idempotency_key"] = str(normalized["idempotency_key"] or "").strip()
    if not normalized["approved_action_envelope_id"]:
        raise ValueError("Approved Action Envelope missing approved_action_envelope_id")
    if not normalized["repo_or_workspace"]:
        raise ValueError("Approved Action Envelope requires repo_or_workspace")
    if not normalized["idempotency_key"]:
        raise ValueError("Approved Action Envelope requires idempotency_key")

    normalized["scope"] = _normalize_required_list(normalized["scope"], field_name="scope")
    normalized["out_of_scope"] = _normalize_required_list(normalized["out_of_scope"], field_name="out_of_scope")
    normalized["acceptance_criteria"] = _normalize_required_list(
        normalized["acceptance_criteria"],
        field_name="acceptance_criteria",
    )
    normalized["approval_boundaries"] = _normalize_required_list(
        normalized["approval_boundaries"],
        field_name="approval_boundaries",
    )
    normalized["stop_conditions"] = _normalize_required_list(
        normalized["stop_conditions"],
        field_name="stop_conditions",
    )
    normalized["source_truth"] = _normalize_required_list(
        normalized["source_truth"],
        field_name="source_truth",
    )
    normalized["silence_protocol"] = _normalize_silence_protocol(normalized["silence_protocol"])

    if not normalized["scope"]:
        raise ValueError("Approved Action Envelope requires scope")
    if not normalized["out_of_scope"]:
        raise ValueError("Approved Action Envelope requires out_of_scope")
    if not normalized["approval_boundaries"]:
        raise ValueError("Approved Action Envelope requires approval_boundaries")
    return normalized


def brief_packet_from_approved_action_envelope(envelope: Any) -> dict[str, Any]:
    validated = validate_approved_action_envelope(envelope)
    if any(key in validated for key in ("raw_chat", "chat_history", "conversation")):
        raise ValueError("Approved Action Envelope rejects raw chat input")

    brief = dict(validated["approved_brief"])
    packet = dict(brief)
    packet.setdefault("title", brief.get("title") or brief.get("objective") or "Approved production workflow brief")
    packet.setdefault("target repo or workspace", validated["repo_or_workspace"])
    packet.setdefault("target_repo_or_workspace", validated["repo_or_workspace"])
    packet["repo_or_workspace"] = validated["repo_or_workspace"]
    packet.setdefault("scope", list(validated["scope"]))
    packet.setdefault("out of scope", list(validated["out_of_scope"]))
    packet.setdefault("out_of_scope", list(validated["out_of_scope"]))
    packet.setdefault("acceptance criteria", list(validated["acceptance_criteria"]))
    packet.setdefault("acceptance_criteria", list(validated["acceptance_criteria"]))
    packet.setdefault("approval boundaries", list(validated["approval_boundaries"]))
    packet.setdefault("approval_boundaries", list(validated["approval_boundaries"]))
    packet.setdefault("stop conditions", list(validated["stop_conditions"]))
    packet.setdefault("stop_conditions", list(validated["stop_conditions"]))
    packet.setdefault("source_truth", list(validated["source_truth"]))
    packet["approved_by"] = validated["approved_by"]
    packet["approved_at"] = validated["approved_at"]
    packet["approval_phrase"] = validated["approval_phrase"]
    packet["priority_lane"] = validated["priority_lane"]
    packet["silence_protocol"] = dict(validated["silence_protocol"])
    packet["approved_action_envelope_id"] = validated["approved_action_envelope_id"]
    packet["idempotency_key"] = validated["idempotency_key"]
    return packet


def run_approved_action_envelope_autonomously(
    conn: sqlite3.Connection,
    envelope: Any,
    *,
    runner: Runner | None = None,
    max_steps: int = 20,
    max_retries: int = 1,
    timeout_seconds: int | None = None,
) -> dict[str, Any]:
    validated = validate_approved_action_envelope(envelope)
    brief_packet = brief_packet_from_approved_action_envelope(validated)
    frozen_source_brief = json.dumps(brief_packet, indent=2, sort_keys=True)
    production_order = _find_existing_order(conn, validated["idempotency_key"])
    if production_order is None:
        production_order = run_full_bridge(
            conn,
            title=str(brief_packet.get("title") or "Approved production workflow brief"),
            source_brief=frozen_source_brief,
            approved_by=validated["approved_by"],
            priority_lane=validated["priority_lane"],
            repo_or_workspace=validated["repo_or_workspace"],
            idempotency_key=validated["idempotency_key"],
        )

    refreshed_order = _load_production_order(conn, production_order.production_order_id)
    if refreshed_order.current_state in _TERMINAL_OR_PAUSE_STATES:
        run_result = _result_from_order(
            refreshed_order,
            steps_run=0,
            terminal_reason="already_terminal",
            applied_actions=[],
            errors=[],
        )
    else:
        run_result = run_production_order_autonomously(
            conn,
            production_order.production_order_id,
            runner=runner,
            max_steps=max_steps,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
        )
        refreshed_order = _load_production_order(conn, production_order.production_order_id)
    return {
        "approved_action_envelope_id": validated["approved_action_envelope_id"],
        "production_order_id": refreshed_order.production_order_id,
        "approved_action_envelope": validated,
        "brief_packet": brief_packet,
        "production_order": _production_order_metadata(refreshed_order),
        "production_run_result": run_result.to_dict(),
        "silence_protocol": dict(validated["silence_protocol"]),
    }


def _coerce_profile_task_envelope(envelope_payload: Any) -> ProfileTaskEnvelope:
    if isinstance(envelope_payload, ProfileTaskEnvelope):
        return envelope_payload
    if not isinstance(envelope_payload, dict):
        raise TypeError("hermes_profile_runner requires a ProfileTaskEnvelope payload dict")
    return ProfileTaskEnvelope(
        production_order_id=str(envelope_payload["production_order_id"]),
        dispatch_attempt=int(envelope_payload["dispatch_attempt"]),
        idempotency_key=str(envelope_payload["idempotency_key"]),
        parent_kanban_card_id=str(envelope_payload["parent_kanban_card_id"]),
        child_kanban_card_id=str(envelope_payload["child_kanban_card_id"]),
        target_profile=str(envelope_payload["target_profile"]),
        source_state=str(envelope_payload["source_state"]),
        expected_next_state=str(envelope_payload["expected_next_state"]),
        objective=str(envelope_payload["objective"]),
        source_truth=tuple(str(item) for item in (envelope_payload.get("source_truth") or ())),
        frozen_brief=str(envelope_payload["frozen_brief"]),
        input_packet=dict(envelope_payload["input_packet"]),
        expected_output_packet=dict(envelope_payload["expected_output_packet"]),
        acceptance_criteria=tuple(
            str(item) for item in (envelope_payload.get("acceptance_criteria") or ())
        ),
        stop_conditions=tuple(str(item) for item in (envelope_payload.get("stop_conditions") or ())),
        approval_boundaries=tuple(
            str(item) for item in (envelope_payload.get("approval_boundaries") or ())
        ),
        allowed_files_or_scope=envelope_payload.get("allowed_files_or_scope"),
        repo_or_workspace=envelope_payload.get("repo_or_workspace"),
    )


def _resolved_profile_runtime_config(hermes_home: Any) -> dict[str, str]:
    """Read the profile-scoped runtime model config before invoking AIAgent."""
    cfg = load_config()
    model_cfg = cfg.get("model") or {}
    if not isinstance(model_cfg, dict):
        model_cfg = {}
    model_default = str(
        model_cfg.get("model")
        or model_cfg.get("default")
        or model_cfg.get("name")
        or model_cfg.get("default_model")
        or ""
    ).strip()
    return {
        "resolved_hermes_home": str(hermes_home),
        "resolved_model_default": model_default,
        "resolved_model_provider": str(model_cfg.get("provider") or "").strip(),
        "resolved_model_base_url": str(model_cfg.get("base_url") or "").strip(),
    }


def _profile_runtime_config_error(runtime_config: dict[str, str]) -> str | None:
    missing: list[str] = []
    if not runtime_config.get("resolved_model_provider"):
        missing.append("provider")
    if not runtime_config.get("resolved_model_default"):
        missing.append("model")
    if not missing:
        return None
    return (
        "profile runtime config missing "
        + ", ".join(missing)
        + f"; resolved_hermes_home={runtime_config.get('resolved_hermes_home')!r} "
        + f"resolved_model_provider={runtime_config.get('resolved_model_provider')!r} "
        + f"resolved_model_default={runtime_config.get('resolved_model_default')!r} "
        + f"resolved_model_base_url={runtime_config.get('resolved_model_base_url')!r}"
    )


def _run_profile_agent_once(
    envelope: ProfileTaskEnvelope,
    *,
    timeout_seconds: int | None,
    runtime_session_id: str,
) -> dict[str, Any]:
    holder: dict[str, Any] = {}

    def _worker() -> dict[str, Any]:
        target_profile = normalize_profile_name(envelope.target_profile)
        hermes_home = resolve_profile_env(target_profile)
        previous_hermes_home_env = os.environ.get("HERMES_HOME")
        os.environ["HERMES_HOME"] = str(hermes_home)
        token = set_hermes_home_override(hermes_home)
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        runtime_config: dict[str, str] = {}
        try:
            runtime_config = _resolved_profile_runtime_config(hermes_home)
            config_error = _profile_runtime_config_error(runtime_config)
            if config_error:
                return {
                    "stdout": stdout_buffer.getvalue(),
                    "stderr": config_error,
                    "exit_code": 1,
                    "final_response": None,
                    **runtime_config,
                }
            try:
                with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
                    agent = AIAgent(
                        quiet_mode=True,
                        skip_context_files=False,
                        load_soul_identity=True,
                        skip_memory=True,
                        platform="production_order_runtime",
                        session_id=runtime_session_id,
                    )
                    holder["agent"] = agent
                    result = agent.run_conversation(_build_profile_runtime_prompt(envelope))
            except Exception:
                return {
                    "stdout": stdout_buffer.getvalue(),
                    "stderr": traceback.format_exc(),
                    "exit_code": 1,
                    "final_response": None,
                    **runtime_config,
                }
            return {
                "stdout": stdout_buffer.getvalue(),
                "stderr": stderr_buffer.getvalue(),
                "exit_code": 0,
                "final_response": result.get("final_response"),
                **runtime_config,
            }
        finally:
            holder.pop("agent", None)
            reset_hermes_home_override(token)
            if previous_hermes_home_env is None:
                os.environ.pop("HERMES_HOME", None)
            else:
                os.environ["HERMES_HOME"] = previous_hermes_home_env

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_worker)
    try:
        if timeout_seconds is None:
            return future.result()
        return future.result(timeout=timeout_seconds)
    except concurrent.futures.TimeoutError as exc:
        agent = holder.get("agent")
        if agent is not None:
            with contextlib.suppress(Exception):
                agent.interrupt("Production order profile runtime timed out")
        future.cancel()
        raise TimeoutError(
            f"profile runtime exceeded timeout_seconds={timeout_seconds}"
        ) from exc
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _build_profile_runtime_prompt(envelope: ProfileTaskEnvelope) -> str:
    contract = {
        "task": "Return exactly one JSON object result packet and no surrounding prose.",
        "target_profile": envelope.target_profile,
        "production_order_id": envelope.production_order_id,
        "source_state": envelope.source_state,
        "expected_next_state": envelope.expected_next_state,
        "objective": envelope.objective,
        "source_truth": list(envelope.source_truth),
        "acceptance_criteria": list(envelope.acceptance_criteria),
        "stop_conditions": list(envelope.stop_conditions),
        "approval_boundaries": list(envelope.approval_boundaries),
        "repo_or_workspace": envelope.repo_or_workspace,
        "allowed_files_or_scope": envelope.allowed_files_or_scope,
        "input_packet": envelope.input_packet,
        "expected_output_packet": envelope.expected_output_packet,
        "frozen_brief": envelope.frozen_brief,
        "rules": [
            "Accept exactly one profile task envelope.",
            "Use the explicit target profile already selected.",
            "FINAL ANSWER REQUIREMENTS: The final assistant output MUST be exactly one raw JSON object only — no surrounding prose, no explanation, no Markdown, no code fences, and no leading or trailing text.",
            "The JSON must be a single JSON object (top-level {}). It must parse with json.loads(...) without error.",
            "The JSON object must match the expected result packet schema (fields/semantics) for this task and MUST NOT include workflow-mutation fields such as 'current_owner_profile' or 'current_state'.",
            "If you are blocked or cannot complete, still RETURN exactly one well-formed JSON object using the rejected/blocked result packet shape compatible with validation (do not return free-text).",
            "Do not mutate production-order state directly.",
            "Do not rely on chat memory.",
            "Do not perform hidden retries.",
        ],
        "final_output_instructions": (
            "Return only the single JSON object. Any deviation (multiple JSON objects, malformed JSON, or any non-JSON text) will be treated as a rejection."
        ),
    }
    return json.dumps(contract, indent=2, sort_keys=True)


def _adapter_error_result(
    envelope_payload: Any,
    *,
    error: str,
    started_at: float,
    log_ref: str,
    stderr: str = "",
) -> dict[str, Any]:
    return {
        "stdout": "",
        "stderr": stderr or error,
        "exit_code": 1,
        "duration_ms": int((time.monotonic() - started_at) * 1000),
        "log_ref": log_ref,
        "error": error,
        "envelope_payload": envelope_payload,
    }


def _stringify_result_channel(result_channel: Any) -> str:
    if isinstance(result_channel, str):
        return result_channel
    return json.dumps(result_channel, sort_keys=True)


def _normalize_required_list(value: list[str] | tuple[str, ...] | str, *, field_name: str) -> list[str]:
    if isinstance(value, str):
        items = [value.strip()] if value.strip() else []
    elif isinstance(value, (list, tuple)):
        items = [str(item).strip() for item in value if str(item).strip()]
    else:
        raise ValueError(f"Approved Action Envelope field {field_name} must be a string or list")
    return items


def _normalize_silence_protocol(value: dict[str, Any] | str) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    text = str(value or "").strip()
    if not text:
        return {}
    return {"mode": text}


def _production_order_metadata(production_order: ProductionOrder) -> dict[str, Any]:
    return {
        "production_order_id": production_order.production_order_id,
        "title": production_order.title,
        "current_state": production_order.current_state,
        "current_owner_profile": production_order.current_owner_profile,
        "repo_or_workspace": production_order.repo_or_workspace,
        "parent_kanban_card_id": production_order.parent_kanban_card_id,
        "child_kanban_card_ids": list(production_order.child_kanban_card_ids),
    }
