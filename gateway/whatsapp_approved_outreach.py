from __future__ import annotations

import json
import os
import shlex
import threading
from pathlib import Path
from typing import Any
from uuid import uuid4

from hermes_constants import get_hermes_home

from gateway.whatsapp_message_store import (
    query_whatsapp_records_any_time,
    utc_isoformat,
    utc_now,
)
from gateway.whatsapp_transcript_summary import ResolveWhatsAppExactTarget

WHATSAPP_APPROVED_OUTREACH_EXACT_SELECTOR_FIELDS = (
    "conversation_key",
    "destination_key",
    "group_chat_id",
    "dm_counterparty_id",
)
WHATSAPP_APPROVED_OUTREACH_ALLOWED_FIELDS = (
    *WHATSAPP_APPROVED_OUTREACH_EXACT_SELECTOR_FIELDS,
    "operator_objective",
    "message_text",
)
_OUTREACH_PREFIXES = (
    "whatsapp outreach",
    "whatsapp-outreach",
)
WHATSAPP_OUTREACH_STATE_SCHEMA_VERSION = 1

_OUTREACH_STATE_LOCK = threading.Lock()


def is_whatsapp_approved_outreach_instruction(text: str | None) -> bool:
    normalized = str(text or "").strip().lower()
    return any(
        normalized == prefix or normalized.startswith(f"{prefix} ")
        for prefix in _OUTREACH_PREFIXES
    )


def parse_whatsapp_approved_outreach_instruction(text: str | None) -> dict[str, Any]:
    raw_text = str(text or "").strip()
    lowered = raw_text.lower()
    prefix = next(
        (
            candidate
            for candidate in _OUTREACH_PREFIXES
            if lowered == candidate or lowered.startswith(f"{candidate} ")
        ),
        None,
    )
    if prefix is None:
        raise ValueError("instruction does not match WhatsApp approved-outreach syntax")

    raw_args = raw_text[len(prefix) :].strip()
    if not raw_args:
        return {}

    parsed: dict[str, Any] = {}
    for token in shlex.split(raw_args):
        if "=" not in token:
            raise ValueError("instruction arguments must use field=value syntax")
        field, value = token.split("=", 1)
        normalized_field = field.strip()
        if normalized_field not in WHATSAPP_APPROVED_OUTREACH_ALLOWED_FIELDS:
            raise ValueError(f"unsupported field: {normalized_field}")
        parsed[normalized_field] = value.strip()
    return parsed


def normalize_whatsapp_approved_outreach_request(
    request: dict[str, Any] | None,
) -> dict[str, Any]:
    raw_request = request or {}
    normalized = {
        field: (str(raw_request.get(field) or "").strip() or None)
        for field in WHATSAPP_APPROVED_OUTREACH_EXACT_SELECTOR_FIELDS
    }
    selected_fields = [field for field, value in normalized.items() if value]
    if len(selected_fields) != 1:
        raise ValueError("approved outreach requires exactly one exact selector")

    operator_objective = str(raw_request.get("operator_objective") or "").strip()
    if not operator_objective:
        raise ValueError("operator_objective is required")

    message_text = str(raw_request.get("message_text") or "").strip() or None
    return {
        **normalized,
        "operator_objective": operator_objective,
        "message_text": message_text,
        "trigger_source": "owner_instruction",
    }


def _outreach_store_path(*, base_dir: Path | None = None) -> Path:
    hermes_home = base_dir or get_hermes_home()
    return hermes_home / "gateway" / "whatsapp-approved-outreach-state.json"


def _default_outreach_state() -> dict[str, Any]:
    return {
        "schema_version": WHATSAPP_OUTREACH_STATE_SCHEMA_VERSION,
        "plans": [],
        "plan_targets": [],
        "runs": [],
        "target_executions": [],
        "reports": [],
    }


def _normalized_outreach_state(raw_state: dict[str, Any] | None) -> dict[str, Any]:
    state = _default_outreach_state()
    if not isinstance(raw_state, dict):
        return state

    schema_version = raw_state.get("schema_version")
    if isinstance(schema_version, int) and schema_version > 0:
        state["schema_version"] = schema_version

    for field in ("plans", "plan_targets", "runs", "target_executions", "reports"):
        value = raw_state.get(field)
        if isinstance(value, list):
            state[field] = [item for item in value if isinstance(item, dict)]
    return state


def load_whatsapp_outreach_state(*, base_dir: Path | None = None) -> dict[str, Any]:
    path = _outreach_store_path(base_dir=base_dir)
    if not path.exists():
        return _default_outreach_state()

    with path.open("r", encoding="utf-8") as handle:
        return _normalized_outreach_state(json.load(handle))


def _write_whatsapp_outreach_state(
    state: dict[str, Any], *, base_dir: Path | None = None
) -> Path:
    path = _outreach_store_path(base_dir=base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(state, ensure_ascii=False, indent=2) + "\n"
    with path.open("w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return path


def _find_first(
    rows: list[dict[str, Any]], field_name: str, expected: str | None
) -> dict[str, Any] | None:
    for row in rows:
        if row.get(field_name) == expected:
            return row
    return None


def _exact_selector_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        field: (str(row.get(field) or "").strip() or None)
        for field in WHATSAPP_APPROVED_OUTREACH_EXACT_SELECTOR_FIELDS
    }


def _resolved_target_from_execution(execution: dict[str, Any]) -> dict[str, Any]:
    return {
        "conversation_key": execution.get("resolved_conversation_key"),
        "destination_key": execution.get("resolved_destination_key"),
        "group_chat_id": execution.get("group_chat_id"),
        "dm_counterparty_id": execution.get("dm_counterparty_id"),
        "destination_context_type": execution.get("destination_context_type"),
        "destination_chat_id": execution.get("resolved_destination_chat_id"),
        "destination_target_id": execution.get("destination_target_id"),
    }


def _enriched_plan(
    state: dict[str, Any], plan: dict[str, Any] | None
) -> dict[str, Any] | None:
    if not isinstance(plan, dict):
        return None
    enriched = dict(plan)
    enriched["approved_targets"] = [
        dict(target)
        for target in state["plan_targets"]
        if target.get("plan_id") == plan.get("plan_id")
    ]
    return enriched


def _enriched_execution(execution: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(execution)
    enriched["resolved_target"] = _resolved_target_from_execution(execution)
    return enriched


def _enriched_run(
    state: dict[str, Any], run: dict[str, Any] | None
) -> dict[str, Any] | None:
    if not isinstance(run, dict):
        return None
    enriched = dict(run)
    enriched["target_executions"] = [
        _enriched_execution(execution)
        for execution in state["target_executions"]
        if execution.get("run_id") == run.get("run_id")
    ]
    return enriched


def _matching_plan_and_target(
    state: dict[str, Any],
    *,
    operator_objective: str,
    approved_by_principal: str,
    selector: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    for plan_target in reversed(state["plan_targets"]):
        if plan_target.get("target_status") != "active":
            continue
        if _exact_selector_from_row(plan_target) != selector:
            continue
        plan = _find_first(state["plans"], "plan_id", plan_target.get("plan_id"))
        if plan is None:
            continue
        if plan.get("approved_by_principal") != approved_by_principal:
            continue
        if plan.get("operator_objective") != operator_objective:
            continue
        if plan.get("trigger_mode") != "instruction_only":
            continue
        if plan.get("plan_status") not in {"approved", "active"}:
            continue
        return plan, plan_target
    return None, None


def _observable_status_for_execution_status(execution_status: str) -> str:
    if execution_status == "sent":
        return "awaiting_reply"
    if execution_status == "send_failed":
        return "send_failed"
    return "unresolved"


def _status_basis_for_execution_status(execution_status: str) -> str:
    if execution_status == "sent":
        return "bounded_model_synthesis"
    return "direct_conversation_evidence"


def _persist_whatsapp_outreach_run_record(record: dict[str, Any]) -> Path:
    with _OUTREACH_STATE_LOCK:
        state = load_whatsapp_outreach_state()
        plan = record.get("plan") if isinstance(record, dict) else None
        run = record.get("run") if isinstance(record, dict) else None
        report = record.get("report") if isinstance(record, dict) else None
        if isinstance(plan, dict):
            plan_targets = plan.get("approved_targets") or []
            state["plans"] = [
                row
                for row in state["plans"]
                if row.get("plan_id") != plan.get("plan_id")
            ]
            state["plans"].append({
                key: value for key, value in plan.items() if key != "approved_targets"
            })
            if isinstance(plan_targets, list):
                target_ids = {
                    target.get("plan_target_id")
                    for target in plan_targets
                    if isinstance(target, dict)
                }
                state["plan_targets"] = [
                    row
                    for row in state["plan_targets"]
                    if row.get("plan_target_id") not in target_ids
                ]
                state["plan_targets"].extend(
                    dict(target) for target in plan_targets if isinstance(target, dict)
                )
        if isinstance(run, dict):
            target_executions = run.get("target_executions") or []
            state["runs"] = [
                row for row in state["runs"] if row.get("run_id") != run.get("run_id")
            ]
            state["runs"].append({
                key: value for key, value in run.items() if key != "target_executions"
            })
            if isinstance(target_executions, list):
                execution_ids = {
                    execution.get("target_execution_id")
                    for execution in target_executions
                    if isinstance(execution, dict)
                }
                state["target_executions"] = [
                    row
                    for row in state["target_executions"]
                    if row.get("target_execution_id") not in execution_ids
                ]
                for execution in target_executions:
                    if not isinstance(execution, dict):
                        continue
                    execution_row = dict(execution)
                    resolved_target = execution_row.pop("resolved_target", None)
                    if isinstance(resolved_target, dict):
                        execution_row.setdefault(
                            "resolved_conversation_key",
                            resolved_target.get("conversation_key"),
                        )
                        execution_row.setdefault(
                            "resolved_destination_key",
                            resolved_target.get("destination_key"),
                        )
                        execution_row.setdefault(
                            "resolved_destination_chat_id",
                            resolved_target.get("destination_chat_id"),
                        )
                        execution_row.setdefault(
                            "group_chat_id", resolved_target.get("group_chat_id")
                        )
                        execution_row.setdefault(
                            "dm_counterparty_id",
                            resolved_target.get("dm_counterparty_id"),
                        )
                        execution_row.setdefault(
                            "destination_context_type",
                            resolved_target.get("destination_context_type"),
                        )
                        execution_row.setdefault(
                            "destination_target_id",
                            resolved_target.get("destination_target_id"),
                        )
                    state["target_executions"].append(execution_row)
        if isinstance(report, dict):
            state["reports"] = [
                row
                for row in state["reports"]
                if row.get("report_id") != report.get("report_id")
            ]
            state["reports"].append(dict(report))
        return _write_whatsapp_outreach_state(state)


def append_whatsapp_outreach_run_record(record: dict[str, Any]) -> Path:
    return _persist_whatsapp_outreach_run_record(record)


def load_whatsapp_outreach_run_records(
    *, base_dir: Path | None = None
) -> list[dict[str, Any]]:
    state = load_whatsapp_outreach_state(base_dir=base_dir)
    results: list[dict[str, Any]] = []
    for run in state["runs"]:
        plan = _find_first(state["plans"], "plan_id", run.get("plan_id"))
        report = _find_first(state["reports"], "report_id", run.get("report_id"))
        results.append({
            "recorded_at_utc": run.get("run_completed_at_utc")
            or run.get("run_started_at_utc"),
            "plan": _enriched_plan(state, plan),
            "run": _enriched_run(state, run),
            "report": dict(report) if isinstance(report, dict) else None,
        })
    return results


def _result(
    *,
    workflow_status: str,
    plan: dict[str, Any] | None = None,
    run: dict[str, Any] | None = None,
    execution: dict[str, Any] | None = None,
    founder_summary: str = "",
    reason: str | None = None,
) -> dict[str, Any]:
    return {
        "workflow_status": workflow_status,
        "plan": plan,
        "run": run,
        "execution": execution,
        "founder_summary": founder_summary,
        "reason": reason,
    }


async def execute_whatsapp_approved_outreach(
    request: dict[str, Any] | None,
    *,
    authorized: bool,
    adapter: Any,
) -> dict[str, Any]:
    if not authorized:
        return _result(
            workflow_status="forbidden",
            founder_summary=(
                "WhatsApp approved outreach stayed blocked because this request "
                "was not owner-authorized."
            ),
            reason="owner authorization required",
        )

    try:
        normalized_request = normalize_whatsapp_approved_outreach_request(request)
    except ValueError as exc:
        return _result(
            workflow_status="invalid_request",
            founder_summary=(
                "WhatsApp approved outreach could not start because the "
                "instruction did not provide one exact target plus an "
                "operator objective."
            ),
            reason=str(exc),
        )

    resolution = ResolveWhatsAppExactTarget(normalized_request)
    resolved_target = resolution.get("resolved_target")
    if resolution.get("resolution_status") != "resolved" or not isinstance(
        resolved_target, dict
    ):
        return _result(
            workflow_status="blocked",
            founder_summary=(
                "WhatsApp approved outreach stayed blocked because the exact "
                "approved target could not be resolved from preserved history."
            ),
            reason="exact target could not be resolved from preserved history",
        )

    continuity_rows = query_whatsapp_records_any_time(
        conversation_key=resolved_target.get("conversation_key"),
        destination_key=resolved_target.get("destination_key"),
        destination_context_type=resolved_target.get("destination_context_type"),
        group_chat_id=resolved_target.get("group_chat_id"),
        dm_counterparty_id=resolved_target.get("dm_counterparty_id"),
    )
    history_window_start_utc = (
        continuity_rows[0].get("effective_event_at_utc") if continuity_rows else None
    )
    history_window_end_utc = (
        continuity_rows[-1].get("effective_event_at_utc") if continuity_rows else None
    )
    if history_window_start_utc is None:
        history_window_start_utc = utc_isoformat(utc_now())
    if history_window_end_utc is None:
        history_window_end_utc = history_window_start_utc

    approved_by_principal = "owner_operator"
    plan_selector = _exact_selector_from_row(resolved_target)
    run_started_at_utc = utc_isoformat(utc_now())

    with _OUTREACH_STATE_LOCK:
        state = load_whatsapp_outreach_state()
        plan_row, plan_target_row = _matching_plan_and_target(
            state,
            operator_objective=normalized_request["operator_objective"],
            approved_by_principal=approved_by_principal,
            selector=plan_selector,
        )

        approved_at_utc = utc_isoformat(utc_now())
        if plan_row is None or plan_target_row is None:
            plan_id = f"waplan-{uuid4()}"
            plan_target_id = f"watarget-{uuid4()}"
            plan_row = {
                "plan_id": plan_id,
                "plan_status": "active",
                "trigger_mode": "instruction_only",
                "operator_objective": normalized_request["operator_objective"],
                "created_by_session_id": None,
                "approved_by_principal": approved_by_principal,
                "approved_at_utc": approved_at_utc,
                "report_delivery_policy": "initiating_session",
                "default_report_cadence": "end_of_run",
                "linked_cron_job_id": None,
            }
            plan_target_row = {
                "plan_target_id": plan_target_id,
                "plan_id": plan_id,
                "target_status": "active",
                **plan_selector,
                "target_objective_override": None,
                "max_outbound_messages_per_run": 1,
                "last_resolved_conversation_key": resolved_target.get(
                    "conversation_key"
                ),
                "last_observed_message_at_utc": history_window_end_utc,
            }
            state["plans"].append(plan_row)
            state["plan_targets"].append(plan_target_row)
        else:
            plan_row["plan_status"] = "active"
            plan_row["approved_at_utc"] = (
                plan_row.get("approved_at_utc") or approved_at_utc
            )
            plan_target_row.update({
                **plan_selector,
                "last_resolved_conversation_key": resolved_target.get(
                    "conversation_key"
                ),
                "last_observed_message_at_utc": history_window_end_utc,
            })

        run_id = f"warun-{uuid4()}"
        target_execution_id = f"waexec-{uuid4()}"
        run = {
            "run_id": run_id,
            "plan_id": plan_row["plan_id"],
            "run_status": "running",
            "trigger_source": "owner_instruction",
            "trigger_reference_id": None,
            "run_started_at_utc": run_started_at_utc,
            "run_completed_at_utc": None,
            "target_count": 1,
            "completed_target_count": 0,
            "failed_target_count": 0,
            "report_delivery_target": None,
            "report_id": None,
        }
        execution = {
            "target_execution_id": target_execution_id,
            "run_id": run_id,
            "plan_target_id": plan_target_row["plan_target_id"],
            "execution_status": "resolved",
            "resolved_conversation_key": resolved_target.get("conversation_key"),
            "resolved_destination_key": resolved_target.get("destination_key"),
            "resolved_destination_chat_id": resolved_target.get("destination_chat_id"),
            "group_chat_id": resolved_target.get("group_chat_id"),
            "dm_counterparty_id": resolved_target.get("dm_counterparty_id"),
            "destination_context_type": resolved_target.get("destination_context_type"),
            "destination_target_id": resolved_target.get("destination_target_id"),
            "history_window_start_utc": history_window_start_utc,
            "history_window_end_utc": history_window_end_utc,
            "history_record_count": len(continuity_rows),
            "dispatch_group_id": None,
            "message_id": None,
            "last_error": None,
        }
        state["plans"] = [
            row for row in state["plans"] if row.get("plan_id") != plan_row["plan_id"]
        ]
        state["plans"].append(dict(plan_row))
        state["plan_targets"] = [
            row
            for row in state["plan_targets"]
            if row.get("plan_target_id") != plan_target_row["plan_target_id"]
        ]
        state["plan_targets"].append(dict(plan_target_row))
        state["runs"] = [
            row for row in state["runs"] if row.get("run_id") != run["run_id"]
        ]
        state["runs"].append(dict(run))
        state["target_executions"] = [
            row
            for row in state["target_executions"]
            if row.get("target_execution_id") != execution["target_execution_id"]
        ]
        state["target_executions"].append(dict(execution))
        _write_whatsapp_outreach_state(state)

    if adapter is None:
        execution["execution_status"] = "blocked"
        execution["last_error"] = "WhatsApp adapter unavailable"
        run["run_status"] = "blocked"
        founder_summary = (
            "WhatsApp approved outreach stayed blocked because the WhatsApp "
            "adapter was unavailable."
        )
    elif normalized_request["message_text"] is None:
        execution["execution_status"] = "no_send_required"
        run["run_status"] = "completed"
        founder_summary = (
            "WhatsApp approved outreach resolved one exact approved target, "
            "loaded preserved continuity, and determined that no send was "
            "needed now."
        )
    else:
        send_result = await adapter.send(
            str(resolved_target["destination_chat_id"]),
            normalized_request["message_text"],
        )
        raw_response = getattr(send_result, "raw_response", None) or {}
        if isinstance(raw_response, dict):
            execution["dispatch_group_id"] = raw_response.get("dispatch_group_id")
        execution["message_id"] = getattr(send_result, "message_id", None)

        if getattr(send_result, "success", False):
            execution["execution_status"] = "sent"
            run["run_status"] = "completed"
            founder_summary = (
                "WhatsApp approved outreach resolved one exact approved target, "
                "used preserved continuity, and sent one logical outbound "
                "follow-up through the existing WhatsApp adapter boundary."
            )
        else:
            execution["execution_status"] = "send_failed"
            execution["last_error"] = getattr(send_result, "error", None)
            run["run_status"] = "failed"
            founder_summary = (
                "WhatsApp approved outreach resolved one exact approved target "
                "but the bounded outbound send failed."
            )

    run["run_completed_at_utc"] = utc_isoformat(utc_now())
    run["completed_target_count"] = 1
    run["failed_target_count"] = (
        1 if execution["execution_status"] in {"blocked", "send_failed"} else 0
    )

    report = {
        "report_id": f"wareport-{uuid4()}",
        "plan_id": plan_row["plan_id"],
        "run_id": run_id,
        "report_status": "ready" if run["run_status"] == "completed" else "partial",
        "report_window_start_utc": history_window_start_utc,
        "report_window_end_utc": history_window_end_utc,
        "report_text": founder_summary,
        "target_rows": [
            {
                "plan_target_id": plan_target_row["plan_target_id"],
                "resolved_target": resolved_target,
                "observable_status": _observable_status_for_execution_status(
                    execution["execution_status"]
                ),
                "status_basis": _status_basis_for_execution_status(
                    execution["execution_status"]
                ),
                "latest_observed_message_at_utc": history_window_end_utc,
                "open_items": [],
                "uncertainties": [],
            }
        ],
        "uncertainties": [],
    }
    run["report_id"] = report["report_id"]

    with _OUTREACH_STATE_LOCK:
        state = load_whatsapp_outreach_state()
        persisted_plan = _find_first(state["plans"], "plan_id", plan_row["plan_id"])
        persisted_target = _find_first(
            state["plan_targets"], "plan_target_id", plan_target_row["plan_target_id"]
        )
        if persisted_plan is None or persisted_target is None:
            state["plans"].append(dict(plan_row))
            state["plan_targets"].append(dict(plan_target_row))
            persisted_plan = state["plans"][-1]
            persisted_target = state["plan_targets"][-1]
        else:
            persisted_plan.update(dict(plan_row))
            persisted_target.update({
                **dict(plan_target_row),
                "last_resolved_conversation_key": resolved_target.get(
                    "conversation_key"
                ),
                "last_observed_message_at_utc": history_window_end_utc,
            })
        persisted_run = _find_first(state["runs"], "run_id", run["run_id"])
        if persisted_run is None:
            state["runs"].append(dict(run))
            persisted_run = state["runs"][-1]
        else:
            persisted_run.update(dict(run))
        persisted_execution = _find_first(
            state["target_executions"],
            "target_execution_id",
            execution["target_execution_id"],
        )
        if persisted_execution is None:
            state["target_executions"].append(dict(execution))
            persisted_execution = state["target_executions"][-1]
        else:
            persisted_execution.update(dict(execution))
        state["reports"] = [
            row
            for row in state["reports"]
            if row.get("report_id") != report["report_id"]
        ]
        state["reports"].append(dict(report))
        _write_whatsapp_outreach_state(state)

        plan = _enriched_plan(state, persisted_plan) or {}
        run = _enriched_run(state, persisted_run) or {}
        execution = _enriched_execution(persisted_execution)

    return _result(
        workflow_status="ready",
        plan=plan,
        run=run,
        execution=execution,
        founder_summary=founder_summary,
        reason=execution.get("last_error"),
    )


def format_whatsapp_approved_outreach_result(result: dict[str, Any]) -> str:
    lines = [
        "WhatsApp approved outreach",
        f"Status: {result.get('workflow_status', 'invalid_request')}",
    ]
    workflow_status = result.get("workflow_status")
    if workflow_status != "ready":
        if result.get("reason"):
            lines.append(f"Reason: {result['reason']}")
        if result.get("founder_summary"):
            lines.append(result["founder_summary"])
        return "\n".join(lines)

    plan = result.get("plan") or {}
    run = result.get("run") or {}
    execution = result.get("execution") or {}
    resolved_target = execution.get("resolved_target") or {}

    lines.extend([
        f"plan_id: {plan.get('plan_id')}",
        f"run_id: {run.get('run_id')}",
        f"trigger_source: {run.get('trigger_source')}",
        f"operator_objective: {plan.get('operator_objective')}",
        f"run_status: {run.get('run_status')}",
        f"execution_status: {execution.get('execution_status')}",
        f"destination_chat_id: {resolved_target.get('destination_chat_id')}",
        f"conversation_key: {resolved_target.get('conversation_key')}",
        f"destination_key: {resolved_target.get('destination_key')}",
        f"group_chat_id: {resolved_target.get('group_chat_id')}",
        f"dm_counterparty_id: {resolved_target.get('dm_counterparty_id')}",
        f"history_window_start_utc: {execution.get('history_window_start_utc')}",
        f"history_window_end_utc: {execution.get('history_window_end_utc')}",
        f"history_record_count: {execution.get('history_record_count')}",
        f"dispatch_group_id: {execution.get('dispatch_group_id')}",
        f"message_id: {execution.get('message_id')}",
    ])
    if execution.get("last_error"):
        lines.append(f"last_error: {execution.get('last_error')}")
    if result.get("founder_summary"):
        lines.extend(["", result["founder_summary"]])
    return "\n".join(lines)


__all__ = [
    "WHATSAPP_APPROVED_OUTREACH_ALLOWED_FIELDS",
    "WHATSAPP_APPROVED_OUTREACH_EXACT_SELECTOR_FIELDS",
    "WHATSAPP_OUTREACH_STATE_SCHEMA_VERSION",
    "append_whatsapp_outreach_run_record",
    "execute_whatsapp_approved_outreach",
    "format_whatsapp_approved_outreach_result",
    "is_whatsapp_approved_outreach_instruction",
    "load_whatsapp_outreach_state",
    "load_whatsapp_outreach_run_records",
    "normalize_whatsapp_approved_outreach_request",
    "parse_whatsapp_approved_outreach_instruction",
]
