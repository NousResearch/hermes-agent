from __future__ import annotations

import json
import shlex
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


def _outreach_store_path() -> Path:
    return get_hermes_home() / "gateway" / "whatsapp-approved-outreach-runs.jsonl"


def append_whatsapp_outreach_run_record(record: dict[str, Any]) -> Path:
    path = _outreach_store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def load_whatsapp_outreach_run_records(
    *, base_dir: Path | None = None
) -> list[dict[str, Any]]:
    if base_dir is None:
        path = _outreach_store_path()
    else:
        path = base_dir / "gateway" / "whatsapp-approved-outreach-runs.jsonl"
    if not path.exists():
        return []

    results: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            results.append(json.loads(line))
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

    plan_id = f"waplan-{uuid4()}"
    plan_target_id = f"watarget-{uuid4()}"
    run_id = f"warun-{uuid4()}"
    target_execution_id = f"waexec-{uuid4()}"

    plan = {
        "plan_id": plan_id,
        "plan_status": "approved",
        "trigger_mode": "instruction_only",
        "operator_objective": normalized_request["operator_objective"],
        "approved_by_principal": "owner_operator",
        "report_delivery_policy": "initiating_session",
        "approved_targets": [
            {
                "plan_target_id": plan_target_id,
                "conversation_key": resolved_target.get("conversation_key"),
                "destination_key": resolved_target.get("destination_key"),
                "group_chat_id": resolved_target.get("group_chat_id"),
                "dm_counterparty_id": resolved_target.get("dm_counterparty_id"),
                "target_status": "active",
                "max_outbound_messages_per_run": 1,
            }
        ],
    }
    run = {
        "run_id": run_id,
        "plan_id": plan_id,
        "run_status": "running",
        "trigger_source": "owner_instruction",
        "trigger_reference_id": None,
    }
    execution = {
        "target_execution_id": target_execution_id,
        "plan_target_id": plan_target_id,
        "execution_status": "resolved",
        "resolved_target": resolved_target,
        "history_window_start_utc": history_window_start_utc,
        "history_window_end_utc": history_window_end_utc,
        "history_record_count": len(continuity_rows),
        "dispatch_group_id": None,
        "message_id": None,
        "last_error": None,
    }

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

    persisted_record = {
        "recorded_at_utc": utc_isoformat(utc_now()),
        "plan": plan,
        "run": {
            **run,
            "target_executions": [execution],
        },
        "report": {
            "report_id": f"wareport-{uuid4()}",
            "plan_id": plan_id,
            "run_id": run_id,
            "report_status": (
                "ready" if run["run_status"] == "completed" else "partial"
            ),
            "report_window_start_utc": history_window_start_utc,
            "report_window_end_utc": history_window_end_utc,
            "report_text": founder_summary,
            "target_rows": [
                {
                    "plan_target_id": plan_target_id,
                    "resolved_target": resolved_target,
                    "observable_status": (
                        "awaiting_reply"
                        if execution["execution_status"] == "sent"
                        else "unresolved"
                        if execution["execution_status"] == "no_send_required"
                        else "send_failed"
                    ),
                    "status_basis": (
                        "direct_conversation_evidence"
                        if execution["execution_status"] == "no_send_required"
                        else "bounded_model_synthesis"
                        if execution["execution_status"] == "sent"
                        else "direct_conversation_evidence"
                    ),
                    "latest_observed_message_at_utc": history_window_end_utc,
                    "open_items": [],
                    "uncertainties": [],
                }
            ],
            "uncertainties": [],
        },
    }
    append_whatsapp_outreach_run_record(persisted_record)

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
    "append_whatsapp_outreach_run_record",
    "execute_whatsapp_approved_outreach",
    "format_whatsapp_approved_outreach_result",
    "is_whatsapp_approved_outreach_instruction",
    "load_whatsapp_outreach_run_records",
    "normalize_whatsapp_approved_outreach_request",
    "parse_whatsapp_approved_outreach_instruction",
]
