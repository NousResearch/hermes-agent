"""Telegram KB journey renderer plugin.

Intercepts a small set of Telegram slash commands and renders concise,
read-only KB status, sync, and review receipts from the configured KB MCP
target.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import hashlib
import inspect
import json
import logging
import os
import re
import time
from types import SimpleNamespace
from typing import Any, Callable, Iterable

logger = logging.getLogger(__name__)

DEFAULT_MCP_TARGET = "kb_engine_prod"
MENU_COMMANDS = {"kb"}
LEGACY_COMMANDS = {"kbtoday", "kbstatus", "kbruns", "kbqueue", "kbreview", "kbrun", "kbsync"}
SUPPORTED_COMMANDS = MENU_COMMANDS
KB_REASONING_LEVELS = {"none", "minimal", "low", "medium", "high", "xhigh"}
QUEUE_REPLY_DECISIONS = {"approve", "reject", "archive", "skip", "complete", "keep", "demote", "detail"}
QUEUE_REPLY_TOOL_DECISIONS = {"approve", "reject", "archive", "skip", "complete", "keep", "demote"}
QUEUE_REPLY_STATE_TTL_SECONDS = 15 * 60
QUEUE_SCOPE_STATE_TTL_SECONDS = 15 * 60
MEETING_HANDOFF_STATE_TTL_SECONDS = 15 * 60
SYNC_PREVIEW_STATE_TTL_SECONDS = 15 * 60
SEMANTIC_WRITE_RECEIPT_PACKET_TYPES = {
    "semantic_write_receipt",
    "semantic_write_through_receipt",
    "semantic_write_through.receipt",
    "semantic_write.receipt",
    "kb_semantic_write_receipt",
}
SEMANTIC_WRITE_SHADOW_PACKET_TYPES = {
    "semantic_write_through.shadow_preview",
}
SUPPORTED_RESULT_PACKET_TYPES = {
    "durable_graph_validation",
    "lifecycle_proposal_draft.packet",
    "lifecycle_review.packet",
    "lifecycle_update.packet",
    "publication_observation",
    "request.receipt",
    "report_admission_receipt",
    *SEMANTIC_WRITE_RECEIPT_PACKET_TYPES,
    *SEMANTIC_WRITE_SHADOW_PACKET_TYPES,
}
DESCRIPTOR_READONLY_TARGET_KINDS = {
    "closeout",
    "component",
    "dashboard_surface",
    "event",
    "lifecycle_candidate",
    "object_graph",
    "proposal_queue",
    "publication",
    "receipt",
    "report",
    "run",
    "situation",
    "sync",
    "todo",
}
DESCRIPTOR_WRITE_TARGET_KINDS = DESCRIPTOR_READONLY_TARGET_KINDS.difference(
    {"dashboard_surface", "lifecycle_candidate", "run"}
)


def _sanitize_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", str(value or ""))


def _mcp_target() -> str:
    return os.getenv("HERMES_KB_MCP_TARGET", DEFAULT_MCP_TARGET).strip() or DEFAULT_MCP_TARGET


def _mcp_tool_name(target: str, tool_name: str) -> str:
    return f"mcp_{_sanitize_component(target)}_{_sanitize_component(tool_name)}"


def _queue_reply_state_path():
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "state" / "kb_queue_reply_state.json"


def _queue_scope_state_path():
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "state" / "kb_queue_scope_state.json"


def _meeting_handoff_state_path():
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "state" / "kb_meeting_handoff_state.json"


def _sync_preview_state_path():
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "state" / "kb_sync_preview_state.json"


def _load_queue_reply_states() -> dict[str, Any]:
    path = _queue_reply_state_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_queue_reply_states(states: dict[str, Any]) -> None:
    path = _queue_reply_state_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(states, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    except OSError:
        logger.debug("kb_journeys: failed to persist iterative queue state", exc_info=True)


def _load_queue_scope_states() -> dict[str, Any]:
    path = _queue_scope_state_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_queue_scope_states(states: dict[str, Any]) -> None:
    path = _queue_scope_state_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(states, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    except OSError:
        logger.debug("kb_journeys: failed to persist queue scope state", exc_info=True)


def _load_meeting_handoff_states() -> dict[str, Any]:
    path = _meeting_handoff_state_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _load_sync_preview_states() -> dict[str, Any]:
    path = _sync_preview_state_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_meeting_handoff_states(states: dict[str, Any]) -> None:
    path = _meeting_handoff_state_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(states, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    except OSError:
        logger.debug("kb_journeys: failed to persist meeting handoff state", exc_info=True)


def _save_sync_preview_states(states: dict[str, Any]) -> None:
    path = _sync_preview_state_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(states, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    except OSError:
        logger.debug("kb_journeys: failed to persist sync preview state", exc_info=True)


def _clear_meeting_handoff_state(session_id: str) -> None:
    if not session_id:
        return
    states = _load_meeting_handoff_states()
    if session_id in states:
        states.pop(session_id, None)
        _save_meeting_handoff_states(states)


def _clear_sync_preview_state(session_id: str) -> None:
    if not session_id:
        return
    states = _load_sync_preview_states()
    if session_id in states:
        states.pop(session_id, None)
        _save_sync_preview_states(states)


def _clear_iterative_queue_reply_state(session_id: str) -> None:
    if not session_id:
        return
    states = _load_queue_reply_states()
    if session_id in states:
        states.pop(session_id, None)
        _save_queue_reply_states(states)


def _platform_name(platform: Any) -> str:
    return str(getattr(platform, "value", platform) or "").lower()


def _command_from_text(text: str) -> str | None:
    stripped = (text or "").strip()
    if not stripped.startswith("/"):
        return None
    token = stripped.split(maxsplit=1)[0][1:]
    command = token.split("@", 1)[0].lower()
    return command if command in MENU_COMMANDS or command in LEGACY_COMMANDS else None


def _command_args_from_text(text: str) -> str:
    stripped = (text or "").strip()
    if not stripped.startswith("/"):
        return ""
    parts = stripped.split(maxsplit=1)
    return parts[1].strip() if len(parts) > 1 else ""


def _prose_kb_command_from_text(text: str) -> tuple[str, str] | None:
    stripped = (text or "").strip()
    if not stripped or stripped.startswith("/"):
        return None
    normalized = re.sub(r"[?!.,;:]+", "", stripped.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    match = re.match(r"^kb\s+(status|sync|review)\b(?:\s+(.*))?$", normalized)
    if match:
        verb = match.group(1)
        rest = (match.group(2) or "").strip()
        return {"status": "kbstatus", "sync": "kbsync", "review": "kblifecycle"}[verb], rest
    if re.search(r"\breview queue\b", normalized) and re.search(
        r"\b(?:what(?: is|'s)?|show|list|open|view|display|check|pending|in)\b",
        normalized,
    ):
        return "kblifecycle", ""
    return None


def _short(value: Any, default: str = "unknown") -> str:
    if value is None:
        return default
    if isinstance(value, bool):
        return "yes" if value else "no"
    text = str(value).strip()
    return text if text else default


def _clip(value: Any, limit: int = 220) -> str:
    text = re.sub(r"\s+", " ", _short(value, "")).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "..."


def _request_receipt(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    receipt = payload.get("receipt") or payload.get("request_receipt")
    if isinstance(receipt, dict):
        return receipt
    if payload.get("packet_type") == "request.receipt":
        return payload
    return {}


def _request_outcome(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    outcome = payload.get("outcome") or payload.get("request_outcome")
    return outcome if isinstance(outcome, dict) else {}


def _request_envelope(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    request = payload.get("request") or payload.get("request_envelope")
    return request if isinstance(request, dict) else {}


def _dedupe_list(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _object_reference_lines(*packets: dict[str, Any], include_report_ref: bool = False) -> list[str]:
    object_family = ""
    report_refs: list[str] = []
    related_refs: list[str] = []
    for packet in packets:
        if not isinstance(packet, dict):
            continue
        if not object_family:
            object_family = _short(packet.get("object_family"), "")
        if include_report_ref:
            report_refs.extend(_id_list(packet.get("report_ref")))
        report_refs.extend(_id_list(packet.get("report_refs")))
        related_refs.extend(_id_list(packet.get("related_object_refs")))
        related_refs.extend(_id_list(packet.get("related_objects")))
    lines: list[str] = []
    if object_family:
        lines.append(f"Object family: {object_family}")
    report_refs = _dedupe_list(report_refs)
    related_refs = _dedupe_list(related_refs)
    if report_refs:
        lines.append(_id_line("Report refs", report_refs))
    if related_refs:
        lines.append(_id_line("Related objects", related_refs))
    return lines


def _receipt_lines(payload: Any, *, include_request: bool = False) -> list[str]:
    receipt = _request_receipt(payload)
    outcome = _request_outcome(payload)
    request = _request_envelope(payload)
    lines: list[str] = []
    if not receipt and not outcome and not request:
        return lines
    if receipt:
        state = _short(receipt.get("state") or receipt.get("status"), "")
        if state:
            lines.append(f"Receipt: {state}")
        effect = _short(receipt.get("durable_effect"), "")
        if effect:
            lines.append(f"Effect: {effect}")
        if receipt.get("llm_invoked_by_read_surface") is not None:
            lines.append(
                "Read-surface LLM: "
                + ("yes" if receipt.get("llm_invoked_by_read_surface") else "no")
            )
        next_step = _short(receipt.get("next_step"), "")
        if next_step:
            lines.append(f"Next: {_clip(next_step, 180)}")
    if outcome:
        family = _short(outcome.get("family") or outcome.get("status"), "")
        if family:
            lines.append(f"Outcome: {family}")
    lines.extend(_object_reference_lines(outcome, receipt))
    if include_request and request:
        kind = _short(request.get("kind") or request.get("request_kind"), "")
        route = _short(request.get("route"), "")
        if kind or route:
            lines.append(f"Request: {kind or 'request'}" + (f" via {route}" if route else ""))
    return lines


def _packet_kind(packet: Any) -> str:
    if not isinstance(packet, dict):
        return ""
    return _short(
        packet.get("packet_type")
        or packet.get("kind")
        or packet.get("packet_kind")
        or packet.get("type"),
        "",
    )


def _first_result_packet(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    if _packet_kind(payload) in SUPPORTED_RESULT_PACKET_TYPES:
        return payload
    for key in (
        "output",
        "result",
        "receipt",
        "packet",
        "preview",
        "shadow_preview",
        "lifecycle_update",
        "semantic_preview",
        "publication_observation",
        "graph_validation",
        "report_admission_receipt",
    ):
        nested = payload.get(key)
        if isinstance(nested, dict) and _packet_kind(nested) in SUPPORTED_RESULT_PACKET_TYPES:
            return nested
    return {}


def _warning_lines(warnings: Any, *, limit: int = 3) -> list[str]:
    if not isinstance(warnings, list) or not warnings:
        return []
    lines = [f"Warnings: {len(warnings)}"]
    for warning in warnings[:limit]:
        if isinstance(warning, dict):
            code = _short(warning.get("code") or warning.get("warning"), "")
            ref = _short(warning.get("ref") or warning.get("path") or warning.get("object_ref"), "")
            detail = _clip(warning.get("message") or warning.get("detail") or warning.get("summary"), 160)
            text = " - ".join(part for part in (code, ref, detail) if part)
            lines.append(f"- {text or 'warning'}")
        else:
            lines.append(f"- {_clip(warning, 180)}")
    return lines


def _render_report_admission_packet(packet: dict[str, Any]) -> dict[str, Any]:
    title = _short(packet.get("title") or packet.get("report_ref"), "Report Admission")
    status = _short(packet.get("status"))
    report_ref = _short(packet.get("report_ref"), "")
    event_ref = _short(packet.get("event_ref"), "")
    event_role = _short(packet.get("event_role"), "")
    situation_ref = _short(packet.get("situation_ref"), "")
    transfers = packet.get("source_transfers") if isinstance(packet.get("source_transfers"), list) else []
    changed_paths = _changed_paths(packet)
    validation = packet.get("graph_validation") if isinstance(packet.get("graph_validation"), dict) else {}
    lines = [
        "Report Admission",
        f"Status: {status}",
        f"Report: {report_ref or title}",
    ]
    if title and title != report_ref:
        lines.append(f"Title: {title}")
    if event_ref:
        lines.append(f"Event: {event_ref}" + (f" ({event_role})" if event_role else ""))
    if situation_ref:
        lines.append(f"Situation: {situation_ref}")
    lines.extend(_object_reference_lines(packet, include_report_ref=True))
    if transfers:
        lines.append(f"Source files: {len(transfers)}")
    if changed_paths:
        lines.append(f"Changed paths: {len(changed_paths)}")
        lines.extend(_format_changed_paths(changed_paths, limit=5))
    if validation:
        lines.append(
            "Graph validation: "
            + _short(validation.get("status") or ("ok" if validation.get("ok") else "warning"))
        )
    lines.extend(_warning_lines(packet.get("warnings")))
    if status == "preview":
        lines.append("No durable write has been made.")
    return {"title": "Report Admission", "text": "\n".join(lines), "actions": []}


def _render_graph_validation_packet(packet: dict[str, Any]) -> dict[str, Any]:
    status = _short(packet.get("status") or ("ok" if packet.get("ok") else "warning"))
    warning_count = packet.get("warning_count")
    error_count = packet.get("error_count")
    lines = [
        "KB Graph Validation",
        f"Status: {status}",
    ]
    if warning_count is not None or error_count is not None:
        lines.append(
            f"Warnings: {_short(warning_count, '0')} · Errors: {_short(error_count, '0')}"
        )
    lines.extend(_warning_lines(packet.get("warnings"), limit=5))
    return {"title": "KB Graph Validation", "text": "\n".join(lines), "actions": []}


def _lifecycle_values(value: Any) -> list[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if not isinstance(value, list):
        return []
    values: list[str] = []
    for item in value:
        if isinstance(item, dict):
            found = ""
            for key in ("ref", "id", "target_ref", "path", "title", "summary", "message"):
                found = _short(item.get(key), "")
                if found:
                    break
            if found:
                values.append(found)
        else:
            text = str(item).strip()
            if text:
                values.append(text)
    return _dedupe_list(values)


def _lifecycle_signal_lines(signal: Any) -> list[str]:
    if not isinstance(signal, dict):
        return []
    lines: list[str] = []
    source = _short(signal.get("source"), "")
    observed = _short(signal.get("observed_at"), "")
    if source or observed:
        parts = []
        if source:
            parts.append(f"Source: {source}")
        if observed:
            parts.append(f"Observed: {observed}")
        lines.append("   " + " · ".join(parts))
    kind = _short(signal.get("kind"), "")
    polarity = _short(signal.get("polarity"), "")
    confidence = _short(signal.get("confidence"), "")
    signal_parts = [part for part in (kind, polarity) if part]
    if signal_parts or confidence:
        suffix = f" ({confidence})" if confidence else ""
        lines.append(f"   Signal: {' / '.join(signal_parts) or 'lifecycle'}{suffix}")
    evidence_refs = _dedupe_list([
        _short(signal.get("source_ref"), ""),
        *_lifecycle_values(signal.get("evidence_refs")),
    ])
    if evidence_refs:
        lines.append(f"   Signal evidence refs: {len(evidence_refs)}")
        lines.append(_id_line("   Signal evidence", evidence_refs, limit=3))
    summary = _short(signal.get("summary"), "")
    if summary:
        lines.append(f"   Signal summary: {_clip(summary, 220)}")
    return lines


def _is_lifecycle_proposal_descriptor(descriptor: dict[str, Any]) -> bool:
    if not isinstance(descriptor, dict):
        return False
    if descriptor.get("dashboard_owned_write") is True:
        return False
    preview_tool = str(descriptor.get("preview_tool") or descriptor.get("method") or "").strip()
    action_id = str(descriptor.get("action_id") or "").strip()
    target_kind = str(descriptor.get("target_kind") or "").strip()
    return (
        target_kind == "lifecycle_candidate"
        or action_id.startswith("lifecycle.")
        or preview_tool.startswith("lifecycle.proposal_")
    ) and bool(preview_tool)


def _lifecycle_candidate_descriptor(candidate: dict[str, Any]) -> dict[str, Any]:
    descriptor = candidate.get("action_descriptor")
    if isinstance(descriptor, dict) and _is_lifecycle_proposal_descriptor(descriptor):
        return dict(descriptor)
    descriptors = candidate.get("action_descriptors")
    if isinstance(descriptors, list):
        for item in descriptors:
            if isinstance(item, dict) and _is_lifecycle_proposal_descriptor(item):
                return dict(item)
    return {}


def _render_lifecycle_proposal_draft_packet(packet: dict[str, Any]) -> dict[str, Any]:
    workflow = _short(packet.get("workflow"), "Lifecycle Review")
    stewardship = _short(packet.get("stewardship_area"), "")
    proposals = packet.get("proposals") if isinstance(packet.get("proposals"), list) else []
    proposal_count = packet.get("proposal_count")
    if proposal_count is None:
        proposal_count = len(proposals)
    lines = [
        "Lifecycle Proposal Draft",
        f"Workflow: {workflow}",
    ]
    if stewardship:
        lines.append(f"Stewardship: {stewardship}")
    if packet.get("mutation_performed") is not None:
        lines.append("Mutation: " + ("performed" if packet.get("mutation_performed") else "none"))
    lines.append(f"Proposals: {_short(proposal_count, '0')}")
    for index, proposal in enumerate(proposals[:5], start=1):
        if not isinstance(proposal, dict):
            lines.append(f"{index}. {_clip(proposal, 180)}")
            continue
        title = _short(
            proposal.get("proposal_id")
            or proposal.get("id")
            or proposal.get("target_ref")
            or proposal.get("title"),
            "proposal",
        )
        lines.append(f"{index}. {title}")
        action = _short(proposal.get("recommended_action") or proposal.get("action"), "")
        target_ref = _short(proposal.get("target_ref"), "")
        summary = _short(proposal.get("summary") or proposal.get("preview") or proposal.get("description"), "")
        if action:
            lines.append(f"   Action: {action}")
        if target_ref:
            lines.append(f"   Target: {target_ref}")
        if summary:
            lines.append(f"   Summary: {_clip(summary, 180)}")
    if len(proposals) > 5:
        lines.append(f"... {len(proposals) - 5} more proposal(s)")
    lines.append("No durable write has been made.")
    return {"title": "Lifecycle Proposal Draft", "text": "\n".join(lines), "actions": []}


def _render_lifecycle_descriptor_preview(
    ctx: Any,
    target: str,
    *,
    descriptor: dict[str, Any],
    callback_ctx: Any,
) -> dict[str, Any]:
    del callback_ctx
    label = _short(descriptor.get("label") or descriptor.get("action_id") or "Lifecycle proposal", "Lifecycle proposal")
    preview_tool = _descriptor_tool_name(target, descriptor.get("preview_tool") or descriptor.get("method"))
    payload = _result_payload(ctx.dispatch_tool(preview_tool, _descriptor_params(descriptor)))
    if isinstance(payload, dict) and payload.get("error"):
        return {"title": label, "text": f"{label}\n{payload['error']}", "actions": []}
    packet_card = _render_supported_result_packet(payload, ctx=ctx, target=target)
    if packet_card is not None:
        packet_card["actions"] = []
        if "No durable write has been made." not in packet_card["text"]:
            packet_card["text"] += "\nNo durable write has been made."
        return packet_card
    if not isinstance(payload, dict):
        return {"title": label, "text": f"{label}\n{_short(payload, 'No proposal preview returned.')}", "actions": []}
    lines = [
        "Lifecycle Proposal Preview",
        f"Status: {_short(payload.get('status') or payload.get('state'), 'preview')}",
    ]
    summary = _short(payload.get("summary") or payload.get("message"), "")
    if summary:
        lines.append("Summary: " + _clip(summary, 260))
    target_ref = _short(payload.get("target_ref") or descriptor.get("target_ref"), "")
    if target_ref:
        lines.append(f"Target: {target_ref}")
    lines.append("No durable write has been made.")
    return {"title": label, "text": "\n".join(lines), "actions": []}


def _lifecycle_descriptor_action(ctx: Any, target: str, descriptor: dict[str, Any]) -> Any | None:
    if not _is_lifecycle_proposal_descriptor(descriptor):
        return None
    try:
        from tools.kb_callback_registry import KbAction
    except Exception:
        return None
    label = _short(descriptor.get("label") or descriptor.get("action_id") or "Lifecycle proposal", "Lifecycle proposal")
    action_id = _short(descriptor.get("action_id") or label, label)
    preview_tool = _descriptor_tool_name(target, descriptor.get("preview_tool") or descriptor.get("method"))
    confirm_tool = _descriptor_tool_name(target, descriptor.get("confirm_tool"))
    if not preview_tool or not confirm_tool:
        return None
    return KbAction(
        label=label,
        action_id=f"{action_id}.preview",
        handler=lambda callback_ctx, d=dict(descriptor): _render_generic_descriptor_preview(
            ctx,
            target,
            descriptor=d,
            callback_ctx=callback_ctx,
        ),
        metadata={
            "target_kind": descriptor.get("target_kind") or "lifecycle_candidate",
            "target_ref": descriptor.get("target_ref"),
            "preview_tool": preview_tool,
            "confirm_tool": confirm_tool,
            "preview_required": True,
            "durable_write": False,
        },
    )


def _lifecycle_candidate_actions(ctx: Any | None, target: str, candidates: list[Any]) -> list[Any]:
    if ctx is None or not target:
        return []
    actions: list[Any] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        descriptor = _lifecycle_candidate_descriptor(candidate)
        action = _lifecycle_descriptor_action(ctx, target, descriptor)
        if action is not None:
            actions.append(action)
        if len(actions) >= 4:
            break
    return actions


def _render_lifecycle_review_packet(
    packet: dict[str, Any],
    *,
    ctx: Any | None = None,
    target: str = "",
) -> dict[str, Any]:
    workflow = _short(packet.get("workflow"), "Lifecycle Review")
    stewardship = _short(packet.get("stewardship_area"), "")
    candidates = packet.get("candidates") if isinstance(packet.get("candidates"), list) else []
    lines = [workflow]
    if stewardship:
        lines.append(f"Stewardship: {stewardship}")
    review_target = _short(packet.get("target") or packet.get("target_ref") or packet.get("scope"), "")
    if review_target:
        lines.append(f"Target: {review_target}")
    if packet.get("mutation_performed") is not None:
        lines.append("Mutation: " + ("performed" if packet.get("mutation_performed") else "none"))
    lines.append(f"Candidates: {len(candidates)}")
    for index, candidate in enumerate(candidates[:5], start=1):
        if not isinstance(candidate, dict):
            lines.append(f"{index}. {_clip(candidate, 180)}")
            continue
        title = _short(candidate.get("title") or candidate.get("name") or candidate.get("target_ref"), "candidate")
        lines.append(f"{index}. {title}")
        action = _short(candidate.get("recommended_action"), "")
        target_ref = _short(candidate.get("target_ref") or candidate.get("object_ref"), "")
        if action:
            lines.append(f"   Action: {action}")
        if target_ref:
            lines.append(f"   Target: {target_ref}")
        signals = candidate.get("signals") if isinstance(candidate.get("signals"), dict) else {}
        closure_signal = signals.get("closure_signal") if isinstance(signals, dict) else None
        lines.extend(_lifecycle_signal_lines(closure_signal))
        evidence_refs = _lifecycle_values(candidate.get("evidence_refs"))
        evidence_gaps = _lifecycle_values(candidate.get("evidence_gaps"))
        if evidence_refs:
            lines.append(_id_line("   Evidence", evidence_refs, limit=3))
        if evidence_gaps:
            lines.append(_id_line("   Gaps", evidence_gaps, limit=3))
    if len(candidates) > 5:
        lines.append(f"... {len(candidates) - 5} more candidate(s)")
    if packet.get("mutation_performed") is False:
        lines.append("No durable write has been made.")
    return {
        "title": "Lifecycle Review",
        "text": "\n".join(lines),
        "actions": _lifecycle_candidate_actions(ctx, target, candidates),
    }


def _manifest_items(manifest: Any) -> list[Any]:
    if isinstance(manifest, list):
        return manifest
    if not isinstance(manifest, dict):
        return []
    for key in ("sources", "items", "entries", "source_refs", "source_manifest"):
        value = manifest.get(key)
        if isinstance(value, list):
            return value
    return []


def _manifest_count(manifest: Any, items: list[Any]) -> int | None:
    if isinstance(manifest, dict):
        for key in ("source_count", "sources_count", "count", "total"):
            try:
                return int(manifest.get(key))
            except (TypeError, ValueError):
                continue
    if items:
        return len(items)
    return None


def _source_manifest_lines(manifest: Any, *, limit: int = 4) -> list[str]:
    if not manifest:
        return []
    items = _manifest_items(manifest)
    count = _manifest_count(manifest, items)
    lines = [f"Sources: {_short(count, 'manifest')}"]
    for item in items[:limit]:
        if isinstance(item, dict):
            kind = _receipt_public_value(item.get("kind") or item.get("source_kind") or item.get("type"), limit=60)
            ref = _receipt_public_value(
                item.get("source_ref")
                or item.get("ref")
                or item.get("id")
                or item.get("source_id")
                or item.get("uri"),
                limit=120,
            )
            title = _receipt_public_value(item.get("title") or item.get("label") or item.get("summary"), limit=120)
            status = _receipt_public_value(item.get("status") or item.get("state"), limit=60)
            parts = [part for part in (kind, ref, title) if part]
            suffix = f" ({status})" if status else ""
            lines.append(f"- {' - '.join(parts) or 'source'}{suffix}")
        else:
            value = _receipt_public_value(item, limit=140)
            if value:
                lines.append(f"- {value}")
    if len(items) > limit:
        lines.append(f"- +{len(items) - limit} more source(s)")
    return lines


def _authority_entry(surface: Any, value: Any = None) -> str:
    label = _receipt_public_value(surface, limit=80)
    if isinstance(value, dict):
        authority = _receipt_public_value(
            value.get("authority")
            or value.get("role")
            or value.get("status")
            or value.get("state")
            or value.get("owner"),
            limit=80,
        )
    else:
        authority = _receipt_public_value(value, limit=80)
    if label and authority:
        return f"{label}={authority}"
    return label or authority


def _authority_map_lines(authority_map: Any, *, limit: int = 6) -> list[str]:
    if not authority_map:
        return []
    entries: list[str] = []
    if isinstance(authority_map, list):
        for item in authority_map:
            if isinstance(item, dict):
                entries.append(
                    _authority_entry(
                        item.get("surface") or item.get("name") or item.get("id") or item.get("target"),
                        item,
                    )
                )
            else:
                entries.append(_authority_entry(item))
    elif isinstance(authority_map, dict):
        list_entries = []
        for key in ("surfaces", "entries", "authorities", "surface_authorities"):
            value = authority_map.get(key)
            if isinstance(value, list):
                list_entries = value
                break
        if list_entries:
            return _authority_map_lines(list_entries, limit=limit)
        for key, value in authority_map.items():
            if key in {"raw", "raw_text", "private_source_text", "source_body"}:
                continue
            entries.append(_authority_entry(key, value))
    entries = [entry for entry in _dedupe_list(entries) if entry]
    if not entries:
        return []
    shown = entries[:limit]
    suffix = f"; +{len(entries) - limit} more" if len(entries) > limit else ""
    return ["Authority map: " + "; ".join(shown) + suffix]


def _candidate_action_values(candidate: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("candidate_actions", "safe_actions", "actions", "action_descriptors"):
        actions = candidate.get(key)
        if not isinstance(actions, list):
            continue
        for action in actions:
            if isinstance(action, dict):
                values.append(
                    _short(
                        action.get("label")
                        or action.get("action_id")
                        or action.get("recommended_action")
                        or action.get("method")
                        or action.get("preview_tool"),
                        "",
                    )
                )
            else:
                values.append(_short(action, ""))
    descriptor = candidate.get("action_descriptor")
    if isinstance(descriptor, dict):
        values.append(
            _short(
                descriptor.get("label")
                or descriptor.get("action_id")
                or descriptor.get("method")
                or descriptor.get("preview_tool"),
                "",
            )
        )
    recommended = _short(candidate.get("recommended_action") or candidate.get("action"), "")
    if recommended:
        values.append(recommended)
    return _receipt_public_values(values, limit=120)


def _candidate_action_line(candidate: dict[str, Any], *, indent: str = "   ") -> str:
    actions = _candidate_action_values(candidate)
    return _id_line(f"{indent}Candidate actions", actions, limit=4) if actions else ""


def _render_lifecycle_update_packet(
    packet: dict[str, Any],
    *,
    ctx: Any | None = None,
    target: str = "",
) -> dict[str, Any]:
    workflow = _short(packet.get("workflow") or packet.get("title"), "Lifecycle Update")
    stewardship = _short(packet.get("stewardship_area") or packet.get("domain"), "")
    candidates = packet.get("candidates") if isinstance(packet.get("candidates"), list) else []
    lines = ["Lifecycle Update"]
    if workflow and workflow != "Lifecycle Update":
        lines.append(f"Workflow: {workflow}")
    if stewardship:
        lines.append(f"Stewardship: {stewardship}")
    status = _short(packet.get("status") or packet.get("state"), "")
    if status:
        lines.append(f"Status: {status}")
    review_target = _receipt_public_value(packet.get("target") or packet.get("target_ref") or packet.get("scope"), limit=140)
    if review_target:
        lines.append(f"Target: {review_target}")
    if packet.get("mutation_performed") is not None:
        lines.append("Mutation: " + ("performed" if packet.get("mutation_performed") else "none"))
    if packet.get("raw_private_source_text_copied") is not None:
        lines.append(
            "Raw private source text copied: "
            + ("yes" if packet.get("raw_private_source_text_copied") else "no")
        )
    lines.extend(_source_manifest_lines(packet.get("source_manifest")))
    lines.extend(_authority_map_lines(packet.get("surface_authority_map")))
    lines.append(f"Candidates: {len(candidates)}")
    for index, candidate in enumerate(candidates[:5], start=1):
        if not isinstance(candidate, dict):
            lines.append(f"{index}. {_clip(candidate, 180)}")
            continue
        title = _receipt_public_value(
            candidate.get("title")
            or candidate.get("name")
            or candidate.get("candidate_id")
            or candidate.get("target_ref"),
            limit=140,
        ) or "candidate"
        lines.append(f"{index}. {title}")
        action = _receipt_public_value(candidate.get("recommended_action") or candidate.get("action"), limit=80)
        target_ref = _receipt_public_value(candidate.get("target_ref") or candidate.get("object_ref"), limit=140)
        summary = _receipt_public_value(candidate.get("summary") or candidate.get("preview") or candidate.get("reason"), limit=180)
        if action:
            lines.append(f"   Action: {action}")
        if target_ref:
            lines.append(f"   Target: {target_ref}")
        if summary:
            lines.append(f"   Summary: {_clip(summary, 180)}")
        action_line = _candidate_action_line(candidate)
        if action_line:
            lines.append(action_line)
        evidence_refs = _receipt_public_values(_lifecycle_values(candidate.get("evidence_refs")), limit=120)
        evidence_gaps = _receipt_public_values(_lifecycle_values(candidate.get("evidence_gaps")), limit=120)
        if evidence_refs:
            lines.append(_id_line("   Evidence", evidence_refs, limit=3))
        if evidence_gaps:
            lines.append(_id_line("   Gaps", evidence_gaps, limit=3))
    if len(candidates) > 5:
        lines.append(f"... {len(candidates) - 5} more candidate(s)")
    if packet.get("mutation_performed") is False:
        lines.append("No durable write has been made.")
    return {
        "title": "Lifecycle Update",
        "text": "\n".join(lines),
        "actions": _lifecycle_candidate_actions(ctx, target, candidates),
    }


def _render_publication_observation_packet(packet: dict[str, Any]) -> dict[str, Any]:
    state = _short(packet.get("publication_state") or packet.get("status") or packet.get("state"))
    changed_paths = _changed_paths(packet)
    lines = [
        "Publication Observation",
        f"State: {state}",
    ]
    changed_count = packet.get("changed_count")
    if changed_count is None and changed_paths:
        changed_count = len(changed_paths)
    if changed_count is not None:
        lines.append(f"Changed paths: {_short(changed_count, '0')}")
    if changed_paths:
        lines.extend(_format_changed_paths(changed_paths, limit=5))
    if packet.get("secret_values_exposed") is not None:
        lines.append("Secrets exposed: " + ("yes" if packet.get("secret_values_exposed") else "no"))
    lines.extend(_warning_lines(packet.get("warnings")))
    return {"title": "Publication Observation", "text": "\n".join(lines), "actions": []}


_PRIVATE_RECEIPT_PATTERNS = (
    re.compile(r"https?://", re.I),
    re.compile(r"\bwww\.", re.I),
    re.compile(r"(?i)(?:^|/)(?:Users|home|private|tmp)/"),
    re.compile(r"(?i)(?:^|[_-])(?:token|secret|password|api[_-]?key)(?:[_-]|$)"),
    re.compile(r"(?i)^(?:acct|account|login|user):"),
    re.compile(r"(?i)\b(?:bearer|sk-[A-Za-z0-9])"),
    re.compile(r"\S+@\S+"),
    re.compile(r"^[A-Za-z]:[\\/]"),
)


def _receipt_public_value(value: Any, *, limit: int = 120) -> str:
    text = _clip(value, limit)
    if not text:
        return ""
    if text.startswith(("/", "~/", "~\\")):
        return ""
    if any(pattern.search(text) for pattern in _PRIVATE_RECEIPT_PATTERNS):
        return ""
    return text


def _receipt_public_values(values: Iterable[Any], *, limit: int = 120) -> list[str]:
    return _dedupe_list(
        value
        for raw in values
        if (value := _receipt_public_value(raw, limit=limit))
    )


def _semantic_values_from(packet: dict[str, Any], *keys: str) -> list[Any]:
    values: list[Any] = []
    for key in keys:
        value = packet.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    for nested_key in ("id", "object_id", "object_ref", "path", "ref", "operation_id"):
                        if item.get(nested_key):
                            values.append(item.get(nested_key))
                            break
                else:
                    values.append(item)
        elif isinstance(value, dict):
            for nested_key in ("id", "object_id", "object_ref", "path", "ref", "operation_id"):
                if value.get(nested_key):
                    values.append(value.get(nested_key))
                    break
        else:
            values.append(value)
    return values


def _semantic_status(packet: dict[str, Any]) -> str:
    status = (
        packet.get("prod_write_status")
        or packet.get("prod_write_state")
        or packet.get("prod_status")
        or packet.get("write_status")
        or _get_path(packet, "prod_write", "status")
        or _get_path(packet, "prod_write", "state")
        or packet.get("status")
        or packet.get("state")
    )
    return _receipt_public_value(status, limit=80)


def _semantic_publication_status(packet: dict[str, Any]) -> str:
    status = (
        packet.get("publication_status")
        or packet.get("publication_state")
        or _get_path(packet, "publication", "status")
        or _get_path(packet, "publication", "state")
        or _get_path(packet, "publication", "publication_status")
        or _get_path(packet, "sync", "status")
    )
    return _receipt_public_value(status, limit=80)


def _semantic_reconciliation_status(packet: dict[str, Any]) -> str:
    status = (
        packet.get("reconciliation_status")
        or packet.get("offline_reconciliation_status")
        or packet.get("local_reconciliation_status")
        or _get_path(packet, "reconciliation", "status")
        or _get_path(packet, "reconciliation", "state")
        or _get_path(packet, "local_reconciliation", "status")
        or _get_path(packet, "local_reconciliation", "state")
        or _get_path(packet, "offline_reconciliation", "status")
        or _get_path(packet, "offline", "reconciliation_status")
    )
    return _receipt_public_value(status, limit=80)


def _semantic_transaction_id(packet: dict[str, Any]) -> str:
    transaction = packet.get("transaction")
    transaction_id = packet.get("transaction_id") or packet.get("txid")
    if not transaction_id and isinstance(transaction, dict):
        transaction_id = transaction.get("id") or transaction.get("transaction_id")
    elif not transaction_id:
        transaction_id = transaction
    return _receipt_public_value(transaction_id, limit=120)


def _public_id_values(value: Any, *, limit: int = 120) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        values: list[Any] = []
        for key in (
            "id",
            "object_id",
            "object_ref",
            "target_ref",
            "ref",
            "path",
            "family",
            "name",
            "label",
            "action_id",
            "method",
            "preview_tool",
            "status",
            "state",
        ):
            if value.get(key):
                values.append(value.get(key))
        return _receipt_public_values(values, limit=limit)
    if isinstance(value, list):
        values = []
        for item in value:
            if isinstance(item, dict):
                nested = _public_id_values(item, limit=limit)
                values.extend(nested)
            else:
                values.append(item)
        return _receipt_public_values(values, limit=limit)
    return _receipt_public_values([value], limit=limit)


def _classification_line(classification: Any) -> str:
    if not classification:
        return ""
    if isinstance(classification, dict):
        values = [
            _receipt_public_value(
                classification.get(key),
                limit=80,
            )
            for key in ("family", "intent", "classification", "status", "state", "confidence")
        ]
        values = [value for value in values if value]
        if values:
            return "Classification: " + " / ".join(values)
        summary = _receipt_public_value(classification.get("summary") or classification.get("reason"), limit=180)
        return f"Classification: {summary}" if summary else ""
    value = _receipt_public_value(classification, limit=120)
    return f"Classification: {value}" if value else ""


def _semantic_preview_lines(preview: Any) -> list[str]:
    if not isinstance(preview, dict):
        value = _receipt_public_value(preview, limit=180)
        return [f"Preview: {value}"] if value else []
    lines: list[str] = []
    status = _receipt_public_value(preview.get("status") or preview.get("state"), limit=80)
    summary = _receipt_public_value(preview.get("summary") or preview.get("safe_summary"), limit=220)
    if status:
        lines.append(f"Preview status: {status}")
    if summary:
        lines.append("Preview: " + _clip(summary, 220))
    object_refs = _receipt_public_values(
        _semantic_values_from(preview, "object_ids", "object_id", "object_refs", "object_ref", "objects"),
        limit=140,
    )
    operations = _receipt_public_values(
        _semantic_values_from(preview, "operation_ids", "operation_id", "operations", "operation_receipts"),
        limit=120,
    )
    families = _public_id_values(preview.get("durable_write_families") or preview.get("families"), limit=100)
    if families:
        lines.append(_id_line("Preview families", families, limit=4))
    if object_refs:
        lines.append(_id_line("Preview objects", object_refs, limit=4))
    if operations:
        lines.append(_id_line("Preview operations", operations, limit=4))
    return lines


def _comparison_lines(comparison: Any) -> list[str]:
    if not comparison:
        return []
    if not isinstance(comparison, dict):
        value = _receipt_public_value(comparison, limit=180)
        return [f"Comparison: {value}"] if value else []
    status = _receipt_public_value(
        comparison.get("status")
        or comparison.get("state")
        or comparison.get("result")
        or comparison.get("outcome"),
        limit=80,
    )
    lines = [f"Comparison: {status}"] if status else []
    for label, key in (
        ("Matches", "matches"),
        ("Differences", "differences"),
        ("Missing", "missing"),
        ("Extra", "extra"),
    ):
        value = comparison.get(key)
        if isinstance(value, list) and value:
            lines.append(f"{label}: {len(value)}")
        elif isinstance(value, int):
            lines.append(f"{label}: {value}")
    summary = _receipt_public_value(comparison.get("summary") or comparison.get("safe_summary"), limit=180)
    if summary:
        lines.append("Comparison summary: " + _clip(summary, 180))
    return lines


def _status_value(statuses: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = statuses.get(key)
        if isinstance(value, dict):
            found = _receipt_public_value(
                value.get("status")
                or value.get("state")
                or value.get("result")
                or value.get("reason"),
                limit=80,
            )
        else:
            found = _receipt_public_value(value, limit=80)
        if found:
            return found
    return ""


def _receipt_status_lines(statuses: Any) -> list[str]:
    if not statuses:
        return []
    if not isinstance(statuses, dict):
        value = _receipt_public_value(statuses, limit=120)
        return [f"Receipt status: {value}"] if value else []
    lines: list[str] = []
    for label, keys in (
        ("Prod write", ("prod_write_status", "prod_write", "prod", "durable_write", "write_status")),
        ("Publication", ("publication_status", "publication", "sync_status", "sync")),
        (
            "Reconciliation",
            (
                "reconciliation_status",
                "reconciliation",
                "local_reconciliation",
                "offline_reconciliation",
            ),
        ),
        ("Shadow receipt", ("shadow_receipt_status", "shadow_receipt", "receipt_status", "status", "state")),
    ):
        value = _status_value(statuses, *keys)
        if value:
            lines.append(f"{label}: {value}")
    return _dedupe_list(lines)


def _render_semantic_write_shadow_packet(packet: dict[str, Any]) -> dict[str, Any]:
    lines = ["Semantic Write Shadow Preview"]
    if packet.get("shadow_mode") is not None:
        lines.append("Shadow mode: " + ("yes" if packet.get("shadow_mode") else "no"))
    if packet.get("mutation_performed") is not None:
        lines.append("Mutation: " + ("performed" if packet.get("mutation_performed") else "none"))
    if packet.get("raw_private_source_text_copied") is not None:
        lines.append(
            "Raw private source text copied: "
            + ("yes" if packet.get("raw_private_source_text_copied") else "no")
        )
    families = _public_id_values(packet.get("durable_write_families"), limit=100)
    if families:
        lines.append(_id_line("Durable write families", families, limit=5))
    classification = _classification_line(packet.get("classification"))
    if classification:
        lines.append(classification)
    lines.extend(_source_manifest_lines(packet.get("source_manifest")))
    lines.extend(_authority_map_lines(packet.get("surface_authority_map")))
    lines.extend(_semantic_preview_lines(packet.get("semantic_preview")))
    lines.extend(_comparison_lines(packet.get("comparison")))
    lines.extend(_receipt_status_lines(packet.get("receipt_status")))
    candidate_actions = _receipt_public_values(
        _public_id_values(packet.get("candidate_actions"), limit=120)
        or _public_id_values(packet.get("actions"), limit=120),
        limit=120,
    )
    if candidate_actions:
        lines.append(_id_line("Candidate actions", candidate_actions, limit=5))
    if packet.get("mutation_performed") is False:
        lines.append("No durable write has been made.")
    return {"title": "Semantic Write Shadow Preview", "text": "\n".join(lines), "actions": []}


def _render_semantic_write_receipt_packet(packet: dict[str, Any]) -> dict[str, Any]:
    lines = ["Semantic Write Receipt"]
    prod_status = _semantic_status(packet)
    publication_status = _semantic_publication_status(packet)
    reconciliation_status = _semantic_reconciliation_status(packet)
    transaction_id = _semantic_transaction_id(packet)
    changed_paths = _receipt_public_values(_changed_paths(packet), limit=140)
    object_ids = _receipt_public_values(
        _semantic_values_from(packet, "object_ids", "object_id", "object_refs", "object_ref", "objects"),
        limit=140,
    )
    operation_ids = _receipt_public_values(
        _semantic_values_from(packet, "operation_ids", "operation_id", "operations", "operation_receipts"),
        limit=120,
    )

    if prod_status:
        lines.append(f"Prod write: {prod_status}")
    if publication_status:
        lines.append(f"Publication: {publication_status}")
    if reconciliation_status:
        lines.append(f"Reconciliation: {reconciliation_status}")
    if transaction_id:
        lines.append(f"Transaction: {transaction_id}")
    if changed_paths:
        lines.append(f"Changed paths: {len(changed_paths)}")
        lines.extend(_format_changed_paths(changed_paths, limit=4))
    if object_ids:
        lines.append(_id_line("Objects", object_ids, limit=4))
    if operation_ids:
        lines.append(_id_line("Operations", operation_ids, limit=4))
    return {"title": "Semantic Write Receipt", "text": "\n".join(lines), "actions": []}


def _id_list(value: Any) -> list[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _id_line(label: str, values: list[str], *, limit: int = 8) -> str:
    shown = ", ".join(values[:limit])
    if len(values) > limit:
        shown += f", +{len(values) - limit} more"
    return f"{label}: {shown}"


def _restore_hint(receipt: dict[str, Any]) -> dict[str, Any]:
    hint = receipt.get("restore_hint")
    return hint if isinstance(hint, dict) else {}


def _restore_args_from_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    hint = _restore_hint(receipt)
    args = hint.get("params") if isinstance(hint.get("params"), dict) else {}
    args = dict(args)
    for key in ("transaction_id", "receipt_id"):
        value = _short(hint.get(key) or receipt.get(key), "")
        if value:
            args.setdefault(key, value)
    proposal_ids = _id_list(hint.get("proposal_ids")) or _id_list(receipt.get("affected_ids"))
    if proposal_ids:
        args.setdefault("proposal_ids", proposal_ids)
    return {key: value for key, value in args.items() if value not in (None, "", [])}


def _restore_tools(target: str, receipt: dict[str, Any]) -> tuple[str, str]:
    hint = _restore_hint(receipt)
    preview_tool = _descriptor_tool_name(
        target,
        hint.get("preview_tool") or hint.get("restore_preview_tool") or "queue.restore_preview",
    )
    confirm_tool = _descriptor_tool_name(
        target,
        hint.get("confirm_tool") or hint.get("restore_confirm_tool") or "queue.restore_confirmed",
    )
    return preview_tool, confirm_tool


def _restore_preview_text(payload: Any) -> str:
    if isinstance(payload, dict) and payload.get("error"):
        return f"Review restore preview failed\n{payload['error']}"
    if not isinstance(payload, dict):
        return f"Review restore preview\n{_short(payload, 'No structured response returned.')}"
    lines = [
        "Review restore preview",
        f"Status: {_short(payload.get('status') or payload.get('state'))} · ok: {_short(payload.get('ok'))}",
    ]
    restorable = _id_list(payload.get("restorable_ids"))
    incompatible = _id_list(payload.get("incompatible_ids"))
    already_restored = _id_list(payload.get("already_restored_ids"))
    if restorable:
        lines.append(_id_line("Restorable ids", restorable))
    if incompatible:
        lines.append(_id_line("Blocked ids", incompatible))
    if already_restored:
        lines.append(_id_line("Already restored", already_restored))
    lines.extend(_queue_scope_lines(payload))
    return "\n".join(lines)


def _restore_action_from_receipt(ctx: Any | None, target: str, receipt: dict[str, Any]) -> Any | None:
    if ctx is None or receipt.get("restore_available") is not True or not _restore_hint(receipt):
        return None
    preview_tool, confirm_tool = _restore_tools(target, receipt)
    args = _restore_args_from_receipt(receipt)
    if not preview_tool or not confirm_tool or not args:
        return None
    try:
        from tools.kb_callback_registry import KbAction
    except Exception:
        return None
    return KbAction(
        label="Preview Restore",
        action_id="queue.restore.preview",
        handler=lambda callback_ctx, r=dict(receipt): _render_restore_preview(
            ctx,
            target,
            receipt=r,
            callback_ctx=callback_ctx,
        ),
        metadata={
            "target_kind": "proposal_queue",
            "preview_tool": preview_tool,
            "confirm_tool": confirm_tool,
            "preview_required": True,
            "restore_available": True,
        },
    )


def _render_restore_preview(
    ctx: Any,
    target: str,
    *,
    receipt: dict[str, Any],
    callback_ctx: Any,
) -> dict[str, Any]:
    del callback_ctx
    try:
        from tools.kb_callback_registry import KbAction
    except Exception:
        return {"title": "KB Review Restore", "text": "KB Review Restore\nAction buttons are unavailable. Use /kb review to refresh.", "actions": []}
    preview_tool, _confirm_tool = _restore_tools(target, receipt)
    preview_payload = _result_payload(ctx.dispatch_tool(preview_tool, _restore_args_from_receipt(receipt)))
    text = _restore_preview_text(preview_payload)
    if not _preview_allows_confirmation(preview_payload):
        return {"title": "KB Review Restore", "text": text, "actions": []}
    preview_metadata = _queue_preview_metadata(preview_payload)
    confirm_action = KbAction(
        label="Confirm Restore",
        action_id="queue.restore.confirm",
        handler=lambda confirm_ctx, r=dict(receipt), metadata=dict(preview_metadata): _render_restore_confirm(
            ctx,
            target,
            receipt=r,
            callback_ctx=confirm_ctx,
            preview_metadata=metadata,
        ),
        metadata={
            "target_kind": "proposal_queue",
            "preview_required": True,
            "preview_lease": bool(preview_metadata.get("preview_lease")),
            "review_session_id": _review_session_id(preview_metadata),
        },
    )
    return {
        "title": "KB Review Restore",
        "text": text + "\n\nConfirm restore only if these ids match the receipt you intended to undo.",
        "actions": [confirm_action],
    }


def _render_restore_confirm(
    ctx: Any,
    target: str,
    *,
    receipt: dict[str, Any],
    callback_ctx: Any,
    preview_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    preview_tool, confirm_tool = _restore_tools(target, receipt)
    effective_metadata = dict(preview_metadata or {})
    if not effective_metadata.get("preview_lease"):
        preview_payload = _result_payload(ctx.dispatch_tool(preview_tool, _restore_args_from_receipt(receipt)))
        if not _preview_allows_confirmation(preview_payload):
            return {"title": "KB Review Restore", "text": _restore_preview_text(preview_payload), "actions": []}
        effective_metadata.update(_queue_preview_metadata(preview_payload))
    args = _restore_args_from_receipt(receipt)
    review_session_id = _review_session_id(effective_metadata)
    cursor_id = _queue_cursor_id(effective_metadata)
    if review_session_id:
        args.setdefault("review_session_id", review_session_id)
    if cursor_id:
        args.setdefault("cursor_id", cursor_id)
    args.setdefault("actor", _queue_callback_actor(callback_ctx))
    args.setdefault("source", "Hermes Telegram Action Card")
    args.setdefault("session_id", review_session_id or f"telegram-kb-restore-{int(time.time())}")
    args["user_confirmation"] = {
        "confirmed": True,
        "surface": "telegram",
        "action": "queue.restore",
        "preview_required": True,
        "confirmation_text": "Confirm queue restore from Telegram receipt action card.",
        "actor_id": _short(getattr(callback_ctx, "actor_id", ""), ""),
        "actor_name": _short(getattr(callback_ctx, "actor_name", ""), ""),
    }
    _apply_queue_confirmation_preview_metadata(args["user_confirmation"], effective_metadata)
    payload = _result_payload(ctx.dispatch_tool(confirm_tool, args))
    packet_card = _render_supported_result_packet(payload, ctx=ctx, target=target)
    if packet_card is not None:
        return packet_card
    return {"title": "KB Review Restore", "text": _restore_preview_text(payload).replace("preview", "result", 1), "actions": []}


def _render_request_receipt_packet(
    packet: dict[str, Any],
    *,
    ctx: Any | None = None,
    target: str = "",
) -> dict[str, Any]:
    route = _short(packet.get("route"), "")
    title = "KB Review Receipt" if route.startswith("queue.") else "KB Request Receipt"
    lines = [
        title,
        f"State: {_short(packet.get('state') or packet.get('status'))}",
    ]
    if packet.get("saved") is not None:
        lines.append("Saved: " + ("yes" if packet.get("saved") else "no"))
    if route:
        lines.append(f"Route: {route}")
    if packet.get("receipt_id"):
        lines.append(f"Receipt: {_short(packet.get('receipt_id'))}")
    if packet.get("transaction_id"):
        lines.append(f"Transaction: {_short(packet.get('transaction_id'))}")
    count_bits: list[str] = []
    if packet.get("reviewed_count") is not None:
        count_bits.append(f"{_short(packet.get('reviewed_count'), '0')} reviewed")
    if packet.get("confirmed_count") is not None:
        count_bits.append(f"{_short(packet.get('confirmed_count'), '0')} confirmed")
    if count_bits:
        lines.append("Counts: " + " · ".join(count_bits))
    for label, key in (
        ("Affected ids", "affected_ids"),
        ("Restored ids", "restored_ids"),
        ("Skipped ids", "skipped_ids"),
        ("Changed refs", "changed_refs"),
    ):
        ids = _id_list(packet.get(key))
        if ids:
            lines.append(_id_line(label, ids))
    lines.extend(_object_reference_lines(packet))
    message = _short(packet.get("safe_message") or packet.get("message"), "")
    if message:
        lines.append(_clip(message, 260))
    next_step = _short(packet.get("next_step"), "")
    if next_step:
        lines.append("Next: " + _clip(next_step, 220))
    if packet.get("restore_available") is True:
        lines.append("Restore: preview available")
    action = _restore_action_from_receipt(ctx, target, packet)
    next_review = _next_review_packet(packet)
    if next_review:
        status = _short(next_review.get("status"), "")
        if route.startswith("queue.") and status == "ready":
            next_payload = _queue_payload_from_next_review(next_review)
            next_card = _render_queue(
                next_payload,
                ctx=ctx,
                target=target,
                session_id=_review_session_id({"review_session": next_review.get("review_session")})
                or _short(next_review.get("source_review_session_id"), ""),
            )
            actions = list(next_card.get("actions") or [])
            if action and len(actions) < 6:
                actions.append(action)
            return {
                "title": "KB Review",
                "text": "\n".join([*lines, "", "Next review from kb-engine:", next_card["text"]]),
                "actions": actions,
            }
        lines.extend(_next_review_lines(next_review))
    return {"title": title, "text": "\n".join(lines), "actions": [action] if action else []}


def _next_review_packet(packet: dict[str, Any]) -> dict[str, Any]:
    next_review = packet.get("next_review")
    if isinstance(next_review, dict) and next_review.get("packet_type") == "guided_kb_review_next":
        return dict(next_review)
    return {}


def _next_review_lines(next_review: dict[str, Any]) -> list[str]:
    status = _short(next_review.get("status"), "")
    reason = _short(next_review.get("reason"), "")
    if status == "no_more_items":
        return ["Next review: no more proposal items."]
    if status in {"changed_queue", "stale_cursor", "preview_lease_required", "refresh_required"}:
        line = "Next review: refresh required"
        if reason:
            line += f" ({_clip(reason, 160)})"
        return [line]
    if status == "unavailable":
        line = "Next review: unavailable"
        if reason:
            line += f" ({_clip(reason, 160)})"
        return [line]
    if status == "ready":
        target = next_review.get("target") if isinstance(next_review.get("target"), dict) else {}
        title = _short(target.get("title"), "next item")
        proposal_count = len(_id_list(target.get("proposal_ids")))
        suffix = f" · {proposal_count} proposal{'s' if proposal_count != 1 else ''}" if proposal_count else ""
        return [f"Next review: {title}{suffix}"]
    return [f"Next review: {status or 'unknown'}"]


def _queue_payload_from_next_review(next_review: dict[str, Any]) -> dict[str, Any]:
    target = next_review.get("target") if isinstance(next_review.get("target"), dict) else {}
    proposal_ids = _id_list(target.get("proposal_ids"))
    review_session = next_review.get("review_session") if isinstance(next_review.get("review_session"), dict) else {}
    item = {
        "item_id": _short(target.get("target_id"), ""),
        "kind": _short(target.get("kind"), "proposal_entity"),
        "title": _short(target.get("title"), "Next review"),
        "summary": _short(target.get("summary"), ""),
        "status": _short(target.get("status"), ""),
        "entity_path": _short(target.get("entity_path"), ""),
        "safe_actions": target.get("safe_actions") if isinstance(target.get("safe_actions"), list) else [],
        "review_session": review_session,
        "raw": {
            "proposal_ids": proposal_ids,
            "proposal_count": int(target.get("proposal_count") or len(proposal_ids)),
            "sections": target.get("sections") if isinstance(target.get("sections"), list) else [],
            "review_session": review_session,
        },
    }
    return {
        "packet_type": "workbench.queue",
        "schema_version": 1,
        "total": 1,
        "offset": 0,
        "next_offset": None,
        "items": [item],
        "review_session": review_session,
    }


def _render_supported_result_packet(
    payload: Any,
    *,
    ctx: Any | None = None,
    target: str = "",
) -> dict[str, Any] | None:
    packet = _first_result_packet(payload)
    if not packet:
        return None
    packet_type = _packet_kind(packet)
    if packet_type == "report_admission_receipt":
        return _render_report_admission_packet(packet)
    if packet_type == "durable_graph_validation":
        return _render_graph_validation_packet(packet)
    if packet_type == "lifecycle_review.packet":
        return _render_lifecycle_review_packet(packet, ctx=ctx, target=target)
    if packet_type == "lifecycle_update.packet":
        return _render_lifecycle_update_packet(packet, ctx=ctx, target=target)
    if packet_type == "lifecycle_proposal_draft.packet":
        return _render_lifecycle_proposal_draft_packet(packet)
    if packet_type == "publication_observation":
        return _render_publication_observation_packet(packet)
    if packet_type in SEMANTIC_WRITE_SHADOW_PACKET_TYPES:
        return _render_semantic_write_shadow_packet(packet)
    if packet_type in SEMANTIC_WRITE_RECEIPT_PACKET_TYPES:
        return _render_semantic_write_receipt_packet(packet)
    if packet_type == "request.receipt":
        return _render_request_receipt_packet(packet, ctx=ctx, target=target)
    return None


def _maybe_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return value
    if not (stripped.startswith("{") or stripped.startswith("[")):
        return value
    try:
        return json.loads(stripped)
    except Exception:
        return value


def _unwrap_tool_result(raw: Any) -> tuple[Any | None, str | None]:
    parsed = _maybe_json(raw)
    if not isinstance(parsed, dict):
        return parsed, None
    if parsed.get("error"):
        return None, _short(parsed.get("error"))
    payload = parsed.get("structuredContent")
    if payload is None:
        payload = parsed.get("result", parsed)
    payload = _maybe_json(payload)
    return payload, None


def _dispatch_first(
    ctx: Any,
    target: str,
    candidates: Iterable[tuple[str, dict[str, Any]]],
) -> tuple[str | None, Any | None, list[str]]:
    errors: list[str] = []
    for kb_tool, args in candidates:
        registry_name = _mcp_tool_name(target, kb_tool)
        try:
            payload, error = _unwrap_tool_result(ctx.dispatch_tool(registry_name, args))
        except Exception as exc:
            errors.append(f"{registry_name}: {exc}")
            continue
        if error:
            errors.append(f"{registry_name}: {error}")
            continue
        return registry_name, payload, errors
    return None, None, errors


def _get_path(data: Any, *path: str, default: Any = None) -> Any:
    cur = data
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return cur if cur is not None else default


def _count_from(data: Any, *keys: str) -> Any:
    for key in keys:
        value = _get_path(data, key, "count")
        if value is not None:
            return value
        if isinstance(data, dict) and data.get(key) is not None and not isinstance(data.get(key), dict):
            found = data.get(key)
            return len(found) if isinstance(found, list) else found
    return None


def _readiness_status(data: dict[str, Any]) -> Any:
    return (
        _get_path(data, "summary", "readiness_status")
        or _get_path(data, "sections", "readiness", "summary", "status")
        or _get_path(data, "sections", "readiness", "payload", "status")
        or _get_path(data, "readiness", "status")
        or _get_path(data, "readiness", "state")
        or data.get("readiness")
    )


def _publication_status(data: dict[str, Any]) -> Any:
    return (
        _get_path(data, "summary", "publication_status")
        or _get_path(data, "sections", "publication", "summary", "status")
        or _get_path(data, "sections", "publication", "payload", "status")
        or _get_path(data, "publication", "status")
        or _get_path(data, "publication", "state")
        or data.get("publication")
    )


def _item_title(item: Any) -> str:
    if isinstance(item, dict):
        return _short(
            item.get("title")
            or item.get("name")
            or item.get("summary")
            or item.get("id")
            or item.get("proposal_id"),
            "item",
        )
    return _short(item, "item")


def _summary_count(summary: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = summary.get(key)
        if value is not None:
            return value
    return None


def _proposal_count_from_summary(summary: dict[str, Any]) -> Any:
    proposal_count = _summary_count(
        summary,
        "proposal_queue_count",
        "pending_proposal_count",
        "proposal_count",
        "review_proposal_count",
    )
    if proposal_count is not None:
        return proposal_count
    legacy_queue_count = summary.get("queue_item_count")
    todo_count = _summary_count(summary, "active_todo_count", "triage_todo_count", "task_queue_count")
    if legacy_queue_count is not None and todo_count is not None and legacy_queue_count != todo_count:
        return legacy_queue_count
    return None


def _todo_count_from_summary(summary: dict[str, Any]) -> Any:
    return _summary_count(summary, "active_todo_count", "triage_todo_count", "task_queue_count")


def _display_text(value: Any) -> str:
    text = _short(value, "")
    if text in {
        "Review prioritized queue items through workbench.queue.",
        "Review prioritized review items through workbench.queue.",
    }:
        return "Review prioritized attention items; use /kb review for proposal review."
    return text


def _dashboard_section_title(section: dict[str, Any], summary: dict[str, Any]) -> str:
    title = _short(section.get("title") or section.get("id"), "Section")
    key = title.strip().lower()
    section_id = str(section.get("id") or "").strip().lower()
    if key == "queue" or section_id == "queue":
        proposal_count = _proposal_count_from_summary(summary)
        todo_count = _todo_count_from_summary(summary)
        legacy_queue_count = summary.get("queue_item_count")
        if proposal_count is None or (todo_count is not None and legacy_queue_count == todo_count):
            return "Attention Review"
        return "Proposal Review"
    return title


def _items(data: Any, *paths: tuple[str, ...]) -> list[Any]:
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        return []
    for path in paths:
        found = _get_path(data, *path)
        if isinstance(found, list) and found:
            return found
    for key in ("items", "proposals", "queue", "runs", "recent", "active"):
        found = data.get(key)
        if isinstance(found, list) and found:
            return found
    return []


def _public_error(errors: list[str]) -> str:
    if not errors:
        return "No compatible KB MCP tool responded."
    detail = errors[-1]
    if detail.startswith("mcp_") and ": " in detail:
        detail = detail.split(": ", 1)[1]
    return detail or "No compatible KB MCP tool responded."


def _render_error(title: str, target: str, errors: list[str]) -> dict[str, Any]:
    detail = _public_error(errors)
    text = f"{title}\nMCP target: {target}\nKB data is not available yet.\n{detail}"
    return {"title": title, "text": text, "actions": []}


def _render_today(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        text = f"KB Today\n{_short(data, 'No cockpit details returned.')}"
        return {"title": "KB Today", "text": text, "actions": []}

    readiness = _short(_readiness_status(data))
    publication = _short(_publication_status(data))
    summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}
    queue_count = _proposal_count_from_summary(summary)
    if queue_count is None:
        queue_count = _count_from(data, "proposals", "proposal_queue")
    todo_count = _count_from(data, "todo", "todos")
    if todo_count is None:
        todo_count = _todo_count_from_summary(summary)

    active_runs = _items(data, ("runs", "active"), ("active_runs",))
    recent_runs = _items(data, ("runs", "recent"), ("recent_runs",))
    run_bits: list[str] = []
    for run in active_runs[:2]:
        if isinstance(run, dict):
            run_bits.append(f"{_item_title(run)} {_short(run.get('status') or run.get('state'))}")
        else:
            run_bits.append(_short(run))
    for run in recent_runs[:1]:
        if isinstance(run, dict):
            run_bits.append(f"recent {_item_title(run)} {_short(run.get('status') or run.get('state'))}")
        else:
            run_bits.append(f"recent {_short(run)}")

    next_actions = _items(data, ("next_actions",), ("actions",))
    lines = [
        "KB Today",
        f"Readiness: {readiness}",
        f"Publication: {publication}",
    ]
    if queue_count is not None or todo_count is not None:
        count_bits = []
        if queue_count is not None:
            count_bits.append(f"Proposals: {_short(queue_count, 'unknown')}")
        if todo_count is not None:
            count_bits.append(f"TODOs: {_short(todo_count, 'unknown')}")
        lines.append(" · ".join(count_bits))
    if run_bits:
        lines.append("Runs: " + " · ".join(run_bits[:3]))
    if next_actions:
        lines.append("Next: " + "; ".join(_item_title(a) for a in next_actions[:3]))
    return {"title": "KB Today", "text": "\n".join(lines), "actions": []}


def _render_dashboard(data: Any, *, ctx: Any, target: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {"title": "KB", "text": f"KB\n{_short(data, 'No KB details returned.')}", "actions": []}

    summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}
    readiness = _short(
        summary.get("readiness_status")
        or _readiness_status(data)
    )
    publication = _short(
        summary.get("publication_status")
        or _publication_status(data)
    )
    sections = data.get("sections") if isinstance(data.get("sections"), list) else []
    queue_count = _proposal_count_from_summary(summary)
    if queue_count is None:
        queue_count = _count_from(data, "proposals", "proposal_queue")
    todo_count = _todo_count_from_summary(summary)
    if todo_count is None:
        todo_count = _count_from(data, "todo", "todos")
    active_runs = summary.get("active_run_count")
    lines = [
        "KB",
        f"kb status: runtime {readiness} · publication {publication}",
    ]
    if data.get("llm_invoked_by_read_surface") is not None:
        lines.append(
            "Read-surface LLM: "
            + ("yes" if data.get("llm_invoked_by_read_surface") else "no")
        )
    lines.extend(_receipt_lines(data))
    counts: list[str] = []
    if queue_count is not None:
        counts.append(f"Proposals {queue_count}")
    if todo_count is not None:
        counts.append(f"TODOs {todo_count}")
    if counts:
        lines.append("kb review: " + " · ".join(counts))
    if active_runs is not None:
        lines.append(f"kb sync: {active_runs} active run(s)")
    for section in sections[:4]:
        if not isinstance(section, dict):
            continue
        cards = section.get("cards") if isinstance(section.get("cards"), list) else []
        if not cards:
            continue
        lines.append("")
        lines.append(_dashboard_section_title(section, summary))
        for card in cards[:3]:
            if not isinstance(card, dict):
                continue
            detail = _display_text(card.get("detail"))
            suffix = f" — {detail}" if detail else ""
            lines.append(f"- {_display_text(card.get('title') or 'item')}{suffix}")
    next_actions = data.get("next_actions") if isinstance(data.get("next_actions"), list) else []
    if next_actions and not any(
        isinstance(section, dict) and str(section.get("id") or "").strip().lower() == "next"
        for section in sections
    ):
        lines.append("")
        lines.append("Next Actions")
        for action in next_actions[:3]:
            lines.append(f"- {_display_text(action)}")
    warnings = data.get("warnings") if isinstance(data.get("warnings"), list) else []
    if warnings:
        lines.append("")
        lines.append(f"Warnings: {len(warnings)}")
    refresh = data.get("refresh") if isinstance(data.get("refresh"), dict) else {}
    if refresh:
        lines.append(f"Refresh: every {_short(refresh.get('ttl_seconds'), '60')}s target")
    lines.append("")
    lines.append("Commands: /kb status · /kb sync · /kb review")
    return {"title": "KB", "text": "\n".join(lines), "actions": _dashboard_descriptor_actions(ctx, target, sections)}


_WORKBENCH_SECTION_IDS = {"closeout", "now", "reports", "situations", "workbench"}


def _dashboard_card_descriptors(card: dict[str, Any]) -> list[dict[str, Any]]:
    descriptors = card.get("action_descriptors") if isinstance(card.get("action_descriptors"), list) else []
    safe: list[dict[str, Any]] = []
    for descriptor in descriptors:
        if not isinstance(descriptor, dict):
            continue
        if descriptor.get("dashboard_owned_write") is True:
            continue
        if descriptor.get("packet_type") != "dashboard_action_descriptor" or descriptor.get("schema_version") != 2:
            continue
        safe.append(descriptor)
    return safe


def _render_workbench(data: Any, *, ctx: Any, target: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {"title": "KB Review", "text": f"KB Review\n{_short(data, 'No review details returned.')}", "actions": []}

    summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}
    readiness = _short(summary.get("readiness_status") or _readiness_status(data))
    publication = _short(summary.get("publication_status") or _publication_status(data))
    sections = data.get("sections") if isinstance(data.get("sections"), list) else []
    queue_count = _proposal_count_from_summary(summary)
    todo_count = _todo_count_from_summary(summary)
    lines = [
        "KB Review",
        f"Status: runtime {readiness} · publication {publication}",
    ]
    counts: list[str] = []
    if queue_count is not None:
        counts.append(f"Proposals {queue_count}")
    if todo_count is not None:
        counts.append(f"TODOs {todo_count}")
    if counts:
        lines.append(" · ".join(counts))
    lines.extend(_receipt_lines(data))

    decision_cards: list[tuple[str, dict[str, Any], list[dict[str, Any]]]] = []
    for section in sections:
        if not isinstance(section, dict):
            continue
        section_id = str(section.get("id") or "").strip().lower()
        if section_id not in _WORKBENCH_SECTION_IDS:
            continue
        section_title = _dashboard_section_title(section, summary)
        cards = section.get("cards") if isinstance(section.get("cards"), list) else []
        for card in cards:
            if not isinstance(card, dict):
                continue
            descriptors = _dashboard_card_descriptors(card)
            if descriptors:
                decision_cards.append((section_title, card, descriptors))

    lines.append("")
    lines.append("Decision Cards")
    if not decision_cards:
        lines.append("No active kb-engine Decision Cards returned.")
    for index, (section_title, card, descriptors) in enumerate(decision_cards[:5], start=1):
        lines.extend(
            _workbench_compact_card_lines(
                index=index,
                section_title=section_title,
                card=card,
                descriptors=descriptors,
            )
        )
    if len(decision_cards) > 5:
        lines.append(f"... {len(decision_cards) - 5} more Decision Card(s)")

    lines.extend(
        [
            "",
            "Buttons open or preview canonical kb-engine actions; writes still require confirmation.",
            "Fallback: /kb status",
        ]
    )
    return {"title": "KB Review", "text": "\n".join(lines), "actions": _dashboard_descriptor_actions(ctx, target, sections)}


def _workbench_review_kind(card: dict[str, Any], descriptors: list[dict[str, Any]]) -> str:
    kind = str(card.get("kind") or "").strip().lower()
    if kind:
        return kind
    for descriptor in descriptors:
        target_kind = str(descriptor.get("target_kind") or "").strip().lower()
        if target_kind:
            return target_kind
    return "review"


def _workbench_rail_labels(descriptors: list[dict[str, Any]], *, include_skip: bool = True) -> list[str]:
    labels: list[str] = []
    for descriptor in descriptors:
        label = _short(descriptor.get("label") or descriptor.get("action_id") or "", "")
        if label and label not in labels:
            labels.append(label)
    if any(_descriptor_advisory_guidance(descriptor) for descriptor in descriptors) and "Ask LLM" not in labels:
        labels.append("Ask LLM")
    if include_skip and "Skip" not in labels:
        labels.append("Skip")
    return labels


def _workbench_compact_card_lines(
    *,
    index: int,
    section_title: str,
    card: dict[str, Any],
    descriptors: list[dict[str, Any]],
) -> list[str]:
    title = _display_text(card.get("title") or "KB review")
    detail = _display_text(card.get("detail"))
    kind = _workbench_review_kind(card, descriptors)
    target = _short(
        card.get("target")
        or next((descriptor.get("target_ref") for descriptor in descriptors if descriptor.get("target_ref")), ""),
        "",
    )
    labels = _workbench_rail_labels(descriptors)
    lines = [f"{index}. {title}"]
    if kind == "situation":
        lines.append("   Surface: Situation Review")
    elif kind == "lifecycle_candidate":
        lines.append("   Surface: Lifecycle Review")
    elif section_title:
        lines.append(f"   Surface: {section_title}")
    if detail:
        lines.append(f"   Summary: {_clip(detail, 180)}")
    scope_bits = [bit for bit in (target, f"{len(descriptors)} action{'s' if len(descriptors) != 1 else ''}") if bit]
    if scope_bits:
        lines.append("   Scope: " + " · ".join(scope_bits))
    if labels:
        lines.append("   Rail: " + ", ".join(labels[:6]))
    if kind == "situation":
        lines.append("   Writes: handoff-only until kb-engine returns a confirmed workflow.")
    if kind == "lifecycle_candidate":
        lines.append("   Writes: proposal preview only; no Hermes durable write.")
    return lines


def _dashboard_descriptor_actions(ctx: Any, target: str, sections: list[Any]) -> list[Any]:
    try:
        from tools.kb_callback_registry import KbAction
    except Exception:
        return []

    actions: list[Any] = []
    guidance_actions: list[Any] = []
    for section in sections:
        if not isinstance(section, dict):
            continue
        if str(section.get("id") or "").strip().lower() not in _WORKBENCH_SECTION_IDS:
            continue
        cards = section.get("cards") if isinstance(section.get("cards"), list) else []
        for card in cards:
            if not isinstance(card, dict):
                continue
            for descriptor in _dashboard_card_descriptors(card):
                descriptor_copy = dict(descriptor)
                label = _short(descriptor.get("label") or descriptor.get("action_id") or "Open", "Open")
                action_id = _short(descriptor.get("action_id") or label, label)
                mutation = _short(descriptor.get("mutation"), "read_only")
                target_kind = str(descriptor.get("target_kind") or "").strip()
                lifecycle_action = _lifecycle_descriptor_action(ctx, target, descriptor_copy)
                if lifecycle_action is not None:
                    actions.append(lifecycle_action)
                elif mutation == "read_only":
                    if target_kind not in DESCRIPTOR_READONLY_TARGET_KINDS:
                        continue
                    if not (descriptor.get("preview_tool") or descriptor.get("method")):
                        continue
                    actions.append(
                        KbAction(
                            label=label,
                            action_id=f"{action_id}.open",
                            handler=lambda callback_ctx, d=descriptor_copy: _render_readonly_descriptor_action(
                                ctx,
                                target,
                                descriptor=d,
                                callback_ctx=callback_ctx,
                            ),
                            metadata={
                                "target_kind": descriptor.get("target_kind"),
                                "target_ref": descriptor.get("target_ref"),
                                "preview_tool": descriptor.get("preview_tool") or descriptor.get("method"),
                            },
                        )
                    )
                elif mutation == "handoff_only":
                    if target_kind not in DESCRIPTOR_READONLY_TARGET_KINDS:
                        continue
                    actions.append(
                        KbAction(
                            label=label,
                            action_id=f"{action_id}.handoff",
                            handler=lambda callback_ctx, d=descriptor_copy: _render_handoff_descriptor_action(
                                d,
                                callback_ctx=callback_ctx,
                            ),
                            metadata={
                                "target_kind": descriptor.get("target_kind"),
                                "target_ref": descriptor.get("target_ref"),
                                "preview_tool": descriptor.get("preview_tool") or descriptor.get("method"),
                                "handoff_only": True,
                            },
                        )
                    )
                else:
                    generic = _generic_descriptor_action(ctx, target, descriptor_copy)
                    if generic is None:
                        continue
                    actions.append(generic)
                guidance = _descriptor_advisory_guidance(descriptor)
                if guidance:
                    guidance_actions.append(
                        KbAction(
                            label="Ask LLM",
                            action_id=f"{action_id}.guidance",
                            handler=lambda callback_ctx, d=descriptor_copy: _render_descriptor_guidance(
                                d,
                                title="KB LLM Guidance",
                            ),
                            metadata={
                                "target_kind": descriptor.get("target_kind"),
                                "target_ref": descriptor.get("target_ref"),
                                "advisory_only": True,
                            },
                        )
                    )
                if len(actions) >= 4:
                    return actions
    return (actions + guidance_actions)[:4]


def _descriptor_advisory_guidance(descriptor: dict[str, Any]) -> dict[str, Any]:
    guidance = descriptor.get("advisory_guidance")
    if not isinstance(guidance, dict):
        return {}
    if guidance.get("packet_type") != "kb_advisory_guidance":
        return {}
    if guidance.get("mutates_state") is True:
        return {}
    return guidance


def _guidance_status(guidance: dict[str, Any]) -> str:
    status = _short(guidance.get("status") or guidance.get("state"), "").strip().lower()
    if status:
        return status
    if guidance.get("stale") is True or _get_path(guidance, "staleness", "stale") is True:
        return "stale"
    return "available"


def _guidance_field(guidance: dict[str, Any], *keys: str, limit: int = 520) -> str:
    for key in keys:
        value = guidance.get(key)
        if isinstance(value, dict):
            value = value.get("summary") or value.get("label") or value.get("text") or value.get("message")
        if isinstance(value, list):
            value = "; ".join(_short(item, "") for item in value[:4])
        text = _redact_guidance_text(value, limit=limit)
        if text:
            return text
    return ""


def _guidance_facet_text(guidance: dict[str, Any], facet: str) -> tuple[str, str]:
    facet_key = str(facet or "summary").strip().lower()
    if facet_key == "why":
        return "Why", _guidance_field(guidance, "why", "why_now", "why_this_matters", "summary")
    if facet_key == "recommend":
        return "Recommendation", _guidance_field(guidance, "recommendation", "recommend", "recommended_action", "next_best_action")
    if facet_key == "evidence":
        return "Evidence", _guidance_field(guidance, "evidence", "evidence_summary", "evidence_refs", "supporting_evidence")
    if facet_key == "missing":
        return "Missing Context", _guidance_field(guidance, "missing_context", "evidence_gaps", "gaps", "missing")
    return "Guidance", _guidance_field(guidance, "summary")


def _descriptor_guidance_actions(descriptor: dict[str, Any], *, title: str) -> list[Any]:
    try:
        from tools.kb_callback_registry import KbAction
    except Exception:
        return []
    actions: list[Any] = []
    for facet, label in (
        ("why", "Why"),
        ("recommend", "Recommend"),
        ("evidence", "Evidence"),
        ("missing", "Missing Context"),
    ):
        actions.append(
            KbAction(
                label=label,
                action_id=f"guidance.{facet}",
                handler=lambda callback_ctx, d=dict(descriptor), f=facet: _render_descriptor_guidance(
                    d,
                    title=title,
                    facet=f,
                    include_facet_actions=False,
                ),
                metadata={
                    "target_kind": descriptor.get("target_kind"),
                    "target_ref": descriptor.get("target_ref"),
                    "advisory_only": True,
                    "guidance_facet": facet,
                },
            )
        )
    return actions


def _render_descriptor_guidance(
    descriptor: dict[str, Any],
    *,
    title: str = "KB Guidance",
    facet: str = "summary",
    include_facet_actions: bool = True,
) -> dict[str, Any]:
    guidance = _descriptor_advisory_guidance(descriptor)
    if not guidance:
        return {"title": title, "text": f"{title}\nNo advisory guidance was attached to this action.", "actions": []}
    status = _guidance_status(guidance)
    if status in {"unavailable", "blocked", "error", "failed"}:
        reason = _guidance_field(guidance, "unavailable_reason", "reason", "message", limit=240)
        lines = [
            title,
            _short(descriptor.get("label") or descriptor.get("action_id") or "KB action", "KB action"),
            "",
            f"Guidance unavailable: {reason or status}.",
            "Advisory output never confirms, applies, commits, or publishes.",
        ]
        return {"title": title, "text": "\n".join(lines), "actions": []}
    stale = status in {"stale", "expired"}
    facet_title, facet_text = _guidance_facet_text(guidance, facet)
    lines = [
        title,
        _short(descriptor.get("label") or descriptor.get("action_id") or "KB action", "KB action"),
        "",
        facet_text or "Guidance is advisory only and cannot mutate durable KB state.",
        "",
        f"Prompt: {_short(guidance.get('llm_prompt'), 'kb.review_guidance')}",
        f"Mode: {_short(guidance.get('mode'), 'advisory_only')}",
        f"Authority: {_short(guidance.get('authority'), 'no_mutation_authority')}",
        "Mutates KB: no",
    ]
    if facet != "summary":
        lines.insert(2, facet_title)
    if stale:
        lines.append("Status: stale; refresh the KB review card before relying on this guidance.")
    sequence = guidance.get("recommended_sequence") if isinstance(guidance.get("recommended_sequence"), list) else []
    if sequence and facet == "summary":
        lines.append("")
        lines.append("Suggested sequence:")
        for step in sequence[:4]:
            lines.append(f"- {_redact_guidance_text(step, limit=180)}")
    lines.append("")
    lines.append("Advisory output never confirms, applies, commits, or publishes. Use the preview/confirm button path for writes.")
    return {
        "title": title,
        "text": "\n".join(lines),
        "actions": _descriptor_guidance_actions(descriptor, title=title) if include_facet_actions and not stale else [],
    }


def _render_readonly_descriptor_action(
    ctx: Any,
    target: str,
    *,
    descriptor: dict[str, Any],
    callback_ctx: Any,
) -> dict[str, Any]:
    del callback_ctx
    method = _descriptor_tool_name(target, descriptor.get("preview_tool") or descriptor.get("method"))
    params = descriptor.get("params") if isinstance(descriptor.get("params"), dict) else {}
    payload = _result_payload(ctx.dispatch_tool(method, params))
    label = _short(descriptor.get("label") or descriptor.get("action_id") or "KB Context", "KB Context")
    if isinstance(payload, dict) and payload.get("error"):
        return {"title": label, "text": f"{label}\n{payload['error']}", "actions": []}
    packet_card = _render_supported_result_packet(payload, ctx=ctx, target=target)
    if packet_card is not None:
        return packet_card
    if not isinstance(payload, dict):
        return {"title": label, "text": f"{label}\n{_short(payload, 'No context returned.')}", "actions": []}
    title = _short(payload.get("title") or payload.get("name") or label, label)
    summary = _short(payload.get("summary") or payload.get("description") or payload.get("text"), "")
    target_ref = _short(payload.get("target_ref") or descriptor.get("target_ref"), "")
    status = _short(payload.get("status") or payload.get("state"), "")
    lines = [title]
    if summary:
        lines.append(summary)
    if status:
        lines.append(f"Status: {status}")
    if target_ref:
        lines.append(f"Ref: {target_ref}")
    lines.extend(_receipt_lines(payload, include_request=True))
    return {"title": label, "text": "\n".join(lines), "actions": []}


def _render_handoff_descriptor_action(
    descriptor: dict[str, Any],
    *,
    callback_ctx: Any,
) -> dict[str, Any]:
    del callback_ctx
    label = _short(descriptor.get("label") or descriptor.get("action_id") or "KB handoff", "KB handoff")
    target_ref = _short(descriptor.get("target_ref"), "")
    route = _short(descriptor.get("preview_tool") or descriptor.get("method") or descriptor.get("surface"), "kb-engine")
    required = descriptor.get("required_inputs")
    if isinstance(required, str):
        required_values = [required]
    elif isinstance(required, list):
        required_values = [str(value).strip() for value in required if str(value).strip()]
    else:
        required_values = []
    lines = [
        label,
        "This is a kb-engine handoff action, not a durable write.",
    ]
    description = _short(descriptor.get("description") or descriptor.get("expected_result"), "")
    if description:
        lines.append(_clip(description, 220))
    if target_ref:
        lines.append(f"Target: {target_ref}")
    lines.append(f"Route: {route}")
    if required_values:
        lines.append("Required input: " + ", ".join(required_values))
        lines.append("Send the missing context explicitly, then preview through kb-engine before any confirmation.")
    else:
        lines.append("Preview through kb-engine before any confirmation.")
    lines.append("No KB state changed.")
    return {"title": label, "text": "\n".join(lines), "actions": []}


def _descriptor_params(descriptor: dict[str, Any]) -> dict[str, Any]:
    params = descriptor.get("params") if isinstance(descriptor.get("params"), dict) else {}
    return dict(params)


def _generic_descriptor_action(ctx: Any, target: str, descriptor: dict[str, Any]) -> Any | None:
    try:
        from tools.kb_callback_registry import KbAction
    except Exception:
        return None
    if descriptor.get("requires_canonical_tool") is not True:
        return None
    preview_tool = _descriptor_tool_name(target, descriptor.get("preview_tool") or descriptor.get("method"))
    confirm_tool = _descriptor_tool_name(target, descriptor.get("confirm_tool"))
    if not preview_tool or not confirm_tool:
        return None
    if str(descriptor.get("target_kind") or "").strip() not in DESCRIPTOR_WRITE_TARGET_KINDS:
        return None
    label = _short(descriptor.get("label") or descriptor.get("action_id") or "Action", "Action")
    action_id = _short(descriptor.get("action_id") or label, label)
    return KbAction(
        label=f"Preview {label}",
        action_id=f"{action_id}.preview",
        handler=lambda callback_ctx, d=dict(descriptor): _render_generic_descriptor_preview(
            ctx,
            target,
            descriptor=d,
            callback_ctx=callback_ctx,
        ),
        metadata={
            "target_kind": descriptor.get("target_kind"),
            "target_ref": descriptor.get("target_ref"),
            "preview_tool": preview_tool,
            "confirm_tool": confirm_tool,
            "preview_required": True,
        },
    )


def _generic_preview_text(label: str, payload: Any) -> str:
    if isinstance(payload, dict) and payload.get("error"):
        return f"{label} Preview Failed\n{payload['error']}"
    packet_card = _render_supported_result_packet(payload)
    if packet_card is not None:
        return packet_card["text"]
    if isinstance(payload, dict):
        lines = [
            f"{label} Preview",
            f"Status: {_short(payload.get('status') or payload.get('state'))}",
        ]
        lines.extend(_receipt_lines(payload, include_request=True))
        changed_paths = _changed_paths(payload)
        if changed_paths:
            lines.append(f"Changed paths: {len(changed_paths)}")
            lines.extend(_format_changed_paths(changed_paths, limit=5))
        lines.extend(_queue_scope_lines(payload))
        summary = _short(payload.get("summary") or payload.get("message"), "")
        if summary:
            lines.append("Summary: " + _clip(summary, 260))
        return "\n".join(lines)
    return f"{label} Preview\n{_short(payload, 'No structured response returned.')}"


def _render_generic_descriptor_preview(
    ctx: Any,
    target: str,
    *,
    descriptor: dict[str, Any],
    callback_ctx: Any,
) -> dict[str, Any]:
    try:
        from tools.kb_callback_registry import KbAction
    except Exception:
        return {"title": "KB Action", "text": "KB Action\nAction buttons are unavailable.", "actions": []}
    del callback_ctx
    label = _short(descriptor.get("label") or descriptor.get("action_id") or "KB Action", "KB Action")
    action_id = _short(descriptor.get("action_id") or label, label)
    preview_tool = _descriptor_tool_name(target, descriptor.get("preview_tool") or descriptor.get("method"))
    preview_payload = _result_payload(ctx.dispatch_tool(preview_tool, _descriptor_params(descriptor)))
    text = _generic_preview_text(label, preview_payload)
    if not _preview_allows_confirmation(preview_payload):
        return {"title": label, "text": text, "actions": []}
    preview_metadata = _queue_preview_metadata(preview_payload)
    confirm_action = KbAction(
        label=f"Confirm {label}",
        action_id=f"{action_id}.confirm",
        handler=lambda confirm_ctx, d=dict(descriptor), metadata=dict(preview_metadata): _render_generic_descriptor_confirm(
            ctx,
            target,
            descriptor=d,
            callback_ctx=confirm_ctx,
            preview_metadata=metadata,
        ),
        metadata={
            "target_kind": descriptor.get("target_kind"),
            "target_ref": descriptor.get("target_ref"),
            "preview_required": True,
            "preview_lease": bool(preview_metadata.get("preview_lease")),
            "review_session_id": _review_session_id(preview_metadata),
        },
    )
    return {
        "title": label,
        "text": text + "\n\nConfirm with the button below only if the preview matches your intent.",
        "actions": [confirm_action],
    }


def _render_generic_descriptor_confirm(
    ctx: Any,
    target: str,
    *,
    descriptor: dict[str, Any],
    callback_ctx: Any,
    preview_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    label = _short(descriptor.get("label") or descriptor.get("action_id") or "KB Action", "KB Action")
    preview_tool = _descriptor_tool_name(target, descriptor.get("preview_tool") or descriptor.get("method"))
    confirm_tool = _descriptor_tool_name(target, descriptor.get("confirm_tool"))
    effective_metadata = dict(preview_metadata or {})
    if not effective_metadata.get("preview_lease"):
        preview_payload = _result_payload(ctx.dispatch_tool(preview_tool, _descriptor_params(descriptor)))
        if not _preview_allows_confirmation(preview_payload):
            return {"title": label, "text": _generic_preview_text(label, preview_payload), "actions": []}
        effective_metadata.update(_queue_preview_metadata(preview_payload))
    confirm_args = _descriptor_params(descriptor)
    _apply_queue_preview_metadata(confirm_args, effective_metadata)
    confirm_args["user_confirmation"] = {
        "confirmed": True,
        "surface": "telegram",
        "action": _short(descriptor.get("action_id") or label, label),
        "preview_required": True,
        "confirmation_text": str(descriptor.get("confirmation_copy") or f"Confirm {label}"),
        "actor_id": _short(getattr(callback_ctx, "actor_id", ""), ""),
        "actor_name": _short(getattr(callback_ctx, "actor_name", ""), ""),
    }
    _apply_queue_confirmation_preview_metadata(confirm_args["user_confirmation"], effective_metadata)
    confirm_args.setdefault("actor", _queue_callback_actor(callback_ctx))
    confirm_args.setdefault("source", "Hermes Telegram Action Card")
    confirm_args.setdefault("session_id", _review_session_id(effective_metadata) or f"telegram-kb-card-{int(time.time())}")
    confirmed_payload = _result_payload(ctx.dispatch_tool(confirm_tool, confirm_args))
    packet_card = _render_supported_result_packet(confirmed_payload, ctx=ctx, target=target)
    if packet_card is not None:
        return packet_card
    return {"title": label, "text": _generic_preview_text(label.replace("Preview", "Applied"), confirmed_payload), "actions": []}


def _strip_env_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _selected_env_values(keys: set[str]) -> dict[str, str]:
    values = {key: value for key in keys if (value := os.getenv(key))}
    missing = keys.difference(values)
    if not missing:
        return values
    try:
        from hermes_cli.config import get_env_path

        env_path = get_env_path()
        lines = env_path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    except Exception:
        return values
    for line in lines:
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        key = key.strip()
        if key in missing and key not in values:
            values[key] = _strip_env_value(value)
    return values


def _first_env(env: dict[str, str], *keys: str) -> str | None:
    for key in keys:
        value = env.get(key)
        if value:
            return value
    return None


def _config_snapshot() -> dict[str, str]:
    config: dict[str, Any] = {}
    env_keys = {
        "ANTHROPIC_API_KEY",
        "ENVIRONMENT",
        "HERMES_API_MODE",
        "HERMES_ENV",
        "HERMES_ENVIRONMENT",
        "HERMES_KB_LANE",
        "HERMES_KB_LLM_MODEL",
        "HERMES_KB_LLM_PROVIDER",
        "HERMES_KB_MODE",
        "HERMES_KB_MODEL",
        "HERMES_KB_PROVIDER",
        "HERMES_KB_REASONING_EFFORT",
        "HERMES_KB_WORKSPACE",
        "HERMES_MODEL",
        "HERMES_MODEL_API_MODE",
        "HERMES_PROFILE",
        "HERMES_PROVIDER",
        "HERMES_REASONING_EFFORT",
        "KB_LLM_MODEL",
        "KB_LLM_PROVIDER",
        "KB_LLM_REASONING_EFFORT",
        "KB_OPENAI_COMPAT_MODEL",
        "KB_PROVIDER",
        "KB_WORKSPACE",
        "MODEL",
        "MODEL_PROVIDER",
        "NVIDIA_API_KEY",
        "OPENAI_API_KEY",
        "OPENAI_REASONING_EFFORT",
        "OPENROUTER_API_KEY",
    }
    env = _selected_env_values(env_keys)
    try:
        from hermes_cli.config import load_config

        loaded = load_config()
        if isinstance(loaded, dict):
            config = loaded
    except Exception:
        config = {}

    model_cfg = config.get("model")
    agent_cfg = config.get("agent") if isinstance(config.get("agent"), dict) else {}
    if isinstance(model_cfg, dict):
        model = model_cfg.get("default") or model_cfg.get("name") or model_cfg.get("model")
        provider = model_cfg.get("provider")
        api_mode = model_cfg.get("api_mode")
        reasoning = (
            agent_cfg.get("reasoning_effort")
            or model_cfg.get("reasoning_effort")
            or model_cfg.get("reasoning")
        )
    else:
        model = model_cfg
        provider = config.get("provider")
        api_mode = None
        reasoning = None

    provider = provider or _first_env(env, "HERMES_PROVIDER", "MODEL_PROVIDER")
    model = model or _first_env(env, "HERMES_MODEL", "MODEL")
    reasoning = reasoning or _first_env(env, "HERMES_REASONING_EFFORT", "OPENAI_REASONING_EFFORT")
    api_mode = api_mode or _first_env(env, "HERMES_MODEL_API_MODE", "HERMES_API_MODE")

    api_envs = [
        "NVIDIA_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENROUTER_API_KEY",
    ]
    configured = [name.removesuffix("_API_KEY") for name in api_envs if env.get(name)]

    return {
        "lane": _first_env(env, "HERMES_KB_MODE", "HERMES_KB_LANE", "HERMES_PROFILE") or "unknown",
        "environment": (
            _first_env(env, "HERMES_ENVIRONMENT", "HERMES_ENV", "ENVIRONMENT", "HERMES_PROFILE")
            or "unknown"
        ),
        "workspace": _first_env(env, "HERMES_KB_WORKSPACE", "KB_WORKSPACE") or "not set",
        "model": _short(model, "not set"),
        "provider": _short(provider, "not set"),
        "api_mode": _short(api_mode, "not set"),
        "api": ", ".join(configured) if configured else "not detected",
        "reasoning": _short(reasoning, "not set"),
        "kb_provider": _short(
            _first_env(env, "HERMES_KB_LLM_PROVIDER", "HERMES_KB_PROVIDER", "KB_LLM_PROVIDER", "KB_PROVIDER"),
            "unknown",
        ),
        "kb_model": _short(
            _first_env(env, "HERMES_KB_LLM_MODEL", "HERMES_KB_MODEL", "KB_LLM_MODEL", "KB_OPENAI_COMPAT_MODEL"),
            "unknown",
        ),
        "kb_reasoning": _short(
            _first_env(env, "HERMES_KB_REASONING_EFFORT", "KB_LLM_REASONING_EFFORT"),
            "unknown",
        ),
    }


def _primary_provider_target(provider_data: Any) -> dict[str, Any]:
    if not isinstance(provider_data, dict):
        return {}
    targets = provider_data.get("targets")
    if not isinstance(targets, list):
        return {}
    dict_targets = [target for target in targets if isinstance(target, dict)]
    for item in dict_targets:
        if _short(item.get("role"), "").lower() == "primary":
            return item
    return dict_targets[0] if dict_targets else {}


def _provider_status_summary(provider_data: Any, fallback: dict[str, str] | None = None) -> dict[str, str]:
    primary = _primary_provider_target(provider_data)
    if not primary:
        fallback = fallback or {}
        return {
            "provider": fallback.get("kb_provider") or "unknown",
            "model": fallback.get("kb_model") or "unknown",
            "reasoning": fallback.get("kb_reasoning") or "unknown",
            "status": "unknown",
        }
    return {
        "provider": _short(primary.get("adapter") or primary.get("provider")),
        "model": _short(primary.get("model")),
        "reasoning": _short(primary.get("reasoning_effort"), "not set"),
        "status": _short(primary.get("status")),
    }


def _live_hermes_reasoning(gateway: Any, source: Any) -> str | None:
    resolver = getattr(gateway, "_resolve_session_reasoning_config", None)
    if not callable(resolver):
        return None
    try:
        reasoning_config = resolver(source=source)
    except TypeError:
        try:
            reasoning_config = resolver()
        except Exception:
            return None
    except Exception:
        return None
    if not isinstance(reasoning_config, dict):
        return None
    if reasoning_config.get("enabled") is False:
        return "none"
    effort = str(reasoning_config.get("effort") or "").strip().lower()
    return effort or None


def _render_status(
    data: Any,
    target: str,
    provider_data: Any | None = None,
    *,
    hermes_reasoning: str | None = None,
) -> dict[str, Any]:
    snap = _config_snapshot()
    if hermes_reasoning:
        snap["reasoning"] = hermes_reasoning
    kb = _provider_status_summary(provider_data, snap)
    if isinstance(data, dict) and data.get("kind") in {"kb_status_proof_packet", "noc_kb_status_receipt"}:
        return _render_status_proof(data, target, kb, snap)
    readiness = "unknown"
    publication = "unknown"
    if isinstance(data, dict):
        readiness = _short(_readiness_status(data))
        publication = _short(_publication_status(data))
    lines = [
        "KB Status",
        f"Lane: {snap['lane']}",
        f"Environment: {snap['environment']}",
        f"MCP target: {target}",
        f"Workspace: {snap['workspace']}",
        f"Hermes model: {snap['model']}",
        f"Hermes provider/API: {snap['provider']} / {snap['api_mode']} / {snap['api']}",
        f"Hermes reasoning: {snap['reasoning']}",
        f"KB provider: {kb['provider']}",
        f"KB model: {kb['model']}",
        f"KB reasoning: {kb['reasoning']}",
        f"Readiness: {readiness}",
        f"Publication: {publication}",
    ]
    return {"title": "KB Status", "text": "\n".join(lines), "actions": []}


def _status_line_value(packet: dict[str, Any], *paths: tuple[str, ...], default: str = "unknown") -> str:
    for path in paths:
        value = _get_path(packet, *path)
        if value not in (None, "", [], {}):
            return _short(value, default)
    return default


def _dirty_summary(packet: dict[str, Any]) -> str:
    dirty_scope = packet.get("dirty_scope") if isinstance(packet.get("dirty_scope"), dict) else {}
    worktrees = packet.get("worktrees") if isinstance(packet.get("worktrees"), dict) else {}
    publication = packet.get("publication") if isinstance(packet.get("publication"), dict) else {}
    dirty_count = None
    for candidate in (
        dirty_scope.get("count"),
        dirty_scope.get("dirty_path_count"),
        worktrees.get("dirty_path_count"),
        publication.get("dirty_path_count"),
        publication.get("changed_count"),
    ):
        if candidate is not None:
            dirty_count = candidate
            break
    if dirty_count not in (None, ""):
        return f"{_short(dirty_count, '0')} dirty"
    dirty_bits = [
        name
        for name, value in dirty_scope.items()
        if isinstance(value, bool) and value
    ]
    if dirty_bits:
        return ", ".join(dirty_bits[:4])
    return _status_line_value(packet, ("worktrees", "status"), ("workspace", "status"), default="unknown")


def _next_action_summary(packet: dict[str, Any]) -> str:
    next_action = packet.get("next_action")
    if isinstance(next_action, dict):
        return _short(
            next_action.get("command")
            or next_action.get("label")
            or next_action.get("summary")
            or next_action.get("next_safe_action"),
            "",
        )
    return _short(next_action, "")


def _render_status_proof(
    packet: dict[str, Any],
    target: str,
    kb: dict[str, str],
    snap: dict[str, str],
) -> dict[str, Any]:
    status = _short(packet.get("status") or packet.get("state"), "unknown")
    lane = _status_line_value(
        packet,
        ("active_target", "target"),
        ("active_target", "name"),
        ("workspace", "lane"),
        ("workspace", "target"),
        default=target,
    )
    runtime = _status_line_value(
        packet,
        ("runtime", "version"),
        ("runtime", "installed_ref"),
        ("runtime", "status"),
    )
    transport = _status_line_value(
        packet,
        ("transport", "status"),
        ("runtime", "transport_status"),
        ("runtime", "mcp_transport_status"),
    )
    publication = _status_line_value(
        packet,
        ("publication", "status"),
        ("publication", "state"),
        ("publication", "publication_state"),
    )
    review = _status_line_value(
        packet,
        ("review", "pending_count"),
        ("review", "status"),
        ("review", "state"),
    )
    sync = _status_line_value(
        packet,
        ("sync", "status"),
        ("sync", "state"),
        ("sync", "last_run_status"),
    )
    privacy = packet.get("privacy") if isinstance(packet.get("privacy"), dict) else {}
    privacy_ok = not any(bool(value) for value in privacy.values())
    next_action = _next_action_summary(packet)
    lines = [
        "KB Status",
        "Request: /kb status",
        f"Outcome: {status}",
        f"Lane: {lane}",
        f"Runtime: {runtime}",
        f"Transport: {transport}",
        f"Publication: {publication}",
        f"Pending review: {review}",
        f"Sync: {sync}",
        f"Dirty: {_dirty_summary(packet)}",
        f"Privacy: {'ok' if privacy_ok else 'check receipt'}",
        f"KB model: {kb['provider']} / {kb['model']} / {kb['reasoning']}",
        f"KB reasoning: {kb['reasoning']}",
        f"Hermes reasoning: {snap['reasoning']}",
    ]
    if next_action:
        lines.append(f"Next: {next_action}")
    lines.append("Commands: /kb sync · /kb review")
    return {"title": "KB Status", "text": "\n".join(lines), "actions": []}


def _render_runs(data: Any) -> dict[str, Any]:
    if isinstance(data, str):
        return {"title": "KB Runs", "text": f"KB Runs\n{data}", "actions": []}
    if isinstance(data, dict):
        runs = []
        for found in (
            _get_path(data, "active"),
            _get_path(data, "runs", "active"),
            _get_path(data, "recent"),
            _get_path(data, "runs", "recent"),
        ):
            if isinstance(found, list):
                runs.extend(found)
        if not runs:
            found = _get_path(data, "runs")
            runs = found if isinstance(found, list) else []
    else:
        runs = []
    lines = ["KB Runs"]
    if not runs:
        lines.append("No active or recent run details returned.")
    for idx, run in enumerate(runs[:6], start=1):
        if isinstance(run, dict):
            status = _short(run.get("status") or run.get("state") or run.get("phase"))
            detail = _short(run.get("summary") or run.get("message") or run.get("updated_at"), "")
            staleness = run.get("staleness") if isinstance(run.get("staleness"), dict) else {}
            if staleness.get("stale"):
                detail = f"stalled {_short(staleness.get('last_trace_age_seconds'), 'unknown')}s"
                action = _short(run.get("recommended_next_action"), "")
                if action:
                    detail += f"; {action}"
            suffix = f" - {detail}" if detail else ""
            title = _short(run.get("run_id") or run.get("workflow_id") or _item_title(run))
            lines.append(f"{idx}. {title}: {status}{suffix}")
        else:
            lines.append(f"{idx}. {_short(run)}")
    return {"title": "KB Runs", "text": "\n".join(lines), "actions": []}


def _proposal_ids_for_item(item: Any) -> list[str]:
    if not isinstance(item, dict):
        return []
    raw = item.get("raw") if isinstance(item.get("raw"), dict) else {}
    proposal_ids = raw.get("proposal_ids") or item.get("proposal_ids") or []
    if isinstance(proposal_ids, str):
        proposal_ids = [proposal_ids]
    return [str(pid).strip() for pid in proposal_ids if str(pid).strip()]


def _preview_lease_payload(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return None
    for candidate in (
        payload.get("preview_lease"),
        payload.get("lease"),
        _get_path(payload, "preview", "preview_lease"),
        _get_path(payload, "preview", "lease"),
    ):
        if isinstance(candidate, dict):
            return dict(candidate)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def _review_session_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    for candidate in (
        payload.get("review_session"),
        payload.get("preview_session"),
        _get_path(payload, "preview", "review_session"),
        _get_path(payload, "preview", "preview_session"),
    ):
        if isinstance(candidate, dict):
            return dict(candidate)
    return {}


def _safe_scope_label(value: Any) -> str:
    text = _clip(value, 120)
    if not text:
        return ""
    # Scope text is user-facing; avoid rendering path-like or secret-like blobs
    # from backend metadata. Counts are rendered separately.
    if any(marker in text for marker in ("/", "\\", "~", "$", "://")):
        return ""
    return text


def _redact_guidance_text(value: Any, *, limit: int = 520) -> str:
    text = _clip(value, limit)
    if not text:
        return ""
    text = re.sub(r"(?i)\b(?:token|secret|api[_-]?key|private[_-]?key)\s*[:=]\s*\S+", "[redacted]", text)
    text = re.sub(r"(?i)\b(?:sk|ghp|gho|github_pat|xox[abprs])_[A-Za-z0-9_\-]{12,}", "[redacted]", text)
    text = re.sub(r"(?i)(?:/Users|/home|/private|/tmp|~)/\S+", "[redacted-path]", text)
    text = re.sub(r"(?i)\b\S*(?:\.env|id_rsa|id_ed25519|credentials|token|cache)\S*\b", "[redacted]", text)
    return text


def _queue_count_value(*values: Any) -> int | None:
    for value in values:
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        if number >= 0:
            return number
    return None


def _queue_preview_metadata(payload: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    lease = _preview_lease_payload(payload)
    if lease:
        metadata["preview_lease"] = lease
    review_session = _review_session_payload(payload)
    if review_session:
        metadata["review_session"] = review_session
        metadata["preview_session"] = review_session
    if isinstance(payload, dict):
        expected_before_hash = _short(
            payload.get("expected_before_hash")
            or payload.get("expectedBeforeHash")
            or payload.get("before_hash")
            or _get_path(payload, "preview", "expected_before_hash")
            or _get_path(payload, "preview", "before_hash"),
            "",
        )
        if expected_before_hash:
            metadata["expected_before_hash"] = expected_before_hash
    return metadata


def _apply_queue_preview_metadata(args: dict[str, Any], metadata: dict[str, Any]) -> None:
    if not metadata:
        return
    expected_before_hash = _short(metadata.get("expected_before_hash"), "")
    if expected_before_hash:
        args.setdefault("expected_before_hash", expected_before_hash)
    review_session_id = _review_session_id(metadata)
    if review_session_id:
        args.setdefault("review_session_id", review_session_id)
    cursor_id = _queue_cursor_id(metadata)
    if cursor_id:
        args.setdefault("cursor_id", cursor_id)
    decision_scope = _queue_decision_scope(metadata)
    if decision_scope:
        # The confirmed write must follow the backend preview lease, even when a
        # descriptor supplied a conservative default scope.
        args["decision_scope"] = decision_scope


def _apply_queue_confirmation_preview_metadata(user_confirmation: dict[str, Any], metadata: dict[str, Any]) -> None:
    if not metadata:
        return
    lease = metadata.get("preview_lease")
    if isinstance(lease, dict) and lease:
        user_confirmation["preview_lease"] = dict(lease)
    elif isinstance(lease, str) and lease.strip():
        user_confirmation["preview_lease_id"] = lease.strip()
    review_session_id = _review_session_id(metadata)
    if review_session_id:
        user_confirmation["review_session_id"] = review_session_id


def _review_session_id(metadata: dict[str, Any]) -> str:
    review_session = metadata.get("review_session") or metadata.get("preview_session")
    if not isinstance(review_session, dict):
        return ""
    return _short(
        review_session.get("session_id")
        or review_session.get("review_session_id")
        or review_session.get("preview_session_id"),
        "",
    )


def _queue_cursor_id(metadata: dict[str, Any]) -> str:
    lease = metadata.get("preview_lease")
    if isinstance(lease, dict):
        cursor_id = _short(lease.get("cursor_id"), "")
        if cursor_id:
            return cursor_id
    review_session = metadata.get("review_session") or metadata.get("preview_session")
    if isinstance(review_session, dict):
        cursor_id = _short(review_session.get("cursor_id"), "")
        if cursor_id:
            return cursor_id
        cursor = review_session.get("cursor")
        if isinstance(cursor, dict):
            return _short(cursor.get("cursor_id"), "")
    return ""


def _queue_decision_scope(metadata: dict[str, Any]) -> str:
    lease = metadata.get("preview_lease")
    if isinstance(lease, dict):
        scope = _queue_scope_value(lease.get("decision_scope")) or _queue_scope_value(lease.get("scope"))
        if scope:
            return scope
    review_session = metadata.get("review_session") or metadata.get("preview_session")
    if isinstance(review_session, dict):
        return _queue_scope_value(review_session.get("decision_scope")) or _queue_scope_value(review_session.get("scope"))
    return ""


def _queue_scope_value(value: Any) -> str:
    if isinstance(value, str):
        return _short(value, "")
    if isinstance(value, dict):
        return _short(value.get("scope_type") or value.get("scope") or value.get("type"), "")
    return ""


def _queue_scope_display_label(value: Any) -> str:
    scope = _queue_scope_value(value).strip().lower()
    if not scope:
        return ""
    aliases = {
        "all_viewed": "Visible",
        "all_window": "Window",
        "all_filtered": "Filter",
        "explicit_ids": "Selected",
    }
    return aliases.get(scope, _safe_scope_label(scope.replace("_", " ").title()))


def _queue_scope_lines(payload: Any) -> list[str]:
    review_session = _review_session_payload(payload)
    if not review_session:
        return []
    lease = _preview_lease_payload(payload)
    cursor = review_session.get("cursor") if isinstance(review_session.get("cursor"), dict) else {}
    scope = review_session.get("scope") if isinstance(review_session.get("scope"), dict) else {}
    scope_label = _safe_scope_label(
        review_session.get("scope_label")
        or review_session.get("scope_description")
        or scope.get("label")
        or scope.get("description")
        or _queue_scope_display_label(review_session.get("decision_scope"))
        or _queue_scope_display_label(review_session.get("scope"))
    )
    item_count = _queue_count_value(
        review_session.get("item_count"),
        review_session.get("selected_item_count"),
        review_session.get("selected_count"),
        cursor.get("displayed_count"),
        scope.get("item_count"),
        scope.get("selected_count"),
    )
    proposal_count = _queue_count_value(
        review_session.get("proposal_count"),
        review_session.get("selected_proposal_count"),
        len(lease.get("proposal_ids", [])) if isinstance(lease, dict) and isinstance(lease.get("proposal_ids"), list) else None,
        scope.get("proposal_count"),
    )
    lines: list[str] = []
    if scope_label:
        lines.append(f"Scope: {scope_label}")
    count_bits: list[str] = []
    if item_count is not None:
        count_bits.append(f"{item_count} item(s)")
    if proposal_count is not None:
        count_bits.append(f"{proposal_count} proposal(s)")
    if count_bits:
        lines.append("Review session: " + " · ".join(count_bits))
    return lines


def _result_payload(raw: Any) -> Any:
    payload, error = _unwrap_tool_result(raw)
    if error:
        return {"error": error}
    return payload


def _item_kind(item: Any) -> str:
    if not isinstance(item, dict):
        return ""
    raw = item.get("raw") if isinstance(item.get("raw"), dict) else {}
    return _short(item.get("kind") or item.get("type") or raw.get("kind") or raw.get("type"), "")


def _item_target(item: Any) -> str:
    if not isinstance(item, dict):
        return ""
    raw = item.get("raw") if isinstance(item.get("raw"), dict) else {}
    return _short(
        item.get("entity_path")
        or item.get("target")
        or item.get("item_id")
        or raw.get("entity_path")
        or raw.get("target")
        or raw.get("item_id"),
        "",
    )


def _item_detail(item: Any) -> str:
    if not isinstance(item, dict):
        return _short(item, "")
    raw = item.get("raw") if isinstance(item.get("raw"), dict) else {}
    return _short(
        item.get("preview")
        or item.get("why")
        or item.get("summary")
        or item.get("description")
        or item.get("detail")
        or raw.get("preview")
        or raw.get("why")
        or raw.get("summary")
        or raw.get("description")
        or raw.get("detail"),
        "",
    )


def _safe_actions_for_item(item: Any) -> list[dict[str, Any]]:
    if not isinstance(item, dict):
        return []
    actions = item.get("safe_actions")
    if not isinstance(actions, list):
        return []
    return [action for action in actions if isinstance(action, dict)]


def _queue_action_decisions(item: dict[str, Any]) -> list[tuple[str, str]]:
    decisions: list[tuple[str, str]] = []
    seen: set[str] = set()
    for action in _safe_actions_for_item(item):
        params = action.get("params") if isinstance(action.get("params"), dict) else {}
        decision = str(params.get("decision") or "").strip().lower()
        if not decision or decision in seen:
            continue
        label = _short(action.get("label") or decision.replace("_", " ").title(), "")
        if not label:
            continue
        seen.add(decision)
        decisions.append((decision, label))
    if any(decision in {"complete", "keep", "demote"} for decision, _ in decisions):
        order = {"complete": 0, "keep": 1, "demote": 2, "archive": 3, "skip": 4}
    else:
        order = {"approve": 0, "reject": 1, "archive": 2, "skip": 3}
    decisions.sort(key=lambda pair: (order.get(pair[0], 99), pair[0]))
    return decisions


def _queue_descriptor_decisions(item: dict[str, Any]) -> list[tuple[str, str]]:
    descriptor_item = dict(item)
    descriptor_item["safe_actions"] = [
        action
        for action in _safe_actions_for_item(item)
        if action.get("dashboard_owned_write") is not True
        and action.get("preview_tool")
        and action.get("confirm_tool")
    ]
    return _queue_action_decisions(descriptor_item)


def _queue_decision_commands(item: dict[str, Any], *, index: int) -> list[str]:
    decisions = _queue_action_decisions(item)
    if not decisions:
        decisions = [
            ("reject", "Reject"),
            ("archive", "Archive"),
            ("approve", "Approve"),
        ]
    lines: list[str] = []
    for decision, label in decisions:
        if decision == "approve":
            label = "Approve proposal"
        elif decision == "reject":
            label = "Reject proposal"
        elif decision == "archive" and not any(d in {"complete", "keep", "demote"} for d, _ in decisions):
            label = "Archive proposal"
        lines.append(f"- {label}: /kb review {decision} {index}")
    if decisions:
        example_decision = decisions[0][0]
        lines.append(f"Confirm from the preview button; text fallback: /kb review {example_decision} {index} confirm")
    return lines


def _descriptor_tool_name(target: str, tool_name: Any) -> str:
    value = str(tool_name or "").strip()
    if not value:
        return ""
    if value.startswith("mcp_"):
        return value
    return _mcp_tool_name(target, value)


def _queue_descriptor_actions(
    ctx: Any,
    target: str,
    item: dict[str, Any],
    *,
    index: int,
    preview_label_prefix: bool = True,
    limit: int | None = 4,
) -> list[Any]:
    try:
        from tools.kb_callback_registry import KbAction
    except Exception:
        return []

    actions: list[Any] = []
    guidance_descriptor = next(
        (
            descriptor
            for descriptor in _safe_actions_for_item(item)
            if isinstance(descriptor, dict) and _descriptor_advisory_guidance(descriptor)
        ),
        None,
    )
    for descriptor in _safe_actions_for_item(item):
        if descriptor.get("dashboard_owned_write") is True:
            continue
        params = descriptor.get("params") if isinstance(descriptor.get("params"), dict) else {}
        decision = str(params.get("decision") or "").strip().lower()
        preview_tool = _descriptor_tool_name(target, descriptor.get("preview_tool"))
        confirm_tool = _descriptor_tool_name(target, descriptor.get("confirm_tool"))
        if not decision or not preview_tool or not confirm_tool:
            continue
        label = _short(descriptor.get("label") or decision.replace("_", " ").title(), decision.title())
        action_id = _short(descriptor.get("action_id") or f"queue.{decision}", f"queue.{decision}")
        descriptor_copy = dict(descriptor)
        actions.append(
            KbAction(
                label=f"Preview {label}" if preview_label_prefix else label,
                action_id=f"{action_id}.preview",
                handler=lambda callback_ctx, d=descriptor_copy: _render_queue_descriptor_preview(
                    ctx,
                    target,
                    item,
                    index=index,
                    descriptor=d,
                    callback_ctx=callback_ctx,
                ),
                metadata={
                    "target_kind": "proposal_queue",
                    "target_ref": _item_target(item),
                    "decision": decision,
                    "preview_tool": preview_tool,
                    "confirm_tool": confirm_tool,
                },
            )
        )
    if guidance_descriptor:
        descriptor_copy = dict(guidance_descriptor)
        actions.append(
            KbAction(
                label="Ask LLM",
                action_id="queue.advisory_guidance",
                handler=lambda callback_ctx, d=descriptor_copy: _render_descriptor_guidance(
                    d,
                    title="KB Review LLM Guidance",
                ),
                metadata={
                    "target_kind": "proposal_queue",
                    "target_ref": _item_target(item),
                    "advisory_only": True,
                },
            )
        )
    if limit is None:
        return actions
    return actions[:limit]


def _queue_descriptor_call_args(
    descriptor: dict[str, Any],
    item: dict[str, Any],
    *,
    decision: str,
    actor: str,
    source: str,
    note: str,
) -> dict[str, Any]:
    params = descriptor.get("params") if isinstance(descriptor.get("params"), dict) else {}
    proposal_ids = [str(proposal_id) for proposal_id in (params.get("proposal_ids") or []) if str(proposal_id)]
    if not proposal_ids:
        proposal_ids = _proposal_ids_for_item(item)
    args = dict(params)
    args["proposal_ids"] = proposal_ids
    args["decision"] = decision
    args.setdefault("decision_scope", "explicit_ids")
    args.setdefault("displayed_count", len(proposal_ids))
    args.setdefault("candidate_count", len(proposal_ids))
    args["actor"] = actor
    args["source"] = source
    args["note"] = note
    review_metadata = _queue_item_review_metadata(item)
    _apply_queue_preview_metadata(args, review_metadata)
    review_session_id = _review_session_id(review_metadata)
    if review_session_id:
        args.setdefault("session_id", review_session_id)
    return args


def _queue_item_review_metadata(item: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    raw = item.get("raw") if isinstance(item.get("raw"), dict) else {}
    for candidate in (item.get("preview_lease"), raw.get("preview_lease")):
        if isinstance(candidate, dict) and candidate:
            metadata["preview_lease"] = dict(candidate)
            break
    for candidate in (item.get("review_session"), raw.get("review_session")):
        if isinstance(candidate, dict) and candidate:
            metadata["review_session"] = dict(candidate)
            metadata["preview_session"] = dict(candidate)
            break
    return metadata


def _review_target_payload(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {}
    raw = item.get("raw") if isinstance(item.get("raw"), dict) else {}
    for candidate in (item.get("review_target"), raw.get("review_target")):
        if isinstance(candidate, dict) and candidate.get("packet_type") == "guided_kb_review_target":
            return dict(candidate)
    return {}


def _review_target_policy(item: Any) -> dict[str, Any]:
    target = _review_target_payload(item)
    policy = target.get("policy") if isinstance(target.get("policy"), dict) else {}
    return dict(policy)


def _control_actions_for_item(item: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for action in _safe_actions_for_item(item):
        route = action.get("confirmed_write_route")
        if isinstance(route, dict) and route.get("operation_id"):
            result.append(action)
    return result


def _action_required_inputs(action: dict[str, Any]) -> list[str]:
    route = action.get("confirmed_write_route") if isinstance(action.get("confirmed_write_route"), dict) else {}
    values = route.get("required_input")
    if values is None:
        values = action.get("required_inputs")
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        return []
    return [str(value).strip() for value in values if str(value).strip()]


def _control_action_decisions(item: dict[str, Any]) -> list[tuple[str, str]]:
    decisions: list[tuple[str, str]] = []
    for action in _control_actions_for_item(item):
        route = action.get("confirmed_write_route") if isinstance(action.get("confirmed_write_route"), dict) else {}
        operation_id = _short(route.get("operation_id") or _get_path(action, "params", "operation_id"), "")
        if not operation_id:
            continue
        label = _short(action.get("label") or operation_id.rsplit(".", 1)[-1].title(), "")
        if label:
            decisions.append((operation_id, label))
    order = {"todo.complete": 0, "todo.delegate": 1, "todo.archive": 2}
    decisions.sort(key=lambda pair: (order.get(pair[0], 99), pair[0]))
    return decisions


def _queue_review_action_labels(item: dict[str, Any]) -> list[str]:
    descriptor_labels = [label for _decision, label in _queue_descriptor_decisions(item)]
    if descriptor_labels:
        return descriptor_labels
    control_labels = [label for _operation, label in _control_action_decisions(item)]
    if control_labels:
        return control_labels
    target = _review_target_payload(item)
    action_ids = target.get("action_ids") if isinstance(target.get("action_ids"), list) else []
    labels: list[str] = []
    for action_id in action_ids:
        label = _review_action_label(action_id)
        if label and label not in labels:
            labels.append(label)
    return labels


def _review_action_label(action_id: Any) -> str:
    value = str(action_id or "").strip()
    aliases = {
        "todo.complete": "Complete",
        "todo.archive": "Archive",
        "todo.delegate": "Delegate",
        "open_situation": "Details",
        "propose_situation_update": "Add Update",
        "propose_child_commitment": "Add Commitment",
    }
    if value in aliases:
        return aliases[value]
    tail = value.rsplit(".", 1)[-1].replace("_", " ").strip()
    return tail.title() if tail else ""


def _review_guidance_available(item: dict[str, Any]) -> bool:
    return bool(_descriptor_advisory_guidance_for_item(item) or _review_target_payload(item))


def _render_review_target_guidance(item: dict[str, Any]) -> dict[str, Any]:
    target = _review_target_payload(item)
    if not target:
        return {
            "title": "KB Review Guidance",
            "text": "KB Review Guidance\nNo backend review target metadata was available. Refresh the card before asking for guidance.",
            "actions": [],
        }
    policy = target.get("policy") if isinstance(target.get("policy"), dict) else {}
    scope = target.get("scope") if isinstance(target.get("scope"), dict) else {}
    lines = [
        "KB Review Guidance",
        f"Target: {_short(target.get('title') or _item_title(item))}",
        f"Kind: {_short(target.get('kind') or _item_kind(item), 'review target')}",
        "Advisory only: cannot preview, confirm, or mutate KB state.",
    ]
    affected_count = scope.get("affected_count")
    viewed_count = scope.get("viewed_count")
    if affected_count is not None or viewed_count is not None:
        lines.append(
            "Scope: "
            + " · ".join(
                bit
                for bit in (
                    f"{affected_count} affected" if affected_count is not None else "",
                    f"{viewed_count} viewed" if viewed_count is not None else "",
                )
                if bit
            )
        )
    if policy:
        lines.append(
            "Write posture: "
            + (
                "preview and confirmed envelope required"
                if policy.get("preview_required") or policy.get("confirmed_envelope_required")
                else "read/proposal handoff only"
            )
        )
    summary = _short(target.get("summary") or _item_detail(item), "")
    if summary:
        lines.append("Context: " + _clip(summary, 360))
    return {"title": "KB Review Guidance", "text": "\n".join(lines), "actions": []}


def _control_action_plan(action: dict[str, Any], *, reason: str) -> dict[str, Any]:
    route = action.get("confirmed_write_route") if isinstance(action.get("confirmed_write_route"), dict) else {}
    operation_id = _short(route.get("operation_id") or _get_path(action, "params", "operation_id"), "")
    arguments = route.get("arguments") if isinstance(route.get("arguments"), dict) else {}
    label = _short(action.get("label") or operation_id.rsplit(".", 1)[-1].title(), "Apply")
    return {
        "summary": f"{label} from Telegram KB Review.",
        "operations": [
            {
                "operation_id": operation_id,
                "arguments": dict(arguments),
                "reason": reason,
            }
        ],
        "confirmation": {
            "question": f"Confirm {label}?",
            "operation_ids": [operation_id] if operation_id else [],
        },
    }


def _control_action_object(action: dict[str, Any]) -> dict[str, Any]:
    route = action.get("confirmed_write_route") if isinstance(action.get("confirmed_write_route"), dict) else {}
    obj = route.get("object_ref") if isinstance(route.get("object_ref"), dict) else {}
    return dict(obj)


def _control_preview_text(label: str, item: dict[str, Any], payload: Any) -> str:
    if isinstance(payload, dict) and payload.get("error"):
        return f"KB Control Preview Failed\n{payload['error']}"
    if not isinstance(payload, dict):
        return "KB Control Preview\n" + _short(payload, "No structured response returned.")
    lines = [
        "KB Control Preview",
        f"Action: {label}",
        f"Target: {_item_title(item)}",
        f"Status: {_short(payload.get('status') or payload.get('state'), 'preview')}",
    ]
    results = payload.get("results") if isinstance(payload.get("results"), list) else []
    for result in results[:3]:
        if isinstance(result, dict):
            message = _short(result.get("message") or result.get("status"), "")
            operation = _short(result.get("operation_id"), "")
            if message or operation:
                lines.append("- " + " · ".join(bit for bit in (operation, message) if bit))
    lines.extend(_receipt_lines(payload, include_request=True))
    if _preview_allows_confirmation(payload):
        lines.append("Confirm only if this preview matches the action you intend.")
    return "\n".join(lines)


def _control_result_text(label: str, payload: Any) -> str:
    if isinstance(payload, dict) and payload.get("error"):
        return f"KB Control Result Failed\n{payload['error']}"
    if not isinstance(payload, dict):
        return "KB Control Result\n" + _short(payload, "No structured response returned.")
    lines = [
        "KB Control Result",
        f"Action: {label}",
        f"Status: {_short(payload.get('status') or payload.get('state'), 'unknown')}",
    ]
    lines.extend(_receipt_lines(payload, include_request=True))
    results = payload.get("results") if isinstance(payload.get("results"), list) else []
    for result in results[:3]:
        if isinstance(result, dict):
            message = _short(result.get("message") or result.get("status"), "")
            operation = _short(result.get("operation_id"), "")
            if message or operation:
                lines.append("- " + " · ".join(bit for bit in (operation, message) if bit))
    return "\n".join(lines)


def _render_control_action_confirm(
    ctx: Any,
    target: str,
    item: dict[str, Any],
    action: dict[str, Any],
    *,
    packet: dict[str, Any],
    plan: dict[str, Any],
    preview_payload: dict[str, Any],
    callback_ctx: Any,
) -> dict[str, Any]:
    label = _short(action.get("label") or "Apply", "Apply")
    actor = _queue_callback_actor(callback_ctx)
    source = "Hermes Telegram Action Card"
    session_id = _review_session_id(_queue_item_review_metadata(item)) or f"telegram-kb-control-{int(time.time())}"
    envelope_payload = _result_payload(
        ctx.dispatch_tool(
            _descriptor_tool_name(target, "control.build_confirmed_envelope"),
            {
                "packet": packet,
                "plan": plan,
                "actor": actor,
                "source": source,
                "session_id": session_id,
                "user_confirmation": {
                    "confirmed": True,
                    "confirmed_by": actor,
                    "confirmation_text": f"Confirmed {label} from Telegram KB Review.",
                    "preview_status": _short(preview_payload.get("status"), ""),
                    "review_session_id": session_id,
                },
            },
        )
    )
    envelope = envelope_payload.get("envelope") if isinstance(envelope_payload, dict) else None
    if not isinstance(envelope, dict):
        return {"title": "KB Control", "text": _control_result_text(label, envelope_payload), "actions": []}
    applied = _result_payload(
        ctx.dispatch_tool(
            _descriptor_tool_name(target, "control.apply_confirmed"),
            {"envelope": envelope},
        )
    )
    return {"title": "KB Control", "text": _control_result_text(label, applied), "actions": []}


def _render_control_action_preview(
    ctx: Any,
    target: str,
    item: dict[str, Any],
    action: dict[str, Any],
    *,
    callback_ctx: Any,
) -> dict[str, Any]:
    try:
        from tools.kb_callback_registry import KbAction
    except Exception:
        return {"title": "KB Control", "text": "KB Control\nAction buttons are unavailable. Refresh from /kb review.", "actions": []}

    label = _short(action.get("label") or "Apply", "Apply")
    missing_inputs = _action_required_inputs(action)
    if missing_inputs:
        return {
            "title": "KB Control",
            "text": (
                f"KB Control\n{label} needs additional input first: "
                + ", ".join(missing_inputs)
                + ". Refresh in a fuller workbench surface."
            ),
            "actions": [],
        }
    actor = _queue_callback_actor(callback_ctx)
    source = "Hermes Telegram Action Card"
    reason = f"{label} previewed from Telegram KB Review for {_item_title(item)}."
    obj = _control_action_object(action)
    packet = _result_payload(
        ctx.dispatch_tool(
            _descriptor_tool_name(target, "control.context"),
            {"object": obj, "user_input": reason},
        )
    )
    if not isinstance(packet, dict) or packet.get("error"):
        return {"title": "KB Control", "text": _control_preview_text(label, item, packet), "actions": []}
    plan = _control_action_plan(action, reason=reason)
    preview_payload = _result_payload(
        ctx.dispatch_tool(
            _descriptor_tool_name(target, "control.apply_preview"),
            {
                "packet": packet,
                "plan": plan,
                "actor": actor,
                "source": source,
            },
        )
    )
    actions: list[Any] = []
    if isinstance(preview_payload, dict) and _preview_allows_confirmation(preview_payload):
        preview_plan = preview_payload.get("plan") if isinstance(preview_payload.get("plan"), dict) else plan
        actions.append(
            KbAction(
                label=f"Confirm {label}",
                action_id="control.apply_confirmed.confirm",
                handler=lambda confirm_ctx, p=dict(packet), pl=dict(preview_plan), pp=dict(preview_payload), a=dict(action): _render_control_action_confirm(
                    ctx,
                    target,
                    item,
                    a,
                    packet=p,
                    plan=pl,
                    preview_payload=pp,
                    callback_ctx=confirm_ctx,
                ),
                metadata={
                    "target_kind": "todo",
                    "target_ref": _item_target(item),
                    "preview_required": True,
                    "review_session_id": _review_session_id(_queue_item_review_metadata(item)),
                },
            )
        )
    return {"title": "KB Control", "text": _control_preview_text(label, item, preview_payload), "actions": actions}


def _queue_control_actions(
    ctx: Any,
    target: str,
    item: dict[str, Any],
    *,
    limit: int | None = 3,
) -> list[Any]:
    try:
        from tools.kb_callback_registry import KbAction
    except Exception:
        return []
    actions: list[Any] = []
    for action in _control_actions_for_item(item):
        label = _short(action.get("label") or _get_path(action, "confirmed_write_route", "operation_id"), "")
        if not label:
            continue
        action_copy = dict(action)
        actions.append(
            KbAction(
                label=label,
                action_id=f"{_short(_get_path(action, 'confirmed_write_route', 'operation_id'), 'control.action')}.preview",
                handler=lambda callback_ctx, a=action_copy: _render_control_action_preview(
                    ctx,
                    target,
                    item,
                    a,
                    callback_ctx=callback_ctx,
                ),
                metadata={
                    "target_kind": "todo",
                    "target_ref": _item_target(item),
                    "preview_required": True,
                },
            )
        )
    if limit is None:
        return actions
    return actions[:limit]


def _queue_callback_actor(callback_ctx: Any) -> str:
    actor_id = str(getattr(callback_ctx, "actor_id", "") or "").strip()
    return f"telegram:{actor_id}" if actor_id else "telegram:operator"


def _render_queue_descriptor_preview(
    ctx: Any,
    target: str,
    item: dict[str, Any],
    *,
    index: int,
    descriptor: dict[str, Any],
    callback_ctx: Any,
) -> dict[str, Any]:
    try:
        from tools.kb_callback_registry import KbAction
    except Exception:
        return {"title": "KB Review", "text": "KB Review\nAction buttons are unavailable. Use /kb review to refresh.", "actions": []}

    params = descriptor.get("params") if isinstance(descriptor.get("params"), dict) else {}
    decision = str(params.get("decision") or "").strip().lower()
    if not decision:
        return {"title": "KB Review", "text": "KB Review\nThis action is missing a proposal decision.", "actions": []}
    proposal_ids = [str(proposal_id) for proposal_id in (params.get("proposal_ids") or []) if str(proposal_id)] or _proposal_ids_for_item(item)
    actor = _queue_callback_actor(callback_ctx)
    source = "Hermes Telegram Action Card"
    preview_tool = _descriptor_tool_name(target, descriptor.get("preview_tool"))
    preview_payload = _result_payload(
        ctx.dispatch_tool(
            preview_tool,
            _queue_descriptor_call_args(
                descriptor,
                item,
                decision=decision,
                actor=actor,
                source=source,
                note=f"Previewed from Telegram action card for {_item_title(item)}",
            ),
        )
    )
    selection = [(index, item)]
    text = _preview_text(decision, proposal_ids, preview_payload, selection=selection)
    if not _preview_allows_confirmation(preview_payload):
        return {"title": "KB Review", "text": text, "actions": []}
    preview_metadata = _queue_preview_metadata(preview_payload)
    label = _short(descriptor.get("label") or decision.replace("_", " ").title(), decision.title())
    action_id = _short(descriptor.get("action_id") or f"queue.{decision}", f"queue.{decision}")
    confirm_action = KbAction(
        label=f"Confirm {label}",
        action_id=f"{action_id}.confirm",
        handler=lambda confirm_ctx: _render_queue_descriptor_confirm(
            ctx,
            target,
            item,
            index=index,
            descriptor=descriptor,
            callback_ctx=confirm_ctx,
            preview_metadata=preview_metadata,
        ),
        metadata={
            "target_kind": "proposal_queue",
            "target_ref": _item_target(item),
            "decision": decision,
            "preview_required": True,
            "preview_lease": bool(preview_metadata.get("preview_lease")),
            "review_session_id": _review_session_id(preview_metadata),
        },
    )
    return {
        "title": "KB Review",
        "text": text + "\n\nConfirm with the button below only if the preview matches your intent.",
        "actions": [confirm_action],
    }


def _render_queue_descriptor_confirm(
    ctx: Any,
    target: str,
    item: dict[str, Any],
    *,
    index: int,
    descriptor: dict[str, Any],
    callback_ctx: Any,
    preview_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    params = descriptor.get("params") if isinstance(descriptor.get("params"), dict) else {}
    decision = str(params.get("decision") or "").strip().lower()
    if not decision:
        return {"title": "KB Review", "text": "KB Review\nThis action is missing a proposal decision.", "actions": []}
    proposal_ids = [str(proposal_id) for proposal_id in (params.get("proposal_ids") or []) if str(proposal_id)] or _proposal_ids_for_item(item)
    actor = _queue_callback_actor(callback_ctx)
    source = "Hermes Telegram Action Card"
    preview_tool = _descriptor_tool_name(target, descriptor.get("preview_tool"))
    confirmed_tool = _descriptor_tool_name(target, descriptor.get("confirm_tool"))
    selection = [(index, item)]
    effective_metadata = dict(preview_metadata or {})
    if not effective_metadata.get("preview_lease"):
        preview_payload = _result_payload(
            ctx.dispatch_tool(
                preview_tool,
                _queue_descriptor_call_args(
                    descriptor,
                    item,
                    decision=decision,
                    actor=actor,
                    source=source,
                    note=f"Re-previewed before Telegram action-card confirmation for {_item_title(item)}",
                ),
            )
        )
        if not _preview_allows_confirmation(preview_payload):
            return {"title": "KB Review", "text": _preview_text(decision, proposal_ids, preview_payload, selection=selection), "actions": []}
        effective_metadata.update(_queue_preview_metadata(preview_payload))
    confirmed_args = _queue_descriptor_call_args(
        descriptor,
        item,
        decision=decision,
        actor=actor,
        source=source,
        note=f"Confirmed from Telegram action card for {_item_title(item)}",
    )
    _apply_queue_preview_metadata(confirmed_args, effective_metadata)
    confirmed_args["session_id"] = _review_session_id(effective_metadata) or f"telegram-kb-card-{int(time.time())}"
    confirmed_args["user_confirmation"] = {
        "confirmed": True,
        "surface": "telegram",
        "action": f"queue.{decision}",
        "preview_required": True,
        "confirmation_text": str(descriptor.get("confirmation_copy") or f"Confirm {decision}"),
        "proposal_ids": proposal_ids,
    }
    _apply_queue_confirmation_preview_metadata(confirmed_args["user_confirmation"], effective_metadata)
    confirmed_payload = _result_payload(ctx.dispatch_tool(confirmed_tool, confirmed_args))
    packet_card = _render_supported_result_packet(confirmed_payload, ctx=ctx, target=target)
    if packet_card is not None:
        return packet_card
    return {
        "title": "KB Review",
        "text": _confirmed_text(decision, confirmed_payload, selection=selection, proposal_ids=proposal_ids),
        "actions": [],
    }


def _queue_item_text(item: dict[str, Any], *, index: int) -> str:
    proposal_ids = _proposal_ids_for_item(item)
    lines = [
        f"Review Item {index}",
        f"Title: {_item_title(item)}",
    ]
    kind = _item_kind(item)
    target = _item_target(item)
    detail = _item_detail(item)
    if kind:
        lines.append(f"Type: {kind}")
    if target:
        lines.append(f"Target: {target}")
    if detail:
        lines.append("")
        lines.append("Summary: " + _clip(detail, 420))
    if proposal_ids:
        lines.append("")
        lines.append(f"Proposal ids: {', '.join(proposal_ids[:5])}")
        descriptor_actions = _queue_descriptor_decisions(item)
        if descriptor_actions:
            labels = [label for _decision, label in descriptor_actions]
            lines.append("Decision rail: " + ", ".join(labels))
            lines.append("Nothing applies until a kb-engine preview returns and you confirm from that preview.")
        else:
            lines.append("Fallback text actions:")
            lines.extend(_queue_decision_commands(item, index=index))
    else:
        control_actions = _control_action_decisions(item)
        lines.append("")
        if control_actions:
            lines.append("Decision rail: " + ", ".join(label for _operation, label in control_actions))
            lines.append("Nothing applies until kb-engine previews the control route and you confirm.")
        elif _review_target_payload(item):
            lines.append("Review target: backend-owned session metadata is available.")
            lines.append("No direct write is available from this card; use Details or Ask LLM.")
        else:
            lines.append(
                "This item did not include backend review metadata, so Telegram cannot apply a decision yet. Refresh the KB workbench."
            )
    return "\n".join(lines)


def _queue_review_text(data: Any, visible_items: list[Any], *, total: int | None, offset: int) -> str:
    item = visible_items[0] if visible_items and isinstance(visible_items[0], dict) else None
    if item is None:
        return "KB Review\nNo proposal previews returned."
    current = offset + 1
    total_label = total if total is not None else len(visible_items)
    title = _item_title(item)
    detail = _item_detail(item)
    target = _item_target(item)
    proposal_ids = _proposal_ids_for_item(item)
    rail_labels = _queue_review_action_labels(item)
    rail_labels.append("Details")
    if _review_guidance_available(item):
        rail_labels.append("Ask LLM")
    if _queue_item_at(data, 2) is not None:
        rail_labels.append("Skip")
    lines = [
        "KB Review",
        f"{current} of {total_label} · Visible scope",
        title,
    ]
    if detail:
        lines.append(_clip(detail, 260))
    scope_bits: list[str] = []
    if target:
        scope_bits.append(target)
    if proposal_ids:
        scope_bits.append(f"{len(proposal_ids)} proposal{'s' if len(proposal_ids) != 1 else ''}")
    elif _review_target_payload(item):
        scope_bits.append("1 review target")
    if visible_items:
        scope_bits.append(f"{len(visible_items)} visible")
    if total is not None and total != len(visible_items):
        scope_bits.append(f"{total} total")
    if scope_bits:
        lines.append("Scope: " + " · ".join(scope_bits))
    if rail_labels:
        lines.append("Rail: " + ", ".join(rail_labels[:6]))
    policy = _review_target_policy(item)
    if proposal_ids:
        lines.append("Nothing applies until kb-engine returns a preview lease and you confirm.")
    elif policy.get("preview_required") or _control_actions_for_item(item):
        lines.append("Nothing applies until kb-engine previews the control route and you confirm.")
    else:
        lines.append("No direct write is available from this card; use Details or Ask LLM.")
    waiting = max(len(visible_items) - 1, 0)
    if waiting:
        lines.append(f"{waiting} more item{'s' if waiting != 1 else ''} waiting in this Telegram window.")
    return "\n".join(lines)


def _descriptor_advisory_guidance_for_item(item: dict[str, Any]) -> dict[str, Any] | None:
    for descriptor in _safe_actions_for_item(item):
        guidance = _descriptor_advisory_guidance(descriptor)
        if guidance:
            return guidance
    return None


def _selection_lines(selection: list[tuple[int, dict[str, Any]]]) -> list[str]:
    lines: list[str] = []
    for index, item in selection:
        lines.append(f"{index}. {_item_title(item)}")
        target = _item_target(item)
        kind = _item_kind(item)
        detail = _item_detail(item)
        if target:
            lines.append(f"   Target: {target}")
        if kind:
            lines.append(f"   Type: {kind}")
        if detail:
            lines.append(f"   Summary: {_clip(detail, 180)}")
    return lines


def _format_indices(indices: list[int]) -> str:
    return ",".join(str(index) for index in indices)


def _proposal_ids_for_selection(selection: list[tuple[int, dict[str, Any]]]) -> list[str]:
    proposal_ids: list[str] = []
    seen: set[str] = set()
    for _, item in selection:
        for proposal_id in _proposal_ids_for_item(item):
            if proposal_id not in seen:
                seen.add(proposal_id)
                proposal_ids.append(proposal_id)
    return proposal_ids


def _queue_selection_snapshot(selection: list[tuple[int, dict[str, Any]]]) -> list[dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []
    for index, item in selection:
        snapshots.append(
            {
                "index": int(index),
                "title": _item_title(item),
                "kind": _item_kind(item),
                "target": _item_target(item),
                "detail": _item_detail(item),
                "proposal_ids": _proposal_ids_for_item(item),
            }
        )
    return snapshots


def _queue_selection_from_snapshot(snapshots: Any) -> list[tuple[int, dict[str, Any]]]:
    selection: list[tuple[int, dict[str, Any]]] = []
    if not isinstance(snapshots, list):
        return selection
    for offset, snapshot in enumerate(snapshots, start=1):
        if not isinstance(snapshot, dict):
            continue
        try:
            index = int(snapshot.get("index") or offset)
        except (TypeError, ValueError):
            index = offset
        proposal_ids = [str(item) for item in (snapshot.get("proposal_ids") or []) if str(item).strip()]
        item = {
            "title": _short(snapshot.get("title"), "Review item"),
            "kind": _short(snapshot.get("kind"), ""),
            "target": _short(snapshot.get("target"), ""),
            "detail": _short(snapshot.get("detail"), ""),
            "raw": {"proposal_ids": proposal_ids},
        }
        selection.append((index, item))
    return selection


def _queue_scope_state(session_id: str, *, create: bool = False) -> tuple[dict[str, Any], dict[str, Any] | None]:
    states = _load_queue_scope_states()
    if not session_id:
        return states, None
    state = states.get(session_id)
    if not isinstance(state, dict):
        if not create:
            return states, None
        state = {"schema_version": 1}
        states[session_id] = state
    return states, state


def _queue_scope_stale(record: Any) -> bool:
    if not isinstance(record, dict):
        return True
    try:
        recorded_at = float(record.get("recorded_at") or 0.0)
    except (TypeError, ValueError):
        return True
    if recorded_at <= 0:
        return True
    return bool(time.time() - recorded_at > QUEUE_SCOPE_STATE_TTL_SECONDS)


def _queue_total(data: Any) -> int | None:
    if not isinstance(data, dict):
        return None
    return _queue_count_value(
        data.get("total"),
        data.get("count"),
        _get_path(data, "counts", "proposals"),
        _get_path(data, "queue", "count"),
    )


def _queue_offset(data: Any) -> int:
    if not isinstance(data, dict):
        return 0
    return _queue_count_value(data.get("offset"), _get_path(data, "page", "offset")) or 0


def _queue_next_offset(data: Any) -> int | None:
    if not isinstance(data, dict):
        return None
    return _queue_count_value(data.get("next_offset"), _get_path(data, "page", "next_offset"))


def _store_visible_queue_scope(
    session_id: str,
    items: list[Any],
    *,
    total: int | None = None,
    offset: int = 0,
    next_offset: int | None = None,
) -> None:
    if not session_id:
        return
    selection = [(idx, item) for idx, item in enumerate(items, start=1) if isinstance(item, dict)]
    states, state = _queue_scope_state(session_id, create=True)
    if state is None:
        return
    state["visible"] = {
        "kind": "visible_queue_window",
        "recorded_at": time.time(),
        "selection": _queue_selection_snapshot(selection),
        "offset": int(offset),
        "displayed_count": len(selection),
    }
    if total is not None:
        state["visible"]["candidate_count"] = int(total)
    if next_offset is not None:
        state["visible"]["next_offset"] = int(next_offset)
    _save_queue_scope_states(states)


def _get_visible_queue_scope_record(session_id: str) -> dict[str, Any]:
    states, state = _queue_scope_state(session_id)
    if state is None:
        return {}
    visible = state.get("visible")
    if _queue_scope_stale(visible):
        state.pop("visible", None)
        _save_queue_scope_states(states)
        return {}
    return dict(visible) if isinstance(visible, dict) else {}


def _get_visible_queue_scope(session_id: str) -> list[tuple[int, dict[str, Any]]]:
    visible = _get_visible_queue_scope_record(session_id)
    if not visible:
        return []
    return _queue_selection_from_snapshot(visible.get("selection"))


def _store_queue_text_preview_scope(
    session_id: str,
    *,
    decision: str,
    indices: list[int],
    selection: list[tuple[int, dict[str, Any]]],
    preview_payload: Any = None,
) -> None:
    if not session_id:
        return
    states, state = _queue_scope_state(session_id, create=True)
    if state is None:
        return
    state["preview"] = {
        "kind": "queue_text_preview",
        "recorded_at": time.time(),
        "decision": str(decision or "").strip().lower(),
        "indices": [int(index) for index in indices],
        "selection": _queue_selection_snapshot(selection),
        "proposal_ids": _proposal_ids_for_selection(selection),
    }
    state["preview"]["preview_metadata"] = _queue_preview_metadata(preview_payload)
    _save_queue_scope_states(states)


def _get_queue_text_preview_scope(
    session_id: str,
    *,
    decision: str,
    indices: list[int],
) -> tuple[list[tuple[int, dict[str, Any]]], dict[str, Any]]:
    states, state = _queue_scope_state(session_id)
    if state is None:
        return [], {}
    preview = state.get("preview")
    if _queue_scope_stale(preview):
        state.pop("preview", None)
        _save_queue_scope_states(states)
        return [], {}
    if str(preview.get("decision") or "").strip().lower() != str(decision or "").strip().lower():
        return [], {}
    try:
        recorded_indices = [int(index) for index in (preview.get("indices") or [])]
    except (TypeError, ValueError):
        recorded_indices = []
    if recorded_indices != [int(index) for index in indices]:
        return [], {}
    preview_metadata = preview.get("preview_metadata") if isinstance(preview.get("preview_metadata"), dict) else {}
    return _queue_selection_from_snapshot(preview.get("selection")), dict(preview_metadata)


def _preview_text(
    decision: str,
    proposal_ids: list[str],
    payload: Any,
    *,
    selection: list[tuple[int, dict[str, Any]]] | None = None,
    item: dict[str, Any] | None = None,
) -> str:
    if selection is None:
        selection = [(0, item)] if isinstance(item, dict) else []
    if isinstance(payload, dict) and payload.get("error"):
        return f"Review {decision} preview failed\n{payload['error']}"
    if isinstance(payload, dict):
        status = _short(payload.get("status"))
        ok = _short(payload.get("ok"))
        preview = payload.get("preview") if isinstance(payload.get("preview"), dict) else {}
        summary = _short(
            preview.get("summary")
            or _get_path(payload, "plan", "summary")
            or f"{decision.title()} {len(proposal_ids)} proposal(s).",
        )
        lines = [f"Review {decision} preview"]
        if selection:
            lines.append(f"Items: {len(selection)}")
            lines.extend(_selection_lines(selection))
        lines.extend(
            [
                f"Status: {status} · ok: {ok}",
                f"Proposal ids: {', '.join(proposal_ids[:5])}",
                "Plan: " + _clip(summary, 260),
                "Confirm only if this item and decision match what you intend.",
            ]
        )
        lines.extend(_queue_scope_lines(payload))
        return "\n".join(lines)
    lines = [f"Review {decision} preview"]
    if selection:
        lines.extend(_selection_lines(selection))
    lines.append(f"Proposal ids: {', '.join(proposal_ids[:5])}")
    return "\n".join(lines)


def _preview_allows_confirmation(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    if payload.get("error") or payload.get("isError"):
        return False
    if payload.get("ok") is False:
        return False
    status = str(payload.get("status") or payload.get("state") or "").strip().lower()
    if status in {
        "blocked",
        "error",
        "failed",
        "operator_blocked",
        "preview_lease_expired",
        "preview_lease_missing",
        "preview_lease_mismatch",
        "preview_lease_stale",
        "stale_cursor",
        "stale_preview_lease",
        "validation_failed",
    }:
        return False
    return True


def _git_summary(git_state: dict[str, Any]) -> str:
    after = git_state.get("after") if isinstance(git_state.get("after"), dict) else {}
    before = git_state.get("before") if isinstance(git_state.get("before"), dict) else {}
    branch = _short(after.get("branch") or git_state.get("branch") or before.get("branch"), "")
    changed = after.get("changed_count")
    if changed is None and isinstance(after.get("changes"), list):
        changed = len(after["changes"])
    if changed is None:
        changed = git_state.get("changed_count")
    if changed is not None:
        suffix = f" on {branch}" if branch else ""
        return f"{changed} changed path(s){suffix}"
    return _short(git_state.get("summary") or git_state.get("status"), "")


def _decision_past_tense(decision: str) -> str:
    return {
        "approve": "Approved",
        "reject": "Rejected",
        "archive": "Archived",
        "complete": "Completed",
        "keep": "Kept unchanged",
        "demote": "Demoted",
        "skip": "Skipped",
    }.get(decision, f"{decision.title()}ed")


def _confirmed_text(
    decision: str,
    payload: Any,
    *,
    selection: list[tuple[int, dict[str, Any]]] | None = None,
    proposal_ids: list[str] | None = None,
) -> str:
    if isinstance(payload, dict) and payload.get("error"):
        return f"Review {decision} failed\n{payload['error']}"
    packet_card = _render_supported_result_packet(payload)
    if packet_card is not None:
        return packet_card["text"]
    selection = selection or []
    proposal_ids = proposal_ids or []
    past_tense = _decision_past_tense(decision)
    if isinstance(payload, dict):
        status = _short(payload.get("status") or payload.get("state"), "")
        reason = _short(payload.get("reason") or payload.get("message"), "")
        if payload.get("ok") is False or status.lower() in {
            "blocked",
            "error",
            "failed",
            "operator_blocked",
            "preview_lease_expired",
            "preview_lease_missing",
            "preview_lease_stale",
            "stale_preview_lease",
            "validation_failed",
        }:
            lines = [
                f"Review {decision.title()} Blocked",
                f"Status: {status or 'blocked'} · ok: {_short(payload.get('ok'))}",
            ]
            if reason:
                lines.append("Reason: " + _clip(reason, 220))
            lines.extend(_receipt_lines(payload))
            lines.append("Next: /kb review")
            return "\n".join(lines)
        publication = payload.get("publication") if isinstance(payload.get("publication"), dict) else {}
        git_state = payload.get("git") if isinstance(payload.get("git"), dict) else {}
        lines = [
            f"Review {decision.title()} Applied",
            f"{past_tense} {len(proposal_ids) or len(selection)} proposal(s).",
        ]
        if selection:
            lines.append("")
            lines.append("Changed:")
            lines.extend(_selection_lines(selection))
        if proposal_ids:
            lines.append("")
            lines.append(f"Proposal ids: {', '.join(proposal_ids[:8])}")
        lines.extend(
            [
                f"Status: {_short(payload.get('status'))} · ok: {_short(payload.get('ok'))}",
            ]
        )
        lines.extend(_receipt_lines(payload))
        if publication:
            lines.append(
                "Publication: "
                + _short(publication.get("status") or publication.get("state") or publication.get("reason"))
            )
        if git_state:
            git_summary = _git_summary(git_state)
            if git_summary:
                lines.append("Git: " + git_summary)
        lines.append("Next: /kb review")
        return "\n".join(lines)
    lines = [
        f"Review {decision.title()} Applied",
        f"{past_tense} {len(proposal_ids) or len(selection)} proposal(s).",
    ]
    if selection:
        lines.extend(["", "Changed:", *_selection_lines(selection)])
    lines.append("Next: /kb review")
    return "\n".join(lines)


def _queue_reply_choices_from_text(text: str) -> list[str]:
    lowered = str(text or "").lower()
    lowered = re.sub(r"\bdetails\b", "detail", lowered)
    ordered = ["complete", "keep", "demote", "archive", "reject", "approve", "detail", "skip"]
    return [decision for decision in ordered if re.search(rf"\b{re.escape(decision)}\b", lowered)]


def _infer_iterative_queue_title(body: str) -> str:
    lines = body.splitlines()
    last_proposal_line = -1
    for idx, line in enumerate(lines):
        if re.search(r"(?i)\bproposal ids?\b", line) and re.search(r"\bact_[A-Za-z0-9]+\b", line):
            last_proposal_line = idx
    search_lines = lines[:last_proposal_line] if last_proposal_line >= 0 else lines
    skip_prefixes = (
        "next item",
        "proposal",
        "todo",
        "summary",
        "created",
        "reply",
        "applied",
        "archived proposal",
        "proposal ids",
        "options",
    )
    for raw_line in reversed(search_lines):
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^\s*#+\s*", "", line).strip()
        line = re.sub(r"^\s*\d+[\).]\s*", "", line).strip()
        proposal_title_match = re.match(r"(?i)^proposal\s+\d+\s*[—:-]\s*(.+)$", line)
        if proposal_title_match:
            return proposal_title_match.group(1).strip("`*_ ")
        if line.startswith("-"):
            continue
        if line.lower().startswith(skip_prefixes):
            continue
        return line.strip("`*_ ")
    return ""


def _parse_iterative_queue_reply_state(response_text: str) -> dict[str, Any] | None:
    text = str(response_text or "")
    reply_matches = list(
        re.finditer(
            r"(?im)^\s*(?:Reply(?:\s+with)?|Options(?:\s+presented)?)\s*:\s*(.+)$",
            text,
        )
    )
    if not reply_matches:
        return None
    reply_match = reply_matches[-1]
    choices = _queue_reply_choices_from_text(reply_match.group(1))
    if not choices:
        return None
    body = text[: reply_match.start()].rstrip()
    proposal_ids: list[str] = []
    for line in body.splitlines():
        if re.search(r"(?i)\bproposal ids?\b", line):
            ids = re.findall(r"\bact_[A-Za-z0-9]+\b", line)
            if ids:
                proposal_ids = ids
    if not proposal_ids:
        all_ids = re.findall(r"\bact_[A-Za-z0-9]+\b", body)
        proposal_ids = all_ids[-1:] if all_ids else []
    if not proposal_ids:
        return None
    title = _infer_iterative_queue_title(body)
    return {
        "schema_version": 1,
        "proposal_ids": proposal_ids,
        "title": title,
        "choices": choices,
        "recorded_at": time.time(),
    }


def _record_iterative_queue_reply_state(session_id: str, response_text: str) -> dict[str, Any] | None:
    if not session_id:
        return None
    parsed = _parse_iterative_queue_reply_state(response_text)
    states = _load_queue_reply_states()
    if not parsed:
        if session_id in states:
            states.pop(session_id, None)
            _save_queue_reply_states(states)
        return None
    states[session_id] = parsed
    _save_queue_reply_states(states)
    return parsed


def _get_iterative_queue_reply_state(session_id: str) -> dict[str, Any] | None:
    if not session_id:
        return None
    states = _load_queue_reply_states()
    state = states.get(session_id)
    if not isinstance(state, dict):
        return None
    recorded_at = float(state.get("recorded_at") or 0.0)
    if recorded_at and time.time() - recorded_at > QUEUE_REPLY_STATE_TTL_SECONDS:
        states.pop(session_id, None)
        _save_queue_reply_states(states)
        return None
    proposal_ids = [str(item) for item in (state.get("proposal_ids") or []) if str(item).strip()]
    if not proposal_ids:
        return None
    state["proposal_ids"] = proposal_ids
    return state


def _bare_queue_reply_decision(text: str) -> str:
    lowered = str(text or "").strip().lower()
    lowered = {"details": "detail", "show": "detail"}.get(lowered, lowered)
    return lowered if lowered in QUEUE_REPLY_DECISIONS else ""


def _visible_scope_all_decision(text: str) -> str:
    lowered = str(text or "").strip().lower()
    match = re.match(r"^(approve|reject|archive|skip)\b", lowered)
    if not match:
        return ""
    rest = lowered[match.end() :].strip()
    if not rest:
        return ""
    if re.search(r"\b(all|these|shown|visible|listed|everything)\b", rest):
        return match.group(1)
    if re.search(r"\b(?:the\s+)?(?:\d+|five)\b.*\b(showed|shown|presented|listed|visible|items?|proposals?)\b", rest):
        return match.group(1)
    return ""


def scoped_mcp_tool_allowlist_for_message(
    *,
    session_id: str,
    message: str,
    target: str | None = None,
) -> set[str]:
    """Return exact MCP tools allowed by a matched pending queue action.

    This is the stateful bridge between Telegram action cards and the generic
    MCP posture filter.  A bare reply such as ``Reject`` should stay preview
    only.  Confirmed queue tools are reserved for explicit confirm commands or
    action-card confirmation gestures.
    """
    decision = _bare_queue_reply_decision(message)
    if decision not in QUEUE_REPLY_TOOL_DECISIONS:
        decision = _visible_scope_all_decision(message)
        if decision not in QUEUE_REPLY_TOOL_DECISIONS:
            return set()
        if not _get_visible_queue_scope(session_id):
            return set()
    else:
        state = _get_iterative_queue_reply_state(session_id)
        if not state:
            return set()
        choices = {str(choice).strip().lower() for choice in (state.get("choices") or []) if str(choice).strip()}
        if choices and decision not in choices:
            return set()
    mcp_target = target or _mcp_target()
    return {_mcp_tool_name(mcp_target, "queue.decision_preview")}


def _session_id_for_queue_reply_state(session_store: Any, source: Any) -> str:
    if session_store is None:
        return ""
    try:
        session_store._ensure_loaded()
    except Exception:
        pass
    try:
        session_key = session_store._generate_session_key(source)
        entry = getattr(session_store, "_entries", {}).get(session_key)
        return str(getattr(entry, "session_id", "") or "")
    except Exception:
        logger.debug("kb_journeys: failed to resolve gateway session id", exc_info=True)
        return ""


def _conversation_state_id(session_store: Any, source: Any) -> str:
    session_id = _session_id_for_queue_reply_state(session_store, source)
    if session_id:
        return session_id
    parts = [
        _platform_name(getattr(source, "platform", None)),
        _short(getattr(source, "chat_id", ""), ""),
        _short(getattr(source, "thread_id", ""), ""),
        _short(getattr(source, "user_id", ""), ""),
    ]
    return ":".join(part for part in parts if part) or "telegram"


def _get_meeting_handoff_state(session_id: str) -> dict[str, Any] | None:
    if not session_id:
        return None
    states = _load_meeting_handoff_states()
    state = states.get(session_id)
    if not isinstance(state, dict):
        return None
    recorded_at = float(state.get("recorded_at") or 0.0)
    if recorded_at and time.time() - recorded_at > MEETING_HANDOFF_STATE_TTL_SECONDS:
        states.pop(session_id, None)
        _save_meeting_handoff_states(states)
        return None
    plan = state.get("plan")
    if not isinstance(plan, dict):
        return None
    return state


def _telegram_user_id(source: Any) -> str:
    return _short(getattr(source, "user_id", ""), "")


def _sync_preview_lease(plan: dict[str, Any], *, recorded_at: float | None = None) -> dict[str, Any]:
    preview_lease = plan.get("preview_lease") if isinstance(plan.get("preview_lease"), dict) else {}
    if preview_lease:
        return dict(preview_lease)
    workflow = plan.get("workflow") if isinstance(plan.get("workflow"), dict) else {}
    payload = {
        "workflow_id": str(workflow.get("workflow_id") or ""),
        "request_id": str(plan.get("request_id") or ""),
        "idempotency_key": str(plan.get("idempotency_key") or ""),
        "recorded_at": int(recorded_at or time.time()),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return {
        "kind": "telegram_workflow_preview",
        "preview_lease_id": f"sha256:{digest}",
        **{key: value for key, value in payload.items() if value},
    }


def _get_sync_preview_state(session_id: str, source: Any) -> tuple[dict[str, Any] | None, str]:
    if not session_id:
        return None, "missing_session"
    states = _load_sync_preview_states()
    state = states.get(session_id)
    if not isinstance(state, dict):
        return None, "missing"
    recorded_at = float(state.get("recorded_at") or 0.0)
    if not recorded_at or time.time() - recorded_at > SYNC_PREVIEW_STATE_TTL_SECONDS:
        states.pop(session_id, None)
        _save_sync_preview_states(states)
        return None, "stale"
    actor_id = _short(state.get("actor_id"), "")
    current_actor = _telegram_user_id(source)
    if actor_id and current_actor and actor_id != current_actor:
        return None, "wrong_actor"
    plan = state.get("plan")
    if not isinstance(plan, dict):
        return None, "invalid"
    return state, ""


def _store_sync_preview_state(
    session_id: str,
    *,
    source: Any,
    target: str,
    workflow_id: str,
    intent: str,
    plan: dict[str, Any],
) -> None:
    if not session_id:
        return
    recorded_at = time.time()
    preview_lease = _sync_preview_lease(plan, recorded_at=recorded_at)
    stored_plan = dict(plan)
    stored_plan.setdefault("preview_lease", preview_lease)
    states = _load_sync_preview_states()
    states[session_id] = {
        "schema_version": 1,
        "recorded_at": recorded_at,
        "actor_id": _telegram_user_id(source),
        "actor_name": _short(getattr(source, "user_name", ""), ""),
        "target": target,
        "workflow_id": workflow_id,
        "intent": intent,
        "preview_lease": preview_lease,
        "plan": stored_plan,
    }
    _save_sync_preview_states(states)


def _store_meeting_handoff_state(
    session_id: str,
    *,
    plan: dict[str, Any],
    meeting_file: str,
    notes_text: str,
) -> None:
    if not session_id:
        return
    states = _load_meeting_handoff_states()
    states[session_id] = {
        "schema_version": 1,
        "recorded_at": time.time(),
        "meeting_file": _short(meeting_file, ""),
        "notes_sha256": "sha256:" + hashlib.sha256(str(notes_text or "").encode("utf-8")).hexdigest(),
        "notes_chars": len(str(notes_text or "")),
        "plan": plan,
    }
    _save_meeting_handoff_states(states)


def _queue_count(data: Any) -> int | None:
    if not isinstance(data, dict):
        return None
    counts = data.get("counts") if isinstance(data.get("counts"), dict) else {}
    for value in (counts.get("proposals"), data.get("total"), data.get("count"), _count_from(data, "queue", "proposals")):
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return None


def _iterative_state_from_item(item: dict[str, Any]) -> dict[str, Any] | None:
    proposal_ids = _proposal_ids_for_item(item)
    if not proposal_ids:
        return None
    decisions = [decision for decision, _ in _queue_action_decisions(item)]
    for fallback in ("detail", "skip"):
        if fallback not in decisions:
            decisions.append(fallback)
    return {
        "schema_version": 1,
        "proposal_ids": proposal_ids,
        "title": _item_title(item),
        "choices": decisions,
        "recorded_at": time.time(),
    }


def _store_iterative_state_from_item(session_id: str, item: dict[str, Any] | None) -> None:
    if not session_id:
        return
    if not isinstance(item, dict):
        _clear_iterative_queue_reply_state(session_id)
        return
    state = _iterative_state_from_item(item)
    if not state:
        _clear_iterative_queue_reply_state(session_id)
        return
    states = _load_queue_reply_states()
    states[session_id] = state
    _save_queue_reply_states(states)


def _iterative_queue_next_text(data: Any, *, session_id: str = "") -> str:
    items = _queue_items_from_payload(data)
    if not items:
        if session_id:
            _clear_iterative_queue_reply_state(session_id)
        return "Review is empty."
    item = items[0] if isinstance(items[0], dict) else {}
    if not item:
        return "Next review item could not be rendered. Use /kb review to refresh."
    _store_iterative_state_from_item(session_id, item)
    count = _queue_count(data)
    proposal_ids = _proposal_ids_for_item(item)
    decisions = [decision for decision, _ in _queue_action_decisions(item)]
    for fallback in ("detail", "skip"):
        if fallback not in decisions:
            decisions.append(fallback)
    lines: list[str] = []
    if count is not None:
        lines.append(f"Review now has {count} proposal(s).")
        lines.append("")
    lines.extend(["Next item:", "", _item_title(item)])
    detail = _item_detail(item)
    if detail:
        lines.append("- Summary: " + _clip(detail, 240))
    target = _item_target(item)
    if target:
        lines.append("- Target: " + target)
    if proposal_ids:
        label = "Proposal ids" if len(proposal_ids) != 1 else "Proposal id"
        lines.append(f"- {label}: {', '.join(proposal_ids[:8])}")
    lines.append("")
    lines.append("Reply: " + ", ".join(decisions) + ".")
    return "\n".join(lines)


def _iterative_selection_from_state(state: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    return [
        (
            1,
            {
                "title": _short(state.get("title"), "Current review item"),
                "raw": {"proposal_ids": list(state.get("proposal_ids") or [])},
            },
        )
    ]


def _render_iterative_queue_reply_decision(
    ctx: Any,
    target: str,
    *,
    session_id: str,
    state: dict[str, Any],
    decision: str,
) -> dict[str, Any]:
    proposal_ids = [str(item) for item in (state.get("proposal_ids") or []) if str(item).strip()]
    title = _short(state.get("title"), "current review item")
    selection = _iterative_selection_from_state(state)
    if decision == "detail":
        lines = [
            "Review Item",
            f"Title: {title}",
            f"Proposal ids: {', '.join(proposal_ids)}",
            "Reply: " + ", ".join(state.get("choices") or sorted(QUEUE_REPLY_DECISIONS)) + ".",
        ]
        return {"title": "KB Review", "text": "\n".join(lines), "actions": []}
    if decision not in QUEUE_REPLY_TOOL_DECISIONS:
        return {"title": "KB Review", "text": "That review reply is not supported. Use /kb review to refresh.", "actions": []}
    actor = "telegram:operator"
    source = "Hermes Telegram iterative queue"
    preview_tool = _mcp_tool_name(target, "queue.decision_preview")
    preview_payload = _result_payload(
        ctx.dispatch_tool(
            preview_tool,
            {
                "proposal_ids": proposal_ids,
                "decision": decision,
                "actor": actor,
                "source": source,
                "note": f"Previewed from Telegram iterative review reply for {title}",
            },
        )
    )
    text = _preview_text(decision, proposal_ids, preview_payload, selection=selection)
    if _preview_allows_confirmation(preview_payload):
        text += f"\nTo apply: /kb review {decision} 1 confirm"
    return {"title": "KB Review", "text": text, "actions": []}


def _render_visible_scope_all_decision(
    ctx: Any,
    target: str,
    *,
    session_id: str,
    decision: str,
) -> dict[str, Any]:
    visible_record = _get_visible_queue_scope_record(session_id)
    selection = _queue_selection_from_snapshot(visible_record.get("selection")) if visible_record else []
    if not selection:
        return {
            "title": "KB Review",
            "text": (
                "KB Review\n"
                f"I can only {decision} all against the proposals currently shown in this Telegram thread. "
                "Run /kb review first, then ask again."
            ),
            "actions": [],
        }
    proposal_ids = _proposal_ids_for_selection(selection)
    if not proposal_ids:
        return {"title": "KB Review", "text": "KB Review\nThe visible review window did not include proposal ids.", "actions": []}
    actor = "telegram:operator"
    source = "Hermes Telegram visible queue"
    preview_tool = _mcp_tool_name(target, "queue.decision_preview")
    candidate_count = _queue_count_value(visible_record.get("candidate_count"), len(selection))
    displayed_count = _queue_count_value(visible_record.get("displayed_count"), len(selection))
    preview_payload = _result_payload(
        ctx.dispatch_tool(
            preview_tool,
            {
                "proposal_ids": proposal_ids,
                "decision": decision,
                "decision_scope": "all_viewed",
                "candidate_count": candidate_count,
                "displayed_count": displayed_count,
                "actor": actor,
                "source": source,
                "note": f"Previewed from Telegram visible queue scope for {len(selection)} shown item(s)",
            },
        )
    )
    text = _preview_text(decision, proposal_ids, preview_payload, selection=selection)
    text += "\nScope: visible Telegram review window only, not the full pending review inbox."
    if _preview_allows_confirmation(preview_payload):
        indices = [index for index, _ in selection]
        _store_queue_text_preview_scope(
            session_id,
            decision=decision,
            indices=indices,
            selection=selection,
            preview_payload=preview_payload,
        )
        text += f"\nTo apply: /kb review {decision} {_format_indices(indices)} confirm"
    return {"title": "KB Review", "text": text, "actions": []}


def _queue_summary_payload(
    ctx: Any,
    target: str,
    *,
    scope: str = "proposals",
    limit: int = 5,
    offset: int = 0,
    selected_id: str = "",
) -> tuple[Any | None, list[str]]:
    args = {"scope": _queue_requested_scope(scope), "limit": limit}
    if offset:
        args["offset"] = int(offset)
    if selected_id:
        args["selected_id"] = selected_id
    _, data, errors = _dispatch_first(
        ctx,
        target,
        [
            ("review.inbox", dict(args)),
            ("queue.summary", dict(args)),
            ("workbench.queue", dict(args)),
            ("queue.preview", {"limit": limit}),
        ],
    )
    return data, errors


def _queue_requested_scope(value: str | None) -> str:
    scope = str(value or "").strip().lower()
    aliases = {
        "task": "tasks",
        "tasks": "tasks",
        "todo": "tasks",
        "todos": "tasks",
        "stale": "stale",
        "delegated": "delegated",
        "done": "done",
        "proposal": "proposals",
        "proposals": "proposals",
        "queue": "proposals",
        "review": "proposals",
    }
    return aliases.get(scope, "proposals")


def _queue_scope_and_args(args: str) -> tuple[str, str]:
    text = (args or "").strip()
    if not text:
        return "proposals", ""
    head, _, tail = text.partition(" ")
    scope = _queue_requested_scope(head)
    if scope != "proposals" or head.strip().lower() in {"proposal", "proposals"}:
        return scope, tail.strip()
    return "proposals", text


def _queue_payload_scope(data: Any) -> str:
    if not isinstance(data, dict):
        return "proposals"
    return _queue_requested_scope(data.get("scope") or data.get("queue_scope"))


def _changed_paths(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return []
    paths = payload.get("changed_paths")
    if paths is None:
        paths = _get_path(payload, "publication", "changed_paths")
    if isinstance(paths, str):
        return [paths]
    if isinstance(paths, list):
        return [str(path).strip() for path in paths if str(path).strip()]
    return []


def _format_changed_paths(paths: list[str], *, limit: int = 10) -> list[str]:
    if not paths:
        return []
    lines = [f"- {path}" for path in paths[:limit]]
    remaining = len(paths) - limit
    if remaining > 0:
        lines.append(f"- ... {remaining} more")
    return lines


def _publish_args(args: str) -> tuple[bool, str]:
    parts = (args or "").strip().split()
    confirm = any(part.lower() in {"confirm", "confirmed", "apply", "publish", "push"} for part in parts)
    message_parts = [part for part in parts if part.lower() not in {"confirm", "confirmed", "apply", "publish", "push"}]
    if message_parts and message_parts[0].lower() in {"message", "msg"}:
        message_parts = message_parts[1:]
    message = " ".join(message_parts).strip() or "Publish KB update"
    return confirm, message


def _publication_git_line(git_state: Any) -> str:
    if not isinstance(git_state, dict):
        return ""
    branch = _short(git_state.get("branch"), "")
    head = _short(git_state.get("head"), "")
    upstream = _short(git_state.get("upstream"), "")
    bits: list[str] = []
    if branch:
        bits.append(branch)
    if head:
        bits.append(head[:12])
    if upstream:
        bits.append(upstream)
    return " · ".join(bits)


def _closeout_packet(ctx: Any, target: str) -> Any:
    return _result_payload(ctx.dispatch_tool(_mcp_tool_name(target, "closeout.packet"), {"limit": 5}))


def _closeout_action_descriptors_from_payload(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    actions = payload.get("action_descriptors")
    if not isinstance(actions, list):
        return []
    return [action for action in actions if isinstance(action, dict)]


def _closeout_action_descriptors(ctx: Any, target: str) -> list[dict[str, Any]]:
    return _closeout_action_descriptors_from_payload(_closeout_packet(ctx, target))


def _closeout_publication_lines(payload: Any) -> list[str]:
    if not isinstance(payload, dict) or payload.get("error"):
        return []
    publication = payload.get("publication") if isinstance(payload.get("publication"), dict) else {}
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    lines: list[str] = []
    status = _short(
        publication.get("status")
        or publication.get("state")
        or publication.get("publication_status")
        or summary.get("publication_status"),
        "",
    )
    if status:
        lines.append(f"Publication posture: {status}")
    manual_expected = bool(
        publication.get("manual_publication_expected")
        or publication.get("manual_publication")
        or summary.get("manual_publication_expected")
        or payload.get("manual_publication_expected")
    )
    if manual_expected:
        lines.append("Manual publication expected.")
    changed_count = publication.get("changed_count")
    if changed_count is None:
        changed_count = summary.get("changed_count")
    if changed_count is None:
        changed_count = payload.get("changed_count")
    if changed_count is None:
        changed_paths = _changed_paths(publication) or _changed_paths(payload)
        changed_count = len(changed_paths) if changed_paths else None
    if changed_count is not None:
        lines.append(f"Closeout changed paths: {_short(changed_count, '0')}")
    reason = _short(publication.get("reason") or summary.get("publication_reason"), "")
    if reason:
        lines.append("Posture reason: " + _clip(reason, 220))
    return lines


def _publication_descriptor(descriptors: list[dict[str, Any]], method: str) -> dict[str, Any] | None:
    for descriptor in descriptors:
        if descriptor.get("dashboard_owned_write") is True:
            continue
        if descriptor.get("target_kind") != "publication":
            continue
        if descriptor.get("method") == method or descriptor.get("preview_tool") == method or descriptor.get("confirm_tool") == method:
            return descriptor
    return None


def _publication_descriptor_args(descriptor: dict[str, Any] | None, *, message: str) -> dict[str, Any]:
    params = descriptor.get("params") if isinstance(descriptor, dict) and isinstance(descriptor.get("params"), dict) else {}
    args = dict(params)
    if message:
        args["message"] = message
    return args


def _render_publication_preflight_descriptor(
    ctx: Any,
    target: str,
    *,
    descriptor: dict[str, Any],
    callback_ctx: Any,
) -> dict[str, Any]:
    del callback_ctx
    tool = _descriptor_tool_name(target, descriptor.get("preview_tool") or descriptor.get("method") or "publication.preflight")
    payload = _result_payload(ctx.dispatch_tool(tool, _descriptor_params(descriptor)))
    if isinstance(payload, dict) and payload.get("error"):
        return {"title": "KB Publish", "text": f"KB Publication Preflight Failed\n{payload['error']}", "actions": []}
    packet_card = _render_supported_result_packet(payload)
    if packet_card is not None:
        return packet_card
    if not isinstance(payload, dict):
        return {
            "title": "KB Publish",
            "text": "KB Publication Preflight\n" + _short(payload, "No structured response returned."),
            "actions": [],
        }
    lines = [
        "KB Publication Preflight",
        f"Status: {_short(payload.get('status') or payload.get('state'))}",
    ]
    changed_paths = _changed_paths(payload)
    if changed_paths:
        lines.append(f"Changed paths: {len(changed_paths)}")
        lines.extend(_format_changed_paths(changed_paths, limit=5))
    lines.extend(_receipt_lines(payload, include_request=True))
    warnings = payload.get("warnings") if isinstance(payload.get("warnings"), list) else []
    if warnings:
        lines.extend(_warning_lines(warnings))
    summary = _short(payload.get("summary") or payload.get("message"), "")
    if summary:
        lines.append("Summary: " + _clip(summary, 260))
    return {"title": "KB Publish", "text": "\n".join(lines), "actions": []}


def _publication_preflight_action(ctx: Any, target: str, descriptor: dict[str, Any] | None) -> list[Any]:
    if not descriptor:
        return []
    try:
        from tools.kb_callback_registry import KbAction
    except Exception:
        return []
    return [
        KbAction(
            label="Run Preflight",
            action_id="publication.preflight.open",
            handler=lambda callback_ctx, d=dict(descriptor): _render_publication_preflight_descriptor(
                ctx,
                target,
                descriptor=d,
                callback_ctx=callback_ctx,
            ),
            metadata={
                "target_kind": "publication",
                "target_ref": descriptor.get("target_ref") or "publication",
                "preview_tool": descriptor.get("preview_tool") or descriptor.get("method") or "publication.preflight",
                "mutation": "read_only",
            },
        )
    ]


def _render_publish_descriptor_confirm(
    ctx: Any,
    target: str,
    *,
    preview_descriptor: dict[str, Any] | None,
    confirm_descriptor: dict[str, Any],
    message: str,
    callback_ctx: Any,
) -> dict[str, Any]:
    preview_tool = _descriptor_tool_name(
        target,
        (preview_descriptor or {}).get("preview_tool") or (preview_descriptor or {}).get("method") or "publication.preview_commit",
    )
    commit_tool = _descriptor_tool_name(target, confirm_descriptor.get("confirm_tool") or confirm_descriptor.get("method"))
    push_tool = _mcp_tool_name(target, "publication.push_confirmed")
    actor = _queue_callback_actor(callback_ctx)
    source = "Hermes Telegram Action Card"
    session_id = f"telegram-kb-publish-{int(time.time())}"
    preview_payload = _result_payload(
        ctx.dispatch_tool(preview_tool, _publication_descriptor_args(preview_descriptor, message=message))
    )
    if not isinstance(preview_payload, dict) or preview_payload.get("error"):
        return _render_publish_preview(preview_payload)
    changed_paths = _changed_paths(preview_payload)
    if not changed_paths:
        return {
            "title": "KB Publish",
            "text": _render_publish_preview(preview_payload)["text"].replace("KB Publish Preview", "KB Publish"),
            "actions": [],
        }
    confirmation = {
        "confirmed": True,
        "surface": "telegram",
        "action": "publication.commit_and_push",
        "preview_required": True,
        "confirmation_text": str(confirm_descriptor.get("confirmation_copy") or "Confirm publication after preview."),
    }
    commit_args = _publication_descriptor_args(confirm_descriptor, message=message)
    commit_args.update(
        {
            "expected_git_head": _short(_get_path(preview_payload, "git", "head"), ""),
            "expected_changed_paths": changed_paths,
            "push": False,
            "actor": actor,
            "source": source,
            "session_id": session_id,
            "user_confirmation": confirmation,
        }
    )
    commit_payload = _result_payload(ctx.dispatch_tool(commit_tool, commit_args))
    if not isinstance(commit_payload, dict) or not commit_payload.get("ok"):
        return _render_publish_result(preview_payload, commit_payload, None)
    push_payload = _result_payload(
        ctx.dispatch_tool(
            push_tool,
            {
                "actor": actor,
                "source": source,
                "session_id": session_id,
                "user_confirmation": confirmation,
            },
        )
    )
    return _render_publish_result(preview_payload, commit_payload, push_payload)


def _publish_confirm_action(
    ctx: Any,
    target: str,
    *,
    preview_descriptor: dict[str, Any] | None,
    confirm_descriptor: dict[str, Any] | None,
    message: str,
) -> list[Any]:
    if not confirm_descriptor:
        return []
    try:
        from tools.kb_callback_registry import KbAction
    except Exception:
        return []
    return [
        KbAction(
            label="Confirm Publish",
            action_id="publication.commit_confirmed.confirm",
            handler=lambda callback_ctx: _render_publish_descriptor_confirm(
                ctx,
                target,
                preview_descriptor=preview_descriptor,
                confirm_descriptor=confirm_descriptor,
                message=message,
                callback_ctx=callback_ctx,
            ),
            metadata={
                "target_kind": "publication",
                "preview_tool": (preview_descriptor or {}).get("preview_tool") or "publication.preview_commit",
                "confirm_tool": confirm_descriptor.get("confirm_tool") or confirm_descriptor.get("method"),
                "preview_required": True,
            },
        )
    ]


def _render_publish_preview(
    payload: Any,
    *,
    confirm_hint: str = "/kb publish confirm",
    actions: list[Any] | None = None,
    closeout_packet: Any = None,
) -> dict[str, Any]:
    if isinstance(payload, dict) and payload.get("error"):
        return {"title": "KB Publish", "text": f"KB Publish Preview Failed\n{payload['error']}", "actions": []}
    if not isinstance(payload, dict):
        return {"title": "KB Publish", "text": "KB Publish Preview Failed\nPublication preview returned an unexpected response.", "actions": []}
    packet_card = _render_supported_result_packet(payload)
    if packet_card is not None:
        closeout_lines = _closeout_publication_lines(closeout_packet)
        if closeout_lines:
            packet_card["text"] = packet_card["text"] + "\n" + "\n".join(closeout_lines)
        packet_card["actions"] = actions or []
        return packet_card
    changed_paths = _changed_paths(payload)
    status = _short(payload.get("status"))
    message = _short(payload.get("message"), "Publish KB update")
    git_line = _publication_git_line(payload.get("git"))
    closeout_lines = _closeout_publication_lines(closeout_packet)
    if not changed_paths:
        lines = [
            "KB Publish Preview",
            "Decision Card: Publication",
            "Nothing to publish.",
            f"Status: {status}",
            f"Message: {message}",
        ]
        lines.extend(closeout_lines)
        lines.extend(_receipt_lines(payload))
        if git_line:
            lines.append(f"Git: {git_line}")
        return {"title": "KB Publish", "text": "\n".join(lines), "actions": actions or []}
    lines = [
        "KB Publish Preview",
        "Decision Card: Publication",
        f"Status: {status}",
        f"Message: {message}",
        f"Changed paths: {len(changed_paths)}",
    ]
    lines.extend(closeout_lines)
    lines.extend(_receipt_lines(payload))
    if git_line:
        lines.append(f"Git: {git_line}")
    lines.append("")
    lines.extend(_format_changed_paths(changed_paths))
    lines.extend(
        [
            "",
            f"To publish: {confirm_hint}",
            "No commit or push has been made.",
        ]
    )
    return {"title": "KB Publish", "text": "\n".join(lines), "actions": actions or []}


def _render_publish_result(preview: Any, commit: Any, push: Any) -> dict[str, Any]:
    changed_paths = _changed_paths(preview)
    if isinstance(commit, dict) and commit.get("error"):
        return {"title": "KB Publish", "text": f"KB Publish Failed\nCommit failed: {commit['error']}", "actions": []}
    if not isinstance(commit, dict):
        return {"title": "KB Publish", "text": "KB Publish Failed\nCommit returned an unexpected response.", "actions": []}
    packet_card = _render_supported_result_packet(commit)
    if packet_card is not None:
        return packet_card
    commit_status = _short(commit.get("status"))
    commit_ok = bool(commit.get("ok"))
    if not commit_ok:
        reason = _short(commit.get("reason") or _get_path(commit, "publication", "reason"), "unknown")
        lines = [
            "KB Publish Blocked",
            f"Committed: {commit_status}",
            f"Reason: {reason}",
        ]
        if changed_paths:
            lines.append(f"Changed paths: {len(changed_paths)}")
            lines.extend(_format_changed_paths(changed_paths))
        lines.append("Next: /kb publish")
        return {"title": "KB Publish", "text": "\n".join(lines), "actions": []}
    push_status = "not run"
    push_ok = False
    if isinstance(push, dict):
        push_status = _short(push.get("status"))
        push_ok = bool(push.get("ok"))
    elif push is not None:
        push_status = "unexpected response"
    publication = commit.get("publication") if isinstance(commit.get("publication"), dict) else {}
    commit_hash = _short(publication.get("commit") or publication.get("head"), "")
    lines = [
        "KB Published",
        f"Committed: {commit_status}",
        f"Pushed: {push_status}",
    ]
    lines.extend(_receipt_lines(commit))
    if commit_hash:
        lines.append(f"Commit: {commit_hash[:12]}")
    if changed_paths:
        lines.append(f"Changed paths: {len(changed_paths)}")
        lines.extend(_format_changed_paths(changed_paths))
    if not push_ok:
        lines.append("Warning: commit succeeded but push did not report success.")
        lines.append("Next: /kb publish push confirm")
    else:
        lines.append("Next: /kb status")
    return {"title": "KB Publish", "text": "\n".join(lines), "actions": []}


def _render_publish_command(ctx: Any, target: str, args: str) -> dict[str, Any]:
    confirm, message = _publish_args(args)
    closeout = _closeout_packet(ctx, target)
    descriptors = _closeout_action_descriptors_from_payload(closeout)
    preflight_descriptor = _publication_descriptor(descriptors, "publication.preflight")
    preview_descriptor = _publication_descriptor(descriptors, "publication.preview_commit")
    commit_descriptor = _publication_descriptor(descriptors, "publication.commit_confirmed")
    preview_tool = _descriptor_tool_name(
        target,
        (preview_descriptor or {}).get("preview_tool") or (preview_descriptor or {}).get("method") or "publication.preview_commit",
    )
    commit_tool = _descriptor_tool_name(
        target,
        (commit_descriptor or {}).get("confirm_tool") or (commit_descriptor or {}).get("method") or "publication.commit_confirmed",
    )
    push_tool = _mcp_tool_name(target, "publication.push_confirmed")
    actor = "telegram:operator"
    source = "Hermes Telegram"
    session_id = f"telegram-kb-publish-{int(time.time())}"
    preview_payload = _result_payload(ctx.dispatch_tool(preview_tool, _publication_descriptor_args(preview_descriptor, message=message)))
    if not confirm:
        actions = _publication_preflight_action(ctx, target, preflight_descriptor)
        if _changed_paths(preview_payload):
            actions.extend(
                _publish_confirm_action(
                    ctx,
                    target,
                    preview_descriptor=preview_descriptor,
                    confirm_descriptor=commit_descriptor,
                    message=message,
                )
            )
        preview_card = _render_publish_preview(
            preview_payload,
            actions=actions,
            closeout_packet=closeout,
        )
        return preview_card
    if not isinstance(preview_payload, dict) or preview_payload.get("error"):
        return _render_publish_preview(preview_payload, closeout_packet=closeout)
    changed_paths = _changed_paths(preview_payload)
    if not changed_paths:
        return {
            "title": "KB Publish",
            "text": _render_publish_preview(preview_payload, closeout_packet=closeout)["text"].replace("KB Publish Preview", "KB Publish"),
            "actions": [],
        }
    confirmation = {
        "confirmed": True,
        "surface": "telegram",
        "action": "publication.commit_and_push",
        "preview_required": True,
        "confirmation_text": "/kb publish confirm",
    }
    commit_payload = _result_payload(
        ctx.dispatch_tool(
            commit_tool,
            {
                **_publication_descriptor_args(commit_descriptor, message=message),
                "message": message,
                "expected_git_head": _short(_get_path(preview_payload, "git", "head"), ""),
                "expected_changed_paths": changed_paths,
                "push": False,
                "actor": actor,
                "source": source,
                "session_id": session_id,
                "user_confirmation": confirmation,
            },
        )
    )
    if not isinstance(commit_payload, dict) or not commit_payload.get("ok"):
        return _render_publish_result(preview_payload, commit_payload, None)
    push_payload = _result_payload(
        ctx.dispatch_tool(
            push_tool,
            {
                "actor": actor,
                "source": source,
                "session_id": session_id,
                "user_confirmation": confirmation,
            },
        )
    )
    return _render_publish_result(preview_payload, commit_payload, push_payload)


def _queue_items_from_payload(data: Any) -> list[Any]:
    return _items(data, ("items",), ("proposals",), ("queue", "items"))


def _queue_item_at(data: Any, index: int) -> dict[str, Any] | None:
    if index < 1:
        return None
    items = _queue_items_from_payload(data)
    if index > len(items):
        return None
    item = items[index - 1]
    return item if isinstance(item, dict) else None


def _parse_queue_indices(tokens: list[str]) -> list[int]:
    text = " ".join(tokens)
    indices: list[int] = []
    seen: set[int] = set()
    for match in re.finditer(r"\d+\s*-\s*\d+|\d+", text):
        token = match.group(0).strip()
        if re.fullmatch(r"\d+\s*-\s*\d+", token):
            start_text, end_text = re.split(r"\s*-\s*", token, maxsplit=1)
            start, end = int(start_text), int(end_text)
            step = 1 if end >= start else -1
            candidates = range(start, end + step, step)
        else:
            candidates = [int(token)]
        for index in candidates:
            if index > 0 and index not in seen:
                seen.add(index)
                indices.append(index)
    return indices


def _queue_items_at(data: Any, indices: list[int]) -> tuple[list[tuple[int, dict[str, Any]]], list[int]]:
    selection: list[tuple[int, dict[str, Any]]] = []
    missing: list[int] = []
    for index in indices:
        item = _queue_item_at(data, index)
        if item is None:
            missing.append(index)
        else:
            selection.append((index, item))
    return selection, missing


def _parse_queue_command_args(args: str, *, command: str) -> tuple[str, list[int], str | None, bool]:
    text = (args or "").strip()
    if not text:
        return "dashboard", [], None, False
    parts = text.split()
    first = parts[0].lower()
    if command == "kbreview":
        if first in {"review", "show", "detail", "details"}:
            indices = _parse_queue_indices(parts[1:])
        else:
            indices = _parse_queue_indices(parts)
        return ("review", indices[:1], None, False) if indices else ("help", [], None, False)
    if first.isdigit():
        return "review", [int(first)], None, False
    if first in {"review", "show", "detail", "details"}:
        indices = _parse_queue_indices(parts[1:])
        return ("review", indices[:1], None, False) if indices else ("help", [], None, False)
    if first in {"approve", "reject", "archive", "skip", "complete", "keep", "demote"}:
        confirm = any(part.lower() in {"confirm", "confirmed", "apply"} for part in parts[1:])
        index_tokens = [part for part in parts[1:] if part.lower() not in {"confirm", "confirmed", "apply"}]
        indices = _parse_queue_indices(index_tokens)
        return ("decision", indices, first, confirm) if indices else ("help", [], None, False)
    return "help", [], None, False


def _queue_command_help() -> dict[str, Any]:
    return {
        "title": "KB Review",
        "text": "\n".join(
            [
                "KB Review",
                "Use /kb review to list proposals.",
                "Use /kb review 1 to inspect one item.",
                "Use /kb review reject 1 to preview a decision.",
                "Use /kb review complete 1 for a TODO-backed proposal.",
                "Confirm from the Telegram preview button when available.",
                "Text fallback: /kb review reject 1 confirm only after that exact Telegram preview.",
                "Reply Reject all to preview only the review items currently shown in Telegram.",
            ]
        ),
        "actions": [],
    }


def _render_queue_item(data: Any, *, index: int, ctx: Any, target: str) -> dict[str, Any]:
    item = _queue_item_at(data, index)
    if item is None:
        total = len(_queue_items_from_payload(data))
        return {
            "title": "KB Review",
            "text": f"KB Review\nNo item {index} in the current review window ({total} shown). Use /kb review to refresh.",
            "actions": [],
        }
    return {
        "title": "KB Review",
        "text": _queue_item_text(item, index=index),
        "actions": _queue_descriptor_actions(
            ctx,
            target,
            item,
            index=index,
            preview_label_prefix=False,
            limit=6,
        )
        or _queue_control_actions(ctx, target, item, limit=6),
    }


def _render_queue_skip(
    data: Any,
    *,
    index: int,
    ctx: Any,
    target: str,
    session_id: str = "",
) -> dict[str, Any]:
    current_item = _queue_item_at(data, index)
    offset = _queue_offset(data)
    scope = _queue_payload_scope(data)
    server_data = None
    if ctx is not None and target:
        server_data, _errors = _queue_summary_payload(
            ctx,
            target,
            scope=scope,
            limit=5,
            offset=offset + index,
        )
    if server_data is not None:
        server_item = _queue_item_at(server_data, 1)
        current_ids = _proposal_ids_for_item(current_item) if isinstance(current_item, dict) else []
        server_ids = _proposal_ids_for_item(server_item) if isinstance(server_item, dict) else []
        if server_item is not None and (not current_ids or server_ids != current_ids):
            _store_iterative_state_from_item(session_id, server_item)
            card = _render_queue_item(server_data, index=1, ctx=ctx, target=target)
            card["text"] = (
                f"Skipped item {offset + index} locally. No KB state changed.\n"
                "Advanced to the next kb-engine review window.\n\n"
                + card["text"]
            )
            return card

    next_index = index + 1
    next_item = _queue_item_at(data, next_index)
    if next_item is None:
        if session_id:
            _clear_iterative_queue_reply_state(session_id)
        return {
            "title": "KB Review",
            "text": "KB Review\nNo more items are visible in this Telegram window. Refresh with /kb review.",
            "actions": [],
        }
    _store_iterative_state_from_item(session_id, next_item)
    card = _render_queue_item(data, index=next_index, ctx=ctx, target=target)
    card["text"] = f"Skipped item {index} locally. No KB state changed.\n\n{card['text']}"
    return card


def _queue_guided_actions(
    ctx: Any | None,
    target: str | None,
    data: Any,
    *,
    session_id: str = "",
) -> list[Any]:
    if ctx is None or not target:
        return []
    item = _queue_item_at(data, 1)
    if item is None:
        return []
    try:
        from tools.kb_callback_registry import KbAction
    except Exception:
        return []

    descriptor_actions = _queue_descriptor_actions(
        ctx,
        target,
        item,
        index=1,
        preview_label_prefix=False,
        limit=None,
    )
    guidance_actions = [action for action in descriptor_actions if getattr(action, "label", "") == "Ask LLM"]
    decision_actions = [action for action in descriptor_actions if getattr(action, "label", "") != "Ask LLM"]
    if not decision_actions:
        decision_actions = _queue_control_actions(ctx, target, item, limit=None)
    if not guidance_actions and _review_target_payload(item):
        guidance_actions = [
            KbAction(
                label="Ask LLM",
                action_id="review_target.guidance",
                handler=lambda callback_ctx: _render_review_target_guidance(item),
                metadata={
                    "target_kind": _item_kind(item) or "review_target",
                    "target_ref": _item_target(item),
                    "advisory_only": True,
                },
            )
        ]
    detail_action = KbAction(
        label="Details",
        action_id="queue.details",
        handler=lambda callback_ctx: _render_queue_item(data, index=1, ctx=ctx, target=target),
        metadata={
            "target_kind": "proposal_queue",
            "target_ref": _item_target(item),
            "review_index": 1,
        },
    )
    actions: list[Any] = [*decision_actions, detail_action, *guidance_actions]
    if _queue_item_at(data, 2) is not None:
        actions.append(
            KbAction(
                label="Skip",
                action_id="queue.skip",
                handler=lambda callback_ctx: _render_queue_skip(
                    data,
                    index=1,
                    ctx=ctx,
                    target=target,
                    session_id=session_id,
                ),
                metadata={
                    "target_kind": "proposal_queue",
                    "target_ref": _item_target(item),
                    "review_index": 1,
                    "mutates_state": False,
                },
            )
        )
    return actions[:6]


def _render_queue_text_decision(
    ctx: Any,
    target: str,
    data: Any,
    *,
    indices: list[int],
    decision: str,
    confirm: bool,
    session_id: str = "",
    callback_ctx: Any | None = None,
) -> dict[str, Any]:
    preview_metadata: dict[str, Any] = {}
    if confirm:
        selection, preview_metadata = _get_queue_text_preview_scope(session_id, decision=decision, indices=indices)
        missing: list[int] = []
        if not selection:
            return {
                "title": "KB Review",
                "text": (
                    "KB Review\n"
                    "That confirmation is not tied to a current Telegram preview. "
                    "Preview the exact item(s) again, then confirm from that preview."
                ),
                "actions": [],
            }
    else:
        selection, missing = _queue_items_at(data, indices)
        if not selection:
            total = len(_queue_items_from_payload(data))
            return {
                "title": "KB Review",
                "text": f"KB Review\nNo selected items in the current review window ({total} shown). Use /kb review to refresh.",
                "actions": [],
            }
        if missing:
            # Partial selections can be previewed, but the stored confirmation
            # lease only covers the concrete items that were actually shown.
            pass
    proposal_ids = _proposal_ids_for_selection(selection)
    if not proposal_ids:
        return {"title": "KB Review", "text": "No proposal ids were available for the selected review item(s).", "actions": []}
    selected_titles = ", ".join(_item_title(item) for _, item in selection)
    index_text = _format_indices([index for index, _ in selection])
    preview_tool = _mcp_tool_name(target, "queue.decision_preview")
    confirmed_tool = _mcp_tool_name(target, "queue.batch_decide_confirmed")
    actor = _queue_callback_actor(callback_ctx) if callback_ctx is not None else "telegram:operator"
    source = "Hermes Telegram"
    preview_payload: Any = None
    if not confirm or not preview_metadata.get("preview_lease"):
        preview_payload = _result_payload(
            ctx.dispatch_tool(
                preview_tool,
                {
                    "proposal_ids": proposal_ids,
                    "decision": decision,
                    "decision_scope": "explicit_ids",
                    "candidate_count": len(proposal_ids),
                    "displayed_count": len(proposal_ids),
                    "actor": actor,
                    "source": source,
                    "note": f"Previewed from Telegram /kb review text command for {selected_titles}",
                },
            )
        )
        if confirm:
            preview_metadata.update(_queue_preview_metadata(preview_payload))
    if not confirm:
        text = _preview_text(decision, proposal_ids, preview_payload, selection=selection)
        if missing:
            text += "\nMissing review item(s): " + ", ".join(str(index) for index in missing)
        actions: list[Any] = []
        if _preview_allows_confirmation(preview_payload):
            _store_queue_text_preview_scope(
                session_id,
                decision=decision,
                indices=[index for index, _ in selection],
                selection=selection,
                preview_payload=preview_payload,
            )
            text += "\nConfirm with the button below when it matches your intent."
            text += f"\nText fallback: /kb review {decision} {index_text} confirm"
            try:
                from tools.kb_callback_registry import KbAction

                metadata = _queue_preview_metadata(preview_payload)
                actions = [
                    KbAction(
                        label=f"Confirm {decision.title()}",
                        action_id=f"queue.{decision}.confirm",
                        handler=lambda confirm_ctx: _render_queue_text_decision(
                            ctx,
                            target,
                            data,
                            indices=[index for index, _ in selection],
                            decision=decision,
                            confirm=True,
                            session_id=session_id,
                            callback_ctx=confirm_ctx,
                        ),
                        metadata={
                            "target_kind": "proposal_queue",
                            "decision": decision,
                            "preview_required": True,
                            "preview_lease": bool(metadata.get("preview_lease")),
                            "review_session_id": _review_session_id(metadata),
                        },
                    )
                ]
            except Exception:
                actions = []
        return {"title": "KB Review", "text": text, "actions": actions}
    if not preview_metadata.get("preview_lease") and not _preview_allows_confirmation(preview_payload):
        return {
            "title": "KB Review",
            "text": _preview_text(decision, proposal_ids, preview_payload, selection=selection),
            "actions": [],
        }
    confirmed_args = {
        "proposal_ids": proposal_ids,
        "decision": decision,
        "actor": actor,
        "source": source,
        "session_id": _review_session_id(preview_metadata) or f"telegram-kb-text-{int(time.time())}",
        "user_confirmation": {
            "confirmed": True,
            "surface": "telegram",
            "action": f"queue.{decision}",
            "preview_required": True,
            "confirmation_text": f"/kb review {decision} {index_text} confirm",
            "proposal_ids": proposal_ids,
        },
        "note": f"Confirmed from Telegram /kb review text command for {selected_titles}",
    }
    _apply_queue_preview_metadata(confirmed_args, preview_metadata)
    _apply_queue_confirmation_preview_metadata(confirmed_args["user_confirmation"], preview_metadata)
    confirmed_payload = _result_payload(ctx.dispatch_tool(confirmed_tool, confirmed_args))
    packet_card = _render_supported_result_packet(confirmed_payload, ctx=ctx, target=target)
    if packet_card is not None:
        if missing:
            packet_card["text"] += "\nSkipped missing review item(s): " + ", ".join(str(index) for index in missing)
        return packet_card
    text = _confirmed_text(decision, confirmed_payload, selection=selection, proposal_ids=proposal_ids)
    if missing:
        text += "\nSkipped missing review item(s): " + ", ".join(str(index) for index in missing)
    return {"title": "KB Review", "text": text, "actions": []}


def _workflow_id_from_args(args: str) -> tuple[str, str]:
    text = (args or "").strip()
    lowered = text.lower()
    if not text or lowered in {"sync", "kb sync", "sync kb", "update kb", "update_kb"}:
        return "update_kb", text or "kb sync"
    if lowered.startswith("meeting"):
        return "meeting_process", text
    return text.split(maxsplit=1)[0], text


def _workflow_args_from_text(args: str) -> tuple[str, str, bool]:
    text = (args or "").strip()
    parts = text.split()
    confirm = bool(parts and parts[-1].lower() in {"confirm", "confirmed", "start", "apply"})
    if confirm:
        text = " ".join(parts[:-1]).strip()
    workflow_id, intent = _workflow_id_from_args(text)
    return workflow_id, intent, confirm


def _workflow_envelope(plan: dict[str, Any], callback_ctx: Any) -> dict[str, Any]:
    workflow = plan.get("workflow") if isinstance(plan.get("workflow"), dict) else {}
    request = plan.get("request") if isinstance(plan.get("request"), dict) else {}
    actor_id = _short(getattr(callback_ctx, "actor_id", ""), "unknown")
    actor_name = _short(getattr(callback_ctx, "actor_name", ""), "")
    confirmed_at = _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat()
    return {
        "schema_version": int(plan.get("schema_version") or 1),
        "tool": plan.get("tool") or "workflow.start_confirmed",
        "plan": {
            "workflow_id": str(workflow.get("workflow_id") or ""),
            "args": dict(request.get("args") or {}),
            "queue_gate_limit": int(request.get("queue_gate_limit") or 0),
            "force": bool(request.get("force", False)),
            "request_id": str(plan.get("request_id") or ""),
            "idempotency_key": str(plan.get("idempotency_key") or ""),
            "preconditions": list(plan.get("preconditions") or []),
        },
        "provenance": dict(plan.get("provenance") or {}),
        "user_confirmation": {
            "confirmed": True,
            "confirmed_by": actor_name or actor_id,
            "confirmed_at": confirmed_at,
            "confirmation_text": "Confirmed by Telegram text command after workflow preview.",
            "preview_required": True,
            "preview_lease": _sync_preview_lease(plan),
            "preview_status": _short(plan.get("status")),
            "surface": "telegram",
            "actor_id": actor_id,
            "actor_name": actor_name,
        },
    }


def _workflow_start_text(ctx: Any, target: str, plan: dict[str, Any]) -> str:
    callback_ctx = SimpleNamespace(
        callback_id=f"text-{int(time.time())}",
        actor_id="operator",
        actor_name="Telegram",
    )
    envelope = _workflow_envelope(plan, callback_ctx)
    payload = _result_payload(
        ctx.dispatch_tool(
            _mcp_tool_name(target, "workflow.start_confirmed"),
            {"envelope": envelope},
        )
    )
    text = _workflow_status_text("Workflow start result", payload, include_run_details=False)
    run_id = _workflow_run_id(payload)
    if run_id:
        progress_text = _workflow_initial_progress_text(ctx, target, run_id, include_run_details=False)
        if progress_text:
            text += "\n" + progress_text
        text += "\nDetails: /kb runs"
    return text


def _meeting_handoff_args(args: str) -> dict[str, Any]:
    text = str(args or "").strip()
    if text.lower() in {"confirm", "confirmed", "start", "apply"}:
        return {"confirm_pending": True}
    if not text:
        return {"error": "meeting_file and notes are required"}
    if text.lower().startswith("process "):
        text = text.split(maxsplit=1)[1].strip()
    if " -- " in text:
        meeting_file, notes_text = text.split(" -- ", 1)
    elif "\n" in text:
        first, notes_text = text.split("\n", 1)
        meeting_file = first
    else:
        return {"error": "separate the meeting file from notes with --"}
    meeting_file = _strip_wrapping_quotes(meeting_file.strip())
    notes_text = notes_text.strip()
    if not meeting_file:
        return {"error": "meeting_file is required"}
    if not notes_text:
        return {"error": "notes are required"}
    return {
        "confirm_pending": False,
        "meeting_file": meeting_file,
        "notes_text": notes_text,
    }


def _strip_wrapping_quotes(value: str) -> str:
    text = str(value or "").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1].strip()
    return text


def _telegram_actor(source: Any) -> str:
    name = _short(getattr(source, "user_name", ""), "")
    user_id = _short(getattr(source, "user_id", ""), "")
    return f"telegram:{name or user_id or 'operator'}"


def _render_meeting_handoff_command(
    ctx: Any,
    target: str,
    args: str,
    *,
    source: Any = None,
    session_store: Any = None,
    adapter: Any = None,
) -> dict[str, Any]:
    parsed = _meeting_handoff_args(args)
    session_id = _conversation_state_id(session_store, source)
    if parsed.get("confirm_pending"):
        state = _get_meeting_handoff_state(session_id)
        if not state:
            return {
                "title": "Meeting Notes",
                "text": "Meeting Notes\nNo pending meeting handoff found for this chat.",
                "actions": [],
            }
        plan = state.get("plan") if isinstance(state.get("plan"), dict) else {}
        text = _workflow_start_text(ctx, target, plan)
        _clear_meeting_handoff_state(session_id)
        return {"title": "Meeting Notes", "text": text, "actions": []}
    if parsed.get("error"):
        return {"title": "Meeting Notes", "text": f"Meeting Notes\n{parsed['error']}", "actions": []}

    meeting_file = str(parsed.get("meeting_file") or "")
    notes_text = str(parsed.get("notes_text") or "")
    plan_args = {
        "meeting_file": meeting_file,
        "source_kind": "telegram",
        "source_notes_source": "telegram",
        "source_notes_text": notes_text,
        "harness_id": "telegram-hermes",
        "harness_session_id": session_id,
    }
    _, data, errors = _dispatch_first(
        ctx,
        target,
        [
            (
                "workflow.plan_request",
                {
                    "workflow_id": "meeting_process",
                    "args": plan_args,
                    "actor": _telegram_actor(source),
                    "source": "Hermes Telegram",
                    "session_id": session_id or f"telegram-meeting-{int(time.time())}",
                },
            )
        ],
    )
    if data is None:
        return _render_error("Meeting Notes", target, errors)
    if isinstance(data, dict) and data.get("status") == "confirmation_required":
        _store_meeting_handoff_state(
            session_id,
            plan=data,
            meeting_file=meeting_file,
            notes_text=notes_text,
        )
    card = _render_workflow_plan(
        data,
        ctx=ctx,
        target=target,
        adapter=adapter,
        start_hint="/kb meeting confirm",
    )
    card["title"] = "Meeting Notes"
    return card


def _workflow_initial_progress_text(
    ctx: Any,
    target: str,
    run_id: str,
    *,
    include_run_details: bool = True,
) -> str:
    payload = _result_payload(
        ctx.dispatch_tool(
            _mcp_tool_name(target, "run.watch"),
            {
                "run_id": run_id,
                "timeout_seconds": 0,
                "poll_interval_seconds": 1,
                "timeline_limit": 5,
            },
        )
    )
    if isinstance(payload, dict) and payload.get("error"):
        return "Initial progress: unavailable - " + _short(payload.get("error"))
    if not isinstance(payload, dict):
        return ""
    digest = payload.get("progress_digest") if isinstance(payload.get("progress_digest"), dict) else payload
    progress = digest.get("progress") if isinstance(digest.get("progress"), dict) else {}
    stage = digest.get("stage") if isinstance(digest.get("stage"), dict) else {}
    provider = digest.get("provider") if isinstance(digest.get("provider"), dict) else {}
    staleness = digest.get("staleness") if isinstance(digest.get("staleness"), dict) else {}
    phase = _short(progress.get("current_phase") or progress.get("current_step") or digest.get("status"), "")
    detail = _short(progress.get("current_detail") or progress.get("latest_message"), "")
    lines: list[str] = []
    if phase:
        lines.append(
            f"Initial progress: {phase}" + (f" - {detail}" if detail and include_run_details else "")
        )
    if include_run_details:
        stage_id = _short(stage.get("stage_id") or stage.get("call_name"), "")
        total = stage.get("total")
        if stage_id and total not in {None, ""}:
            try:
                failed_count = int(stage.get("failed") or 0)
            except (TypeError, ValueError):
                failed_count = 0
            lines.append(
                f"Stage: {stage_id} {_short(stage.get('completed'), '0')}/{_short(total, '0')}"
                + (f" failed {_short(stage.get('failed'), '0')}" if failed_count else "")
            )
        provider_name = _short(provider.get("provider"), "")
        model = _short(provider.get("model"), "")
        if provider_name or model:
            lines.append(f"Provider: {provider_name or 'unknown'} / {model or 'unknown'}")
    if staleness.get("stale"):
        age = _short(staleness.get("last_trace_age_seconds"), "unknown")
        lines.append(f"Attention: run appears stalled; no trace progress for {age}s")
    if include_run_details:
        if payload.get("terminal") is False:
            lines.append("Watch: still running")
        elif payload.get("terminal") is True:
            lines.append("Watch: terminal")
    return "\n".join(lines)


def _workflow_run_id(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    run = payload.get("run") if isinstance(payload.get("run"), dict) else {}
    return str(payload.get("run_id") or run.get("run_id") or "")


def _workflow_status_text(prefix: str, payload: Any, *, include_run_details: bool = True) -> str:
    if isinstance(payload, dict) and payload.get("error"):
        return f"{prefix}\n{payload['error']}"
    if not isinstance(payload, dict):
        return f"{prefix}\n{_short(payload, 'No structured response returned.')}"
    lines = [
        prefix,
        f"Status: {_short(payload.get('status'))}",
    ]
    lines.extend(_receipt_lines(payload))
    run_id = _workflow_run_id(payload)
    if run_id and include_run_details:
        lines.append(f"Run: {run_id}")
    if payload.get("started") is not None:
        lines.append(f"Started: {_short(payload.get('started'))}")
    follow = payload.get("followthrough_contract") if isinstance(payload.get("followthrough_contract"), dict) else {}
    if follow and include_run_details:
        lines.append(f"Next: {_short(follow.get('recommended_next_action'))}")
    if isinstance(payload.get("readiness"), dict):
        lines.append("Readiness: " + _short(payload["readiness"].get("status")))
    return "\n".join(lines)


def _render_workflow_plan(
    data: Any,
    *,
    ctx: Any,
    target: str,
    adapter: Any,
    start_hint: str = "/kb run sync confirm",
    title: str = "Workflow",
    heading: str = "Workflow Preview",
) -> dict[str, Any]:
    if isinstance(data, dict) and data.get("error"):
        return {"title": title, "text": f"{heading} failed\n{data['error']}", "actions": []}
    if not isinstance(data, dict):
        return {"title": title, "text": f"{title}\n{_short(data, 'No plan returned.')}", "actions": []}
    workflow = data.get("workflow") if isinstance(data.get("workflow"), dict) else {}
    request = data.get("request") if isinstance(data.get("request"), dict) else {}
    effect_plan = data.get("effect_plan") if isinstance(data.get("effect_plan"), dict) else {}
    effects = effect_plan.get("effects") if isinstance(effect_plan.get("effects"), list) else []
    lines = [
        heading,
        f"Request: {start_hint.removesuffix(' confirm') if start_hint.endswith(' confirm') else start_hint}",
        f"Workflow: {_short(workflow.get('workflow_id'))}",
        f"Status: {_short(data.get('status'))}",
        f"Risk: {_short(workflow.get('risk') or effect_plan.get('risk'))}",
        f"Force: {_short(request.get('force'))}",
    ]
    if data.get("message"):
        lines.append("Message: " + _short(data.get("message")))
    lines.extend(_receipt_lines(data, include_request=True))
    if isinstance(data.get("readiness"), dict):
        lines.append("Readiness: " + _short(data["readiness"].get("status")))
    if effects:
        lines.append("Effects: " + ", ".join(_short(effect.get("id")) for effect in effects[:4] if isinstance(effect, dict)))
    follow = data.get("followthrough_contract") if isinstance(data.get("followthrough_contract"), dict) else {}
    if follow:
        lines.append("Follow-through: " + _short(follow.get("watch_tool")) + " -> " + _short(follow.get("terminal_summary_tool")))
    if data.get("status") == "confirmation_required":
        lines.append(f"To start: {start_hint}")
    return {"title": title, "text": "\n".join(lines), "actions": []}


def _sync_confirm_blocked_text(reason: str) -> str:
    messages = {
        "stale": "The pending /kb sync preview is stale. Run /kb sync again, then confirm from that fresh preview.",
        "wrong_actor": "The pending /kb sync preview belongs to another Telegram user. Run /kb sync yourself, then confirm.",
        "invalid": "The pending /kb sync preview is invalid. Run /kb sync again before confirming.",
        "missing_session": "Hermes could not identify this chat session. Run /kb sync again before confirming.",
    }
    return "\n".join(
        [
            "KB Sync",
            messages.get(reason, "No fresh /kb sync preview is pending for this chat. Run /kb sync first."),
            "No KB state changed.",
        ]
    )


def _render_queue(
    data: Any,
    *,
    ctx: Any | None = None,
    target: str | None = None,
    session_id: str = "",
) -> dict[str, Any]:
    if isinstance(data, str):
        return {"title": "KB Review", "text": f"KB Review\n{data}", "actions": []}
    count = None
    if isinstance(data, dict):
        count = data.get("total") or data.get("count") or _count_from(data, "queue", "proposals")
    items = _items(data, ("items",), ("proposals",), ("queue", "items"))
    visible_items = items[:5]
    total = _queue_total(data)
    offset = _queue_offset(data)
    next_offset = _queue_next_offset(data)
    _store_visible_queue_scope(session_id, visible_items, total=total, offset=offset, next_offset=next_offset)
    if visible_items and isinstance(visible_items[0], dict):
        _store_iterative_state_from_item(session_id, visible_items[0])
    elif session_id:
        _clear_iterative_queue_reply_state(session_id)
    if not items:
        lines = ["KB Review"]
        if count is not None:
            lines.append(f"{count} pending")
        lines.append("No proposal previews returned.")
        return {"title": "KB Review", "text": "\n".join(lines), "actions": []}
    return {
        "title": "KB Review",
        "text": _queue_review_text(data, visible_items, total=total, offset=offset),
        "actions": _queue_guided_actions(ctx, target, data, session_id=session_id),
    }


def _lifecycle_review_args(args: str) -> dict[str, Any]:
    text = str(args or "").strip()
    if text.lower().startswith("lifecycle "):
        text = text.split(maxsplit=1)[1].strip()
    if text.lower().startswith("for "):
        text = text.split(maxsplit=1)[1].strip()
    return {
        "target": text or "situations",
        "dry_run": True,
    }


def _render_lifecycle_review_command(ctx: Any, target: str, args: str) -> dict[str, Any]:
    _, data, errors = _dispatch_first(
        ctx,
        target,
        [("lifecycle.review", _lifecycle_review_args(args))],
    )
    if data is None:
        return _render_error("Lifecycle Review", target, errors)
    packet_card = _render_supported_result_packet(data, ctx=ctx, target=target)
    if packet_card is not None:
        return packet_card
    return _render_lifecycle_review_packet(
        {
            "packet_type": "lifecycle_review.packet",
            "workflow": "Lifecycle Review",
            "mutation_performed": False,
            "candidates": [],
            "summary": data,
        },
        ctx=ctx,
        target=target,
    )


def _kb_root_command(args: str) -> tuple[str, str]:
    text = (args or "").strip()
    if not text:
        return "kb", ""
    head, _, tail = text.partition(" ")
    key = head.strip().lower()
    rest = tail.strip()
    if key in {"dashboard", "home"}:
        return "kb", rest
    if key in {"workbench", "wb", "cards", "decision-cards"}:
        return "kbworkbench", rest
    if key in {"help", "commands"}:
        return "kbhelp", rest
    if key == "today":
        return "kbtoday", rest
    if key in {"status", "info"}:
        return "kbstatus", rest
    if key in {"reasoning", "reasoning-effort", "kb-reasoning"}:
        return "kbreasoning", rest
    if key in {"runs", "runlog", "history"}:
        return "kbruns", rest
    if key in {"queue", "q"}:
        return "kbqueue", rest
    if key == "review":
        review_head, _, review_tail = rest.partition(" ")
        review_mode = review_head.strip().lower()
        if not review_mode:
            return "kblifecycle", ""
        if review_mode in {"lifecycle", "stewardship"}:
            return "kblifecycle", review_tail.strip()
        if review_mode in {"proposal", "proposals", "queue", "inbox"}:
            return "kbqueue", review_tail.strip()
        if review_mode.isdigit() or review_mode in QUEUE_REPLY_DECISIONS:
            return "kbqueue", rest
        if "," in review_mode and all(part.strip().isdigit() for part in review_mode.split(",")):
            return "kbqueue", rest
        return "kbqueue", rest
    if key in {"lifecycle", "stewardship"}:
        return "kblifecycle", rest
    if key in {"publish", "publication"}:
        return "kbpublish", rest
    if key in {"run", "workflow"}:
        return "kbrun", rest
    if key in {"meeting", "meetings", "notes"}:
        return "kbmeeting", rest
    if key == "sync":
        return "kbsync", rest
    return "kbhelp", text


def _kb_command_help() -> dict[str, Any]:
    return {
        "title": "KB",
        "text": "\n".join(
            [
                "KB Commands",
                "/kb status - prove lane, runtime, transport, publication, review, sync, dirtiness, and next action",
                "/kb sync - preview evidence gathering and factual KB update workflow",
                "/kb review - lifecycle and proposal judgment inbox",
                "Advanced/debug aliases are still accepted for operators, but these three verbs are the normal KB surface.",
            ]
        ),
        "actions": [],
    }


def _normalize_kb_reasoning_effort(args: str) -> tuple[str, str]:
    effort = ((args or "").strip().split(maxsplit=1) or [""])[0].lower()
    if not effort:
        return "", f"Send /kb reasoning <level>. Options: {', '.join(sorted(KB_REASONING_LEVELS))}."
    if effort not in KB_REASONING_LEVELS:
        return "", f"Unknown KB reasoning effort '{effort}'. Options: {', '.join(sorted(KB_REASONING_LEVELS))}."
    return effort, ""


def _render_kb_reasoning_command(args: str, *, reload_available: bool) -> dict[str, Any]:
    effort, error = _normalize_kb_reasoning_effort(args)
    if error:
        return {"title": "KB Reasoning", "text": f"KB Reasoning\n{error}", "actions": []}
    try:
        from hermes_cli.config import get_env_path, save_env_value

        save_env_value("HERMES_KB_REASONING_EFFORT", effort)
        env_path = get_env_path()
    except Exception as exc:
        logger.warning("kb_journeys: failed to save KB reasoning effort", exc_info=True)
        return {
            "title": "KB Reasoning",
            "text": f"KB Reasoning\nCould not save KB reasoning effort: {_short(exc)}",
            "actions": [],
        }

    reload_line = "MCP reload started." if reload_available else "Run /reload-mcp to apply it to the KB MCP server."
    return {
        "title": "KB Reasoning",
        "text": "\n".join(
            [
                f"KB reasoning set to {effort}.",
                f"Saved: {env_path}:HERMES_KB_REASONING_EFFORT",
                reload_line,
            ]
        ),
        "actions": [],
        "_reload_mcp": reload_available,
    }


def _card_for_command(
    ctx: Any,
    command: str,
    *,
    args: str = "",
    adapter: Any = None,
    gateway: Any = None,
    source: Any = None,
    session_store: Any = None,
) -> dict[str, Any]:
    target = _mcp_target()
    queue_session_id = _conversation_state_id(session_store, source)
    cockpit_args = {
        "attention_limit": 5,
        "include_publication": True,
        "include_readiness": True,
        "run_limit": 3,
    }
    if command == "kb":
        routed_command, routed_args = _kb_root_command(args)
        if routed_command == "kbhelp":
            return _kb_command_help()
        if routed_command != "kb":
            return _card_for_command(
                ctx,
                routed_command,
                args=routed_args,
                adapter=adapter,
                gateway=gateway,
                source=source,
                session_store=session_store,
            )
        _, data, errors = _dispatch_first(
            ctx,
            target,
            [
                (
                    "dashboard.live",
                    {
                        "limit": 5,
                        "include_feedback": True,
                        "include_publication": True,
                        "include_readiness": True,
                    },
                ),
                ("attention.cockpit", cockpit_args),
            ],
        )
        return _render_error("KB Dashboard", target, errors) if data is None else _render_dashboard(data, ctx=ctx, target=target)
    if command == "kbhelp":
        return _kb_command_help()
    if command == "kbworkbench":
        _, data, errors = _dispatch_first(
            ctx,
            target,
            [
                (
                    "dashboard.live",
                    {
                        "limit": 5,
                        "include_feedback": True,
                        "include_publication": True,
                        "include_readiness": True,
                    },
                ),
                ("attention.cockpit", cockpit_args),
            ],
        )
        return _render_error("KB Workbench", target, errors) if data is None else _render_workbench(data, ctx=ctx, target=target)
    if command == "kbtoday":
        _, data, errors = _dispatch_first(ctx, target, [("attention.cockpit", cockpit_args)])
        return _render_error("KB Today", target, errors) if data is None else _render_today(data)
    if command == "kbstatus":
        _, data, _errors = _dispatch_first(
            ctx,
            target,
            [
                ("status.proof", {}),
                ("attention.cockpit", cockpit_args),
            ],
        )
        _, provider_data, _provider_errors = _dispatch_first(ctx, target, [("provider.status", {})])
        hermes_reasoning = _live_hermes_reasoning(gateway, source)
        return _render_status(data, target, provider_data, hermes_reasoning=hermes_reasoning)
    if command == "kbreasoning":
        reload_available = callable(getattr(gateway, "_execute_mcp_reload", None))
        return _render_kb_reasoning_command(args, reload_available=reload_available)
    if command == "kbruns":
        _, data, errors = _dispatch_first(
            ctx,
            target,
            [
                ("run.health", {}),
                ("run.watch", {"mode": "progress_digest"}),
                ("progress_digest", {}),
            ],
        )
        return _render_error("KB Runs", target, errors) if data is None else _render_runs(data)
    if command == "kbreview" and not (args or "").strip():
        return _render_lifecycle_review_command(ctx, target, "")
    if command in {"kbqueue", "kbreview"}:
        queue_scope, queue_args = _queue_scope_and_args(args)
        mode, indices, decision, confirm = _parse_queue_command_args(queue_args, command=command)
        if mode == "help":
            return _queue_command_help()
        data, errors = _queue_summary_payload(ctx, target, scope=queue_scope, limit=5)
        if data is None:
            return _render_error("KB Review", target, errors)
        packet_card = _render_supported_result_packet(data, ctx=ctx, target=target)
        if packet_card is not None:
            return packet_card
        if mode == "review" and indices:
            return _render_queue_item(data, index=indices[0], ctx=ctx, target=target)
        if mode == "decision" and indices and decision:
            return _render_queue_text_decision(
                ctx,
                target,
                data,
                indices=indices,
                decision=decision,
                confirm=confirm,
                session_id=queue_session_id,
            )
        return _render_queue(data, ctx=ctx, target=target, session_id=queue_session_id)
    if command == "kblifecycle":
        return _render_lifecycle_review_command(ctx, target, args)
    if command == "kbpublish":
        return _render_publish_command(ctx, target, args)
    if command == "kbmeeting":
        return _render_meeting_handoff_command(
            ctx,
            target,
            args,
            source=source,
            session_store=session_store,
            adapter=adapter,
        )
    if command == "kbsync":
        sync_args = f"sync {args}".strip()
        workflow_id, intent, confirm = _workflow_args_from_text(sync_args)
        if confirm:
            state, reason = _get_sync_preview_state(queue_session_id, source)
            if state is None:
                return {"title": "KB Sync", "text": _sync_confirm_blocked_text(reason), "actions": []}
            plan = state.get("plan") if isinstance(state.get("plan"), dict) else {}
            text = _workflow_start_text(ctx, target, plan)
            _clear_sync_preview_state(queue_session_id)
            return {"title": "KB Sync", "text": text, "actions": []}
        _, data, errors = _dispatch_first(
            ctx,
            target,
            [
                (
                    "workflow.plan_request",
                    {
                        "workflow_id": workflow_id or "sync",
                        "intent": intent,
                        "actor": "telegram:operator",
                        "source": "Hermes Telegram",
                        "session_id": f"telegram-kb-sync-{int(time.time())}",
                    },
                )
            ],
        )
        if data is None:
            return _render_error("KB Sync", target, errors)
        if isinstance(data, dict) and data.get("status") == "confirmation_required":
            _store_sync_preview_state(
                queue_session_id,
                source=source,
                target=target,
                workflow_id=workflow_id or "sync",
                intent=intent,
                plan=data,
            )
        else:
            _clear_sync_preview_state(queue_session_id)
        return _render_workflow_plan(
            data,
            ctx=ctx,
            target=target,
            adapter=adapter,
            start_hint="/kb sync confirm",
            title="KB Sync",
            heading="KB Sync Preview",
        )
    if command == "kbrun":
        workflow_id, intent, confirm = _workflow_args_from_text(args)
        if not workflow_id:
            return {
                "title": "Workflow",
                "text": "Workflow\nSend /kb run sync or /kb run <workflow_id>.",
                "actions": [],
            }
        _, data, errors = _dispatch_first(
            ctx,
            target,
            [
                (
                    "workflow.plan_request",
                    {
                        "workflow_id": workflow_id,
                        "intent": intent,
                        "actor": "telegram:operator",
                        "source": "Hermes Telegram",
                        "session_id": f"telegram-kb-{int(time.time())}",
                    },
                )
            ],
        )
        if data is None:
            return _render_error("Workflow", target, errors)
        if confirm and isinstance(data, dict) and data.get("status") == "confirmation_required":
            return {"title": "Workflow", "text": _workflow_start_text(ctx, target, data), "actions": []}
        hint_args = (args or "sync").strip()
        hint_parts = hint_args.split()
        if hint_parts and hint_parts[-1].lower() in {"confirm", "confirmed", "start", "apply"}:
            hint_args = " ".join(hint_parts[:-1]).strip()
        return _render_workflow_plan(
            data,
            ctx=ctx,
            target=target,
            adapter=adapter,
            start_hint=f"/kb run {hint_args or 'sync'} confirm",
        )
    return {"title": "KB", "text": "Unsupported KB command.", "actions": []}


def _adapter_for(gateway: Any, source: Any) -> Any | None:
    adapters = getattr(gateway, "adapters", {}) or {}
    platform = getattr(source, "platform", None)
    return (
        adapters.get(platform)
        or adapters.get(_platform_name(platform))
        or adapters.get("telegram")
    )


def _authorized_for_gateway(gateway: Any, source: Any) -> bool:
    checker = getattr(gateway, "_is_user_authorized", None)
    if checker is None:
        return True
    try:
        return bool(checker(source))
    except Exception:
        logger.debug("kb_journeys: authorization check failed", exc_info=True)
        return False


def _reply_anchor_and_metadata(event: Any) -> tuple[str | None, dict[str, Any] | None]:
    source = getattr(event, "source", None)
    try:
        from gateway.platforms.base import _reply_anchor_for_event, _thread_metadata_for_source

        return _reply_anchor_for_event(event), _thread_metadata_for_source(source)
    except Exception:
        metadata = None
        if getattr(source, "thread_id", None):
            metadata = {"thread_id": getattr(source, "thread_id")}
        return getattr(event, "message_id", None), metadata


async def _send_card(adapter: Any, event: Any, card: dict[str, Any]) -> None:
    source = getattr(event, "source", None)
    chat_id = getattr(source, "chat_id", None)
    if not chat_id:
        return
    reply_to, metadata = _reply_anchor_and_metadata(event)
    actions = card.get("actions", []) or []
    if actions and hasattr(adapter, "send_kb_actions"):
        result = adapter.send_kb_actions(
            chat_id,
            card["text"],
            actions,
            reply_to=reply_to,
            metadata=metadata,
        )
    else:
        result = adapter.send(chat_id, card["text"], reply_to=reply_to, metadata=metadata)
    if inspect.isawaitable(result):
        result = await result
    if actions and not getattr(result, "success", True):
        labels = []
        for action in actions:
            label = getattr(action, "label", None)
            if label is None and isinstance(action, dict):
                label = action.get("label")
            if label:
                labels.append(str(label))
        fallback_text = card["text"]
        if labels:
            fallback_text = f"{card['text']}\n\nActions: {', '.join(labels)}"
        fallback = adapter.send(chat_id, fallback_text, reply_to=reply_to, metadata=metadata)
        if inspect.isawaitable(fallback):
            await fallback


async def _send_mcp_reload_result(adapter: Any, event: Any, gateway: Any) -> None:
    reload_fn = getattr(gateway, "_execute_mcp_reload", None)
    if not callable(reload_fn):
        return
    try:
        result = reload_fn(event)
        if inspect.isawaitable(result):
            result = await result
        text = "KB MCP Reload\n" + _short(result, "complete")
    except Exception as exc:
        logger.warning("kb_journeys: MCP reload after KB reasoning change failed", exc_info=True)
        text = f"KB MCP Reload\nReload failed: {_short(exc)}"
    await _send_card(adapter, event, {"title": "KB MCP Reload", "text": text, "actions": []})


def _run_delivery(coro: Any) -> None:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(coro)
        return
    loop.create_task(coro)


def build_pre_gateway_dispatch_hook(ctx: Any) -> Callable[..., dict[str, str] | None]:
    def _hook(event: Any = None, gateway: Any = None, session_store: Any = None, **_: Any) -> dict[str, str] | None:
        source = getattr(event, "source", None)
        if _platform_name(getattr(source, "platform", None)) != "telegram":
            return None
        text = getattr(event, "text", "")
        command = _command_from_text(text)
        prose_command = _prose_kb_command_from_text(text) if command is None else None
        bare_decision = _bare_queue_reply_decision(text)
        visible_all_decision = _visible_scope_all_decision(text)
        if command is None and prose_command is None and not bare_decision and not visible_all_decision:
            return None
        if not _authorized_for_gateway(gateway, source):
            return None
        adapter = _adapter_for(gateway, source)
        if adapter is None:
            logger.debug("kb_journeys: no Telegram adapter available")
            return None
        if visible_all_decision:
            session_id = _conversation_state_id(session_store, source)
            card = _render_visible_scope_all_decision(
                ctx,
                _mcp_target(),
                session_id=session_id,
                decision=visible_all_decision,
            )
        elif bare_decision:
            session_id = _session_id_for_queue_reply_state(session_store, source)
            state = _get_iterative_queue_reply_state(session_id)
            if not state:
                return None
            card = _render_iterative_queue_reply_decision(
                ctx,
                _mcp_target(),
                session_id=session_id,
                state=state,
                decision=bare_decision,
            )
        else:
            if prose_command is not None:
                command, args = prose_command
            else:
                args = _command_args_from_text(text)
            card = _card_for_command(
                ctx,
                command,
                args=args,
                adapter=adapter,
                gateway=gateway,
                source=source,
                session_store=session_store,
            )
        reload_mcp = bool(card.pop("_reload_mcp", False))
        _run_delivery(_send_card(adapter, event, card))
        if reload_mcp:
            _run_delivery(_send_mcp_reload_result(adapter, event, gateway))
        return {"action": "skip", "reason": "kb_journeys"}

    return _hook


def _on_post_llm_call(
    *,
    session_id: str = "",
    assistant_response: str = "",
    platform: str = "",
    **_: Any,
) -> None:
    if str(platform or "").lower() != "telegram":
        return
    _record_iterative_queue_reply_state(session_id, assistant_response)


def register(ctx: Any) -> None:
    def _command_help(_: str = "") -> str:
        return "Use /kb in Telegram. Try: /kb status, /kb sync, or /kb review."

    for command in sorted(MENU_COMMANDS):
        try:
            ctx.register_command(
                command,
                _command_help,
                description="KB status, sync, and review.",
            )
        except Exception:
            logger.debug("kb_journeys: failed to register /%s", command, exc_info=True)
    ctx.register_hook("pre_gateway_dispatch", build_pre_gateway_dispatch_hook(ctx))
    ctx.register_hook("post_llm_call", _on_post_llm_call)
