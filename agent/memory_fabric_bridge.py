"""Shared Hermes Memory Fabric bridge.

This module is intentionally provider-light: it reads Hermes-owned memory
surfaces and creates write proposals, but it never mutates curated memory or
the Memory Graph directly. MCP clients such as Codex and OpenClaw can use it as
the single shared memory entry point.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from hermes_constants import get_hermes_home
from tools.memory_tool import MemoryStore, _scan_memory_content


VALID_SEARCH_SCOPES = {
    "all",
    "graph",
    "prompt_cases",
    "knowledge",
    "legacy_memory",
}
VALID_PROPOSAL_SCOPES = {
    "global",
    "project",
    "agent_private",
    "user",
    "memory",
    "procedural",
}
VALID_GATE_OPERATIONS = {
    "status",
    "audit",
    "search",
    "graph_read",
    "snapshot_export",
    "auto_precheck",
    "external_auto_recall",
    "write_proposal",
    "direct_write",
    "promote_memory",
    "delete_memory",
}
INTERNAL_MEMORY_CLIENTS = {"hermes", "codex", "openclaw"}
DURABLE_WRITE_OPERATIONS = {"direct_write", "promote_memory", "delete_memory"}
VALID_LEDGER_EVENT_TYPES = {
    "gate_decision",
    "search",
    "graph_read",
    "write_proposal_created",
    "snapshot_export_created",
}
VALID_POLICY_PROPOSAL_DECISIONS = {"approved", "rejected", "deferred"}
VALID_POLICY_PROPOSAL_STATUSES = {
    "proposed",
    "approved",
    "rejected",
    "deferred",
}
POLICY_EXECUTE_CONFIRM_TOKEN = "HERMES_EXECUTE_APPROVED_POLICY_PLAN"
VALID_POLICY_EXECUTE_ACTIONS = {
    "verify",
    "run_check",
    "diagnose",
    "review",
    "manual_review",
    "manual_repair",
}
EXTERNAL_CHANNEL_ROOTS = {
    "telegram",
    "wechat",
    "whatsapp",
    "discord",
    "irc",
    "googlechat",
    "slack",
    "signal",
    "imessage",
    "feishu",
    "nostr",
    "msteams",
    "mattermost",
    "nextcloud-talk",
    "matrix",
    "bluebubbles",
    "line",
    "zalo",
    "zalouser",
    "synology-chat",
    "tlon",
    "qa-channel",
    "qqbot",
    "twitch",
}


def memory_bridge_status() -> dict[str, Any]:
    """Return read-only status for Hermes shared memory surfaces."""

    home = get_hermes_home()
    graph_path = _graph_path(home)
    prompt_index_path = _prompt_index_path(home)
    knowledge_dir = home / "knowledge"
    proposal_path = _proposal_path(home)
    return {
        "success": True,
        "status": "available",
        "hermes_home": str(home),
        "surfaces": {
            "graph": {
                "path": str(graph_path),
                "exists": graph_path.exists(),
                "node_count": _sqlite_count(graph_path, "graph_nodes"),
                "edge_count": _sqlite_count(graph_path, "graph_edges"),
            },
            "gpt_image_prompt_cases": {
                "path": str(prompt_index_path),
                "exists": prompt_index_path.exists(),
                "case_count": _sqlite_count(prompt_index_path, "cases"),
            },
            "knowledge": {
                "path": str(knowledge_dir),
                "exists": knowledge_dir.exists(),
                "file_count": _knowledge_file_count(knowledge_dir),
            },
            "legacy_memory": {
                "path": str(home / "memories"),
                "exists": (home / "memories").exists(),
            },
            "write_proposals": {
                "path": str(proposal_path),
                "exists": proposal_path.exists(),
                "proposal_count": _jsonl_count(proposal_path),
            },
            "operation_ledger": {
                "path": str(_operation_ledger_path(home)),
                "exists": _operation_ledger_path(home).exists(),
                "event_count": _jsonl_count(_operation_ledger_path(home)),
            },
            "policy_proposals": {
                "path": str(_policy_proposal_path(home)),
                "exists": _policy_proposal_path(home).exists(),
                "event_count": _jsonl_count(_policy_proposal_path(home)),
            },
        },
        "policy": {
            "hermes_is_primary_memory": True,
            "external_clients_should_not_clone_as_primary": True,
            "writes_are_proposal_only": True,
            "read_only_memory": True,
            "would_mutate_memory": False,
        },
    }


def memory_federation_status() -> dict[str, Any]:
    """Return read-only status for clients sharing Hermes Memory Fabric."""

    home = get_hermes_home()
    codex = _codex_memory_client_status()
    openclaw = _openclaw_memory_client_status()
    hermes = {
        "role": "primary_memory_owner",
        "hermes_home": str(home),
        "mcp_server": {
            "path": str(Path(__file__).resolve().parents[1] / "mcp_serve.py"),
            "exists": (Path(__file__).resolve().parents[1] / "mcp_serve.py").exists(),
            "tools": [
                "memory_bridge_status",
                "memory_fabric_search",
                "memory_graph_read",
                "memory_write_proposal",
                "memory_snapshot_export",
                "memory_federation_status",
                "memory_federation_audit",
                "memory_federation_gate",
                "memory_operation_ledger",
                "memory_ledger_intelligence",
                "memory_policy_autotune",
                "memory_policy_proposal_create",
                "memory_policy_proposal_ledger",
                "memory_policy_proposal_decision",
                "memory_policy_apply_plan",
                "memory_policy_apply_execute",
                "memory_policy_outcome_monitor",
                "memory_policy_stale_resolution_preview",
                "memory_policy_stale_closure_payload_preview",
                "memory_policy_stale_closure_execute_plan",
                "memory_policy_stale_closure_handoff_bundle",
            ],
        },
        "surfaces": memory_bridge_status()["surfaces"],
    }
    clients = {
        "hermes": hermes,
        "codex": codex,
        "openclaw": openclaw,
    }
    warnings = _federation_warnings(clients)
    return {
        "success": True,
        "federation_type": "hermes_primary_shared_memory",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "clients": clients,
        "policy": {
            "primary_memory_owner": "hermes",
            "codex_access": "mcp_read_and_governed_write_proposal",
            "openclaw_access": "plugin_read_and_governed_write_proposal",
            "external_clients_should_not_clone_as_primary": True,
            "snapshots_are_cache_or_backup_only": True,
            "writes_are_proposal_only": True,
            "external_channel_auto_recall_requires_allowlist": True,
        },
        "ready": not warnings,
        "warnings": warnings,
        "read_only": True,
        "read_only_memory": True,
        "would_mutate_memory": False,
    }


def memory_federation_audit(*, log_limit: int = 200) -> dict[str, Any]:
    """Return a read-only audit report for the shared memory federation."""

    log_limit = _clamp_int(log_limit, default=200, minimum=20, maximum=2000)
    status = memory_federation_status()
    openclaw_logs = _read_recent_openclaw_log_lines(limit=log_limit)
    proposal_summary = _memory_proposal_summary()
    checks = _build_federation_audit_checks(status, openclaw_logs, proposal_summary)
    health_score = _audit_health_score(checks)
    return {
        "success": True,
        "audit_type": "hermes_memory_federation_audit",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "health_score": health_score,
        "risk_level": _audit_risk_level(health_score),
        "ready": status.get("ready") is True and health_score >= 80,
        "federation_status": {
            "ready": status.get("ready"),
            "warnings": status.get("warnings", []),
            "clients": sorted((status.get("clients") or {}).keys()),
        },
        "proposal_summary": proposal_summary,
        "log_summary": {
            "source": openclaw_logs["source"],
            "line_count": len(openclaw_logs["lines"]),
            "openclaw_loaded_events": openclaw_logs["loaded_events"],
            "auto_precheck_injections": openclaw_logs["auto_precheck_injections"],
            "auto_precheck_misses": openclaw_logs["auto_precheck_misses"],
            "rate_limit_events": openclaw_logs["rate_limit_events"],
        },
        "checks": checks,
        "policy": {
            "audit_is_read_only": True,
            "does_not_write_memory": True,
            "ledger_is_synthesized_from_current_state": True,
            "durable_memory_writes_remain_proposal_only": True,
        },
        "read_only": True,
        "read_only_memory": True,
        "would_mutate_memory": False,
    }


def memory_federation_gate(
    *,
    client: str = "codex",
    operation: str = "search",
    target_scope: str = "project",
    channel_id: str = "",
    log_limit: int = 200,
) -> dict[str, Any]:
    """Return a read-only allow/review/block decision for a memory operation."""

    requested_operation = _clean_text(operation).lower()
    normalized_operation = (
        requested_operation if requested_operation in VALID_GATE_OPERATIONS else "unknown"
    )
    normalized_client = _clean_text(client).lower() or "unknown"
    normalized_scope = _proposal_scope(target_scope)
    normalized_channel_id = _clean_text(channel_id)
    log_limit = _clamp_int(log_limit, default=200, minimum=20, maximum=2000)

    audit = memory_federation_audit(log_limit=log_limit)
    checks = audit.get("checks", []) if isinstance(audit.get("checks"), list) else []
    failed_checks = [check for check in checks if check.get("status") != "pass"]
    critical_failures = [
        check for check in failed_checks if check.get("severity") == "critical"
    ]
    status = memory_federation_status()
    openclaw = (
        status.get("clients", {}).get("openclaw", {})
        if isinstance(status.get("clients"), dict)
        else {}
    )
    allowed_channels = (
        openclaw.get("external_auto_precheck_allowed_channels", [])
        if isinstance(openclaw, dict)
        else []
    )
    rules = _build_federation_gate_rules(
        audit=audit,
        client=normalized_client,
        operation=normalized_operation,
        requested_operation=requested_operation,
        target_scope=normalized_scope,
        channel_id=normalized_channel_id,
        allowed_channels=allowed_channels,
        critical_failures=critical_failures,
    )
    decision = _gate_decision(rules)
    result = {
        "success": True,
        "gate_type": "hermes_memory_federation_gate",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "allowed": decision == "allow",
        "client": normalized_client,
        "operation": normalized_operation,
        "requested_operation": requested_operation,
        "target_scope": normalized_scope,
        "channel_id": normalized_channel_id,
        "audit_summary": {
            "ready": audit.get("ready"),
            "health_score": audit.get("health_score"),
            "risk_level": audit.get("risk_level"),
            "failed_checks": [check.get("id") for check in failed_checks],
            "critical_failures": [check.get("id") for check in critical_failures],
        },
        "rules": rules,
        "required_action": _gate_required_action(decision, rules),
        "policy": {
            "gate_is_read_only": True,
            "does_not_write_memory": True,
            "durable_writes_must_use_proposals": True,
            "external_auto_recall_requires_allowlist": True,
            "snapshots_are_cache_or_backup_only": True,
        },
        "read_only": True,
        "read_only_memory": True,
        "would_mutate_memory": False,
    }
    result["operation_ledger"] = _append_memory_operation_event(
        event_type="gate_decision",
        client=normalized_client,
        operation=normalized_operation,
        decision=decision,
        metadata={
            "requested_operation": requested_operation,
            "target_scope": normalized_scope,
            "channel_id": normalized_channel_id,
            "audit_health_score": audit.get("health_score"),
            "audit_risk_level": audit.get("risk_level"),
            "failed_checks": [check.get("id") for check in failed_checks],
            "rule_statuses": {
                rule.get("id"): rule.get("status") for rule in rules
            },
        },
    )
    return result


def memory_operation_ledger(
    *,
    limit: int = 50,
    client: str = "",
    operation: str = "",
    decision: str = "",
    event_type: str = "",
) -> dict[str, Any]:
    """Read the append-only operation ledger without mutating memory."""

    limit = _clamp_int(limit, default=50, minimum=1, maximum=500)
    client = _clean_text(client).lower()
    operation = _clean_text(operation).lower()
    decision = _clean_text(decision).lower()
    event_type = _clean_text(event_type).lower()
    path = _operation_ledger_path(get_hermes_home())
    rows = _read_jsonl(path)
    parse_errors = [row for row in rows if row.get("_parse_error")]
    events = [row for row in rows if not row.get("_parse_error")]
    filtered = []
    for event in events:
        if client and _clean_text(event.get("client")).lower() != client:
            continue
        if operation and _clean_text(event.get("operation")).lower() != operation:
            continue
        if decision and _clean_text(event.get("decision")).lower() != decision:
            continue
        if event_type and _clean_text(event.get("event_type")).lower() != event_type:
            continue
        filtered.append(event)
    filtered.sort(key=lambda row: _clean_text(row.get("created_at")), reverse=True)
    limited = filtered[:limit]
    return {
        "success": True,
        "ledger_type": "hermes_memory_operation_ledger",
        "path": str(path),
        "exists": path.exists(),
        "total_events": len(events),
        "matched_events": len(filtered),
        "returned_events": len(limited),
        "events": limited,
        "summary": _operation_ledger_summary(events),
        "parse_error_count": len(parse_errors),
        "parse_errors": parse_errors[:5],
        "policy": {
            "ledger_is_append_only_audit": True,
            "ledger_does_not_store_memory_content": True,
            "does_not_write_memory": True,
        },
        "read_only": True,
        "read_only_memory": True,
        "would_mutate_memory": False,
    }


def memory_ledger_intelligence(
    *,
    limit: int = 500,
    client: str = "",
    operation: str = "",
) -> dict[str, Any]:
    """Analyze memory operation ledger patterns without mutating memory."""

    limit = _clamp_int(limit, default=500, minimum=20, maximum=5000)
    client = _clean_text(client).lower()
    operation = _clean_text(operation).lower()
    path = _operation_ledger_path(get_hermes_home())
    rows = _read_jsonl(path)
    parse_errors = [row for row in rows if row.get("_parse_error")]
    events = [row for row in rows if not row.get("_parse_error")]
    events.sort(key=lambda row: _clean_text(row.get("created_at")), reverse=True)
    scoped_events = []
    for event in events:
        if client and _clean_text(event.get("client")).lower() != client:
            continue
        if operation and _clean_text(event.get("operation")).lower() != operation:
            continue
        scoped_events.append(event)
    analyzed_events = scoped_events[:limit]
    metrics = _ledger_intelligence_metrics(analyzed_events)
    findings = _ledger_intelligence_findings(metrics, parse_errors)
    health_score = _ledger_intelligence_health_score(findings)
    return {
        "success": True,
        "intelligence_type": "hermes_memory_ledger_intelligence",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "path": str(path),
        "exists": path.exists(),
        "total_events": len(events),
        "matched_events": len(scoped_events),
        "analyzed_events": len(analyzed_events),
        "health_score": health_score,
        "risk_level": _audit_risk_level(health_score),
        "metrics": metrics,
        "findings": findings,
        "recommended_next_actions": _ledger_intelligence_actions(findings),
        "parse_error_count": len(parse_errors),
        "parse_errors": parse_errors[:5],
        "policy": {
            "analysis_is_read_only": True,
            "does_not_append_ledger_events": True,
            "does_not_write_memory": True,
            "does_not_store_memory_content": True,
        },
        "read_only": True,
        "read_only_memory": True,
        "would_mutate_memory": False,
    }


def memory_policy_autotune(
    *,
    limit: int = 500,
    client: str = "",
    operation: str = "",
    mode: str = "conservative",
) -> dict[str, Any]:
    """Generate read-only policy tuning suggestions from ledger intelligence."""

    limit = _clamp_int(limit, default=500, minimum=20, maximum=5000)
    client = _clean_text(client).lower()
    operation = _clean_text(operation).lower()
    mode = _policy_autotune_mode(mode)
    intelligence = memory_ledger_intelligence(
        limit=limit,
        client=client,
        operation=operation,
    )
    suggestions = _memory_policy_autotune_suggestions(intelligence, mode=mode)
    summary = _policy_autotune_summary(suggestions)
    return {
        "success": True,
        "autotune_type": "hermes_memory_policy_autotune",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "decision": "review_required" if summary["requires_review_count"] else "no_change",
        "intelligence_summary": {
            "health_score": intelligence.get("health_score"),
            "risk_level": intelligence.get("risk_level"),
            "analyzed_events": intelligence.get("analyzed_events"),
            "finding_ids": [
                finding.get("id")
                for finding in intelligence.get("findings", [])
                if isinstance(finding, dict)
            ],
        },
        "suggestion_count": len(suggestions),
        "suggestions": suggestions,
        "summary": summary,
        "policy": {
            "autotune_is_read_only": True,
            "does_not_modify_config": True,
            "does_not_append_ledger_events": True,
            "does_not_write_memory": True,
            "suggestions_require_human_review": True,
            "external_channel_allowlist_changes_are_never_auto_applied": True,
        },
        "read_only": True,
        "read_only_memory": True,
        "would_mutate_memory": False,
    }


def memory_policy_proposal_create(
    *,
    source_agent: str = "codex",
    limit: int = 500,
    client: str = "",
    operation: str = "",
    mode: str = "conservative",
    suggestion_id: str = "",
) -> dict[str, Any]:
    """Create governed policy proposals from current policy-autotune suggestions."""

    source_agent = _clean_text(source_agent).lower() or "unknown"
    suggestion_id = _clean_text(suggestion_id)
    autotune = memory_policy_autotune(
        limit=limit,
        client=client,
        operation=operation,
        mode=mode,
    )
    suggestions = [
        suggestion
        for suggestion in autotune.get("suggestions", [])
        if isinstance(suggestion, dict)
        and suggestion.get("requires_human_review") is not False
        and _clean_text(suggestion.get("id")) != "policy.no_change"
        and (not suggestion_id or _clean_text(suggestion.get("id")) == suggestion_id)
    ]
    existing = _policy_proposal_states()
    created = []
    skipped_duplicates = []
    for suggestion in suggestions:
        proposal = _policy_proposal_from_suggestion(
            suggestion,
            source_agent=source_agent,
            autotune=autotune,
        )
        current = existing.get(proposal["proposal_id"])
        if current and current.get("latest_status") == "proposed":
            skipped_duplicates.append(
                {
                    "proposal_id": proposal["proposal_id"],
                    "suggestion_id": proposal["suggestion_id"],
                    "latest_status": current.get("latest_status"),
                }
            )
            continue
        event = _append_policy_proposal_event(
            {
                "event_type": "policy_proposal_created",
                **proposal,
            }
        )
        created.append({**proposal, "ledger_event": event})
        existing[proposal["proposal_id"]] = {
            **proposal,
            "latest_status": "proposed",
            "decisions": [],
        }
    return {
        "success": True,
        "proposal_type": "hermes_memory_policy_proposal_create",
        "created_count": len(created),
        "skipped_duplicate_count": len(skipped_duplicates),
        "proposals": created,
        "skipped_duplicates": skipped_duplicates,
        "autotune_summary": {
            "decision": autotune.get("decision"),
            "suggestion_count": autotune.get("suggestion_count"),
            "intelligence_summary": autotune.get("intelligence_summary", {}),
        },
        "policy": {
            "proposal_only": True,
            "does_not_apply_policy": True,
            "does_not_modify_config": True,
            "does_not_write_memory": True,
            "approval_records_intent_only": True,
        },
        "read_only_memory": True,
        "would_mutate_memory": False,
        "would_modify_config": False,
    }


def memory_policy_proposal_ledger(
    *,
    limit: int = 50,
    status: str = "",
    proposal_id: str = "",
) -> dict[str, Any]:
    """Read synthesized memory policy proposal state from the proposal ledger."""

    limit = _clamp_int(limit, default=50, minimum=1, maximum=500)
    status = _policy_proposal_status(status, allow_empty=True)
    proposal_id = _clean_text(proposal_id)
    path = _policy_proposal_path(get_hermes_home())
    rows = _read_jsonl(path)
    parse_errors = [row for row in rows if row.get("_parse_error")]
    proposals = list(_policy_proposal_states(rows).values())
    filtered = []
    for proposal in proposals:
        if proposal_id and proposal.get("proposal_id") != proposal_id:
            continue
        if status and proposal.get("latest_status") != status:
            continue
        filtered.append(proposal)
    filtered.sort(key=lambda row: _clean_text(row.get("updated_at")), reverse=True)
    limited = filtered[:limit]
    return {
        "success": True,
        "ledger_type": "hermes_memory_policy_proposal_ledger",
        "path": str(path),
        "exists": path.exists(),
        "total_proposals": len(proposals),
        "matched_proposals": len(filtered),
        "returned_proposals": len(limited),
        "proposals": limited,
        "summary": _policy_proposal_summary(proposals),
        "parse_error_count": len(parse_errors),
        "parse_errors": parse_errors[:5],
        "policy": {
            "ledger_is_append_only": True,
            "does_not_apply_policy": True,
            "does_not_modify_config": True,
            "does_not_write_memory": True,
        },
        "read_only": True,
        "read_only_memory": True,
        "would_mutate_memory": False,
        "would_modify_config": False,
    }


def memory_policy_proposal_decision(
    *,
    proposal_id: str,
    decision: str,
    reviewer: str = "codex",
    rationale: str = "",
) -> dict[str, Any]:
    """Append an approval/rejection/defer decision for a policy proposal."""

    proposal_id = _clean_text(proposal_id)
    decision = _policy_proposal_decision(decision)
    reviewer = _clean_text(reviewer) or "unknown"
    rationale = _clean_text(rationale)
    states = _policy_proposal_states()
    proposal = states.get(proposal_id)
    if not proposal:
        return _error("policy proposal was not found.", proposal_id=proposal_id)
    event = {
        "event_type": "policy_proposal_decision",
        "proposal_id": proposal_id,
        "decision": decision,
        "reviewer": reviewer,
        "rationale": rationale,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "does_not_apply_policy": True,
        "would_modify_config": False,
        "would_write_memory": False,
    }
    ledger_event = _append_policy_proposal_event(event)
    updated = _policy_proposal_states().get(proposal_id, proposal)
    return {
        "success": True,
        "proposal_id": proposal_id,
        "decision": decision,
        "proposal": updated,
        "ledger_event": ledger_event,
        "policy": {
            "decision_is_record_only": True,
            "does_not_apply_policy": True,
            "does_not_modify_config": True,
            "does_not_write_memory": True,
        },
        "read_only_memory": True,
        "would_mutate_memory": False,
        "would_modify_config": False,
    }


def memory_policy_apply_plan(
    *,
    limit: int = 50,
    status: str = "approved",
    proposal_id: str = "",
) -> dict[str, Any]:
    """Build a dry-run policy application plan from policy proposals."""

    limit = _clamp_int(limit, default=50, minimum=1, maximum=500)
    status = _policy_proposal_status(status, allow_empty=True) or "approved"
    proposal_id = _clean_text(proposal_id)
    ledger = memory_policy_proposal_ledger(
        limit=limit,
        status=status,
        proposal_id=proposal_id,
    )
    proposals = ledger.get("proposals", []) if isinstance(ledger.get("proposals"), list) else []
    plans = [_policy_apply_plan_for_proposal(proposal) for proposal in proposals]
    patch_count = sum(len(plan.get("patches", [])) for plan in plans)
    eligible_count = sum(1 for plan in plans if plan.get("eligible_for_apply"))
    return {
        "success": True,
        "plan_type": "hermes_memory_policy_apply_plan",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": True,
        "status_filter": status,
        "proposal_id_filter": proposal_id,
        "proposal_count": len(proposals),
        "eligible_count": eligible_count,
        "patch_count": patch_count,
        "plans": plans,
        "summary": {
            "by_action": _policy_apply_plan_summary(plans, "action"),
            "by_target_file": _policy_apply_plan_summary(plans, "target_file"),
        },
        "policy": {
            "plan_is_dry_run": True,
            "does_not_apply_policy": True,
            "does_not_modify_config": True,
            "does_not_write_memory": True,
            "approved_proposals_only_by_default": True,
            "requires_separate_apply_step": True,
        },
        "read_only": True,
        "read_only_memory": True,
        "would_mutate_memory": False,
        "would_modify_config": False,
    }


def memory_policy_apply_execute(
    *,
    limit: int = 50,
    proposal_id: str = "",
    execute: bool = False,
    confirm_token: str = "",
    actor: str = "codex",
) -> dict[str, Any]:
    """Guarded executor for approved policy plans.

    The executor only runs non-mutating verification/check actions. It refuses
    any plan patch that would modify config or memory.
    """

    limit = _clamp_int(limit, default=50, minimum=1, maximum=500)
    proposal_id = _clean_text(proposal_id)
    actor = _clean_text(actor) or "unknown"
    execute = _coerce_bool(execute)
    plan = memory_policy_apply_plan(
        limit=limit,
        status="approved",
        proposal_id=proposal_id,
    )
    guard = _policy_apply_guard(plan, execute=execute, confirm_token=confirm_token)
    if not execute:
        return {
            "success": True,
            "executor_type": "hermes_memory_policy_apply_execute",
            "dry_run": True,
            "did_execute": False,
            "plan": plan,
            "guard": guard,
            "results": [],
            "policy": {
                "execute_defaults_to_false": True,
                "requires_confirm_token": True,
                "allowed_actions_are_non_mutating": sorted(VALID_POLICY_EXECUTE_ACTIONS),
                "does_not_modify_config": True,
                "does_not_write_memory": True,
            },
            "read_only_memory": True,
            "would_mutate_memory": False,
            "would_modify_config": False,
        }
    if not guard["allowed"]:
        return {
            "success": False,
            "executor_type": "hermes_memory_policy_apply_execute",
            "dry_run": False,
            "did_execute": False,
            "error": guard["reason"],
            "plan": plan,
            "guard": guard,
            "results": [],
            "policy": {
                "blocked_before_execution": True,
                "does_not_modify_config": True,
                "does_not_write_memory": True,
            },
            "read_only_memory": True,
            "would_mutate_memory": False,
            "would_modify_config": False,
        }

    results = _execute_policy_apply_plan(plan)
    event = _append_policy_proposal_event(
        {
            "event_type": "policy_apply_execute",
            "proposal_id": proposal_id or "",
            "actor": actor,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "plan_summary": {
                "proposal_count": plan.get("proposal_count"),
                "eligible_count": plan.get("eligible_count"),
                "patch_count": plan.get("patch_count"),
            },
            "result_summary": _policy_apply_execute_summary(results),
            "does_not_apply_config_changes": True,
            "would_modify_config": False,
            "would_write_memory": False,
        }
    )
    summary = _policy_apply_execute_summary(results)
    return {
        "success": summary["blocked_count"] == 0,
        "executor_type": "hermes_memory_policy_apply_execute",
        "dry_run": False,
        "did_execute": True,
        "plan": plan,
        "guard": guard,
        "results": results,
        "summary": summary,
        "ledger_event": event,
        "policy": {
            "executed_actions_are_non_mutating": True,
            "does_not_modify_config": True,
            "does_not_write_memory": True,
            "writes_execution_audit_event_only": True,
            "backup_required": False,
            "diff_required": False,
        },
        "read_only_memory": True,
        "would_mutate_memory": False,
        "would_modify_config": False,
    }


def memory_policy_outcome_monitor(
    *,
    limit: int = 100,
    stale_after_hours: int = 72,
) -> dict[str, Any]:
    """Monitor policy proposal lifecycle outcomes without applying policy."""

    limit = _clamp_int(limit, default=100, minimum=1, maximum=500)
    stale_after_hours = _clamp_int(
        stale_after_hours,
        default=72,
        minimum=1,
        maximum=8760,
    )
    path = _policy_proposal_path(get_hermes_home())
    rows = _read_jsonl(path)
    parse_errors = [row for row in rows if row.get("_parse_error")]
    proposals = list(_policy_proposal_states(rows).values())
    metrics = _policy_outcome_metrics(
        proposals,
        rows,
        stale_after_hours=stale_after_hours,
    )
    findings = _policy_outcome_findings(metrics, parse_errors)
    health_score = _ledger_intelligence_health_score(findings)
    risk_level = "high" if health_score < 70 else "medium" if health_score < 90 else "low"
    executions = metrics.get("recent_execution_events", [])
    if not isinstance(executions, list):
        executions = []
    proposals.sort(key=lambda row: _clean_text(row.get("updated_at")), reverse=True)
    return {
        "success": True,
        "monitor_type": "hermes_memory_policy_outcome_monitor",
        "path": str(path),
        "exists": path.exists(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "health_score": health_score,
        "risk_level": risk_level,
        "stale_after_hours": stale_after_hours,
        "analyzed_events": len([row for row in rows if not row.get("_parse_error")]),
        "parse_error_count": len(parse_errors),
        "parse_errors": parse_errors[:5],
        "metrics": metrics,
        "findings": findings,
        "recommended_next_actions": _ledger_intelligence_actions(findings),
        "recent_proposals": proposals[:limit],
        "recent_execution_events": executions[:limit],
        "policy": {
            "monitor_is_read_only": True,
            "does_not_apply_policy": True,
            "does_not_modify_config": True,
            "does_not_write_memory": True,
            "approved_proposals_still_require_explicit_execute": True,
        },
        "read_only": True,
        "read_only_memory": True,
        "would_mutate_memory": False,
        "would_modify_config": False,
    }


def memory_policy_stale_resolution_preview(
    *,
    limit: int = 50,
    stale_after_hours: int = 72,
    proposal_id: str = "",
) -> dict[str, Any]:
    """Preview safe stale policy proposal resolutions without writing state."""

    limit = _clamp_int(limit, default=50, minimum=1, maximum=500)
    stale_after_hours = _clamp_int(
        stale_after_hours,
        default=72,
        minimum=1,
        maximum=8760,
    )
    proposal_id = _clean_text(proposal_id)
    monitor = memory_policy_outcome_monitor(
        limit=limit,
        stale_after_hours=stale_after_hours,
    )
    metrics = monitor.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    stale_items = metrics.get("stale_proposed", [])
    if not isinstance(stale_items, list):
        stale_items = []
    resolutions = []
    for item in stale_items:
        if not isinstance(item, dict):
            continue
        item_proposal_id = _clean_text(item.get("proposal_id"))
        if proposal_id and item_proposal_id != proposal_id:
            continue
        resolutions.append(_stale_policy_resolution_from_monitor_item(item))
        if len(resolutions) >= limit:
            break
    if proposal_id and not resolutions:
        proposal = _policy_proposal_states().get(proposal_id)
        if proposal and _clean_text(proposal.get("latest_status")) != "proposed":
            resolutions.append(_closed_policy_resolution_from_proposal(proposal))
    return {
        "success": True,
        "preview_type": "hermes_memory_policy_stale_resolution_preview",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": True,
        "proposal_id_filter": proposal_id,
        "stale_after_hours": stale_after_hours,
        "stale_count": len(stale_items),
        "matched_resolution_count": len(resolutions),
        "resolutions": resolutions,
        "recommended_next_actions": [
            resolution.get("recommended_action", "")
            for resolution in resolutions
            if resolution.get("recommended_action")
        ],
        "source_monitor": {
            "health_score": monitor.get("health_score"),
            "risk_level": monitor.get("risk_level"),
            "stale_proposed_count": metrics.get("stale_proposed_count"),
        },
        "policy": {
            "preview_is_read_only": True,
            "proposal_only": True,
            "does_not_apply_policy": True,
            "does_not_modify_config": True,
            "does_not_write_memory": True,
            "does_not_write_graph": True,
            "does_not_append_ledger": True,
            "no_direct_graph_write": True,
            "can_auto_apply": False,
            "requires_human_review": True,
        },
        "read_only": True,
        "read_only_memory": True,
        "would_mutate_memory": False,
        "would_modify_config": False,
    }


def memory_policy_stale_closure_payload_preview(
    *,
    limit: int = 50,
    stale_after_hours: int = 72,
    proposal_id: str = "",
) -> dict[str, Any]:
    """Preview human-review closure payloads for stale policy proposals.

    This is deliberately a preview-only bridge between stale proposal detection
    and the governed ``memory_policy_proposal_decision`` flow. It never appends
    decision events, never writes Memory Graph data, and never changes config.
    """

    limit = _clamp_int(limit, default=50, minimum=1, maximum=500)
    stale_after_hours = _clamp_int(
        stale_after_hours,
        default=72,
        minimum=1,
        maximum=8760,
    )
    proposal_id = _clean_text(proposal_id)
    resolution_preview = memory_policy_stale_resolution_preview(
        limit=limit,
        stale_after_hours=stale_after_hours,
        proposal_id=proposal_id,
    )
    resolutions = resolution_preview.get("resolutions", [])
    if not isinstance(resolutions, list):
        resolutions = []
    closure_payloads = [
        _stale_policy_closure_payload_from_resolution(resolution)
        for resolution in resolutions
        if isinstance(resolution, dict)
    ]
    return {
        "success": True,
        "preview_type": "hermes_memory_policy_stale_closure_payload_preview",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": True,
        "preview_only": True,
        "proposal_id_filter": proposal_id,
        "stale_after_hours": stale_after_hours,
        "stale_count": resolution_preview.get("stale_count", 0),
        "matched_resolution_count": resolution_preview.get("matched_resolution_count", 0),
        "closure_payload_count": len(closure_payloads),
        "closure_payloads": closure_payloads,
        "recommended_next_actions": [
            payload.get("recommended_next_action", "")
            for payload in closure_payloads
            if payload.get("recommended_next_action")
        ],
        "source_preview": {
            "preview_type": resolution_preview.get("preview_type"),
            "source_monitor": resolution_preview.get("source_monitor", {}),
        },
        "policy": {
            "preview_is_read_only": True,
            "preview_only": True,
            "proposal_only": True,
            "does_not_apply_policy": True,
            "does_not_modify_config": True,
            "does_not_write_memory": True,
            "does_not_write_graph": True,
            "does_not_append_ledger": True,
            "no_direct_graph_write": True,
            "can_auto_apply": False,
            "requires_human_review": True,
        },
        "read_only": True,
        "read_only_memory": True,
        "would_mutate_memory": False,
        "would_modify_config": False,
    }


def memory_policy_stale_closure_execute_plan(
    *,
    limit: int = 50,
    stale_after_hours: int = 72,
    proposal_id: str = "",
) -> dict[str, Any]:
    """Build a read-only execution gate plan for stale closure payloads."""

    limit = _clamp_int(limit, default=50, minimum=1, maximum=500)
    stale_after_hours = _clamp_int(stale_after_hours, default=72, minimum=1, maximum=8760)
    proposal_id = _clean_text(proposal_id)
    payload_preview = memory_policy_stale_closure_payload_preview(
        limit=limit,
        stale_after_hours=stale_after_hours,
        proposal_id=proposal_id,
    )
    payloads = payload_preview.get("closure_payloads", [])
    if not isinstance(payloads, list):
        payloads = []
    plans = [
        _stale_policy_closure_execute_plan_from_payload(payload)
        for payload in payloads
        if isinstance(payload, dict)
    ]
    return {
        "success": True,
        "plan_type": "hermes_memory_policy_stale_closure_execute_plan",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": True,
        "preview_only": True,
        "proposal_id_filter": proposal_id,
        "stale_after_hours": stale_after_hours,
        "plan_count": len(plans),
        "eligible_for_human_decision_count": sum(
            1 for plan in plans if plan.get("eligible_for_human_decision") is True
        ),
        "eligible_for_auto_execute_count": 0,
        "plans": plans,
        "source_preview": {
            "preview_type": payload_preview.get("preview_type"),
            "closure_payload_count": payload_preview.get("closure_payload_count", 0),
        },
        "policy": {
            "plan_is_read_only": True,
            "preview_only": True,
            "proposal_only": True,
            "does_not_call_policy_proposal_decision": True,
            "does_not_apply_policy": True,
            "does_not_modify_config": True,
            "does_not_write_memory": True,
            "does_not_write_graph": True,
            "does_not_append_ledger": True,
            "does_not_run_audit": True,
            "no_direct_graph_write": True,
            "can_auto_apply": False,
            "eligible_for_auto_execute": False,
            "requires_human_review": True,
        },
        "read_only": True,
        "read_only_memory": True,
        "would_mutate_memory": False,
        "would_modify_config": False,
    }


def memory_policy_stale_closure_handoff_bundle(
    *,
    limit: int = 50,
    stale_after_hours: int = 72,
    proposal_id: str = "",
) -> dict[str, Any]:
    """Build a read-only human handoff bundle for stale closure decisions."""

    limit = _clamp_int(limit, default=50, minimum=1, maximum=500)
    stale_after_hours = _clamp_int(stale_after_hours, default=72, minimum=1, maximum=8760)
    proposal_id = _clean_text(proposal_id)
    execute_plan = memory_policy_stale_closure_execute_plan(
        limit=limit,
        stale_after_hours=stale_after_hours,
        proposal_id=proposal_id,
    )
    plans = execute_plan.get("plans", [])
    if not isinstance(plans, list):
        plans = []
    handoff_bundles = [
        _stale_policy_handoff_bundle_from_plan(plan)
        for plan in plans
        if isinstance(plan, dict)
    ]
    return {
        "success": True,
        "bundle_type": "hermes_memory_policy_stale_closure_handoff_bundle",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": True,
        "preview_only": True,
        "proposal_id_filter": proposal_id,
        "stale_after_hours": stale_after_hours,
        "bundle_count": len(handoff_bundles),
        "handoff_bundles": handoff_bundles,
        "source_plan": {
            "plan_type": execute_plan.get("plan_type"),
            "plan_count": execute_plan.get("plan_count", 0),
            "eligible_for_auto_execute_count": execute_plan.get("eligible_for_auto_execute_count", 0),
        },
        "policy": {
            "bundle_is_read_only": True,
            "preview_only": True,
            "proposal_only": True,
            "does_not_call_policy_proposal_decision": True,
            "does_not_apply_policy": True,
            "does_not_modify_config": True,
            "does_not_write_memory": True,
            "does_not_write_graph": True,
            "does_not_append_ledger": True,
            "does_not_run_audit": True,
            "no_direct_graph_write": True,
            "can_auto_apply": False,
            "eligible_for_auto_execute": False,
            "requires_human_review": True,
        },
        "read_only": True,
        "read_only_memory": True,
        "would_mutate_memory": False,
        "would_modify_config": False,
    }


def search_memory_fabric(
    query: str,
    *,
    scope: str = "all",
    limit: int = 8,
) -> dict[str, Any]:
    """Search Memory Graph, knowledge files, prompt cases, and legacy memory."""

    query = _clean_text(query)
    scope = _scope(scope)
    limit = _clamp_int(limit, default=8, minimum=1, maximum=50)
    if not query:
        return _error("query is required.", query=query, scope=scope)

    home = get_hermes_home()
    results: list[dict[str, Any]] = []
    if scope in ("all", "graph"):
        results.extend(_search_graph(home, query, limit=limit))
    if scope in ("all", "prompt_cases"):
        results.extend(_search_prompt_cases(home, query, limit=limit))
    if scope in ("all", "knowledge"):
        results.extend(_search_knowledge(home, query, limit=limit))
    if scope in ("all", "legacy_memory"):
        results.extend(_search_legacy_memory(home, query, limit=limit))

    results.sort(key=lambda row: row.get("score", 0), reverse=True)
    limited = results[:limit]
    ledger_event = _append_memory_operation_event(
        event_type="search",
        client="hermes",
        operation="search",
        decision="allow",
        metadata={
            "scope": scope,
            "query_digest": _text_digest(query),
            "result_count": len(limited),
            "result_sources": sorted(
                {
                    _clean_text(row.get("source")) or "unknown"
                    for row in limited
                }
            ),
        },
    )
    return {
        "success": True,
        "query": query,
        "scope": scope,
        "count": len(limited),
        "results": limited,
        "operation_ledger": ledger_event,
        "read_only": True,
        "read_only_memory": True,
        "would_mutate_memory": False,
    }


def read_memory_graph(
    *,
    node_id: str = "",
    query: str = "",
    kind: str = "",
    limit: int = 10,
    include_edges: bool = True,
) -> dict[str, Any]:
    """Read Memory Graph nodes plus provenance and optional adjacent edges."""

    home = get_hermes_home()
    path = _graph_path(home)
    limit = _clamp_int(limit, default=10, minimum=1, maximum=100)
    node_id = _clean_text(node_id)
    query = _clean_text(query)
    kind = _clean_text(kind)
    if not path.exists():
        return _error("memory graph database does not exist.", path=str(path))

    try:
        with sqlite3.connect(path) as conn:
            conn.row_factory = sqlite3.Row
            params: list[Any] = []
            conditions: list[str] = []
            if node_id:
                conditions.append("id = ?")
                params.append(node_id)
            if kind:
                conditions.append("kind = ?")
                params.append(kind)
            if query:
                like = f"%{query.lower()}%"
                conditions.append(
                    "lower(title || ' ' || summary || ' ' || kind) LIKE ?"
                )
                params.append(like)
            where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
            rows = conn.execute(
                f"""
                SELECT id, kind, title, summary, source_id, confidence,
                       metadata_json, created_at, updated_at
                FROM graph_nodes
                {where}
                ORDER BY confidence DESC, updated_at DESC
                LIMIT ?
                """,
                (*params, limit),
            ).fetchall()
            nodes = []
            for row in rows:
                node = _row_to_dict(row)
                node["metadata"] = _loads_json(node.pop("metadata_json", "{}"), {})
                node["provenance"] = _graph_provenance(conn, node["id"])
                if include_edges:
                    node["edges"] = _graph_edges(conn, node["id"])
                nodes.append(node)
    except sqlite3.Error as exc:
        return _error(f"failed to read memory graph: {exc}", path=str(path))

    ledger_event = _append_memory_operation_event(
        event_type="graph_read",
        client="hermes",
        operation="graph_read",
        decision="allow",
        metadata={
            "node_id": node_id,
            "query_digest": _text_digest(query) if query else "",
            "kind": kind,
            "include_edges": bool(include_edges),
            "node_count": len(nodes),
        },
    )
    return {
        "success": True,
        "count": len(nodes),
        "nodes": nodes,
        "operation_ledger": ledger_event,
        "read_only": True,
        "read_only_memory": True,
        "would_mutate_memory": False,
    }


def create_memory_write_proposal(
    *,
    source_agent: str,
    target_scope: str,
    content: str,
    rationale: str = "",
    project: str = "",
    tags: Iterable[str] | str | None = None,
) -> dict[str, Any]:
    """Append a governed write proposal without mutating durable memory."""

    source_agent = _clean_text(source_agent) or "unknown"
    target_scope = _proposal_scope(target_scope)
    content = _clean_text(content)
    rationale = _clean_text(rationale)
    project = _clean_text(project)
    tag_list = _list_of_str(tags)
    if not content:
        return _error("content is required.")
    scan_error = _scan_memory_content(content)
    if scan_error:
        return _error(scan_error)

    now = datetime.now(timezone.utc).isoformat()
    seed = {
        "source_agent": source_agent,
        "target_scope": target_scope,
        "content": content,
        "rationale": rationale,
        "project": project,
        "tags": tag_list,
        "created_at": now,
    }
    digest = _digest(seed)
    proposal = {
        "id": f"memory-proposal-{digest[:16]}",
        "status": "proposed",
        "source_agent": source_agent,
        "target_scope": target_scope,
        "project": project,
        "content": content,
        "rationale": rationale,
        "tags": tag_list,
        "created_at": now,
        "proposal_digest": digest,
        "governance": {
            "requires_hermes_review": True,
            "requires_human_approval": True,
            "can_be_promoted_to_memory_graph": True,
            "can_be_promoted_to_curated_memory": target_scope in ("user", "memory"),
        },
        "write_policy": {
            "proposal_only": True,
            "read_only_memory": True,
            "would_mutate_memory": False,
        },
    }
    path = _proposal_path(get_hermes_home())
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(proposal, ensure_ascii=False, sort_keys=True) + "\n")
    ledger_event = _append_memory_operation_event(
        event_type="write_proposal_created",
        client=source_agent,
        operation="write_proposal",
        decision="allow",
        metadata={
            "proposal_id": proposal["id"],
            "proposal_digest": digest,
            "target_scope": target_scope,
            "project": project,
            "tags": tag_list,
            "content_digest": _text_digest(content),
        },
    )
    return {
        "success": True,
        "proposal": proposal,
        "proposal_path": str(path),
        "operation_ledger": ledger_event,
        "read_only_memory": True,
        "would_mutate_memory": False,
    }


def export_memory_snapshot(
    *,
    scope: str = "all",
    limit: int = 500,
) -> dict[str, Any]:
    """Export a portable snapshot for clone/cache use."""

    scope = _scope(scope)
    limit = _clamp_int(limit, default=500, minimum=1, maximum=5000)
    home = get_hermes_home()
    snapshot: dict[str, Any] = {
        "snapshot_type": "hermes_memory_fabric_snapshot",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hermes_home": str(home),
        "scope": scope,
        "policy": {
            "snapshot_is_cache_or_backup": True,
            "hermes_remains_primary_memory": True,
            "external_clients_must_not_treat_snapshot_as_primary": True,
        },
    }
    if scope in ("all", "graph"):
        snapshot["graph"] = _dump_graph(home, limit=limit)
    if scope in ("all", "prompt_cases"):
        snapshot["prompt_cases"] = _dump_prompt_cases(home, limit=limit)
    if scope in ("all", "knowledge"):
        snapshot["knowledge"] = _dump_knowledge_manifest(home, limit=limit)
    if scope in ("all", "legacy_memory"):
        snapshot["legacy_memory"] = _dump_legacy_memory(home)

    digest = _digest(snapshot)
    path = (
        home
        / "memory"
        / "exports"
        / f"hermes-memory-snapshot-{digest[:16]}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, {**snapshot, "snapshot_digest": digest})
    ledger_event = _append_memory_operation_event(
        event_type="snapshot_export_created",
        client="hermes",
        operation="snapshot_export",
        decision="allow",
        metadata={
            "scope": scope,
            "limit": limit,
            "snapshot_digest": digest,
            "snapshot_path": str(path),
        },
    )
    return {
        "success": True,
        "snapshot_path": str(path),
        "snapshot_digest": digest,
        "scope": scope,
        "operation_ledger": ledger_event,
        "read_only_memory": True,
        "would_mutate_memory": False,
    }


def _search_graph(home: Path, query: str, *, limit: int) -> list[dict[str, Any]]:
    path = _graph_path(home)
    if not path.exists():
        return []
    like = f"%{query.lower()}%"
    try:
        with sqlite3.connect(path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT id, kind, title, summary, source_id, confidence, metadata_json
                FROM graph_nodes
                WHERE lower(title || ' ' || summary || ' ' || kind) LIKE ?
                LIMIT ?
                """,
                (like, limit),
            ).fetchall()
    except sqlite3.Error:
        return []
    return [
        {
            "source": "memory_graph",
            "id": row["id"],
            "kind": row["kind"],
            "title": row["title"],
            "summary": _snippet(row["summary"], query),
            "score": _score_text(
                " ".join((row["title"], row["summary"], row["kind"])),
                query,
                base=float(row["confidence"] or 0.5),
            ),
            "metadata": _loads_json(row["metadata_json"], {}),
        }
        for row in rows
    ]


def _search_prompt_cases(home: Path, query: str, *, limit: int) -> list[dict[str, Any]]:
    path = _prompt_index_path(home)
    if not path.exists():
        return []
    like = f"%{query.lower()}%"
    try:
        with sqlite3.connect(path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT id, category, section, title, prompt, source_url,
                       author, tags_json, source_repo
                FROM cases
                WHERE lower(
                    COALESCE(title, '') || ' ' || COALESCE(prompt, '') || ' ' ||
                    COALESCE(category, '') || ' ' || COALESCE(section, '')
                ) LIKE ?
                LIMIT ?
                """,
                (like, limit),
            ).fetchall()
    except sqlite3.Error:
        return []
    return [
        {
            "source": "gpt_image_prompt_cases",
            "id": row["id"],
            "kind": row["category"],
            "title": row["title"],
            "summary": _snippet(row["prompt"], query),
            "score": _score_text(
                " ".join((row["title"], row["prompt"], row["category"] or "")),
                query,
                base=0.7,
            ),
            "metadata": {
                "section": row["section"],
                "source_url": row["source_url"],
                "author": row["author"],
                "tags": _loads_json(row["tags_json"], []),
                "source_repo": row["source_repo"],
            },
        }
        for row in rows
    ]


def _search_knowledge(home: Path, query: str, *, limit: int) -> list[dict[str, Any]]:
    root = home / "knowledge"
    if not root.exists():
        return []
    results: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if len(results) >= limit:
            break
        if not path.is_file() or path.suffix.lower() not in {".md", ".jsonl", ".txt"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if query.lower() not in text.lower() and query.lower() not in path.name.lower():
            continue
        rel = str(path.relative_to(home))
        results.append(
            {
                "source": "knowledge",
                "id": rel,
                "kind": path.suffix.lower().lstrip("."),
                "title": path.stem.replace("-", " ").title(),
                "summary": _snippet(text, query),
                "score": _score_text(f"{rel}\n{text}", query, base=0.6),
                "metadata": {"path": str(path)},
            }
        )
    return results


def _search_legacy_memory(home: Path, query: str, *, limit: int) -> list[dict[str, Any]]:
    store = MemoryStore()
    mem_dir = home / "memories"
    memory_entries = MemoryStore._read_file(mem_dir / "MEMORY.md")
    user_entries = MemoryStore._read_file(mem_dir / "USER.md")
    rows = []
    for target, entries in (("memory", memory_entries), ("user", user_entries)):
        for index, entry in enumerate(entries, start=1):
            if query.lower() not in entry.lower():
                continue
            rows.append(
                {
                    "source": "legacy_memory",
                    "id": f"{target}:{index}",
                    "kind": target,
                    "title": f"{target} memory entry {index}",
                    "summary": _snippet(entry, query),
                    "score": _score_text(entry, query, base=0.8),
                    "metadata": {"target": target},
                }
            )
    del store
    rows.sort(key=lambda row: row["score"], reverse=True)
    return rows[:limit]


def _graph_provenance(conn: sqlite3.Connection, node_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT source_type, source_path, source_trust, observed_at,
               valid_from, valid_to, status, metadata_json
        FROM graph_provenance
        WHERE node_id = ?
        ORDER BY source_trust DESC
        """,
        (node_id,),
    ).fetchall()
    result = []
    for row in rows:
        item = _row_to_dict(row)
        item["metadata"] = _loads_json(item.pop("metadata_json", "{}"), {})
        result.append(item)
    return result


def _graph_edges(conn: sqlite3.Connection, node_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT source_id, target_id, relation, weight, metadata_json, created_at
        FROM graph_edges
        WHERE source_id = ? OR target_id = ?
        ORDER BY weight DESC
        LIMIT 50
        """,
        (node_id, node_id),
    ).fetchall()
    result = []
    for row in rows:
        item = _row_to_dict(row)
        item["metadata"] = _loads_json(item.pop("metadata_json", "{}"), {})
        result.append(item)
    return result


def _dump_graph(home: Path, *, limit: int) -> dict[str, Any]:
    path = _graph_path(home)
    if not path.exists():
        return {"exists": False, "nodes": [], "edges": [], "provenance": []}
    try:
        with sqlite3.connect(path) as conn:
            conn.row_factory = sqlite3.Row
            nodes = [
                _row_to_dict(row)
                for row in conn.execute(
                    "SELECT * FROM graph_nodes ORDER BY updated_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            ]
            edges = [
                _row_to_dict(row)
                for row in conn.execute(
                    "SELECT * FROM graph_edges ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            ]
            provenance = [
                _row_to_dict(row)
                for row in conn.execute(
                    "SELECT * FROM graph_provenance ORDER BY observed_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            ]
    except sqlite3.Error as exc:
        return {"exists": True, "error": str(exc), "nodes": [], "edges": []}
    return {
        "exists": True,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "provenance_count": len(provenance),
        "nodes": nodes,
        "edges": edges,
        "provenance": provenance,
    }


def _dump_prompt_cases(home: Path, *, limit: int) -> dict[str, Any]:
    path = _prompt_index_path(home)
    if not path.exists():
        return {"exists": False, "cases": []}
    try:
        with sqlite3.connect(path) as conn:
            conn.row_factory = sqlite3.Row
            cases = [
                _row_to_dict(row)
                for row in conn.execute(
                    "SELECT * FROM cases ORDER BY case_number ASC LIMIT ?",
                    (limit,),
                ).fetchall()
            ]
    except sqlite3.Error as exc:
        return {"exists": True, "error": str(exc), "cases": []}
    return {"exists": True, "case_count": len(cases), "cases": cases}


def _dump_knowledge_manifest(home: Path, *, limit: int) -> dict[str, Any]:
    root = home / "knowledge"
    if not root.exists():
        return {"exists": False, "files": []}
    files = []
    for path in sorted(root.rglob("*")):
        if len(files) >= limit:
            break
        if path.is_file():
            files.append(
                {
                    "path": str(path.relative_to(home)),
                    "size": path.stat().st_size,
                    "mtime": datetime.fromtimestamp(
                        path.stat().st_mtime,
                        tz=timezone.utc,
                    ).isoformat(),
                }
            )
    return {"exists": True, "file_count": len(files), "files": files}


def _dump_legacy_memory(home: Path) -> dict[str, Any]:
    mem_dir = home / "memories"
    return {
        "memory": MemoryStore._read_file(mem_dir / "MEMORY.md"),
        "user": MemoryStore._read_file(mem_dir / "USER.md"),
    }


def _graph_path(home: Path) -> Path:
    return home / "memory" / "graph" / "memory_graph.sqlite"


def _prompt_index_path(home: Path) -> Path:
    return home / "knowledge" / "gpt-image-prompts" / "index.sqlite"


def _proposal_path(home: Path) -> Path:
    return home / "memory" / "proposals" / "memory_write_proposals.jsonl"


def _operation_ledger_path(home: Path) -> Path:
    return home / "memory" / "audit" / "memory_operation_ledger.jsonl"


def _policy_proposal_path(home: Path) -> Path:
    return home / "memory" / "policy" / "memory_policy_proposals.jsonl"


def _codex_memory_client_status() -> dict[str, Any]:
    config_path = Path.home() / ".codex" / "config.toml"
    config_text = _safe_read_text(config_path)
    has_hermes_server = "hermes-memory" in config_text
    invokes_hermes_mcp = "hermes mcp serve" in config_text or (
        "hermes" in config_text and "mcp" in config_text and "serve" in config_text
    )
    agents_path = Path.home() / ".codex" / "AGENTS.md"
    agents_text = _safe_read_text(agents_path)
    return {
        "role": "memory_client",
        "access_path": "codex_mcp",
        "config_path": str(config_path),
        "config_exists": config_path.exists(),
        "mcp_server_configured": has_hermes_server,
        "invokes_hermes_mcp": invokes_hermes_mcp,
        "guidance_path": str(agents_path),
        "guidance_exists": agents_path.exists(),
        "guidance_mentions_hermes_memory": "Hermes Memory" in agents_text
        or "memory_fabric" in agents_text
        or "Memory Fabric" in agents_text,
        "write_policy": "proposal_only",
        "ready": config_path.exists() and has_hermes_server and invokes_hermes_mcp,
    }


def _openclaw_memory_client_status() -> dict[str, Any]:
    config_path = Path.home() / ".openclaw" / "openclaw.json"
    extension_path = Path.home() / ".openclaw" / "extensions" / "hermes-memory" / "index.ts"
    manifest_path = (
        Path.home()
        / ".openclaw"
        / "extensions"
        / "hermes-memory"
        / "openclaw.plugin.json"
    )
    config = _loads_json(_safe_read_text(config_path), {})
    entries = config.get("plugins", {}).get("entries", {}) if isinstance(config, dict) else {}
    entry = entries.get("hermes-memory", {}) if isinstance(entries, dict) else {}
    plugin_config = entry.get("config", {}) if isinstance(entry, dict) else {}
    hooks = entry.get("hooks", {}) if isinstance(entry, dict) else {}
    profiles = (
        plugin_config.get("autoPrecheckAgentProfiles", {})
        if isinstance(plugin_config, dict)
        else {}
    )
    allowed_channels = (
        plugin_config.get("autoPrecheckAllowedChannelIds", [])
        if isinstance(plugin_config, dict)
        else []
    )
    return {
        "role": "memory_client",
        "access_path": "openclaw_plugin",
        "config_path": str(config_path),
        "config_exists": config_path.exists(),
        "extension_path": str(extension_path),
        "extension_exists": extension_path.exists(),
        "manifest_path": str(manifest_path),
        "manifest_exists": manifest_path.exists(),
        "plugin_enabled": bool(entry.get("enabled")) if isinstance(entry, dict) else False,
        "conversation_access_allowed": bool(hooks.get("allowConversationAccess"))
        if isinstance(hooks, dict)
        else False,
        "auto_precheck_enabled": bool(plugin_config.get("autoPrecheckEnabled"))
        if isinstance(plugin_config, dict)
        else False,
        "auto_precheck_agent_profiles": sorted(profiles.keys())
        if isinstance(profiles, dict)
        else [],
        "external_auto_precheck_allowed_channels": allowed_channels
        if isinstance(allowed_channels, list)
        else [],
        "external_auto_precheck_default": "blocked",
        "write_policy": "proposal_only",
        "ready": (
            config_path.exists()
            and extension_path.exists()
            and manifest_path.exists()
            and bool(entry.get("enabled"))
            and bool(hooks.get("allowConversationAccess"))
        )
        if isinstance(entry, dict) and isinstance(hooks, dict)
        else False,
    }


def _federation_warnings(clients: Mapping[str, Any]) -> list[str]:
    warnings: list[str] = []
    codex = clients.get("codex", {})
    openclaw = clients.get("openclaw", {})
    hermes = clients.get("hermes", {})
    if not hermes.get("mcp_server", {}).get("exists"):
        warnings.append("Hermes MCP server file was not found.")
    if not codex.get("ready"):
        warnings.append("Codex is not fully configured for hermes-memory MCP access.")
    if not openclaw.get("ready"):
        warnings.append("OpenClaw hermes-memory plugin is not fully configured.")
    if openclaw.get("external_auto_precheck_allowed_channels"):
        warnings.append(
            "OpenClaw external channel auto-precheck allowlist is non-empty; review exposure policy."
        )
    return warnings


def _build_federation_audit_checks(
    status: Mapping[str, Any],
    openclaw_logs: Mapping[str, Any],
    proposal_summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    clients = status.get("clients", {}) if isinstance(status.get("clients"), dict) else {}
    codex = clients.get("codex", {}) if isinstance(clients, dict) else {}
    openclaw = clients.get("openclaw", {}) if isinstance(clients, dict) else {}
    policy = status.get("policy", {}) if isinstance(status.get("policy"), dict) else {}
    checks = [
        _audit_check(
            "federation.ready",
            "federation",
            status.get("ready") is True,
            "critical",
            "Hermes, Codex, and OpenClaw federation is ready.",
            "Federation readiness has warnings.",
            status.get("warnings", []),
            "Run memory_federation_status and fix the listed client wiring.",
        ),
        _audit_check(
            "codex.mcp_ready",
            "codex",
            codex.get("ready") is True,
            "critical",
            "Codex hermes-memory MCP is configured.",
            "Codex hermes-memory MCP is not fully configured.",
            {
                "config_exists": codex.get("config_exists"),
                "mcp_server_configured": codex.get("mcp_server_configured"),
                "invokes_hermes_mcp": codex.get("invokes_hermes_mcp"),
            },
            "Run `codex mcp list` and ensure hermes-memory invokes `hermes mcp serve`.",
        ),
        _audit_check(
            "openclaw.plugin_ready",
            "openclaw",
            openclaw.get("ready") is True,
            "critical",
            "OpenClaw hermes-memory plugin is configured.",
            "OpenClaw hermes-memory plugin is not fully configured.",
            {
                "plugin_enabled": openclaw.get("plugin_enabled"),
                "conversation_access_allowed": openclaw.get("conversation_access_allowed"),
                "extension_exists": openclaw.get("extension_exists"),
            },
            "Enable the hermes-memory plugin and hooks.allowConversationAccess.",
        ),
        _audit_check(
            "openclaw.loaded_recently",
            "openclaw",
            int(openclaw_logs.get("loaded_events", 0)) > 0,
            "warning",
            "Recent OpenClaw logs show hermes-memory loaded.",
            "Recent OpenClaw logs do not show hermes-memory loaded.",
            {"log_source": openclaw_logs.get("source")},
            "Restart OpenClaw gateway and inspect the gateway log.",
        ),
        _audit_check(
            "openclaw.auto_precheck_evidence",
            "openclaw",
            int(openclaw_logs.get("auto_precheck_injections", 0)) > 0,
            "warning",
            "Recent OpenClaw logs show memory auto-precheck injections.",
            "Recent OpenClaw logs do not show memory auto-precheck injections.",
            {
                "auto_precheck_injections": openclaw_logs.get("auto_precheck_injections"),
                "auto_precheck_misses": openclaw_logs.get("auto_precheck_misses"),
            },
            "Run a local OpenClaw memory recall test or check trigger/profile matching.",
        ),
        _audit_check(
            "openclaw.external_channels_gated",
            "openclaw",
            not openclaw.get("external_auto_precheck_allowed_channels"),
            "critical",
            "OpenClaw external channel auto-recall allowlist is empty.",
            "OpenClaw external channel auto-recall allowlist is non-empty.",
            {
                "allowed_channels": openclaw.get("external_auto_precheck_allowed_channels"),
                "default": openclaw.get("external_auto_precheck_default"),
            },
            "Review each allowed external channel before exposing long-term memory.",
        ),
        _audit_check(
            "writes.proposal_only",
            "policy",
            policy.get("writes_are_proposal_only") is True,
            "critical",
            "Federated writes remain proposal-only.",
            "Federated writes are not clearly proposal-only.",
            policy,
            "Route client writes through memory_write_proposal.",
        ),
        _audit_check(
            "proposals.parseable",
            "proposal_ledger",
            not proposal_summary.get("parse_errors"),
            "warning",
            "Memory proposal ledger is parseable.",
            "Memory proposal ledger has parse errors.",
            proposal_summary,
            "Inspect memory/proposals/memory_write_proposals.jsonl.",
        ),
    ]
    if int(openclaw_logs.get("rate_limit_events", 0)) > 0:
        checks.append(
            _audit_check(
                "runtime.rate_limit_observed",
                "runtime",
                False,
                "info",
                "No recent OpenClaw/Codex rate-limit events observed.",
                "Recent model rate-limit events were observed; memory wiring can still be healthy.",
                {"rate_limit_events": openclaw_logs.get("rate_limit_events")},
                "Wait for quota reset or route tests through a lower-cost model.",
            )
        )
    return checks


def _build_federation_gate_rules(
    *,
    audit: Mapping[str, Any],
    client: str,
    operation: str,
    requested_operation: str,
    target_scope: str,
    channel_id: str,
    allowed_channels: Any,
    critical_failures: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    health_score = int(audit.get("health_score") or 0)
    channel_external = _is_external_channel(channel_id)
    channel_allowed = _channel_allowed(channel_id, allowed_channels)
    rules = [
        _gate_rule(
            "operation.known",
            "operation",
            "pass" if operation != "unknown" else "block",
            "Memory operation is recognized.",
            "Memory operation is not recognized.",
            {
                "requested_operation": requested_operation,
                "allowed_operations": sorted(VALID_GATE_OPERATIONS),
            },
            "Use a supported memory operation.",
        ),
        _gate_rule(
            "client.trusted",
            "client",
            "pass" if client in INTERNAL_MEMORY_CLIENTS else "review",
            "Client is part of the Hermes/Codex/OpenClaw federation.",
            "Client is not a known internal memory client.",
            {"client": client, "trusted_clients": sorted(INTERNAL_MEMORY_CLIENTS)},
            "Route unknown clients through Hermes review before memory access.",
        ),
        _gate_rule(
            "audit.ready",
            "audit",
            "pass"
            if audit.get("ready") is True or operation in {"audit", "status"}
            else "block"
            if critical_failures
            else "review",
            "Federation audit is ready for governed memory operations.",
            "Federation audit is not ready for this memory operation.",
            {
                "health_score": audit.get("health_score"),
                "risk_level": audit.get("risk_level"),
                "critical_failures": [check.get("id") for check in critical_failures],
            },
            "Run memory_federation_audit and fix failed critical checks.",
        ),
        _gate_rule(
            "audit.health_floor",
            "audit",
            "pass"
            if health_score >= 80 or operation in {"audit", "status"}
            else "review",
            "Federation audit health is at or above the governed-operation floor.",
            "Federation audit health is below the governed-operation floor.",
            {"health_score": health_score, "minimum": 80},
            "Resolve audit warnings before broad memory access.",
        ),
        _gate_rule(
            "policy.durable_write",
            "policy",
            "block" if operation in DURABLE_WRITE_OPERATIONS else "pass",
            "Operation does not directly mutate durable memory.",
            "Operation would directly mutate durable memory.",
            {
                "operation": operation,
                "durable_write_operations": sorted(DURABLE_WRITE_OPERATIONS),
            },
            "Use memory_write_proposal and promote only after Hermes review.",
        ),
    ]
    if operation == "write_proposal":
        rules.append(
            _gate_rule(
                "policy.write_proposal_scope",
                "policy",
                "pass" if target_scope in VALID_PROPOSAL_SCOPES else "review",
                "Write proposal target scope is recognized.",
                "Write proposal target scope needs review.",
                {"target_scope": target_scope, "valid_scopes": sorted(VALID_PROPOSAL_SCOPES)},
                "Choose a valid governed proposal scope.",
            )
        )
    if operation == "snapshot_export" and client not in INTERNAL_MEMORY_CLIENTS:
        rules.append(
            _gate_rule(
                "policy.snapshot_external_client",
                "policy",
                "review",
                "Snapshot export is internal or cache-only.",
                "Snapshot export for an unknown client needs review.",
                {"client": client},
                "Confirm the snapshot will remain cache or backup only.",
            )
        )
    if operation in {"auto_precheck", "external_auto_recall"}:
        if channel_external and not channel_allowed:
            channel_status = "block"
            fail_message = "External channel is not allowlisted for automatic recall."
        elif channel_external:
            channel_status = "review"
            fail_message = "External channel is allowlisted but should be reviewed."
        else:
            channel_status = "pass"
            fail_message = "Channel gate needs review."
        rules.append(
            _gate_rule(
                "policy.external_channel_recall",
                "policy",
                channel_status,
                "Channel is allowed for this automatic memory recall.",
                fail_message,
                {
                    "channel_id": channel_id,
                    "channel_external": channel_external,
                    "allowed_channels": allowed_channels if isinstance(allowed_channels, list) else [],
                },
                "Add the exact channel to the allowlist only after reviewing exposure risk.",
            )
        )
    return rules


def _gate_rule(
    rule_id: str,
    subject: str,
    status: str,
    pass_message: str,
    fail_message: str,
    evidence: Any,
    recommendation: str,
) -> dict[str, Any]:
    return {
        "id": rule_id,
        "subject": subject,
        "status": status,
        "message": pass_message if status == "pass" else fail_message,
        "evidence": evidence,
        "recommendation": "" if status == "pass" else recommendation,
    }


def _gate_decision(rules: Iterable[Mapping[str, Any]]) -> str:
    statuses = {rule.get("status") for rule in rules}
    if "block" in statuses:
        return "block"
    if "review" in statuses:
        return "review"
    return "allow"


def _gate_required_action(decision: str, rules: Iterable[Mapping[str, Any]]) -> str:
    if decision == "allow":
        return "Proceed under Hermes Memory Fabric policy."
    recommendations = [
        _clean_text(rule.get("recommendation"))
        for rule in rules
        if rule.get("status") != "pass" and _clean_text(rule.get("recommendation"))
    ]
    if recommendations:
        return recommendations[0]
    if decision == "block":
        return "Resolve blocking policy rules before continuing."
    return "Review non-passing policy rules before continuing."


def _is_external_channel(channel_id: str) -> bool:
    normalized = _clean_text(channel_id).lower()
    if not normalized:
        return False
    root = normalized.split(":", 1)[0]
    return root in EXTERNAL_CHANNEL_ROOTS


def _channel_allowed(channel_id: str, allowed_channels: Any) -> bool:
    if not _is_external_channel(channel_id):
        return True
    normalized = _clean_text(channel_id).lower()
    root = normalized.split(":", 1)[0]
    allowed = {
        _clean_text(entry).lower()
        for entry in allowed_channels
        if _clean_text(entry)
    } if isinstance(allowed_channels, list) else set()
    return normalized in allowed or root in allowed


def _audit_check(
    check_id: str,
    subject: str,
    passed: bool,
    severity: str,
    pass_message: str,
    fail_message: str,
    evidence: Any,
    recommendation: str,
) -> dict[str, Any]:
    return {
        "id": check_id,
        "subject": subject,
        "status": "pass" if passed else "warn" if severity == "info" else "fail",
        "severity": "info" if passed else severity,
        "message": pass_message if passed else fail_message,
        "evidence": evidence,
        "recommendation": "" if passed else recommendation,
    }


def _audit_health_score(checks: Iterable[Mapping[str, Any]]) -> int:
    penalty = 0
    for check in checks:
        if check.get("status") == "pass":
            continue
        severity = check.get("severity")
        if severity == "critical":
            penalty += 25
        elif severity == "warning":
            penalty += 10
        else:
            penalty += 3
    return max(0, 100 - penalty)


def _audit_risk_level(health_score: int) -> str:
    if health_score >= 90:
        return "low"
    if health_score >= 70:
        return "medium"
    return "high"


def _read_recent_openclaw_log_lines(*, limit: int) -> dict[str, Any]:
    path = _openclaw_log_path()
    lines: list[str] = []
    if path and path.exists():
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()[-limit:]
        except OSError:
            lines = []
    return {
        "source": str(path) if path else "",
        "lines": lines,
        "loaded_events": sum(
            1 for line in lines if "http server listening" in line and "hermes-memory" in line
        ),
        "auto_precheck_injections": sum(
            1 for line in lines if "hermes-memory auto-precheck injected" in line
        ),
        "auto_precheck_misses": sum(
            1 for line in lines if "hermes-memory auto-precheck found no recall" in line
        ),
        "rate_limit_events": sum(
            1
            for line in lines
            if "rate limit" in line.lower() or "usage_limit_reached" in line.lower()
        ),
    }


def _openclaw_log_path() -> Path | None:
    explicit = _clean_text(os.environ.get("HERMES_OPENCLAW_LOG_PATH"))
    if explicit:
        return Path(explicit).expanduser()
    log_dir = Path("/tmp/openclaw")
    candidates = sorted(
        log_dir.glob("openclaw-*.log"),
        key=lambda path: path.stat().st_mtime if path.exists() else 0,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _memory_proposal_summary() -> dict[str, Any]:
    path = _proposal_path(get_hermes_home())
    rows = _read_jsonl(path)
    parse_errors = [row for row in rows if row.get("_parse_error")]
    proposals = [row for row in rows if not row.get("_parse_error")]
    by_source: dict[str, int] = {}
    by_scope: dict[str, int] = {}
    for proposal in proposals:
        source = _clean_text(proposal.get("source_agent")) or "unknown"
        scope = _clean_text(proposal.get("target_scope")) or "unknown"
        by_source[source] = by_source.get(source, 0) + 1
        by_scope[scope] = by_scope.get(scope, 0) + 1
    return {
        "path": str(path),
        "exists": path.exists(),
        "proposal_count": len(proposals),
        "parse_error_count": len(parse_errors),
        "parse_errors": parse_errors[:5],
        "by_source_agent": by_source,
        "by_target_scope": by_scope,
    }


def _append_memory_operation_event(
    *,
    event_type: str,
    client: str,
    operation: str,
    decision: str,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    event_type = _clean_text(event_type).lower()
    if event_type not in VALID_LEDGER_EVENT_TYPES:
        event_type = "gate_decision"
    now = datetime.now(timezone.utc).isoformat()
    event = {
        "event_type": event_type,
        "client": _clean_text(client).lower() or "unknown",
        "operation": _clean_text(operation).lower() or "unknown",
        "decision": _clean_text(decision).lower() or "unknown",
        "metadata": _sanitize_ledger_metadata(metadata or {}),
        "created_at": now,
        "memory_mutation": False,
        "durable_memory_write": False,
    }
    event["id"] = f"memory-operation-{_digest(event)[:16]}"
    path = _operation_ledger_path(get_hermes_home())
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
    except OSError as exc:
        return {
            "success": False,
            "error": str(exc),
            "path": str(path),
            "would_mutate_memory": False,
        }
    return {
        "success": True,
        "event_id": event["id"],
        "path": str(path),
        "event_type": event_type,
        "read_only_memory": True,
        "would_mutate_memory": False,
    }


def _sanitize_ledger_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in metadata.items():
        clean_key = _clean_text(key)
        if not clean_key:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            sanitized[clean_key] = value
        elif isinstance(value, list):
            sanitized[clean_key] = [
                item
                for item in value[:25]
                if isinstance(item, (str, int, float, bool)) or item is None
            ]
        elif isinstance(value, dict):
            sanitized[clean_key] = {
                _clean_text(child_key): child_value
                for child_key, child_value in list(value.items())[:50]
                if _clean_text(child_key)
                and (
                    isinstance(child_value, (str, int, float, bool))
                    or child_value is None
                )
            }
        else:
            sanitized[clean_key] = _clean_text(value)[:500]
    return sanitized


def _operation_ledger_summary(events: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, int] = {}
    by_client: dict[str, int] = {}
    by_operation: dict[str, int] = {}
    by_decision: dict[str, int] = {}
    latest_event_at = ""
    for event in events:
        event_type = _clean_text(event.get("event_type")) or "unknown"
        client = _clean_text(event.get("client")) or "unknown"
        operation = _clean_text(event.get("operation")) or "unknown"
        decision = _clean_text(event.get("decision")) or "unknown"
        created_at = _clean_text(event.get("created_at"))
        by_type[event_type] = by_type.get(event_type, 0) + 1
        by_client[client] = by_client.get(client, 0) + 1
        by_operation[operation] = by_operation.get(operation, 0) + 1
        by_decision[decision] = by_decision.get(decision, 0) + 1
        if created_at > latest_event_at:
            latest_event_at = created_at
    return {
        "by_event_type": by_type,
        "by_client": by_client,
        "by_operation": by_operation,
        "by_decision": by_decision,
        "latest_event_at": latest_event_at,
    }


def _ledger_intelligence_metrics(events: list[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(events)
    by_type: dict[str, int] = {}
    by_client: dict[str, int] = {}
    by_operation: dict[str, int] = {}
    by_decision: dict[str, int] = {}
    blocked_by_operation: dict[str, int] = {}
    blocked_by_client: dict[str, int] = {}
    blocked_channels: dict[str, int] = {}
    unknown_clients: dict[str, int] = {}
    direct_write_attempts = 0
    external_auto_recall_blocks = 0
    search_events = 0
    graph_read_events = 0
    write_proposal_events = 0
    snapshot_export_events = 0
    allow_gate_events = 0
    allow_gate_without_followup_candidates = 0

    for event in events:
        event_type = _clean_text(event.get("event_type")) or "unknown"
        client = _clean_text(event.get("client")).lower() or "unknown"
        operation = _clean_text(event.get("operation")).lower() or "unknown"
        decision = _clean_text(event.get("decision")).lower() or "unknown"
        metadata = event.get("metadata") if isinstance(event.get("metadata"), dict) else {}
        channel_id = _clean_text(metadata.get("channel_id"))

        by_type[event_type] = by_type.get(event_type, 0) + 1
        by_client[client] = by_client.get(client, 0) + 1
        by_operation[operation] = by_operation.get(operation, 0) + 1
        by_decision[decision] = by_decision.get(decision, 0) + 1

        if client not in INTERNAL_MEMORY_CLIENTS:
            unknown_clients[client] = unknown_clients.get(client, 0) + 1
        if operation in DURABLE_WRITE_OPERATIONS:
            direct_write_attempts += 1
        if decision == "block":
            blocked_by_operation[operation] = blocked_by_operation.get(operation, 0) + 1
            blocked_by_client[client] = blocked_by_client.get(client, 0) + 1
            if channel_id:
                blocked_channels[channel_id] = blocked_channels.get(channel_id, 0) + 1
            if operation in {"auto_precheck", "external_auto_recall"} and _is_external_channel(channel_id):
                external_auto_recall_blocks += 1
        if event_type == "search":
            search_events += 1
        elif event_type == "graph_read":
            graph_read_events += 1
        elif event_type == "write_proposal_created":
            write_proposal_events += 1
        elif event_type == "snapshot_export_created":
            snapshot_export_events += 1
        if event_type == "gate_decision" and decision == "allow":
            allow_gate_events += 1
            if operation in {"search", "auto_precheck", "graph_read"}:
                allow_gate_without_followup_candidates += 1

    block_count = by_decision.get("block", 0)
    review_count = by_decision.get("review", 0)
    allow_count = by_decision.get("allow", 0)
    recall_event_count = search_events + graph_read_events
    return {
        "total": total,
        "by_event_type": by_type,
        "by_client": by_client,
        "by_operation": by_operation,
        "by_decision": by_decision,
        "block_count": block_count,
        "review_count": review_count,
        "allow_count": allow_count,
        "block_rate": round(block_count / total, 4) if total else 0,
        "review_rate": round(review_count / total, 4) if total else 0,
        "blocked_by_operation": blocked_by_operation,
        "blocked_by_client": blocked_by_client,
        "blocked_channels": blocked_channels,
        "unknown_clients": unknown_clients,
        "direct_write_attempts": direct_write_attempts,
        "external_auto_recall_blocks": external_auto_recall_blocks,
        "search_events": search_events,
        "graph_read_events": graph_read_events,
        "recall_event_count": recall_event_count,
        "write_proposal_events": write_proposal_events,
        "snapshot_export_events": snapshot_export_events,
        "allow_gate_events": allow_gate_events,
        "allow_gate_without_followup_candidates": allow_gate_without_followup_candidates,
        "allowed_recall_followup_gap": max(
            0,
            allow_gate_without_followup_candidates - recall_event_count,
        ),
    }


def _ledger_intelligence_findings(
    metrics: Mapping[str, Any],
    parse_errors: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    total = int(metrics.get("total") or 0)
    block_count = int(metrics.get("block_count") or 0)
    review_count = int(metrics.get("review_count") or 0)
    block_rate = float(metrics.get("block_rate") or 0)
    if total == 0:
        findings.append(
            _ledger_finding(
                "ledger.no_events",
                "info",
                "ledger",
                "No memory operation events have been recorded yet.",
                {"total": total},
                "Use memory_federation_gate or memory_fabric_search to start producing audit evidence.",
            )
        )
        return findings
    if parse_errors:
        findings.append(
            _ledger_finding(
                "ledger.parse_errors",
                "warning",
                "ledger",
                "Operation ledger has parse errors.",
                {"parse_error_count": len(parse_errors), "parse_errors": parse_errors[:3]},
                "Inspect memory/audit/memory_operation_ledger.jsonl for partial or corrupt lines.",
            )
        )
    if block_count >= 3 and block_rate >= 0.3:
        findings.append(
            _ledger_finding(
                "gate.high_block_rate",
                "warning" if block_rate < 0.5 else "critical",
                "gate",
                "Memory gate is blocking a high share of recent operations.",
                {
                    "block_count": block_count,
                    "block_rate": block_rate,
                    "blocked_by_operation": metrics.get("blocked_by_operation", {}),
                    "blocked_by_client": metrics.get("blocked_by_client", {}),
                },
                "Review blocked operation patterns before widening memory access.",
            )
        )
    if review_count >= 3:
        findings.append(
            _ledger_finding(
                "gate.review_backlog",
                "warning",
                "gate",
                "Several memory operations require review.",
                {"review_count": review_count},
                "Resolve review decisions or tighten client/scoped operation policy.",
            )
        )
    if int(metrics.get("direct_write_attempts") or 0) > 0:
        findings.append(
            _ledger_finding(
                "policy.direct_write_attempts",
                "critical",
                "policy",
                "Durable memory write operations were attempted.",
                {"direct_write_attempts": metrics.get("direct_write_attempts")},
                "Keep durable writes behind memory_write_proposal and Hermes review.",
            )
        )
    if int(metrics.get("external_auto_recall_blocks") or 0) > 0:
        findings.append(
            _ledger_finding(
                "policy.external_auto_recall_blocked",
                "warning",
                "external_channel",
                "External-channel automatic recall was blocked.",
                {
                    "external_auto_recall_blocks": metrics.get("external_auto_recall_blocks"),
                    "blocked_channels": metrics.get("blocked_channels", {}),
                },
                "Only allowlist external channels after confirming privacy and memory exposure policy.",
            )
        )
    unknown_clients = metrics.get("unknown_clients", {})
    if isinstance(unknown_clients, dict) and unknown_clients:
        findings.append(
            _ledger_finding(
                "client.unknown_clients",
                "warning",
                "client",
                "Unknown clients are touching the memory fabric.",
                {"unknown_clients": unknown_clients},
                "Route unknown clients through Hermes review before granting memory operations.",
            )
        )
    if int(metrics.get("allowed_recall_followup_gap") or 0) >= 3:
        findings.append(
            _ledger_finding(
                "recall.allowed_without_followup",
                "warning",
                "recall",
                "Allowed recall gates are not consistently followed by recall reads.",
                {
                    "allowed_recall_followup_gap": metrics.get("allowed_recall_followup_gap"),
                    "allow_gate_events": metrics.get("allow_gate_events"),
                    "recall_event_count": metrics.get("recall_event_count"),
                },
                "Check OpenClaw auto-precheck query matching and MCP client recall behavior.",
            )
        )
    if int(metrics.get("recall_event_count") or 0) == 0 and int(metrics.get("allow_gate_events") or 0) > 0:
        findings.append(
            _ledger_finding(
                "recall.blind_spot",
                "warning",
                "recall",
                "The gate is allowing operations but no recall reads are recorded.",
                {
                    "allow_gate_events": metrics.get("allow_gate_events"),
                    "recall_event_count": metrics.get("recall_event_count"),
                },
                "Verify search and graph_read calls are using Hermes Memory Fabric instead of bypassing it.",
            )
        )
    if int(metrics.get("snapshot_export_events") or 0) >= 3:
        findings.append(
            _ledger_finding(
                "snapshot.frequent_exports",
                "info",
                "snapshot",
                "Memory snapshots are being exported frequently.",
                {"snapshot_export_events": metrics.get("snapshot_export_events")},
                "Confirm snapshots remain cache or backup only and do not become a forked primary memory.",
            )
        )
    if not findings:
        findings.append(
            _ledger_finding(
                "ledger.healthy",
                "info",
                "ledger",
                "No notable memory operation risks were detected in the analyzed ledger window.",
                {"total": total},
                "",
            )
        )
    return findings


def _ledger_finding(
    finding_id: str,
    severity: str,
    subject: str,
    message: str,
    evidence: Any,
    recommendation: str,
) -> dict[str, Any]:
    return {
        "id": finding_id,
        "severity": severity,
        "subject": subject,
        "message": message,
        "evidence": evidence,
        "recommendation": recommendation,
    }


def _ledger_intelligence_health_score(findings: Iterable[Mapping[str, Any]]) -> int:
    penalty = 0
    for finding in findings:
        severity = finding.get("severity")
        finding_id = finding.get("id")
        if finding_id == "ledger.healthy":
            continue
        if severity == "critical":
            penalty += 25
        elif severity == "warning":
            penalty += 10
    return max(0, 100 - penalty)


def _ledger_intelligence_actions(findings: Iterable[Mapping[str, Any]]) -> list[str]:
    actions = []
    for finding in findings:
        recommendation = _clean_text(finding.get("recommendation"))
        if recommendation and recommendation not in actions:
            actions.append(recommendation)
    return actions[:5]


def _policy_autotune_mode(mode: str) -> str:
    cleaned = _clean_text(mode).lower()
    return cleaned if cleaned in {"conservative", "diagnostic"} else "conservative"


def _memory_policy_autotune_suggestions(
    intelligence: Mapping[str, Any],
    *,
    mode: str,
) -> list[dict[str, Any]]:
    findings = [
        finding
        for finding in intelligence.get("findings", [])
        if isinstance(finding, dict)
    ]
    suggestions: list[dict[str, Any]] = []
    for finding in findings:
        finding_id = _clean_text(finding.get("id"))
        evidence = finding.get("evidence", {})
        if finding_id == "policy.external_auto_recall_blocked":
            blocked_channels = (
                evidence.get("blocked_channels", {})
                if isinstance(evidence, dict)
                else {}
            )
            suggestions.append(
                _policy_suggestion(
                    "external_auto_recall.keep_blocked",
                    "medium",
                    "approval_required",
                    "openclaw.autoPrecheckAllowedChannelIds",
                    "Keep external-channel automatic recall blocked by default.",
                    "Ledger shows external-channel recall was blocked; automatic allowlisting would widen memory exposure.",
                    {
                        "candidate_channels_for_review": sorted(blocked_channels.keys())
                        if isinstance(blocked_channels, dict)
                        else [],
                        "source_finding": finding_id,
                    },
                    "Only add an exact channel id after a human privacy review.",
                )
            )
        elif finding_id == "policy.direct_write_attempts":
            suggestions.append(
                _policy_suggestion(
                    "durable_writes.enforce_proposal_only",
                    "high",
                    "policy_guard",
                    "memory_write_policy",
                    "Keep durable memory writes behind memory_write_proposal.",
                    "Ledger shows direct durable write operations were attempted.",
                    {"source_finding": finding_id, "evidence": evidence},
                    "Audit the caller and route future writes through governed proposals.",
                )
            )
        elif finding_id == "client.unknown_clients":
            suggestions.append(
                _policy_suggestion(
                    "clients.require_registration",
                    "medium",
                    "approval_required",
                    "memory_client_registry",
                    "Require unknown memory clients to be explicitly registered.",
                    "Unknown clients should not receive broad memory access without review.",
                    {"source_finding": finding_id, "evidence": evidence},
                    "Add trusted clients intentionally; block or review all others.",
                )
            )
        elif finding_id in {"recall.allowed_without_followup", "recall.blind_spot"}:
            suggestions.append(
                _policy_suggestion(
                    "recall.routing_diagnostics",
                    "medium",
                    "diagnostic",
                    "openclaw.autoPrecheckRouting",
                    "Inspect recall routing because allowed gates are not consistently followed by recall reads.",
                    "The policy gate is allowing recall, but the ledger does not show enough matching search or graph reads.",
                    {"source_finding": finding_id, "evidence": evidence},
                    "Test OpenClaw before_prompt_build query selection and client MCP recall paths.",
                )
            )
        elif finding_id == "gate.high_block_rate":
            suggestions.append(
                _policy_suggestion(
                    "gate.review_block_patterns",
                    "medium",
                    "diagnostic",
                    "memory_federation_gate",
                    "Review high block-rate patterns before loosening policy.",
                    "A high block rate can mean policy is working, or that clients are misrouted.",
                    {"source_finding": finding_id, "evidence": evidence},
                    "Fix client behavior first; widen policy only after reviewing specific blocked operations.",
                )
            )
        elif finding_id == "gate.review_backlog":
            suggestions.append(
                _policy_suggestion(
                    "gate.drain_review_backlog",
                    "medium",
                    "manual_review",
                    "memory_review_queue",
                    "Resolve memory gate review decisions before changing policy.",
                    "Pending review-like outcomes can hide ambiguous access needs.",
                    {"source_finding": finding_id, "evidence": evidence},
                    "Approve or reject review cases, then rerun memory_policy_autotune.",
                )
            )
        elif finding_id == "snapshot.frequent_exports":
            suggestions.append(
                _policy_suggestion(
                    "snapshots.confirm_cache_only",
                    "low",
                    "policy_guard",
                    "memory_snapshot_export",
                    "Keep snapshots cache-only and monitor export frequency.",
                    "Frequent snapshots can accidentally become a forked primary memory.",
                    {"source_finding": finding_id, "evidence": evidence},
                    "Require a stated purpose for repeated exports.",
                )
            )
        elif finding_id == "ledger.parse_errors":
            suggestions.append(
                _policy_suggestion(
                    "ledger.repair_parse_errors",
                    "medium",
                    "maintenance",
                    "memory_operation_ledger",
                    "Repair or quarantine malformed ledger lines.",
                    "Parse errors weaken audit quality.",
                    {"source_finding": finding_id, "evidence": evidence},
                    "Inspect the ledger file and preserve malformed lines for forensic review.",
                )
            )
        elif finding_id == "ledger.no_events":
            suggestions.append(
                _policy_suggestion(
                    "ledger.seed_audit_evidence",
                    "low",
                    "diagnostic",
                    "memory_operation_ledger",
                    "Run basic gate and recall checks to create audit evidence.",
                    "The policy tuner needs ledger evidence before making useful suggestions.",
                    {"source_finding": finding_id},
                    "Run memory_federation_gate and memory_fabric_search dry-run checks.",
                )
            )
        elif finding_id == "ledger.healthy":
            suggestions.append(
                _policy_suggestion(
                    "policy.no_change",
                    "low",
                    "no_change",
                    "memory_policy",
                    "No memory policy change is suggested.",
                    "Ledger intelligence did not detect notable policy risks.",
                    {"source_finding": finding_id},
                    "",
                    requires_human_review=False,
                )
            )

    if mode == "diagnostic" and not any(s["id"] == "diagnostic.rerun_full_audit" for s in suggestions):
        suggestions.append(
            _policy_suggestion(
                "diagnostic.rerun_full_audit",
                "low",
                "diagnostic",
                "memory_federation_audit",
                "Run memory_federation_audit before applying any policy change.",
                "Policy tuning should stay aligned with live federation health.",
                {
                    "intelligence_health_score": intelligence.get("health_score"),
                    "intelligence_risk_level": intelligence.get("risk_level"),
                },
                "Fix critical federation audit findings before changing recall or write policy.",
            )
        )

    if not suggestions:
        suggestions.append(
            _policy_suggestion(
                "policy.no_change",
                "low",
                "no_change",
                "memory_policy",
                "No memory policy change is suggested.",
                "No ledger findings were available for tuning.",
                {},
                "",
                requires_human_review=False,
            )
        )
    return _dedupe_policy_suggestions(suggestions)


def _policy_suggestion(
    suggestion_id: str,
    priority: str,
    change_type: str,
    target: str,
    recommendation: str,
    rationale: str,
    evidence: Any,
    next_step: str,
    *,
    requires_human_review: bool = True,
) -> dict[str, Any]:
    return {
        "id": suggestion_id,
        "priority": priority,
        "change_type": change_type,
        "target": target,
        "recommendation": recommendation,
        "rationale": rationale,
        "evidence": evidence,
        "next_step": next_step,
        "requires_human_review": requires_human_review,
        "can_auto_apply": False,
        "would_modify_config": False,
        "would_write_memory": False,
    }


def _dedupe_policy_suggestions(suggestions: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    priority_order = {"high": 0, "medium": 1, "low": 2}
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for suggestion in suggestions:
        suggestion_id = _clean_text(suggestion.get("id"))
        if not suggestion_id or suggestion_id in seen:
            continue
        seen.add(suggestion_id)
        deduped.append(dict(suggestion))
    deduped.sort(key=lambda row: priority_order.get(_clean_text(row.get("priority")), 3))
    return deduped


def _policy_autotune_summary(suggestions: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    by_priority: dict[str, int] = {}
    by_change_type: dict[str, int] = {}
    requires_review_count = 0
    for suggestion in suggestions:
        priority = _clean_text(suggestion.get("priority")) or "unknown"
        change_type = _clean_text(suggestion.get("change_type")) or "unknown"
        by_priority[priority] = by_priority.get(priority, 0) + 1
        by_change_type[change_type] = by_change_type.get(change_type, 0) + 1
        if suggestion.get("requires_human_review"):
            requires_review_count += 1
    return {
        "by_priority": by_priority,
        "by_change_type": by_change_type,
        "requires_review_count": requires_review_count,
        "auto_apply_count": 0,
        "config_mutation_count": 0,
    }


def _policy_proposal_from_suggestion(
    suggestion: Mapping[str, Any],
    *,
    source_agent: str,
    autotune: Mapping[str, Any],
) -> dict[str, Any]:
    seed = {
        "suggestion_id": suggestion.get("id"),
        "target": suggestion.get("target"),
        "recommendation": suggestion.get("recommendation"),
        "evidence": suggestion.get("evidence", {}),
        "mode": autotune.get("mode"),
    }
    digest = _digest(seed)
    now = datetime.now(timezone.utc).isoformat()
    return {
        "proposal_id": f"memory-policy-proposal-{digest[:16]}",
        "proposal_digest": digest,
        "status": "proposed",
        "latest_status": "proposed",
        "source_agent": source_agent,
        "suggestion_id": _clean_text(suggestion.get("id")),
        "priority": _clean_text(suggestion.get("priority")) or "medium",
        "change_type": _clean_text(suggestion.get("change_type")) or "approval_required",
        "target": _clean_text(suggestion.get("target")),
        "recommendation": _clean_text(suggestion.get("recommendation")),
        "rationale": _clean_text(suggestion.get("rationale")),
        "next_step": _clean_text(suggestion.get("next_step")),
        "evidence": _sanitize_ledger_metadata(
            suggestion.get("evidence", {})
            if isinstance(suggestion.get("evidence"), dict)
            else {}
        ),
        "autotune": {
            "mode": autotune.get("mode"),
            "decision": autotune.get("decision"),
            "intelligence_summary": autotune.get("intelligence_summary", {}),
        },
        "governance": {
            "requires_human_review": True,
            "can_auto_apply": False,
            "does_not_apply_policy": True,
            "approval_records_intent_only": True,
        },
        "created_at": now,
        "updated_at": now,
    }


def _append_policy_proposal_event(event: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(event)
    payload.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    payload.setdefault("would_modify_config", False)
    payload.setdefault("would_write_memory", False)
    payload["event_id"] = f"memory-policy-event-{_digest(payload)[:16]}"
    path = _policy_proposal_path(get_hermes_home())
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    except OSError as exc:
        return {
            "success": False,
            "error": str(exc),
            "path": str(path),
            "would_modify_config": False,
            "would_mutate_memory": False,
        }
    return {
        "success": True,
        "event_id": payload["event_id"],
        "path": str(path),
        "event_type": payload.get("event_type"),
        "read_only_memory": True,
        "would_modify_config": False,
        "would_mutate_memory": False,
    }


def _policy_proposal_states(
    rows: list[dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    if rows is None:
        rows = _read_jsonl(_policy_proposal_path(get_hermes_home()))
    states: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("_parse_error"):
            continue
        event_type = _clean_text(row.get("event_type"))
        proposal_id = _clean_text(row.get("proposal_id"))
        if not proposal_id:
            continue
        if event_type == "policy_proposal_created":
            states[proposal_id] = {
                "proposal_id": proposal_id,
                "proposal_digest": row.get("proposal_digest", ""),
                "suggestion_id": row.get("suggestion_id", ""),
                "source_agent": row.get("source_agent", ""),
                "priority": row.get("priority", ""),
                "change_type": row.get("change_type", ""),
                "target": row.get("target", ""),
                "recommendation": row.get("recommendation", ""),
                "rationale": row.get("rationale", ""),
                "next_step": row.get("next_step", ""),
                "evidence": row.get("evidence", {}),
                "autotune": row.get("autotune", {}),
                "governance": row.get("governance", {}),
                "latest_status": "proposed",
                "created_at": row.get("created_at", ""),
                "updated_at": row.get("created_at", ""),
                "decisions": [],
                "would_modify_config": False,
                "would_write_memory": False,
            }
        elif event_type == "policy_proposal_decision" and proposal_id in states:
            decision = _policy_proposal_decision(row.get("decision"))
            states[proposal_id]["latest_status"] = decision
            states[proposal_id]["updated_at"] = row.get("created_at", "")
            states[proposal_id].setdefault("decisions", []).append(
                {
                    "decision": decision,
                    "reviewer": row.get("reviewer", ""),
                    "rationale": row.get("rationale", ""),
                    "created_at": row.get("created_at", ""),
                }
            )
    return states


def _policy_proposal_summary(proposals: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    by_status: dict[str, int] = {}
    by_priority: dict[str, int] = {}
    by_change_type: dict[str, int] = {}
    for proposal in proposals:
        status = _clean_text(proposal.get("latest_status")) or "unknown"
        priority = _clean_text(proposal.get("priority")) or "unknown"
        change_type = _clean_text(proposal.get("change_type")) or "unknown"
        by_status[status] = by_status.get(status, 0) + 1
        by_priority[priority] = by_priority.get(priority, 0) + 1
        by_change_type[change_type] = by_change_type.get(change_type, 0) + 1
    return {
        "by_status": by_status,
        "by_priority": by_priority,
        "by_change_type": by_change_type,
    }


def _policy_proposal_decision(value: Any) -> str:
    cleaned = _clean_text(value).lower()
    return cleaned if cleaned in VALID_POLICY_PROPOSAL_DECISIONS else "deferred"


def _policy_proposal_status(value: Any, *, allow_empty: bool = False) -> str:
    cleaned = _clean_text(value).lower()
    if allow_empty and not cleaned:
        return ""
    return cleaned if cleaned in VALID_POLICY_PROPOSAL_STATUSES else "proposed"


def _policy_apply_plan_for_proposal(proposal: Mapping[str, Any]) -> dict[str, Any]:
    suggestion_id = _clean_text(proposal.get("suggestion_id"))
    latest_status = _clean_text(proposal.get("latest_status")) or "unknown"
    target = _clean_text(proposal.get("target"))
    eligible = latest_status == "approved"
    base = {
        "proposal_id": proposal.get("proposal_id"),
        "suggestion_id": suggestion_id,
        "status": latest_status,
        "eligible_for_apply": eligible,
        "target": target,
        "recommendation": proposal.get("recommendation", ""),
        "patches": [],
        "notes": [],
        "risk_level": "low",
        "would_modify_config": False,
        "would_write_memory": False,
    }

    if suggestion_id == "external_auto_recall.keep_blocked":
        base["patches"] = [
            {
                "action": "verify",
                "target_file": str(Path.home() / ".openclaw" / "openclaw.json"),
                "json_path": "/plugins/entries/hermes-memory/config/autoPrecheckAllowedChannelIds",
                "expected": [],
                "reason": "Keep external-channel automatic recall blocked; this plan never unblocks or allowlists channels.",
                "would_modify": False,
            }
        ]
        base["notes"] = [
            "No config write is needed when the allowlist is already empty.",
            "Human privacy review is required before any separate exact-channel allowlist proposal.",
            "This proposal does not unblock, add, or widen external-channel automatic recall.",
        ]
        base["governance_classification"] = _classify_stale_policy_proposal(proposal)
    elif suggestion_id == "durable_writes.enforce_proposal_only":
        base["patches"] = [
            {
                "action": "verify",
                "target_file": str(Path(__file__).resolve()),
                "json_path": "memory_write_policy",
                "expected": "proposal_only",
                "reason": "Durable memory writes must continue routing through memory_write_proposal.",
                "would_modify": False,
            }
        ]
        base["risk_level"] = "medium"
    elif suggestion_id == "clients.require_registration":
        base["patches"] = [
            {
                "action": "review",
                "target_file": "Hermes memory client registry",
                "json_path": "/trusted_clients",
                "expected": sorted(INTERNAL_MEMORY_CLIENTS),
                "reason": "Unknown clients need explicit registration before broad memory access.",
                "would_modify": False,
            }
        ]
        base["risk_level"] = "medium"
    elif suggestion_id == "recall.routing_diagnostics":
        base["patches"] = [
            {
                "action": "diagnose",
                "target_file": str(Path.home() / ".openclaw" / "extensions" / "hermes-memory" / "index.ts"),
                "json_path": "before_prompt_build.auto_precheck",
                "expected": "gate allow followed by search or graph_read ledger events",
                "reason": "Allowed recall gates should be followed by actual recall reads.",
                "would_modify": False,
            }
        ]
    elif suggestion_id == "diagnostic.rerun_full_audit":
        base["patches"] = [
            {
                "action": "manual_review",
                "target_file": "Hermes MCP",
                "json_path": "memory_federation_audit",
                "expected": "human-triggered audit review only",
                "reason": "Diagnostic stale governance is proposal-preview only and never runs memory_federation_audit automatically.",
                "would_modify": False,
            }
        ]
        base["eligible_for_apply"] = False
        base["notes"] = [
            "Dry-run proposal preview only; do not execute a full audit from this plan.",
            "Schedule any full audit as a separate explicit human-triggered diagnostic.",
        ]
        base["governance_classification"] = _classify_stale_policy_proposal(proposal)
    elif suggestion_id == "snapshots.confirm_cache_only":
        base["patches"] = [
            {
                "action": "verify",
                "target_file": "Hermes snapshot policy",
                "json_path": "memory_snapshot_export.policy",
                "expected": "cache_or_backup_only",
                "reason": "Snapshots must not become a forked primary memory.",
                "would_modify": False,
            }
        ]
    elif suggestion_id == "ledger.repair_parse_errors":
        base["patches"] = [
            {
                "action": "manual_repair",
                "target_file": str(_operation_ledger_path(get_hermes_home())),
                "json_path": "malformed_jsonl_lines",
                "expected": "quarantine malformed lines after forensic copy",
                "reason": "Ledger parse errors weaken audit quality.",
                "would_modify": False,
            }
        ]
        base["risk_level"] = "medium"
    elif suggestion_id == "ledger.seed_audit_evidence":
        base["patches"] = [
            {
                "action": "run_check",
                "target_file": "Hermes MCP",
                "json_path": "memory_federation_gate + memory_fabric_search",
                "expected": "new ledger events",
                "reason": "Policy planner needs audit evidence before useful recommendations.",
                "would_modify": False,
            }
        ]
    else:
        base["patches"] = [
            {
                "action": "manual_review",
                "target_file": target or "memory_policy",
                "json_path": "",
                "expected": proposal.get("recommendation", ""),
                "reason": "No deterministic dry-run patch is available for this suggestion.",
                "would_modify": False,
            }
        ]
        base["risk_level"] = "medium"

    if not eligible:
        base["notes"].append("Proposal is not approved; plan is preview-only.")
    return base


def _classify_stale_policy_proposal(proposal: Mapping[str, Any]) -> dict[str, Any]:
    """Return read-only closure guidance for stale policy proposals.

    The classifier is intentionally preview-only: it never appends ledger
    events, never writes the Memory Graph, and never modifies runtime config.
    Its output is safe to embed in monitors and dry-run policy plans.
    """

    suggestion_id = _clean_text(proposal.get("suggestion_id"))
    proposal_id = _clean_text(proposal.get("proposal_id"))
    status = _clean_text(proposal.get("latest_status")) or _clean_text(proposal.get("status")) or "proposed"
    stale_reason = _stale_policy_reason(proposal)
    target_ledger_path = str(_policy_proposal_path(get_hermes_home()))
    required_ledger_evidence = _stale_policy_required_ledger_evidence(proposal)
    base = {
        "proposal_id": proposal_id,
        "suggestion_id": suggestion_id,
        "latest_status": status,
        "classification": "human_review_required",
        "stale_reason": stale_reason,
        "proposal_preview_only": True,
        "no_direct_graph_write": True,
        "does_not_modify_config": True,
        "does_not_write_memory": True,
        "does_not_write_graph": True,
        "does_not_append_ledger": True,
        "can_auto_apply": False,
        "requires_human_review": True,
        "approval_requirement": "explicit_human_policy_decision_required",
        "required_ledger_evidence": required_ledger_evidence,
        "safety_notes": [
            "Unknown or stale policy proposals require manual review before closure.",
            "Preview output must not be treated as a recorded policy decision.",
        ],
        "dry_run_no_write_marker": "dry_run_preview_no_write",
        "target_ledger_path": target_ledger_path,
        "rollback_or_noop_statement": "No rollback is needed for this preview because no ledger, config, memory, or graph writes occur.",
        "recommended_action": "Review the stale proposal and record approve, reject, or defer.",
    }
    if suggestion_id == "external_auto_recall.keep_blocked":
        base.update(
            {
                "classification": "privacy_review_keep_blocked",
                "requires_human_privacy_review": True,
                "never_unblocks_external_auto_recall": True,
                "never_adds_allowlist_entries": True,
                "approval_requirement": "human_privacy_review_required_before_any_exact_channel_allowlist_change",
                "safety_notes": [
                    "Keep external automatic recall blocked while stale.",
                    "Do not add external-channel allowlist entries from stale closure previews.",
                    "Any future allowlist change must be a separate exact-channel proposal.",
                ],
                "recommended_action": "Keep external automatic recall blocked until a human privacy review approves a separate exact-channel proposal.",
            }
        )
    elif suggestion_id == "diagnostic.rerun_full_audit":
        base.update(
            {
                "classification": "diagnostic_dry_run_proposal_only",
                "never_runs_audit": True,
                "approval_requirement": "separate_explicit_human_diagnostic_request_required",
                "safety_notes": [
                    "Do not run memory_federation_audit from stale closure planning.",
                    "Treat the stale diagnostic proposal as preview-only until a human requests a diagnostic.",
                ],
                "recommended_action": "Treat rerun_full_audit as a dry-run proposal preview; schedule any full audit as a separate human-triggered diagnostic.",
            }
        )
    return base


def _stale_policy_reason(proposal: Mapping[str, Any]) -> str:
    status = _clean_text(proposal.get("latest_status")) or _clean_text(proposal.get("status")) or "proposed"
    if status != "proposed":
        return f"Proposal is already closed with status {status}; no stale closure write is needed."
    age_hours = proposal.get("age_hours")
    if age_hours not in (None, ""):
        return f"Proposal remained proposed past the stale threshold; age_hours={age_hours}."
    updated_at = _clean_text(proposal.get("updated_at")) or _clean_text(proposal.get("created_at"))
    if updated_at:
        return f"Proposal remained proposed since {updated_at}."
    return "Proposal remained proposed past the stale threshold."


def _stale_policy_required_ledger_evidence(proposal: Mapping[str, Any]) -> list[str]:
    proposal_id = _clean_text(proposal.get("proposal_id")) or "<proposal_id>"
    suggestion_id = _clean_text(proposal.get("suggestion_id")) or "<suggestion_id>"
    return [
        f"policy_proposal_created event for {proposal_id}",
        f"latest proposal state is synthesized from append-only ledger path for {suggestion_id}",
        "no policy_proposal_decision event is appended by preview or execute-plan tools",
    ]


def _stale_policy_handoff_bundle_from_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    proposal_id = _clean_text(plan.get("proposal_id"))
    suggestion_id = _clean_text(plan.get("suggestion_id"))
    decision_args = plan.get("decision_args_preview", {})
    if not isinstance(decision_args, dict):
        decision_args = {}
    decision = _clean_text(decision_args.get("decision")) or "deferred"
    rationale = _clean_text(decision_args.get("rationale"))
    summary = f"Review stale policy proposal {proposal_id} ({suggestion_id}) for a record-only closure decision."
    risk_notes = [
        "This bundle is a preview and must not be treated as an executed decision.",
        "Recording a real decision would append a policy proposal ledger event only after explicit human approval.",
    ]
    guardrails = plan.get("guardrails", [])
    if not isinstance(guardrails, list):
        guardrails = []
    if suggestion_id == "external_auto_recall.keep_blocked":
        risk_notes.append("External automatic recall must remain blocked; telegram:dm:123 must not be allowlisted.")
        for guardrail in ["do_not_allowlist_telegram_dm_123", "do_not_unblock_external_auto_recall"]:
            if guardrail not in guardrails:
                guardrails.append(guardrail)
    elif suggestion_id == "diagnostic.rerun_full_audit":
        risk_notes.append("Full audit must not run automatically; any audit requires a separate explicit diagnostic request.")
        if "do_not_auto_run_full_audit" not in guardrails:
            guardrails.append("do_not_auto_run_full_audit")
    exact_decision_payload_preview = {
        "proposal_id": proposal_id,
        "decision": decision,
        "reviewer": "<human_reviewer>",
        "rationale": rationale or "<human_rationale>",
    }
    exact_decision_command_preview = (
        "memory_policy_proposal_decision("
        f"proposal_id={proposal_id!r}, "
        f"decision={decision!r}, "
        "reviewer='<human_reviewer>', "
        f"rationale={(rationale or '<human_rationale>')!r})"
    )
    return {
        "proposal_id": proposal_id,
        "suggestion_id": suggestion_id,
        "stale_reason": plan.get("stale_reason", ""),
        "recommended_action": plan.get("recommended_action", ""),
        "safety_notes": plan.get("safety_notes", []),
        "required_ledger_evidence": plan.get("required_ledger_evidence", []),
        "approval_requirement": plan.get("approval_requirement", "explicit_human_policy_decision_required"),
        "dry_run_no_write_marker": plan.get("dry_run_no_write_marker", "dry_run_preview_no_write"),
        "target_ledger_path": plan.get("target_ledger_path", str(_policy_proposal_path(get_hermes_home()))),
        "summary": summary,
        "risk_notes": risk_notes,
        "required_human_inputs": plan.get("required_inputs", ["human_reviewer", "human_rationale"]),
        "exact_decision_payload_preview": exact_decision_payload_preview,
        "exact_decision_command_preview": exact_decision_command_preview,
        "expected_ledger_side_effect_preview": {
            "would_append_event_type": "policy_proposal_decision",
            "would_update_proposal_status_to": decision,
            "would_apply_policy": False,
            "would_modify_config": False,
            "would_write_memory": False,
        },
        "rollback_or_undo_note": "If a real decision is recorded incorrectly, append a new corrective policy proposal decision; do not edit existing append-only ledger events.",
        "rollback_or_noop_statement": plan.get(
            "rollback_or_noop_statement",
            "No rollback is needed for this handoff preview because it performs no writes.",
        ),
        "guardrails": guardrails,
        "blocked_reasons": plan.get("blocked_reasons", []),
        "dry_run": True,
        "preview_only": True,
        "proposal_only": True,
        "does_not_call_policy_proposal_decision": True,
        "does_not_apply_policy": True,
        "does_not_modify_config": True,
        "does_not_write_memory": True,
        "does_not_write_graph": True,
        "does_not_append_ledger": True,
        "does_not_run_audit": True,
        "no_direct_graph_write": True,
        "can_auto_apply": False,
        "eligible_for_auto_execute": False,
        "requires_human_review": True,
        "would_modify_config": False,
        "would_write_memory": False,
    }


def _stale_policy_closure_execute_plan_from_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    proposal_id = _clean_text(payload.get("proposal_id"))
    suggestion_id = _clean_text(payload.get("suggestion_id"))
    decision_args = payload.get("decision_args_preview", {})
    if not isinstance(decision_args, dict):
        decision_args = {}
    decision = _clean_text(decision_args.get("decision"))
    reviewer = _clean_text(decision_args.get("reviewer"))
    rationale = _clean_text(decision_args.get("rationale"))
    blocked_reasons: list[str] = []
    if payload.get("dry_run") is not True:
        blocked_reasons.append("payload_not_dry_run")
    if payload.get("preview_only") is not True:
        blocked_reasons.append("payload_not_preview_only")
    if not proposal_id:
        blocked_reasons.append("proposal_id_required")
    if decision not in {"approved", "rejected", "deferred"}:
        blocked_reasons.append("invalid_or_missing_decision_status")
    if not reviewer:
        blocked_reasons.append("reviewer_required")
    if reviewer == "human_required":
        blocked_reasons.append("human_reviewer_not_supplied")
    if not rationale:
        blocked_reasons.append("rationale_required")
    if suggestion_id == "external_auto_recall.keep_blocked":
        blocked_reasons.extend(
            [
                "external_auto_recall_must_remain_blocked",
                "telegram_dm_123_must_not_be_allowlisted",
            ]
        )
    elif suggestion_id == "diagnostic.rerun_full_audit":
        blocked_reasons.append("full_audit_must_not_auto_run")
    required_inputs = []
    if reviewer == "human_required" or not reviewer:
        required_inputs.append("human_reviewer")
    if not rationale:
        required_inputs.append("human_rationale")
    eligible_for_human_decision = not any(
        reason
        in {
            "payload_not_dry_run",
            "payload_not_preview_only",
            "proposal_id_required",
            "invalid_or_missing_decision_status",
            "rationale_required",
        }
        for reason in blocked_reasons
    )
    execution_call_preview = {
        "tool": "memory_policy_proposal_decision",
        "arguments": {
            "proposal_id": proposal_id,
            "decision": decision,
            "reviewer": "<human_reviewer>",
            "rationale": rationale or "<human_rationale>",
        },
        "call_is_not_executed": True,
        "requires_explicit_human_approval": True,
    }
    return {
        "proposal_id": proposal_id,
        "suggestion_id": suggestion_id,
        "stale_reason": payload.get("stale_reason", ""),
        "recommended_action": payload.get("recommended_action", ""),
        "safety_notes": payload.get("safety_notes", []),
        "required_ledger_evidence": payload.get("required_ledger_evidence", []),
        "approval_requirement": payload.get("approval_requirement", "explicit_human_policy_decision_required"),
        "dry_run_no_write_marker": payload.get("dry_run_no_write_marker", "dry_run_preview_no_write"),
        "target_ledger_path": payload.get("target_ledger_path", str(_policy_proposal_path(get_hermes_home()))),
        "rollback_or_noop_statement": payload.get(
            "rollback_or_noop_statement",
            "No rollback is needed for this execute plan because it is plan-only and performs no writes.",
        ),
        "eligible_for_human_decision": eligible_for_human_decision,
        "eligible_for_auto_execute": False,
        "blocked_reasons": blocked_reasons,
        "required_inputs": required_inputs,
        "decision_args_preview": decision_args,
        "execution_call_preview": execution_call_preview,
        "guardrails": payload.get("guardrails", []),
        "plan_only": True,
        "will_not_execute": True,
        "dry_run": True,
        "preview_only": True,
        "proposal_only": True,
        "does_not_call_policy_proposal_decision": True,
        "does_not_apply_policy": True,
        "does_not_modify_config": True,
        "does_not_write_memory": True,
        "does_not_write_graph": True,
        "does_not_append_ledger": True,
        "does_not_run_audit": True,
        "no_direct_graph_write": True,
        "can_auto_apply": False,
        "requires_human_review": True,
        "would_modify_config": False,
        "would_write_memory": False,
    }


def _stale_policy_closure_payload_from_resolution(resolution: Mapping[str, Any]) -> dict[str, Any]:
    suggestion_id = _clean_text(resolution.get("suggestion_id"))
    proposal_id = _clean_text(resolution.get("proposal_id"))
    recommended_resolution = _clean_text(resolution.get("recommended_resolution"))
    decision_status = "deferred"
    reason = "Stale policy proposal requires an explicit human governance decision before closure."
    classification = _clean_text(resolution.get("classification"))
    guardrails = [
        "do_not_write_memory_graph",
        "do_not_modify_runtime_config",
        "do_not_append_policy_ledger",
        "do_not_auto_apply",
    ]
    if classification == "already_closed":
        decision_status = "none"
        reason = "Proposal is already closed; this payload is a no-op preview and must not append another decision."
        guardrails.append("do_not_append_duplicate_closure_decision")
    elif suggestion_id == "external_auto_recall.keep_blocked":
        decision_status = "deferred"
        reason = (
            "Retain blocked external automatic recall posture; close only after human privacy review "
            "confirms the keep-blocked intent or requests a separate exact-channel proposal."
        )
        guardrails.extend(
            [
                "do_not_allowlist_telegram_dm_123",
                "do_not_unblock_external_auto_recall",
                "privacy_review_required",
            ]
        )
    elif suggestion_id == "diagnostic.rerun_full_audit":
        decision_status = "deferred"
        reason = (
            "Convert stale full-audit proposal into a manual diagnostic preview; do not run a full audit "
            "unless a human makes a separate explicit audit request."
        )
        guardrails.extend(
            [
                "do_not_auto_run_full_audit",
                "defer_until_explicit_audit_request",
                "manual_diagnostic_preview_only",
            ]
        )
    decision_args_preview = {
        "proposal_id": proposal_id,
        "decision": decision_status,
        "reviewer": "human_required",
        "rationale": reason,
    }
    return {
        "proposal_id": proposal_id,
        "suggestion_id": suggestion_id,
        "original_status": "closed" if classification == "already_closed" else "proposed",
        "stale_reason": resolution.get("stale_reason", reason),
        "recommended_resolution": recommended_resolution,
        "recommended_action": resolution.get("recommended_action", ""),
        "recommended_decision_status": decision_status,
        "reason": reason,
        "safety_notes": resolution.get("safety_notes", []),
        "required_ledger_evidence": resolution.get("required_ledger_evidence", []),
        "approval_requirement": resolution.get("approval_requirement", "explicit_human_policy_decision_required"),
        "dry_run_no_write_marker": resolution.get("dry_run_no_write_marker", "dry_run_preview_no_write"),
        "target_ledger_path": resolution.get("target_ledger_path", str(_policy_proposal_path(get_hermes_home()))),
        "rollback_or_noop_statement": resolution.get(
            "rollback_or_noop_statement",
            "No rollback is needed for this payload preview because no writes occur.",
        ),
        "decision_args_preview": decision_args_preview,
        "guardrails": guardrails,
        "recommended_next_action": (
            "No closure action is needed for an already closed proposal."
            if classification == "already_closed"
            else "Human reviewer may pass decision_args_preview to memory_policy_proposal_decision after review."
        ),
        "dry_run": True,
        "preview_only": True,
        "proposal_only": True,
        "does_not_apply_policy": True,
        "does_not_modify_config": True,
        "does_not_write_memory": True,
        "does_not_write_graph": True,
        "does_not_append_ledger": True,
        "no_direct_graph_write": True,
        "can_auto_apply": False,
        "requires_human_review": classification != "already_closed",
        "requires_human_privacy_review": resolution.get("requires_human_privacy_review") is True,
        "would_modify_config": False,
        "would_write_memory": False,
    }


def _stale_policy_resolution_from_monitor_item(item: Mapping[str, Any]) -> dict[str, Any]:
    suggestion_id = _clean_text(item.get("suggestion_id"))
    classification = item.get("governance_classification", {})
    if not isinstance(classification, dict):
        classification = {}
    recommended_resolution = "record_human_review_decision"
    decision_recommendation = "defer_until_reviewed"
    if suggestion_id == "external_auto_recall.keep_blocked":
        recommended_resolution = "retain_blocked_posture"
        decision_recommendation = "approve_as_keep_blocked_intent_only_or_defer_for_privacy_review"
    elif suggestion_id == "diagnostic.rerun_full_audit":
        recommended_resolution = "convert_to_manual_diagnostic_preview"
        decision_recommendation = "defer_until_explicit_audit_request"
    return {
        "proposal_id": item.get("proposal_id", ""),
        "suggestion_id": suggestion_id,
        "age_hours": item.get("age_hours"),
        "updated_at": item.get("updated_at", ""),
        "stale_reason": classification.get("stale_reason", ""),
        "classification": classification.get("classification", "human_review_required"),
        "governance_classification": classification,
        "recommended_resolution": recommended_resolution,
        "decision_recommendation": decision_recommendation,
        "recommended_action": classification.get("recommended_action") or item.get("recommended_action", ""),
        "safety_notes": classification.get("safety_notes", []),
        "required_ledger_evidence": classification.get("required_ledger_evidence", []),
        "approval_requirement": classification.get("approval_requirement", "explicit_human_policy_decision_required"),
        "dry_run_no_write_marker": classification.get("dry_run_no_write_marker", "dry_run_preview_no_write"),
        "target_ledger_path": classification.get("target_ledger_path", str(_policy_proposal_path(get_hermes_home()))),
        "rollback_or_noop_statement": classification.get(
            "rollback_or_noop_statement",
            "No rollback is needed for this preview because no writes occur.",
        ),
        "requires_human_review": True,
        "requires_human_privacy_review": classification.get("requires_human_privacy_review") is True,
        "can_auto_apply": False,
        "proposal_only": True,
        "dry_run": True,
        "no_direct_graph_write": True,
        "does_not_modify_config": True,
        "does_not_write_memory": True,
        "does_not_write_graph": True,
        "does_not_append_ledger": True,
        "would_modify_config": False,
        "would_write_memory": False,
    }


def _closed_policy_resolution_from_proposal(proposal: Mapping[str, Any]) -> dict[str, Any]:
    classification = _classify_stale_policy_proposal(proposal)
    status = _clean_text(proposal.get("latest_status")) or "unknown"
    return {
        "proposal_id": proposal.get("proposal_id", ""),
        "suggestion_id": _clean_text(proposal.get("suggestion_id")),
        "age_hours": None,
        "updated_at": proposal.get("updated_at", ""),
        "stale_reason": classification.get("stale_reason", ""),
        "classification": "already_closed",
        "governance_classification": {**classification, "classification": "already_closed"},
        "recommended_resolution": "no_closure_needed",
        "decision_recommendation": f"already_{status}",
        "recommended_action": "No stale closure action is needed because the proposal already has a recorded decision.",
        "safety_notes": [
            "Do not append duplicate closure decisions from stale preview tools.",
            "Use the existing append-only ledger decision as the source of truth.",
        ],
        "required_ledger_evidence": classification.get("required_ledger_evidence", []),
        "approval_requirement": "none_for_preview_already_closed",
        "dry_run_no_write_marker": classification.get("dry_run_no_write_marker", "dry_run_preview_no_write"),
        "target_ledger_path": classification.get("target_ledger_path", str(_policy_proposal_path(get_hermes_home()))),
        "rollback_or_noop_statement": "No-op: proposal is already closed and this preview performs no writes.",
        "requires_human_review": False,
        "requires_human_privacy_review": False,
        "can_auto_apply": False,
        "proposal_only": True,
        "dry_run": True,
        "no_direct_graph_write": True,
        "does_not_modify_config": True,
        "does_not_write_memory": True,
        "does_not_write_graph": True,
        "does_not_append_ledger": True,
        "would_modify_config": False,
        "would_write_memory": False,
    }


def _policy_apply_plan_summary(plans: Iterable[Mapping[str, Any]], field: str) -> dict[str, int]:
    summary: dict[str, int] = {}
    for plan in plans:
        patches = plan.get("patches", [])
        if not isinstance(patches, list):
            continue
        for patch in patches:
            if not isinstance(patch, dict):
                continue
            value = _clean_text(patch.get(field)) or "unknown"
            summary[value] = summary.get(value, 0) + 1
    return summary


def _policy_apply_guard(
    plan: Mapping[str, Any],
    *,
    execute: bool,
    confirm_token: str,
) -> dict[str, Any]:
    plans = plan.get("plans", []) if isinstance(plan.get("plans"), list) else []
    patches = [
        patch
        for item in plans
        if isinstance(item, dict)
        for patch in item.get("patches", [])
        if isinstance(patch, dict)
    ]
    if int(plan.get("eligible_count") or 0) == 0:
        return {
            "allowed": False,
            "reason": "No approved, eligible policy proposals are available for execution.",
            "requires_confirm_token": True,
            "expected_confirm_token": POLICY_EXECUTE_CONFIRM_TOKEN,
        }
    if not all(item.get("eligible_for_apply") is True for item in plans):
        return {
            "allowed": False,
            "reason": "Plan contains proposals that are not approved.",
            "requires_confirm_token": True,
            "expected_confirm_token": POLICY_EXECUTE_CONFIRM_TOKEN,
        }
    mutable_patches = [patch for patch in patches if patch.get("would_modify") is True]
    if mutable_patches:
        return {
            "allowed": False,
            "reason": "Plan contains config-mutating patches; guarded executor only allows non-mutating checks.",
            "mutable_patch_count": len(mutable_patches),
            "requires_confirm_token": True,
            "expected_confirm_token": POLICY_EXECUTE_CONFIRM_TOKEN,
        }
    unsupported_actions = [
        _clean_text(patch.get("action"))
        for patch in patches
        if _clean_text(patch.get("action")) not in VALID_POLICY_EXECUTE_ACTIONS
    ]
    if unsupported_actions:
        return {
            "allowed": False,
            "reason": "Plan contains unsupported executor actions.",
            "unsupported_actions": sorted(set(unsupported_actions)),
            "requires_confirm_token": True,
            "expected_confirm_token": POLICY_EXECUTE_CONFIRM_TOKEN,
        }
    if execute and _clean_text(confirm_token) != POLICY_EXECUTE_CONFIRM_TOKEN:
        return {
            "allowed": False,
            "reason": "Missing or invalid policy execution confirmation token.",
            "requires_confirm_token": True,
            "expected_confirm_token": POLICY_EXECUTE_CONFIRM_TOKEN,
        }
    return {
        "allowed": True,
        "reason": "Plan is approved and contains only non-mutating executor actions.",
        "requires_confirm_token": True,
        "expected_confirm_token": POLICY_EXECUTE_CONFIRM_TOKEN,
        "patch_count": len(patches),
    }


def _execute_policy_apply_plan(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for item in plan.get("plans", []):
        if not isinstance(item, dict):
            continue
        proposal_id = _clean_text(item.get("proposal_id"))
        for patch in item.get("patches", []):
            if not isinstance(patch, dict):
                continue
            results.append(_execute_policy_patch(patch, proposal_id=proposal_id))
    return results


def _execute_policy_patch(
    patch: Mapping[str, Any],
    *,
    proposal_id: str,
) -> dict[str, Any]:
    action = _clean_text(patch.get("action"))
    if patch.get("would_modify") is True:
        return _policy_patch_result(
            patch,
            proposal_id=proposal_id,
            status="blocked",
            passed=False,
            message="Patch would modify config and is blocked by executor guard.",
        )
    if action not in VALID_POLICY_EXECUTE_ACTIONS:
        return _policy_patch_result(
            patch,
            proposal_id=proposal_id,
            status="blocked",
            passed=False,
            message="Patch action is not supported by executor guard.",
        )
    if action == "verify":
        return _execute_policy_verify_patch(patch, proposal_id=proposal_id)
    if action == "run_check":
        return _execute_policy_run_check_patch(patch, proposal_id=proposal_id)
    return _policy_patch_result(
        patch,
        proposal_id=proposal_id,
        status="manual_required",
        passed=True,
        message="Patch is a non-mutating manual or diagnostic action.",
    )


def _execute_policy_verify_patch(
    patch: Mapping[str, Any],
    *,
    proposal_id: str,
) -> dict[str, Any]:
    target_file = _clean_text(patch.get("target_file"))
    json_path = _clean_text(patch.get("json_path"))
    expected = patch.get("expected")
    if target_file.endswith(".json") and json_path.startswith("/"):
        path = Path(target_file).expanduser()
        data = _loads_json(_safe_read_text(path), None)
        actual = _json_pointer_get(data, json_path) if data is not None else None
        return _policy_patch_result(
            patch,
            proposal_id=proposal_id,
            status="checked",
            passed=actual == expected,
            message="JSON pointer verification completed.",
            actual=actual,
        )
    if target_file.endswith(".py") and expected == "proposal_only":
        text = _safe_read_text(Path(target_file).expanduser())
        passed = "proposal_only" in text and "create_memory_write_proposal" in text
        return _policy_patch_result(
            patch,
            proposal_id=proposal_id,
            status="checked",
            passed=passed,
            message="Source policy verification completed.",
        )
    return _policy_patch_result(
        patch,
        proposal_id=proposal_id,
        status="manual_required",
        passed=True,
        message="Verification target is not machine-checkable by this executor.",
    )


def _execute_policy_run_check_patch(
    patch: Mapping[str, Any],
    *,
    proposal_id: str,
) -> dict[str, Any]:
    json_path = _clean_text(patch.get("json_path"))
    if json_path == "memory_federation_audit":
        audit = memory_federation_audit(log_limit=200)
        critical = [
            check
            for check in audit.get("checks", [])
            if isinstance(check, dict)
            and check.get("status") != "pass"
            and check.get("severity") == "critical"
        ]
        return _policy_patch_result(
            patch,
            proposal_id=proposal_id,
            status="checked",
            passed=audit.get("ready") is True and not critical,
            message="memory_federation_audit check completed.",
            actual={
                "ready": audit.get("ready"),
                "health_score": audit.get("health_score"),
                "critical_failure_count": len(critical),
            },
        )
    return _policy_patch_result(
        patch,
        proposal_id=proposal_id,
        status="manual_required",
        passed=True,
        message="Run-check target is not machine-executable by this executor.",
    )


def _policy_patch_result(
    patch: Mapping[str, Any],
    *,
    proposal_id: str,
    status: str,
    passed: bool,
    message: str,
    actual: Any = None,
) -> dict[str, Any]:
    return {
        "proposal_id": proposal_id,
        "action": patch.get("action"),
        "target_file": patch.get("target_file"),
        "json_path": patch.get("json_path"),
        "status": status,
        "passed": passed,
        "message": message,
        "expected": patch.get("expected"),
        "actual": actual,
        "would_modify": False,
    }


def _policy_apply_execute_summary(results: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    summary = {
        "checked_count": 0,
        "manual_required_count": 0,
        "blocked_count": 0,
        "failed_count": 0,
        "passed_count": 0,
    }
    for result in results:
        status = _clean_text(result.get("status"))
        if status == "checked":
            summary["checked_count"] += 1
        elif status == "manual_required":
            summary["manual_required_count"] += 1
        elif status == "blocked":
            summary["blocked_count"] += 1
        if result.get("passed") is True:
            summary["passed_count"] += 1
        elif result.get("passed") is False:
            summary["failed_count"] += 1
    return summary


def _policy_outcome_metrics(
    proposals: Iterable[Mapping[str, Any]],
    rows: Iterable[Mapping[str, Any]],
    *,
    stale_after_hours: int,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    proposal_items = list(proposals)
    summary = _policy_proposal_summary(proposal_items)
    executions_by_proposal: dict[str, list[dict[str, Any]]] = {}
    recent_execution_events: list[dict[str, Any]] = []
    execution_totals = {
        "checked_count": 0,
        "manual_required_count": 0,
        "blocked_count": 0,
        "failed_count": 0,
        "passed_count": 0,
    }

    for row in rows:
        if row.get("_parse_error") or _clean_text(row.get("event_type")) != "policy_apply_execute":
            continue
        result_summary = row.get("result_summary", {})
        if not isinstance(result_summary, dict):
            result_summary = {}
        event = {
            "event_id": row.get("event_id", ""),
            "proposal_id": _clean_text(row.get("proposal_id")),
            "actor": row.get("actor", ""),
            "created_at": row.get("created_at", ""),
            "plan_summary": row.get("plan_summary", {}),
            "result_summary": result_summary,
            "does_not_apply_config_changes": row.get("does_not_apply_config_changes") is True,
            "would_modify_config": row.get("would_modify_config") is True,
        }
        proposal_id = _clean_text(event.get("proposal_id"))
        if proposal_id:
            executions_by_proposal.setdefault(proposal_id, []).append(event)
        recent_execution_events.append(event)
        for key in execution_totals:
            execution_totals[key] += _safe_int(result_summary.get(key))

    stale_proposed: list[dict[str, Any]] = []
    approved_not_executed: list[dict[str, Any]] = []
    for proposal in proposal_items:
        proposal_id = _clean_text(proposal.get("proposal_id"))
        status = _clean_text(proposal.get("latest_status")) or "unknown"
        updated_at = _parse_datetime(proposal.get("updated_at")) or _parse_datetime(
            proposal.get("created_at")
        )
        age_hours = None
        if updated_at is not None:
            age_hours = max(0.0, (now - updated_at).total_seconds() / 3600)
        if status == "proposed" and age_hours is not None and age_hours >= stale_after_hours:
            classification = _classify_stale_policy_proposal(proposal)
            stale_proposed.append(
                {
                    "proposal_id": proposal_id,
                    "suggestion_id": proposal.get("suggestion_id", ""),
                    "age_hours": round(age_hours, 2),
                    "updated_at": proposal.get("updated_at", ""),
                    "governance_classification": classification,
                    "recommended_action": classification.get("recommended_action", ""),
                }
            )
        if status == "approved" and proposal_id and not executions_by_proposal.get(proposal_id):
            approved_not_executed.append(
                {
                    "proposal_id": proposal_id,
                    "suggestion_id": proposal.get("suggestion_id", ""),
                    "updated_at": proposal.get("updated_at", ""),
                }
            )

    recent_execution_events.sort(key=lambda row: _clean_text(row.get("created_at")), reverse=True)
    latest_execution_at = (
        _clean_text(recent_execution_events[0].get("created_at"))
        if recent_execution_events
        else ""
    )
    by_status = summary.get("by_status", {}) if isinstance(summary, dict) else {}
    return {
        "total_proposals": len(proposal_items),
        "proposal_count_by_status": by_status if isinstance(by_status, dict) else {},
        "proposed_count": _safe_int((by_status or {}).get("proposed")),
        "approved_count": _safe_int((by_status or {}).get("approved")),
        "rejected_count": _safe_int((by_status or {}).get("rejected")),
        "deferred_count": _safe_int((by_status or {}).get("deferred")),
        "stale_proposed_count": len(stale_proposed),
        "stale_proposed": stale_proposed,
        "approved_not_executed_count": len(approved_not_executed),
        "approved_not_executed": approved_not_executed,
        "execution_count": len(recent_execution_events),
        "execution_count_with_specific_proposal": sum(
            1 for event in recent_execution_events if _clean_text(event.get("proposal_id"))
        ),
        "execution_totals": execution_totals,
        "latest_execution_at": latest_execution_at,
        "recent_execution_events": recent_execution_events,
    }


def _policy_outcome_findings(
    metrics: Mapping[str, Any],
    parse_errors: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    parse_errors = list(parse_errors)
    if parse_errors:
        findings.append(
            _ledger_finding(
                "policy_ledger.parse_errors",
                "warning",
                "policy_proposal_ledger",
                "Policy proposal ledger contains malformed rows.",
                {"parse_error_count": len(parse_errors), "examples": parse_errors[:3]},
                "Repair or quarantine malformed proposal ledger lines before making policy decisions.",
            )
        )

    total_proposals = _safe_int(metrics.get("total_proposals"))
    proposed_count = _safe_int(metrics.get("proposed_count"))
    stale_count = _safe_int(metrics.get("stale_proposed_count"))
    approved_not_executed_count = _safe_int(metrics.get("approved_not_executed_count"))
    execution_totals = metrics.get("execution_totals", {})
    if not isinstance(execution_totals, dict):
        execution_totals = {}
    blocked_count = _safe_int(execution_totals.get("blocked_count"))
    failed_count = _safe_int(execution_totals.get("failed_count"))

    if total_proposals == 0:
        findings.append(
            _ledger_finding(
                "policy.no_proposals",
                "info",
                "policy_proposal_lifecycle",
                "No policy proposals are present yet.",
                {},
                "Use memory_policy_autotune and memory_policy_proposal_create after enough ledger evidence exists.",
            )
        )
    elif proposed_count:
        findings.append(
            _ledger_finding(
                "policy.proposal_backlog",
                "warning" if proposed_count >= 3 else "info",
                "policy_proposal_lifecycle",
                "Some policy proposals are waiting for human review.",
                {"proposed_count": proposed_count},
                "Review proposals with memory_policy_proposal_ledger and record approve/reject/defer decisions.",
            )
        )
    if stale_count:
        stale_actions = [
            row.get("recommended_action")
            for row in metrics.get("stale_proposed", [])
            if isinstance(row, dict) and row.get("recommended_action")
        ]
        findings.append(
            _ledger_finding(
                "policy.stale_proposals",
                "warning",
                "policy_proposal_lifecycle",
                "Some proposed policy changes have become stale.",
                {"stale_proposed": metrics.get("stale_proposed", [])},
                " ".join(stale_actions) if stale_actions else "Decide stale proposals before creating more policy suggestions.",
            )
        )
    if approved_not_executed_count:
        findings.append(
            _ledger_finding(
                "policy.approved_not_executed",
                "warning",
                "policy_apply_lifecycle",
                "Some approved policy proposals have no matching execution check event.",
                {"approved_not_executed": metrics.get("approved_not_executed", [])},
                "Run memory_policy_apply_execute with explicit confirmation for approved non-mutating checks.",
            )
        )
    if blocked_count or failed_count:
        findings.append(
            _ledger_finding(
                "policy.execution_failures",
                "critical",
                "policy_apply_lifecycle",
                "Policy execution checks reported blocked or failed results.",
                {"blocked_count": blocked_count, "failed_count": failed_count},
                "Inspect the latest policy_apply_execute results before approving more proposals.",
            )
        )
    if not findings:
        findings.append(
            _ledger_finding(
                "policy.outcome_healthy",
                "info",
                "policy_outcome_monitor",
                "Policy proposal and execution outcomes look healthy in the analyzed ledger.",
                {"total_proposals": total_proposals},
                "",
            )
        )
    return findings


def _json_pointer_get(data: Any, pointer: str) -> Any:
    current = data
    if pointer == "":
        return current
    for raw_part in pointer.strip("/").split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return current


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                rows.append(
                    {
                        "_parse_error": str(exc),
                        "line_number": line_number,
                        "line_preview": line[:200],
                    }
                )
                continue
            rows.append(value if isinstance(value, dict) else {"value": value})
    except OSError:
        return []
    return rows


def _safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def _scope(scope: str) -> str:
    cleaned = _clean_text(scope).lower() or "all"
    return cleaned if cleaned in VALID_SEARCH_SCOPES else "all"


def _proposal_scope(scope: str) -> str:
    cleaned = _clean_text(scope).lower() or "project"
    return cleaned if cleaned in VALID_PROPOSAL_SCOPES else "project"


def _sqlite_count(path: Path, table: str) -> int:
    if not path.exists():
        return 0
    try:
        with sqlite3.connect(path) as conn:
            row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            return int(row[0]) if row else 0
    except sqlite3.Error:
        return 0


def _jsonl_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    except OSError:
        return 0


def _knowledge_file_count(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for path in root.rglob("*") if path.is_file())


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def _loads_json(value: str, default: Any) -> Any:
    try:
        return json.loads(value) if value else default
    except (TypeError, json.JSONDecodeError):
        return default


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _snippet(text: str, query: str, *, width: int = 420) -> str:
    text = _clean_text(text)
    if len(text) <= width:
        return text
    lower = text.lower()
    index = lower.find(query.lower())
    if index == -1:
        return text[:width].rstrip() + "..."
    start = max(0, index - width // 3)
    end = min(len(text), start + width)
    prefix = "..." if start else ""
    suffix = "..." if end < len(text) else ""
    return prefix + text[start:end].strip() + suffix


def _score_text(text: str, query: str, *, base: float) -> float:
    text_lower = text.lower()
    terms = [term for term in query.lower().split() if term]
    hits = sum(text_lower.count(term) for term in terms) if terms else 0
    exact = 1 if query.lower() in text_lower else 0
    return round(base + exact + hits * 0.15, 4)


def _list_of_str(value: Iterable[str] | str | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [_clean_text(item) for item in value.split(",") if _clean_text(item)]
    return [_clean_text(item) for item in value if _clean_text(item)]


def _clamp_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(minimum, min(number, maximum))


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _parse_datetime(value: Any) -> datetime | None:
    text = _clean_text(value)
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _text_digest(value: str) -> str:
    return hashlib.sha256(_clean_text(value).encode("utf-8")).hexdigest()


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _error(message: str, **extra: Any) -> dict[str, Any]:
    return {
        "success": False,
        "error": message,
        **extra,
        "read_only_memory": True,
        "would_mutate_memory": False,
    }
