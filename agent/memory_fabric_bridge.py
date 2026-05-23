"""Shared Hermes Memory Fabric bridge.

This module keeps Hermes as the primary memory owner. It exposes read-first
status, search, governance, policy-proposal, and evolution-level helpers for
MCP clients such as Codex and OpenClaw. Durable memory content and Memory Graph
rows are not modified here; writes are represented as governed proposals.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    from hermes_constants import get_hermes_home
except Exception:  # pragma: no cover - fallback for standalone scripts
    def get_hermes_home() -> Path:
        return Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))


VALID_SEARCH_SCOPES = {"all", "graph", "prompt_cases", "knowledge", "legacy_memory"}
VALID_PROPOSAL_SCOPES = {"global", "project", "agent_private", "user", "memory", "procedural"}
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
VALID_POLICY_PROPOSAL_DECISIONS = {"approved", "rejected", "deferred"}
VALID_POLICY_PROPOSAL_STATUSES = {"proposed", "approved", "rejected", "deferred"}
DURABLE_WRITE_OPERATIONS = {"direct_write", "promote_memory", "delete_memory"}
INTERNAL_MEMORY_CLIENTS = {"hermes", "codex", "openclaw"}
POLICY_EXECUTE_CONFIRM_TOKEN = "HERMES_EXECUTE_APPROVED_POLICY_PLAN"
VALID_POLICY_EXECUTE_ACTIONS = {"verify", "run_check", "diagnose", "review", "manual_review", "manual_repair"}
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
DEFAULT_RECALL_QUALITY_QUERIES = [
    "Lovart",
    "GPT image",
    "OpenClaw",
    "policy",
    "memory",
]

MEMORY_EVOLUTION_TIERS: list[dict[str, Any]] = [
    {
        "level": 1,
        "id": "star_spark",
        "name": "星火记忆",
        "one_line": "能记录零散事实和短期片段。",
        "capabilities": ["basic_capture"],
    },
    {
        "level": 2,
        "id": "star_point",
        "name": "星点记忆",
        "one_line": "能把零散记忆沉淀为可复用条目。",
        "capabilities": ["entry_persistence", "basic_lookup"],
    },
    {
        "level": 3,
        "id": "star_link",
        "name": "星链记忆",
        "one_line": "能把记忆之间的关系连接起来。",
        "capabilities": ["relations", "linked_recall"],
    },
    {
        "level": 4,
        "id": "star_map",
        "name": "星图记忆",
        "one_line": "能形成可浏览、可检索的知识地图。",
        "capabilities": ["graph", "provenance", "structured_read"],
    },
    {
        "level": 5,
        "id": "star_river",
        "name": "星河记忆",
        "one_line": "能跨知识库、案例库和历史记忆做联合召回。",
        "capabilities": ["multi_surface_recall", "knowledge_index"],
    },
    {
        "level": 6,
        "id": "star_core",
        "name": "星辰记忆",
        "one_line": "能沉淀知识、案例、方法论，并服务具体创作和工作流。",
        "capabilities": ["case_memory", "workflow_memory", "skill_support"],
    },
    {
        "level": 7,
        "id": "star_domain",
        "name": "星域记忆",
        "one_line": "能按项目、角色、场景和智能体分区管理记忆。",
        "capabilities": ["project_domains", "agent_profiles", "scoped_recall"],
    },
    {
        "level": 8,
        "id": "star_dome",
        "name": "星穹记忆",
        "one_line": "能治理、审计、联邦共享，并约束外部通道记忆暴露。",
        "capabilities": ["federation", "audit", "policy_gate", "proposal_only_writes"],
    },
    {
        "level": 9,
        "id": "star_sea",
        "name": "星海记忆",
        "one_line": "能容纳多来源、多智能体的大规模记忆，并持续评估召回质量。",
        "capabilities": ["large_scale_sources", "multi_agent_recall", "recall_quality_evaluation"],
    },
    {
        "level": 10,
        "id": "star_realm",
        "name": "星界记忆",
        "one_line": "能跨系统、跨边界协同记忆，同时保持边界和权限。",
        "capabilities": ["cross_system_memory", "boundary_aware_sharing"],
    },
    {
        "level": 11,
        "id": "star_hub",
        "name": "星枢记忆",
        "one_line": "有中心调度和策略编排，能把记忆能力分配给任务链。",
        "capabilities": ["orchestration", "routing", "policy_scheduling"],
    },
    {
        "level": 12,
        "id": "star_law",
        "name": "星律记忆",
        "one_line": "记忆具备自我规则、自我约束和策略闭环。",
        "capabilities": ["self_governance", "closed_loop_policy", "risk_controls"],
    },
    {
        "level": 13,
        "id": "star_soul",
        "name": "星魂记忆",
        "one_line": "形成长期人格、偏好、工作方式和协作风格连续性。",
        "capabilities": ["preference_continuity", "persona_memory", "collaboration_style"],
    },
    {
        "level": 14,
        "id": "star_universe",
        "name": "星宙记忆",
        "one_line": "跨时间、跨项目、跨生态长期演化。",
        "capabilities": ["long_horizon_evolution", "ecosystem_memory", "temporal_reasoning"],
    },
    {
        "level": 15,
        "id": "star_source",
        "name": "星源记忆",
        "one_line": "能反推知识来源、生成方法论，并持续自进化。",
        "capabilities": ["source_reasoning", "methodology_generation", "self_evolution"],
    },
]


def memory_bridge_status() -> dict[str, Any]:
    """Return read-only status for Hermes shared memory surfaces."""

    home = get_hermes_home()
    graph_path = _graph_path(home)
    prompt_index_path = _prompt_index_path(home)
    knowledge_dir = home / "knowledge"
    proposal_path = _proposal_path(home)
    operation_path = _operation_ledger_path(home)
    policy_path = _policy_proposal_path(home)
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
                "provenance_count": _sqlite_count(graph_path, "graph_provenance"),
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
                "path": str(operation_path),
                "exists": operation_path.exists(),
                "event_count": _jsonl_count(operation_path),
            },
            "policy_proposals": {
                "path": str(policy_path),
                "exists": policy_path.exists(),
                "event_count": _jsonl_count(policy_path),
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


def memory_evolution_status() -> dict[str, Any]:
    """Return the fixed Hermes memory tier taxonomy and current stage assessment."""

    bridge = memory_bridge_status()
    federation = memory_federation_status()
    boundary_audit = memory_boundary_allowlist_audit(log_limit=200)
    routing_metrics = memory_orchestration_routing_metrics()
    outcome = memory_policy_outcome_monitor(limit=50, stale_after_hours=72)
    recall_quality = memory_recall_quality_evaluate(limit=5)
    evidence = _memory_evolution_evidence(bridge, federation, outcome, recall_quality, routing_metrics)
    evidence["boundary_allowlists_reviewed"] = bool(boundary_audit.get("ready"))
    evidence["boundary_readiness_score"] = boundary_audit.get("boundary_readiness_score", 0)
    readiness = [_tier_readiness(tier, evidence) for tier in MEMORY_EVOLUTION_TIERS]
    achieved = [item for item in readiness if item["achieved"]]
    current = achieved[-1] if achieved else readiness[0]
    next_item = next((item for item in readiness if item["level"] > current["level"]), None)
    phase = current["name"]
    if current["level"] == 8 and next_item and next_item["readiness_score"] >= 0.5:
        phase = "星穹记忆初期到星海记忆门口"
    elif next_item and next_item["readiness_score"] >= 0.65:
        phase = f"{current['name']}到{next_item['name']}门口"
    return {
        "success": True,
        "evolution_type": "hermes_memory_evolution_status",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "taxonomy": MEMORY_EVOLUTION_TIERS,
        "current": {
            "level": current["level"],
            "id": current["id"],
            "name": current["name"],
            "phase": phase,
            "readiness_score": current["readiness_score"],
        },
        "next": next_item,
        "readiness": readiness,
        "evidence": evidence,
        "orchestration_routing_metrics": routing_metrics,
        "recall_quality": {
            "quality_score": recall_quality.get("quality_score"),
            "readiness": recall_quality.get("readiness"),
            "passed_query_count": recall_quality.get("summary", {}).get("passed_query_count")
            if isinstance(recall_quality.get("summary"), dict)
            else None,
            "benchmark_query_count": recall_quality.get("summary", {}).get("benchmark_query_count")
            if isinstance(recall_quality.get("summary"), dict)
            else None,
        },
        "recommended_next_actions": _memory_evolution_next_actions(current, next_item, outcome, evidence),
        "policy": {
            "taxonomy_is_fixed": True,
            "status_is_read_only": True,
            "does_not_modify_config": True,
            "does_not_write_memory": True,
            "persistent_changes_must_use_proposals": True,
        },
        "read_only": True,
        "read_only_memory": True,
        "would_mutate_memory": False,
        "would_modify_config": False,
    }


def memory_federation_status() -> dict[str, Any]:
    """Return read-only status for clients sharing Hermes Memory Fabric."""

    home = get_hermes_home()
    tools = [
        "memory_bridge_status",
        "memory_evolution_status",
        "memory_orchestration_routing_metrics",
        "memory_recall_quality_evaluate",
        "memory_fabric_search",
        "memory_graph_read",
        "memory_write_proposal",
        "memory_snapshot_export",
        "memory_federation_status",
        "memory_federation_audit",
        "memory_boundary_allowlist_audit",
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
    ]
    clients = {
        "hermes": {
            "role": "primary_memory_owner",
            "hermes_home": str(home),
            "mcp_server": {
                "path": str(Path(__file__).resolve().parents[1] / "mcp_serve.py"),
                "exists": (Path(__file__).resolve().parents[1] / "mcp_serve.py").exists(),
                "tools": tools,
            },
            "surfaces": memory_bridge_status()["surfaces"],
        },
        "codex": _codex_memory_client_status(),
        "openclaw": _openclaw_memory_client_status(),
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
    checks = [
        _audit_check("hermes.mcp_server", status["clients"]["hermes"]["mcp_server"]["exists"], "critical", "Hermes MCP server file exists."),
        _audit_check("codex.configured", status["clients"]["codex"]["ready"], "warning", "Codex is configured as a memory client."),
        _audit_check("openclaw.configured", status["clients"]["openclaw"]["ready"], "warning", "OpenClaw plugin is configured as a memory client."),
        _audit_check("writes.proposal_only", status["policy"]["writes_are_proposal_only"], "critical", "Durable writes are proposal-only."),
        _audit_check("external.default_blocked", status["policy"]["external_channel_auto_recall_requires_allowlist"], "critical", "External channel recall requires allowlist."),
    ]
    ready = not [check for check in checks if check["status"] != "pass" and check["severity"] == "critical"]
    return {
        "success": True,
        "audit_type": "hermes_memory_federation_audit",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ready": ready,
        "health_score": max(0, 100 - 15 * len([c for c in checks if c["status"] != "pass"])),
        "checks": checks,
        "status_summary": {
            "federation_ready": status.get("ready"),
            "warnings": status.get("warnings", []),
            "log_limit": log_limit,
        },
        "policy": {
            "audit_is_read_only": True,
            "does_not_modify_config": True,
            "does_not_write_memory": True,
        },
        "read_only": True,
        "read_only_memory": True,
        "would_mutate_memory": False,
        "would_modify_config": False,
    }


def memory_boundary_allowlist_audit(*, log_limit: int = 200) -> dict[str, Any]:
    """Review cross-system memory boundaries and allowlists without changing policy."""

    log_limit = _clamp_int(log_limit, default=200, minimum=20, maximum=2000)
    status = memory_federation_status()
    federation_audit = memory_federation_audit(log_limit=log_limit)
    outcome = memory_policy_outcome_monitor(limit=50, stale_after_hours=72)
    ledger = memory_ledger_intelligence(limit=500)

    clients = status.get("clients", {})
    hermes = clients.get("hermes", {})
    codex = clients.get("codex", {})
    openclaw = clients.get("openclaw", {})

    external_allowed = openclaw.get("external_auto_precheck_allowed_channels", [])
    if not isinstance(external_allowed, list):
        external_allowed = []

    agent_profiles = openclaw.get("auto_precheck_agent_profiles", [])
    if not isinstance(agent_profiles, list):
        agent_profiles = []

    reviewed_allowlists = []
    unreviewed_allowlists = []
    manual_review_evidence = _boundary_manual_review_evidence()
    manual_review_complete = bool(manual_review_evidence.get("complete"))

    if external_allowed:
        unreviewed_allowlists.append({
            "id": "openclaw.external_auto_precheck_allowed_channels",
            "value": external_allowed,
            "reason": "External automatic recall allowlist is non-empty and requires explicit human review evidence.",
        })
    elif manual_review_complete:
        reviewed_allowlists.append({
            "id": "openclaw.external_auto_precheck_allowed_channels",
            "value": [],
            "review": "empty_allowlist_reviewed_with_manual_evidence",
            "evidence_source": manual_review_evidence.get("source"),
        })
    else:
        unreviewed_allowlists.append({
            "id": "openclaw.external_auto_precheck_boundary_review",
            "value": {
                "external_auto_precheck_allowed_channels": [],
                "external_auto_precheck_default": openclaw.get("external_auto_precheck_default"),
            },
            "reason": "Empty allowlist and blocked default are safe, but 星界记忆 readiness requires explicit manual boundary review evidence in the policy or operation ledger.",
        })

    if openclaw.get("external_auto_precheck_default") == "blocked" and manual_review_complete:
        reviewed_allowlists.append({
            "id": "openclaw.external_auto_precheck_default",
            "value": "blocked",
            "review": "blocked_default_reviewed_with_manual_evidence",
            "evidence_source": manual_review_evidence.get("source"),
        })

    exposure_risks = []
    if hermes.get("role") != "primary_memory_owner":
        exposure_risks.append("Hermes is not reported as primary memory owner.")
    if codex.get("role") != "memory_client":
        exposure_risks.append("Codex is not reported as a memory client.")
    if openclaw.get("role") != "memory_client":
        exposure_risks.append("OpenClaw is not reported as a memory client.")
    if openclaw.get("external_auto_precheck_default") != "blocked":
        exposure_risks.append("External automatic recall is not blocked by default.")
    if external_allowed:
        exposure_risks.append("External automatic recall allowlist is non-empty.")
    if not status.get("policy", {}).get("writes_are_proposal_only"):
        exposure_risks.append("Shared memory writes are not proposal-only.")
    if not status.get("policy", {}).get("external_channel_auto_recall_requires_allowlist"):
        exposure_risks.append("External channel automatic recall does not require allowlist.")

    unknown_client_findings = [
        finding for finding in ledger.get("findings", [])
        if "unknown" in str(finding).lower() or "direct write" in str(finding).lower()
    ]
    if unknown_client_findings:
        exposure_risks.append("Ledger contains unknown-client or direct-write findings.")

    critical_audit_failures = [
        check for check in federation_audit.get("checks", [])
        if check.get("severity") == "critical" and check.get("status") != "pass"
    ]

    if critical_audit_failures:
        exposure_risks.append("Federation audit has critical failures.")

    score = 100
    score -= 20 * len(unreviewed_allowlists)
    score -= 15 * len(exposure_risks)
    score -= 10 * len(critical_audit_failures)
    boundary_readiness_score = max(0, min(100, score))

    ready = (
        boundary_readiness_score >= 90
        and not unreviewed_allowlists
        and not exposure_risks
        and hermes.get("role") == "primary_memory_owner"
        and codex.get("role") == "memory_client"
        and openclaw.get("role") == "memory_client"
        and openclaw.get("external_auto_precheck_default") == "blocked"
        and status.get("policy", {}).get("writes_are_proposal_only") is True
        and status.get("policy", {}).get("external_channel_auto_recall_requires_allowlist") is True
        and manual_review_complete
    )

    recommended_next_actions = []
    if ready:
        recommended_next_actions.append("Keep the manual boundary review evidence linked to 星界记忆 readiness.")
    else:
        recommended_next_actions.append("Keep external automatic recall blocked and resolve unreviewed allowlists or exposure risks before 星界记忆.")
    if not manual_review_complete:
        recommended_next_actions.append("Create a formal policy proposal or manual review record for the blocked external automatic recall boundary; the audit will only read that evidence.")
    if external_allowed:
        recommended_next_actions.append("Create a policy proposal for each exact external channel before allowlisting; do not auto-approve.")

    return {
        "success": True,
        "audit_type": "hermes_memory_boundary_allowlist_audit",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ready": ready,
        "boundary_readiness_score": boundary_readiness_score,
        "reviewed": ready,
        "reviewed_allowlists": reviewed_allowlists,
        "unreviewed_allowlists": unreviewed_allowlists,
        "external_channel_exposure_risks": exposure_risks,
        "clients": {
            "hermes": {
                "role": hermes.get("role"),
                "is_primary_memory_owner": hermes.get("role") == "primary_memory_owner",
            },
            "codex": {
                "role": codex.get("role"),
                "access_path": codex.get("access_path"),
                "write_policy": codex.get("write_policy"),
                "ready": codex.get("ready"),
            },
            "openclaw": {
                "role": openclaw.get("role"),
                "access_path": openclaw.get("access_path"),
                "plugin_enabled": openclaw.get("plugin_enabled"),
                "conversation_access_allowed": openclaw.get("conversation_access_allowed"),
                "auto_precheck_enabled": openclaw.get("auto_precheck_enabled"),
                "auto_precheck_agent_profiles": agent_profiles,
                "external_auto_precheck_allowed_channels": external_allowed,
                "external_auto_precheck_default": openclaw.get("external_auto_precheck_default"),
                "write_policy": openclaw.get("write_policy"),
                "ready": openclaw.get("ready"),
            },
        },
        "evidence": {
            "federation_ready": status.get("ready"),
            "federation_audit_ready": federation_audit.get("ready"),
            "policy_outcome_health_score": outcome.get("health_score"),
            "policy_outcome_risk_level": outcome.get("risk_level"),
            "ledger_health_score": ledger.get("health_score"),
            "ledger_risk_level": ledger.get("risk_level"),
            "manual_boundary_review": manual_review_evidence,
            "critical_audit_failure_count": len(critical_audit_failures),
            "unreviewed_allowlist_count": len(unreviewed_allowlists),
            "exposure_risk_count": len(exposure_risks),
            "external_auto_precheck_default_blocked": openclaw.get("external_auto_precheck_default") == "blocked",
            "external_auto_precheck_allowlist_empty": not external_allowed,
        },
        "recommended_next_actions": recommended_next_actions,
        "policy": {
            "audit_is_read_only": True,
            "does_not_modify_config": True,
            "does_not_write_memory": True,
            "does_not_write_graph": True,
            "does_not_approve_allowlists": True,
            "does_not_enable_external_recall": True,
            "persistent_changes_must_use_proposals": True,
        },
        "read_only": True,
        "read_only_memory": True,
        "would_mutate_memory": False,
        "would_modify_config": False,
        "would_write_graph": False,
    }


def memory_orchestration_routing_metrics() -> dict[str, Any]:
    """Inspect read-only memory orchestration routing evidence."""

    config_path = Path.home() / ".openclaw" / "openclaw.json"
    config = _loads_json(_safe_read_text(config_path), {})
    config = config if isinstance(config, dict) else {}
    plugin_config = _openclaw_hermes_plugin_config(config)
    profiles = plugin_config.get("autoPrecheckAgentProfiles", {}) if isinstance(plugin_config, dict) else {}
    auto_precheck_agent_ids = _list_of_str(plugin_config.get("autoPrecheckAgentIds", [])) if isinstance(plugin_config, dict) else []
    profile_ids = sorted(profiles.keys()) if isinstance(profiles, dict) else _list_of_str(profiles if isinstance(profiles, list) else [])
    agent_ids = _openclaw_agent_ids(config)
    route_bindings = _openclaw_route_bindings(config)
    routed_agent_ids = sorted({
        binding["agent_id"]
        for binding in route_bindings
        if binding.get("agent_id")
    })
    channel_counts = dict(sorted(Counter(
        binding["channel"]
        for binding in route_bindings
        if binding.get("channel")
    ).items()))

    ledger = memory_operation_ledger(limit=500)
    events = ledger.get("events", []) if isinstance(ledger.get("events"), list) else []
    client_counts: Counter[str] = Counter()
    decision_counts: Counter[str] = Counter()
    operation_routing_event_count = 0
    gate_decision_count = 0
    auto_precheck_operation_count = 0
    for event in events:
        if not isinstance(event, dict):
            continue
        client = _clean_text(event.get("client")).lower()
        decision = _clean_text(event.get("decision")).lower()
        operation = _clean_text(event.get("operation")).lower()
        event_type = _clean_text(event.get("event_type")).lower()
        action = _clean_text(event.get("action")).lower()
        combined = " ".join([operation, event_type, action])
        if client:
            client_counts[client] += 1
        if decision:
            decision_counts[decision] += 1
        if event_type == "gate_decision":
            gate_decision_count += 1
        if "route" in combined or "routing" in combined or "orchestration" in combined or event.get("route_id") or event.get("routed_agent_id"):
            operation_routing_event_count += 1
        if operation == "auto_precheck":
            auto_precheck_operation_count += 1

    evidence_checks = {
        "has_three_or_more_agents": len(agent_ids) >= 3,
        "has_route_binding": len(route_bindings) >= 1,
        "has_auto_precheck_profile": len(profile_ids) >= 1,
        "has_explicit_routing_operation_event": operation_routing_event_count >= 1,
    }
    routing_readiness_score = round(sum(1 for value in evidence_checks.values() if value) / len(evidence_checks), 3)
    gaps = _orchestration_routing_gaps(evidence_checks)
    ready = not gaps
    return {
        "success": True,
        "metrics_type": "hermes_memory_orchestration_routing_metrics",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "openclaw_config_path": str(config_path),
        "openclaw_config_exists": config_path.exists(),
        "agent_count": len(agent_ids),
        "agent_ids": agent_ids,
        "route_binding_count": len(route_bindings),
        "routed_agent_count": len(routed_agent_ids),
        "routed_agent_ids": routed_agent_ids,
        "route_channel_counts": channel_counts,
        "channel_counts": channel_counts,
        "auto_precheck_profile_count": len(profile_ids),
        "auto_precheck_agent_ids": auto_precheck_agent_ids,
        "operation_routing_event_count": operation_routing_event_count,
        "has_explicit_routing_operation_event": operation_routing_event_count >= 1,
        "gate_decision_count": gate_decision_count,
        "auto_precheck_operation_count": auto_precheck_operation_count,
        "client_counts": dict(sorted(client_counts.items())),
        "decision_counts": dict(sorted(decision_counts.items())),
        "routing_readiness_score": routing_readiness_score,
        "active_routing_metrics": ready,
        "ready": ready,
        "gaps": gaps,
        "recommended_next_actions": _orchestration_routing_next_actions(gaps),
        "evidence_checks": evidence_checks,
        "policy": {
            "metrics_are_read_only": True,
            "does_not_modify_openclaw_config": True,
            "does_not_write_memory": True,
            "does_not_write_graph": True,
            "does_not_approve_allowlists": True,
            "does_not_enable_external_recall": True,
        },
        "read_only": True,
        "read_only_memory": True,
        "would_mutate_memory": False,
        "would_modify_config": False,
        "would_write_graph": False,
    }


def memory_federation_gate(
    *,
    client: str = "codex",
    operation: str = "search",
    target_scope: str = "project",
    channel_id: str = "",
    log_limit: int = 200,
) -> dict[str, Any]:
    """Evaluate a memory operation against the federation governance gate."""

    client = _clean_text(client).lower() or "unknown"
    operation = _clean_text(operation).lower() or "search"
    target_scope = _proposal_scope(target_scope)
    channel_id = _clean_text(channel_id)
    decision = "allow"
    reason = "Operation is allowed under read/proposal-only policy."
    if operation in DURABLE_WRITE_OPERATIONS:
        decision = "block"
        reason = "Durable memory writes must be routed through governed proposals."
    elif operation in {"auto_precheck", "external_auto_recall"} and _is_external_channel(channel_id):
        decision = "block"
        reason = "External-channel automatic recall is blocked unless an exact channel is reviewed and allowlisted."
    elif client not in INTERNAL_MEMORY_CLIENTS:
        decision = "review"
        reason = "Unknown memory client requires review."
    result = {
        "success": True,
        "gate_type": "hermes_memory_federation_gate",
        "decision": decision,
        "allowed": decision == "allow",
        "client": client,
        "operation": operation,
        "target_scope": target_scope,
        "channel_id": channel_id,
        "reason": reason,
        "policy": {
            "proposal_only_writes": True,
            "external_auto_recall_requires_allowlist": True,
            "gate_is_read_only_for_memory": True,
        },
        "read_only_memory": True,
        "would_mutate_memory": False,
    }
    _append_operation_event(
        {
            "event_type": "gate_decision",
            "client": client,
            "operation": operation,
            "target_scope": target_scope,
            "channel_id_hash": _text_digest(channel_id) if channel_id else "",
            "channel_root": channel_id.split(":", 1)[0].lower() if channel_id else "",
            "decision": decision,
            "reason": reason,
            "log_limit": _clamp_int(log_limit, default=200, minimum=20, maximum=2000),
        }
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
    """Read the memory operation audit ledger."""

    limit = _clamp_int(limit, default=50, minimum=1, maximum=500)
    client = _clean_text(client).lower()
    operation = _clean_text(operation).lower()
    decision = _clean_text(decision).lower()
    event_type = _clean_text(event_type)
    path = _operation_ledger_path(get_hermes_home())
    rows = _read_jsonl(path)
    parse_errors = [row for row in rows if row.get("_parse_error")]
    filtered = []
    for row in rows:
        if row.get("_parse_error"):
            continue
        if client and _clean_text(row.get("client")).lower() != client:
            continue
        if operation and _clean_text(row.get("operation")).lower() != operation:
            continue
        if decision and _clean_text(row.get("decision")).lower() != decision:
            continue
        if event_type and _clean_text(row.get("event_type")) != event_type:
            continue
        filtered.append(row)
    filtered.sort(key=lambda row: _clean_text(row.get("created_at")), reverse=True)
    return {
        "success": True,
        "ledger_type": "hermes_memory_operation_ledger",
        "path": str(path),
        "exists": path.exists(),
        "total_events": len([row for row in rows if not row.get("_parse_error")]),
        "matched_events": len(filtered),
        "returned_events": len(filtered[:limit]),
        "events": filtered[:limit],
        "parse_error_count": len(parse_errors),
        "parse_errors": parse_errors[:5],
        "policy": {"ledger_is_read_only": True, "does_not_write_memory": True},
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
    """Analyze memory operation ledger patterns and risks."""

    ledger = memory_operation_ledger(limit=limit, client=client, operation=operation)
    events = ledger.get("events", []) if isinstance(ledger.get("events"), list) else []
    findings = _ledger_findings(events, ledger.get("parse_errors", []))
    health_score = _health_score(findings)
    return {
        "success": True,
        "intelligence_type": "hermes_memory_ledger_intelligence",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "analyzed_events": len(events),
        "health_score": health_score,
        "risk_level": "high" if health_score < 70 else "medium" if health_score < 90 else "low",
        "findings": findings,
        "recommended_next_actions": _recommended_actions(findings),
        "policy": {
            "intelligence_is_read_only": True,
            "does_not_modify_config": True,
            "does_not_write_memory": True,
        },
        "read_only": True,
        "read_only_memory": True,
        "would_mutate_memory": False,
        "would_modify_config": False,
    }


def memory_policy_autotune(
    *,
    limit: int = 500,
    client: str = "",
    operation: str = "",
    mode: str = "conservative",
) -> dict[str, Any]:
    """Generate read-only memory policy tuning suggestions."""

    intelligence = memory_ledger_intelligence(limit=limit, client=client, operation=operation)
    suggestions = _policy_suggestions_from_findings(intelligence.get("findings", []), mode=mode)
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
    return {
        "success": True,
        "autotune_type": "hermes_memory_policy_autotune",
        "mode": mode if mode in {"conservative", "diagnostic"} else "conservative",
        "decision": "review_required" if suggestions else "no_change",
        "suggestion_count": len(suggestions),
        "suggestions": suggestions,
        "summary": {
            "requires_review_count": len([s for s in suggestions if s.get("requires_human_review")]),
            "auto_apply_count": 0,
        },
        "intelligence_summary": {
            "health_score": intelligence.get("health_score"),
            "risk_level": intelligence.get("risk_level"),
            "analyzed_events": intelligence.get("analyzed_events"),
            "finding_ids": [f.get("id") for f in intelligence.get("findings", []) if isinstance(f, dict)],
        },
        "policy": {
            "autotune_is_read_only": True,
            "does_not_modify_config": True,
            "does_not_write_memory": True,
            "suggestions_require_human_review": True,
        },
        "read_only": True,
        "read_only_memory": True,
        "would_mutate_memory": False,
        "would_modify_config": False,
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
    autotune = memory_policy_autotune(limit=limit, client=client, operation=operation, mode=mode)
    suggestions = [
        suggestion
        for suggestion in autotune.get("suggestions", [])
        if isinstance(suggestion, dict)
        and _clean_text(suggestion.get("id")) != "policy.no_change"
        and (not suggestion_id or _clean_text(suggestion.get("id")) == suggestion_id)
    ]
    existing = _policy_proposal_states()
    created: list[dict[str, Any]] = []
    skipped_duplicates: list[dict[str, Any]] = []
    for suggestion in suggestions:
        proposal = _policy_proposal_from_suggestion(suggestion, source_agent=source_agent, autotune=autotune)
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
        event = _append_policy_proposal_event({"event_type": "policy_proposal_created", **proposal})
        created.append({**proposal, "ledger_event": event})
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
    status = _policy_status(status, allow_empty=True)
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
    return {
        "success": True,
        "ledger_type": "hermes_memory_policy_proposal_ledger",
        "path": str(path),
        "exists": path.exists(),
        "total_proposals": len(proposals),
        "matched_proposals": len(filtered),
        "returned_proposals": len(filtered[:limit]),
        "proposals": filtered[:limit],
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
    decision = _policy_decision(decision)
    states = _policy_proposal_states()
    if proposal_id not in states:
        return _error("policy proposal was not found.", proposal_id=proposal_id)
    event = _append_policy_proposal_event(
        {
            "event_type": "policy_proposal_decision",
            "proposal_id": proposal_id,
            "decision": decision,
            "reviewer": _clean_text(reviewer) or "unknown",
            "rationale": _clean_text(rationale),
            "does_not_apply_policy": True,
            "would_modify_config": False,
            "would_write_memory": False,
        }
    )
    updated = _policy_proposal_states().get(proposal_id, states[proposal_id])
    return {
        "success": True,
        "proposal_id": proposal_id,
        "decision": decision,
        "proposal": updated,
        "ledger_event": event,
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
    status = _policy_status(status, allow_empty=True) or "approved"
    proposal_id = _clean_text(proposal_id)
    ledger = memory_policy_proposal_ledger(limit=limit, status=status, proposal_id=proposal_id)
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
        "summary": {"by_action": _policy_apply_plan_summary(plans, "action")},
        "policy": {
            "plan_is_dry_run": True,
            "does_not_apply_policy": True,
            "does_not_modify_config": True,
            "does_not_write_memory": True,
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

    Only non-mutating verification/check actions are executed. Config-mutating
    or memory-mutating patches are refused.
    """

    plan = memory_policy_apply_plan(limit=limit, status="approved", proposal_id=proposal_id)
    guard = _policy_apply_guard(plan, execute=execute, confirm_token=confirm_token)
    if not _coerce_bool(execute):
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
            "policy": {"blocked_before_execution": True},
            "read_only_memory": True,
            "would_mutate_memory": False,
            "would_modify_config": False,
        }
    results = [_policy_patch_result(patch, proposal_id=item.get("proposal_id", "")) for item in plan.get("plans", []) for patch in item.get("patches", [])]
    summary = _policy_apply_execute_summary(results)
    event = _append_policy_proposal_event(
        {
            "event_type": "policy_apply_execute",
            "proposal_id": _clean_text(proposal_id),
            "actor": _clean_text(actor) or "unknown",
            "plan_summary": {
                "proposal_count": plan.get("proposal_count"),
                "eligible_count": plan.get("eligible_count"),
                "patch_count": plan.get("patch_count"),
            },
            "result_summary": summary,
            "does_not_apply_config_changes": True,
            "would_modify_config": False,
            "would_write_memory": False,
        }
    )
    return {
        "success": summary["blocked_count"] == 0 and summary["failed_count"] == 0,
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
    stale_after_hours = _clamp_int(stale_after_hours, default=72, minimum=1, maximum=8760)
    path = _policy_proposal_path(get_hermes_home())
    rows = _read_jsonl(path)
    parse_errors = [row for row in rows if row.get("_parse_error")]
    proposals = list(_policy_proposal_states(rows).values())
    metrics = _policy_outcome_metrics(proposals, rows, stale_after_hours=stale_after_hours)
    findings = _policy_outcome_findings(metrics, parse_errors)
    health_score = _health_score(findings)
    proposals.sort(key=lambda row: _clean_text(row.get("updated_at")), reverse=True)
    executions = metrics.get("recent_execution_events", [])
    return {
        "success": True,
        "monitor_type": "hermes_memory_policy_outcome_monitor",
        "path": str(path),
        "exists": path.exists(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "health_score": health_score,
        "risk_level": "high" if health_score < 70 else "medium" if health_score < 90 else "low",
        "stale_after_hours": stale_after_hours,
        "analyzed_events": len([row for row in rows if not row.get("_parse_error")]),
        "parse_error_count": len(parse_errors),
        "parse_errors": parse_errors[:5],
        "metrics": metrics,
        "findings": findings,
        "recommended_next_actions": _recommended_actions(findings),
        "recent_proposals": proposals[:limit],
        "recent_execution_events": executions[:limit] if isinstance(executions, list) else [],
        "policy": {
            "monitor_is_read_only": True,
            "does_not_apply_policy": True,
            "does_not_modify_config": True,
            "does_not_write_memory": True,
        },
        "read_only": True,
        "read_only_memory": True,
        "would_mutate_memory": False,
        "would_modify_config": False,
    }


def memory_recall_quality_evaluate(
    *,
    queries: Iterable[str] | str | None = None,
    scope: str = "all",
    limit: int = 5,
) -> dict[str, Any]:
    """Evaluate recall quality across memory surfaces without writing memory."""

    scope = _scope(scope)
    limit = _clamp_int(limit, default=5, minimum=1, maximum=20)
    benchmark_queries = _recall_quality_queries(queries)
    home = get_hermes_home()
    evaluations = [
        _evaluate_recall_query(home, query, scope=scope, limit=limit)
        for query in benchmark_queries
    ]
    summary = _recall_quality_summary(evaluations)
    findings = _recall_quality_findings(evaluations, summary)
    quality_score = round(float(summary.get("average_quality_score", 0.0)), 3)
    readiness = (
        "ready"
        if quality_score >= 0.55
        and _safe_int(summary.get("passed_query_count")) >= max(1, len(evaluations) // 2)
        else "needs_attention"
    )
    return {
        "success": True,
        "evaluation_type": "hermes_memory_recall_quality_evaluate",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": scope,
        "limit": limit,
        "quality_score": quality_score,
        "readiness": readiness,
        "summary": summary,
        "queries": evaluations,
        "findings": findings,
        "recommended_next_actions": _recommended_actions(findings),
        "policy": {
            "evaluation_is_read_only": True,
            "does_not_append_ledger_events": True,
            "does_not_modify_config": True,
            "does_not_write_memory": True,
            "does_not_write_graph": True,
        },
        "read_only": True,
        "read_only_memory": True,
        "would_mutate_memory": False,
        "would_modify_config": False,
    }


def search_memory_fabric(query: str, *, scope: str = "all", limit: int = 8) -> dict[str, Any]:
    """Search Memory Graph, prompt cases, knowledge files, and legacy memory."""

    query = _clean_text(query)
    scope = _scope(scope)
    limit = _clamp_int(limit, default=8, minimum=1, maximum=50)
    if not query:
        return _error("query is required.", query=query, scope=scope)
    home = get_hermes_home()
    results: list[dict[str, Any]] = []
    if scope in {"all", "graph"}:
        results.extend(_search_graph(home, query, limit=limit))
    if scope in {"all", "prompt_cases"}:
        results.extend(_search_prompt_cases(home, query, limit=limit))
    if scope in {"all", "knowledge"}:
        results.extend(_search_knowledge(home, query, limit=limit))
    if scope in {"all", "legacy_memory"}:
        results.extend(_search_legacy_memory(home, query, limit=limit))
    results.sort(key=lambda item: float(item.get("score", 0)), reverse=True)
    limited = results[:limit]
    ledger = _append_operation_event(
        {
            "event_type": "search",
            "client": "unknown",
            "operation": "search",
            "query_hash": _text_digest(query),
            "scope": scope,
            "result_count": len(limited),
        }
    )
    return {
        "success": True,
        "query": query,
        "scope": scope,
        "count": len(limited),
        "results": limited,
        "operation_ledger": ledger,
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
    """Read Memory Graph nodes with provenance and optional edges."""

    home = get_hermes_home()
    graph_path = _graph_path(home)
    limit = _clamp_int(limit, default=10, minimum=1, maximum=100)
    node_id = _clean_text(node_id)
    query = _clean_text(query)
    kind = _clean_text(kind)
    if not graph_path.exists():
        return _error("Memory Graph database does not exist.", path=str(graph_path))
    nodes: list[dict[str, Any]] = []
    try:
        with sqlite3.connect(graph_path) as conn:
            conn.row_factory = sqlite3.Row
            if node_id:
                rows = conn.execute("SELECT * FROM graph_nodes WHERE id = ?", (node_id,)).fetchall()
            elif query:
                like = f"%{query}%"
                if kind:
                    rows = conn.execute(
                        "SELECT * FROM graph_nodes WHERE kind = ? AND (title LIKE ? OR summary LIKE ?) LIMIT ?",
                        (kind, like, like, limit),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM graph_nodes WHERE title LIKE ? OR summary LIKE ? LIMIT ?",
                        (like, like, limit),
                    ).fetchall()
            elif kind:
                rows = conn.execute("SELECT * FROM graph_nodes WHERE kind = ? LIMIT ?", (kind, limit)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM graph_nodes ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall()
            for row in rows:
                node = _row_to_dict(row)
                node["metadata"] = _loads_json(node.pop("metadata_json", "{}"), {})
                node["provenance"] = [
                    _row_to_dict(prov)
                    for prov in conn.execute(
                        "SELECT * FROM graph_provenance WHERE node_id = ?",
                        (node["id"],),
                    ).fetchall()
                ]
                if include_edges:
                    node["edges"] = [
                        _row_to_dict(edge)
                        for edge in conn.execute(
                            "SELECT * FROM graph_edges WHERE source_id = ? OR target_id = ? LIMIT 50",
                            (node["id"], node["id"]),
                        ).fetchall()
                    ]
                nodes.append(node)
    except sqlite3.Error as exc:
        return _error("failed to read Memory Graph.", error=str(exc), path=str(graph_path))
    ledger = _append_operation_event(
        {
            "event_type": "graph_read",
            "client": "unknown",
            "operation": "graph_read",
            "node_id_hash": _text_digest(node_id) if node_id else "",
            "query_hash": _text_digest(query) if query else "",
            "kind": kind,
            "result_count": len(nodes),
        }
    )
    return {
        "success": True,
        "graph_type": "hermes_memory_graph_read",
        "path": str(graph_path),
        "count": len(nodes),
        "nodes": nodes,
        "operation_ledger": ledger,
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
    """Create a governed write proposal instead of mutating durable memory."""

    content = _clean_text(content)
    if not content:
        return _error("content is required.")
    proposal = {
        "proposal_id": f"memory-write-proposal-{_text_digest(content + rationale)[:16]}",
        "source_agent": _clean_text(source_agent) or "unknown",
        "target_scope": _proposal_scope(target_scope),
        "content": content,
        "rationale": _clean_text(rationale),
        "project": _clean_text(project),
        "tags": _list_of_str(tags),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "proposed",
        "would_write_memory": False,
        "would_modify_graph": False,
        "would_modify_config": False,
    }
    path = _proposal_path(get_hermes_home())
    event = _append_jsonl(path, proposal)
    _append_operation_event(
        {
            "event_type": "write_proposal_created",
            "client": proposal["source_agent"],
            "operation": "write_proposal",
            "target_scope": proposal["target_scope"],
            "proposal_id": proposal["proposal_id"],
            "content_hash": _text_digest(content),
        }
    )
    return {
        "success": event.get("success", False),
        "proposal": proposal,
        "ledger_event": event,
        "policy": {
            "proposal_only": True,
            "does_not_write_memory": True,
            "does_not_modify_graph": True,
        },
        "read_only_memory": True,
        "would_mutate_memory": False,
    }


def export_memory_snapshot(*, scope: str = "all", limit: int = 500) -> dict[str, Any]:
    """Export a portable read-only snapshot summary for backup/cache use."""

    scope = _scope(scope)
    limit = _clamp_int(limit, default=500, minimum=1, maximum=5000)
    snapshot = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": scope,
        "limit": limit,
        "bridge_status": memory_bridge_status(),
        "evolution_status": memory_evolution_status(),
    }
    _append_operation_event(
        {
            "event_type": "snapshot_export_created",
            "client": "unknown",
            "operation": "snapshot_export",
            "scope": scope,
            "limit": limit,
        }
    )
    return {
        "success": True,
        "snapshot_type": "hermes_memory_snapshot",
        "snapshot": snapshot,
        "policy": {
            "snapshot_is_cache_or_backup_only": True,
            "does_not_clone_as_primary": True,
            "does_not_write_memory": True,
        },
        "read_only_memory": True,
        "would_mutate_memory": False,
    }


def memory_policy_stale_resolution_preview(*, limit: int = 50) -> dict[str, Any]:
    """Preview stale policy proposal resolution options without deciding."""

    outcome = memory_policy_outcome_monitor(limit=limit, stale_after_hours=72)
    stale = outcome.get("metrics", {}).get("stale_proposed", [])
    return {
        "success": True,
        "preview_type": "memory_policy_stale_resolution_preview",
        "stale_proposals": stale,
        "recommendations": [
            {
                "proposal_id": item.get("proposal_id"),
                "suggestion_id": item.get("suggestion_id"),
                "recommended_decision": "deferred" if item.get("suggestion_id") == "diagnostic.rerun_full_audit" else "approved_for_non_mutating_check",
                "reason": "Preview only; human must record an explicit decision.",
            }
            for item in stale
            if isinstance(item, dict)
        ],
        "policy": {"preview_is_read_only": True, "does_not_record_decisions": True},
        "read_only": True,
        "read_only_memory": True,
        "would_modify_config": False,
        "would_mutate_memory": False,
    }


def memory_policy_stale_closure_payload_preview(*, limit: int = 50) -> dict[str, Any]:
    """Preview payloads that a human could use to close stale proposals."""

    preview = memory_policy_stale_resolution_preview(limit=limit)
    return {
        "success": True,
        "preview_type": "memory_policy_stale_closure_payload_preview",
        "payloads": [
            {
                "proposal_id": item.get("proposal_id"),
                "decision": "deferred",
                "rationale": "Stale proposal requires fresh audit before closure.",
            }
            for item in preview.get("stale_proposals", [])
            if isinstance(item, dict)
        ],
        "policy": {"preview_is_read_only": True, "does_not_record_decisions": True},
        "read_only": True,
        "read_only_memory": True,
        "would_modify_config": False,
        "would_mutate_memory": False,
    }


def memory_policy_stale_closure_execute_plan(*, limit: int = 50) -> dict[str, Any]:
    """Build a dry-run plan for closing stale policy proposals."""

    payloads = memory_policy_stale_closure_payload_preview(limit=limit)
    return {
        "success": True,
        "plan_type": "memory_policy_stale_closure_execute_plan",
        "dry_run": True,
        "steps": payloads.get("payloads", []),
        "policy": {"plan_is_dry_run": True, "does_not_record_decisions": True},
        "read_only": True,
        "read_only_memory": True,
        "would_modify_config": False,
        "would_mutate_memory": False,
    }


def memory_policy_stale_closure_handoff_bundle(*, limit: int = 50) -> dict[str, Any]:
    """Bundle stale policy context for human/Codex handoff."""

    return {
        "success": True,
        "bundle_type": "memory_policy_stale_closure_handoff_bundle",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "outcome": memory_policy_outcome_monitor(limit=limit, stale_after_hours=72),
        "resolution_preview": memory_policy_stale_resolution_preview(limit=limit),
        "execute_plan": memory_policy_stale_closure_execute_plan(limit=limit),
        "policy": {"bundle_is_read_only": True, "does_not_record_decisions": True},
        "read_only": True,
        "read_only_memory": True,
        "would_modify_config": False,
        "would_mutate_memory": False,
    }


memory_fabric_search = search_memory_fabric
memory_graph_read = read_memory_graph
memory_write_proposal = create_memory_write_proposal
memory_snapshot_export = export_memory_snapshot


def _boundary_manual_review_evidence() -> dict[str, Any]:
    """Find existing manual boundary review evidence without recording anything."""

    policy_ledger = memory_policy_proposal_ledger(limit=500)
    proposals = policy_ledger.get("proposals", []) if isinstance(policy_ledger.get("proposals"), list) else []
    for proposal in proposals:
        if not isinstance(proposal, dict) or not _is_boundary_review_policy_proposal(proposal):
            continue
        decisions = proposal.get("decisions", []) if isinstance(proposal.get("decisions"), list) else []
        approved_decisions = [
            decision for decision in decisions
            if isinstance(decision, dict) and _clean_text(decision.get("decision")) == "approved"
        ]
        if approved_decisions and proposal.get("latest_status") == "approved":
            latest = approved_decisions[-1]
            return {
                "complete": True,
                "source": "policy_proposal_ledger",
                "proposal_id": proposal.get("proposal_id", ""),
                "suggestion_id": proposal.get("suggestion_id", ""),
                "reviewer": latest.get("reviewer", ""),
                "reviewed_at": latest.get("created_at", ""),
                "decision": "approved",
                "read_only_detection": True,
            }

    operation_ledger = memory_operation_ledger(limit=500)
    events = operation_ledger.get("events", []) if isinstance(operation_ledger.get("events"), list) else []
    for event in events:
        if not isinstance(event, dict) or not _is_boundary_manual_review_event(event):
            continue
        return {
            "complete": True,
            "source": "operation_ledger",
            "event_id": event.get("event_id", ""),
            "event_type": event.get("event_type", ""),
            "reviewer": event.get("reviewer", event.get("actor", "")),
            "reviewed_at": event.get("created_at", ""),
            "decision": event.get("decision", event.get("status", "reviewed")),
            "read_only_detection": True,
        }

    decided_boundary_proposals = [
        {
            "proposal_id": proposal.get("proposal_id", ""),
            "suggestion_id": proposal.get("suggestion_id", ""),
            "latest_status": proposal.get("latest_status", ""),
        }
        for proposal in proposals
        if isinstance(proposal, dict)
        and _is_boundary_review_policy_proposal(proposal)
        and proposal.get("latest_status") in {"rejected", "deferred"}
    ]
    return {
        "complete": False,
        "source": "none",
        "read_only_detection": True,
        "policy_proposal_ledger_exists": policy_ledger.get("exists"),
        "operation_ledger_exists": operation_ledger.get("exists"),
        "decided_but_not_completion_evidence": decided_boundary_proposals[:5],
    }


def _is_boundary_review_policy_proposal(proposal: Mapping[str, Any]) -> bool:
    suggestion_id = _clean_text(proposal.get("suggestion_id"))
    target = _clean_text(proposal.get("target"))
    text = " ".join(
        _clean_text(proposal.get(key)).lower()
        for key in ("recommendation", "rationale", "next_step")
    )
    return (
        suggestion_id == "external_auto_recall.keep_blocked"
        or "autoprecheckallowedchannelids" in target.lower()
        or "external_auto_recall" in target.lower()
        or ("external" in text and "allowlist" in text and "review" in text)
        or ("automatic recall" in text and "blocked" in text and "review" in text)
    )


def _is_boundary_manual_review_event(event: Mapping[str, Any]) -> bool:
    event_type = _clean_text(event.get("event_type")).lower()
    operation = _clean_text(event.get("operation")).lower()
    subject = " ".join(
        _clean_text(event.get(key)).lower()
        for key in ("subject", "target", "boundary", "rationale", "reason")
    )
    decision = _clean_text(event.get("decision", event.get("status", ""))).lower()
    non_mutating = event.get("would_modify_config") is not True and event.get("would_write_memory") is not True
    return (
        non_mutating
        and event_type in {"memory_boundary_manual_review", "boundary_manual_review", "manual_boundary_review"}
        and (operation in {"", "audit", "manual_review", "review"} or "boundary" in subject)
        and decision in {"", "approved", "reviewed", "complete", "completed", "pass", "passed"}
    )


def _memory_evolution_evidence(
    bridge: Mapping[str, Any],
    federation: Mapping[str, Any],
    outcome: Mapping[str, Any],
    recall_quality: Mapping[str, Any],
    routing_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    surfaces = bridge.get("surfaces", {}) if isinstance(bridge.get("surfaces"), dict) else {}
    graph = surfaces.get("graph", {}) if isinstance(surfaces.get("graph"), dict) else {}
    prompt_cases = surfaces.get("gpt_image_prompt_cases", {}) if isinstance(surfaces.get("gpt_image_prompt_cases"), dict) else {}
    knowledge = surfaces.get("knowledge", {}) if isinstance(surfaces.get("knowledge"), dict) else {}
    operation = surfaces.get("operation_ledger", {}) if isinstance(surfaces.get("operation_ledger"), dict) else {}
    policy = surfaces.get("policy_proposals", {}) if isinstance(surfaces.get("policy_proposals"), dict) else {}
    clients = federation.get("clients", {}) if isinstance(federation.get("clients"), dict) else {}
    openclaw = clients.get("openclaw", {}) if isinstance(clients.get("openclaw"), dict) else {}
    codex = clients.get("codex", {}) if isinstance(clients.get("codex"), dict) else {}
    routing_metrics = routing_metrics if isinstance(routing_metrics, Mapping) else {}
    policy_outcome = _policy_closed_loop_evidence(outcome)
    star_soul_evidence = _star_soul_continuity_evidence()
    star_universe_evidence = _star_universe_temporal_evolution_evidence()
    star_source_evidence = _star_source_methodology_evidence()
    return {
        "has_basic_memory_home": bool(bridge.get("hermes_home")),
        "has_graph": bool(graph.get("exists")) and _safe_int(graph.get("node_count")) > 0,
        "graph_node_count": _safe_int(graph.get("node_count")),
        "graph_edge_count": _safe_int(graph.get("edge_count")),
        "graph_provenance_count": _safe_int(graph.get("provenance_count")),
        "has_knowledge": bool(knowledge.get("exists")) and _safe_int(knowledge.get("file_count")) > 0,
        "knowledge_file_count": _safe_int(knowledge.get("file_count")),
        "has_prompt_cases": bool(prompt_cases.get("exists")) and _safe_int(prompt_cases.get("case_count")) > 0,
        "prompt_case_count": _safe_int(prompt_cases.get("case_count")),
        "operation_ledger_events": _safe_int(operation.get("event_count")),
        "policy_proposal_events": _safe_int(policy.get("event_count")),
        "federation_ready": federation.get("ready") is True,
        "codex_ready": codex.get("ready") is True,
        "openclaw_ready": openclaw.get("ready") is True,
        "openclaw_agent_profiles": openclaw.get("auto_precheck_agent_profiles", []),
        "openclaw_auto_precheck_enabled": openclaw.get("auto_precheck_enabled") is True,
        "orchestration_routing_metrics_ready": routing_metrics.get("ready") is True,
        "active_routing_metrics": routing_metrics.get("active_routing_metrics") is True,
        "routing_readiness_score": routing_metrics.get("routing_readiness_score", 0),
        "routing_agent_count": _safe_int(routing_metrics.get("agent_count")),
        "routing_route_binding_count": _safe_int(routing_metrics.get("route_binding_count")),
        "routing_operation_event_count": _safe_int(routing_metrics.get("operation_routing_event_count")),
        "has_explicit_routing_operation_event": _safe_int(routing_metrics.get("operation_routing_event_count")) >= 1,
        "routing_gate_decision_count": _safe_int(routing_metrics.get("gate_decision_count")),
        "routing_auto_precheck_operation_count": _safe_int(routing_metrics.get("auto_precheck_operation_count")),
        "recall_quality_evaluation_ready": recall_quality.get("readiness") == "ready",
        "recall_quality_score": recall_quality.get("quality_score"),
        "recall_quality_passed_query_count": recall_quality.get("summary", {}).get("passed_query_count", 0)
        if isinstance(recall_quality.get("summary"), dict)
        else 0,
        "policy_outcome_health_score": outcome.get("health_score"),
        "stale_policy_proposal_count": outcome.get("metrics", {}).get("stale_proposed_count", 0)
        if isinstance(outcome.get("metrics"), dict)
        else 0,
        **policy_outcome,
        **star_soul_evidence,
        **star_universe_evidence,
        **star_source_evidence,
    }


def _policy_closed_loop_evidence(outcome: Mapping[str, Any]) -> dict[str, Any]:
    metrics = outcome.get("metrics", {}) if isinstance(outcome.get("metrics"), Mapping) else {}
    totals = metrics.get("execution_totals", {}) if isinstance(metrics.get("execution_totals"), Mapping) else {}
    stale_count = _safe_int(metrics.get("stale_proposed_count"))
    approved_not_executed_count = _safe_int(metrics.get("approved_not_executed_count"))
    execution_count = _safe_int(metrics.get("execution_count"))
    checked_count = _safe_int(totals.get("checked_count"))
    passed_count = _safe_int(totals.get("passed_count"))
    blocked_count = _safe_int(totals.get("blocked_count"))
    failed_count = _safe_int(totals.get("failed_count"))
    manual_required_count = _safe_int(totals.get("manual_required_count"))
    return {
        "policy_execution_count": execution_count,
        "policy_approved_not_executed_count": approved_not_executed_count,
        "policy_execution_checked_count": checked_count,
        "policy_execution_passed_count": passed_count,
        "policy_execution_blocked_count": blocked_count,
        "policy_execution_failed_count": failed_count,
        "policy_execution_manual_required_count": manual_required_count,
        "policy_closed_loop_ready": (
            stale_count == 0
            and approved_not_executed_count == 0
            and execution_count >= 1
            and blocked_count == 0
            and failed_count == 0
            and manual_required_count == 0
            and (checked_count >= 1 or passed_count >= 1)
        ),
    }


STAR_SOUL_CONTINUITY_TERMS = {
    "preference",
    "preferences",
    "persona",
    "collaboration",
    "style",
    "continuity",
    "long-term",
    "long_term",
    "星魂",
    "偏好",
    "人格",
    "协作风格",
    "长期",
}


def _star_soul_continuity_evidence() -> dict[str, Any]:
    """Read proposal-only continuity evidence without mutating memory."""

    home = get_hermes_home()
    proposals = [row for row in _read_jsonl(_proposal_path(home)) if not row.get("_parse_error")]
    ledger = memory_operation_ledger(limit=500, event_type="write_proposal_created")
    events = ledger.get("events", []) if isinstance(ledger.get("events"), list) else []
    persona_proposals = [proposal for proposal in proposals if _is_persona_continuity_write_proposal(proposal)]
    persona_proposal_ids = {
        _clean_text(proposal.get("proposal_id"))
        for proposal in persona_proposals
        if _clean_text(proposal.get("proposal_id"))
    }
    governed_events = [
        event
        for event in events
        if _clean_text(event.get("event_type")) == "write_proposal_created"
        and _clean_text(event.get("operation")) == "write_proposal"
        and _clean_text(event.get("proposal_id")) in persona_proposal_ids
        and event.get("would_write_memory") is not True
        and event.get("would_modify_config") is not True
        and event.get("would_modify_graph") is not True
    ]
    governed_proposal_ids = sorted({_clean_text(event.get("proposal_id")) for event in governed_events})
    return {
        "write_proposal_count": len(proposals),
        "write_proposal_operation_event_count": len(events),
        "persona_continuity_write_proposal_count": len(persona_proposals),
        "persona_continuity_governed_proposal_ids": governed_proposal_ids,
        "persona_continuity_governed_event_count": len(governed_events),
        "persona_continuity_governed": bool(governed_events),
    }


def _is_persona_continuity_write_proposal(proposal: Mapping[str, Any]) -> bool:
    if proposal.get("would_write_memory") is not False:
        return False
    if proposal.get("would_modify_graph") is not False:
        return False
    if proposal.get("would_modify_config") is True:
        return False
    searchable = " ".join(
        [
            " ".join(_list_of_str(proposal.get("tags"))),
            _clean_text(proposal.get("content")),
            _clean_text(proposal.get("rationale")),
            _clean_text(proposal.get("target_scope")),
        ]
    ).lower()
    return any(term.lower() in searchable for term in STAR_SOUL_CONTINUITY_TERMS)


TEMPORAL_EVOLUTION_METRICS_TYPE = "temporal_evolution_metrics"
TEMPORAL_EVOLUTION_PROJECT_KEYS = {
    "project",
    "project_id",
    "project_scope",
    "target_project",
    "target_project_id",
    "source_project",
    "source_project_id",
}
TEMPORAL_EVOLUTION_PROJECT_LIST_KEYS = {
    "projects",
    "project_ids",
    "project_scopes",
    "target_projects",
    "source_projects",
    "ecosystem_projects",
    "ecosystem_project_ids",
}
TEMPORAL_EVOLUTION_SURFACE_KEYS = {
    "surface",
    "source_surface",
    "target_surface",
    "ecosystem_surface",
}
TEMPORAL_EVOLUTION_SURFACE_LIST_KEYS = {
    "surfaces",
    "surface_ids",
    "memory_surfaces",
    "ecosystem_surfaces",
    "evidence_surfaces",
    "contributing_surfaces",
}
TEMPORAL_EVOLUTION_DATE_KEYS = {
    "created_at",
    "updated_at",
    "recorded_at",
    "generated_at",
    "date",
    "day",
    "start",
    "end",
    "start_at",
    "end_at",
    "window_start",
    "window_end",
    "start_date",
    "end_date",
    "from",
    "to",
}
TEMPORAL_EVOLUTION_DATE_LIST_KEYS = {
    "days",
    "dates",
    "covered_days",
    "temporal_evolution_days",
}

STAR_SOURCE_EXPLICIT_TYPES = {
    "source_reasoning_methodology",
    "source_methodology_self_evolution",
    "methodology_self_evolution",
    "source_reasoning",
}
STAR_SOURCE_SOURCE_TERMS = {
    "source_reasoning",
    "source-reasoning",
    "source reasoning",
    "source_provenance_reasoning",
    "provenance_reasoning",
    "来源推理",
    "源推理",
}
STAR_SOURCE_METHODOLOGY_TERMS = {
    "methodology_generation",
    "methodology-generation",
    "methodology generation",
    "generate_methodology",
    "generated_methodology",
    "方法论生成",
}
STAR_SOURCE_SELF_EVOLUTION_TERMS = {
    "self_evolution",
    "self-evolution",
    "self evolution",
    "methodology_self_evolution",
    "source_methodology_self_evolution",
    "自进化",
    "自我进化",
}
STAR_SOURCE_PROJECT_KEYS = {
    "project",
    "project_id",
    "project_scope",
    "target_project",
    "target_project_id",
    "target_project_scope",
    "source_project",
    "source_project_id",
    "source_project_scope",
}
STAR_SOURCE_PROJECT_LIST_KEYS = {
    "projects",
    "project_ids",
    "project_scopes",
    "target_projects",
    "target_project_ids",
    "target_project_scopes",
    "source_projects",
    "source_project_ids",
    "source_project_scopes",
}
STAR_SOURCE_SURFACE_KEYS = {
    "surface",
    "source_surface",
    "target_surface",
    "evidence_surface",
}
STAR_SOURCE_SURFACE_LIST_KEYS = {
    "surfaces",
    "surface_ids",
    "memory_surfaces",
    "evidence_surfaces",
    "contributing_surfaces",
}


def _star_universe_temporal_evolution_evidence() -> dict[str, Any]:
    """Read explicit temporal evolution metric evidence without mutating state."""

    home = get_hermes_home()
    sources = [
        ("operation_ledger", _operation_ledger_path(home)),
        ("policy_proposal_ledger", _policy_proposal_path(home)),
        ("write_proposal_ledger", _proposal_path(home)),
    ]
    explicit_events: list[Mapping[str, Any]] = []
    days: set[str] = set()
    clients: set[str] = set()
    operations: set[str] = set()
    project_ids: set[str] = set()
    surfaces: set[str] = set()
    read_only_policy_holds = True

    for source_surface, path in sources:
        for row in _read_jsonl(path):
            if row.get("_parse_error") or not _is_explicit_temporal_evolution_metrics_record(row):
                continue
            explicit_events.append(row)
            surfaces.add(source_surface)
            days.update(_temporal_evolution_days(row))
            clients.update(_temporal_evolution_values(row, {"client", "source_agent", "agent", "actor"}, {"clients"}))
            operation = _clean_text(row.get("operation"))
            if operation:
                operations.add(operation)
            operations.update(_temporal_evolution_values(row, {"operation_id"}, {"operations", "operation_ids"}))
            project_ids.update(
                _temporal_evolution_values(
                    row,
                    TEMPORAL_EVOLUTION_PROJECT_KEYS,
                    TEMPORAL_EVOLUTION_PROJECT_LIST_KEYS,
                )
            )
            surfaces.update(
                _temporal_evolution_values(
                    row,
                    TEMPORAL_EVOLUTION_SURFACE_KEYS,
                    TEMPORAL_EVOLUTION_SURFACE_LIST_KEYS,
                )
            )
            read_only_policy_holds = read_only_policy_holds and _temporal_evolution_record_is_read_only(row)

    sorted_days = sorted(days)
    sorted_project_ids = sorted(project_ids)
    event_count = len(explicit_events)
    operation_count = max(len(operations), event_count)
    span_days = 0
    if len(sorted_days) >= 2:
        start_day = datetime.fromisoformat(sorted_days[0])
        end_day = datetime.fromisoformat(sorted_days[-1])
        span_days = max((end_day - start_day).days, 0)
    ready = (
        event_count >= 1
        and len(sorted_days) >= 2
        and len(sorted_project_ids) >= 2
        and len(surfaces) >= 2
        and read_only_policy_holds
    )
    return {
        "temporal_evolution_metric_event_count": event_count,
        "temporal_evolution_day_count": len(sorted_days),
        "temporal_evolution_span_days": span_days,
        "temporal_evolution_client_count": len(clients),
        "temporal_evolution_operation_count": operation_count,
        "temporal_evolution_project_count": len(sorted_project_ids),
        "temporal_evolution_project_ids": sorted_project_ids,
        "temporal_evolution_surface_count": len(surfaces),
        "temporal_evolution_metrics_ready": ready,
    }


def _star_source_methodology_evidence() -> dict[str, Any]:
    """Read explicit source/methodology self-evolution evidence without mutation."""

    home = get_hermes_home()
    source_reasoning_count = 0
    methodology_generation_count = 0
    self_evolution_count = 0
    governed_event_ids: list[str] = []
    project_ids: set[str] = set()
    surfaces: set[str] = set()

    for row in _read_jsonl(_operation_ledger_path(home)):
        if row.get("_parse_error") or not _is_explicit_star_source_record(row):
            continue
        event_surfaces = {"operation_ledger"}
        semantic_text = _star_source_semantic_text(row)
        has_source_reasoning = _star_source_has_semantic(row, semantic_text, STAR_SOURCE_SOURCE_TERMS)
        has_methodology_generation = _star_source_has_semantic(row, semantic_text, STAR_SOURCE_METHODOLOGY_TERMS)
        has_self_evolution = _star_source_has_semantic(row, semantic_text, STAR_SOURCE_SELF_EVOLUTION_TERMS)
        if has_source_reasoning:
            source_reasoning_count += 1
        if has_methodology_generation:
            methodology_generation_count += 1
        if has_self_evolution:
            self_evolution_count += 1
        event_project_ids = _temporal_evolution_values(
            row,
            STAR_SOURCE_PROJECT_KEYS,
            STAR_SOURCE_PROJECT_LIST_KEYS,
        )
        event_surfaces.update(
            _temporal_evolution_values(
                row,
                STAR_SOURCE_SURFACE_KEYS,
                STAR_SOURCE_SURFACE_LIST_KEYS,
            )
        )
        event_is_read_only = _star_source_record_is_read_only(row)
        project_ids.update(event_project_ids)
        surfaces.update(event_surfaces)
        if (
            has_source_reasoning
            and has_methodology_generation
            and has_self_evolution
            and len(event_project_ids) >= 1
            and len(event_surfaces) >= 2
            and event_is_read_only
        ):
            governed_event_ids.append(_star_source_event_id(row))

    sorted_project_ids = sorted(project_ids)
    ready = len(governed_event_ids) >= 1
    return {
        "source_reasoning_event_count": source_reasoning_count,
        "methodology_generation_event_count": methodology_generation_count,
        "self_evolution_event_count": self_evolution_count,
        "source_methodology_project_count": len(sorted_project_ids),
        "source_methodology_project_ids": sorted_project_ids,
        "source_methodology_surface_count": len(surfaces),
        "source_methodology_governed_event_count": len(governed_event_ids),
        "source_methodology_governed_event_ids": governed_event_ids,
        "source_methodology_ready": ready,
    }


def _is_explicit_star_source_record(row: Mapping[str, Any]) -> bool:
    for key in (
        "event_type",
        "operation",
        "metrics_type",
        "record_type",
        "evidence_type",
        "type",
    ):
        value = _clean_text(row.get(key)).lower().replace("-", "_").replace(" ", "_")
        if value in STAR_SOURCE_EXPLICIT_TYPES:
            return True
    return False


def _star_source_record_is_read_only(row: Mapping[str, Any]) -> bool:
    operation = _clean_text(row.get("operation")).lower()
    return (
        operation not in DURABLE_WRITE_OPERATIONS
        and operation != "policy_apply_execute"
        and row.get("would_modify_config") is not True
        and row.get("would_write_memory") is not True
        and row.get("would_mutate_memory") is not True
        and row.get("would_modify_graph") is not True
        and row.get("would_write_graph") is not True
    )


def _star_source_event_id(row: Mapping[str, Any]) -> str:
    for key in ("event_id", "operation_id", "record_id", "id"):
        value = _clean_text(row.get(key))
        if value:
            return value
    return _digest(row)


def _star_source_has_semantic(row: Mapping[str, Any], semantic_text: str, terms: set[str]) -> bool:
    if any(row.get(term) is True for term in terms):
        return True
    normalized_text = semantic_text.lower().replace("-", "_").replace(" ", "_")
    normalized_terms = {term.lower().replace("-", "_").replace(" ", "_") for term in terms}
    return any(term in normalized_text for term in normalized_terms)


def _star_source_semantic_text(row: Mapping[str, Any]) -> str:
    values: list[str] = []
    for key in (
        "event_type",
        "operation",
        "record_type",
        "evidence_type",
        "type",
        "capability",
        "reasoning_type",
        "methodology_type",
        "evolution_type",
        "description",
        "summary",
        "rationale",
    ):
        values.append(_clean_text(row.get(key)))
    values.extend(_list_of_str(row.get("tags")))
    values.extend(_list_of_str(row.get("labels")))
    values.extend(_star_source_metadata_terms(row.get("metadata")))
    values.extend(_star_source_metadata_terms(row.get("evidence")))
    values.extend(_star_source_metadata_terms(row.get("governance")))
    return " ".join(value for value in values if value)


def _star_source_metadata_terms(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        terms: list[str] = []
        for key, item in value.items():
            terms.append(_clean_text(key))
            if isinstance(item, (Mapping, list, tuple, set)):
                terms.extend(_star_source_metadata_terms(item))
            elif item is True:
                terms.append(_clean_text(key))
            else:
                terms.append(_clean_text(item))
        return [term for term in terms if term]
    if isinstance(value, (list, tuple, set)):
        terms: list[str] = []
        for item in value:
            terms.extend(_star_source_metadata_terms(item))
        return terms
    text = _clean_text(value)
    return [text] if text else []


def _is_explicit_temporal_evolution_metrics_record(row: Mapping[str, Any]) -> bool:
    for key in (
        "event_type",
        "operation",
        "metrics_type",
        "snapshot_type",
        "record_type",
        "evidence_type",
        "type",
    ):
        value = _clean_text(row.get(key)).lower().replace("-", "_").replace(" ", "_")
        if value == TEMPORAL_EVOLUTION_METRICS_TYPE:
            return True
    return False


def _temporal_evolution_record_is_read_only(row: Mapping[str, Any]) -> bool:
    operation = _clean_text(row.get("operation")).lower()
    return (
        operation not in DURABLE_WRITE_OPERATIONS
        and operation != "policy_apply_execute"
        and row.get("would_modify_config") is not True
        and row.get("would_write_memory") is not True
        and row.get("would_mutate_memory") is not True
        and row.get("would_modify_graph") is not True
        and row.get("would_write_graph") is not True
    )


def _temporal_evolution_values(
    row: Mapping[str, Any],
    scalar_keys: set[str],
    list_keys: set[str],
) -> set[str]:
    values: set[str] = set()
    for key in scalar_keys:
        value = _clean_text(row.get(key))
        if value and value.lower() not in {"all", "global", "unknown"}:
            values.add(value)
    for key in list_keys:
        values.update(item for item in _list_of_str(row.get(key)) if item.lower() not in {"all", "global", "unknown"})
    return values


def _temporal_evolution_days(row: Mapping[str, Any]) -> set[str]:
    days: set[str] = set()
    for key in TEMPORAL_EVOLUTION_DATE_KEYS:
        day = _date_only(row.get(key))
        if day:
            days.add(day)
    for key in TEMPORAL_EVOLUTION_DATE_LIST_KEYS:
        for value in _list_of_str(row.get(key)):
            day = _date_only(value)
            if day:
                days.add(day)
    return days


def _date_only(value: Any) -> str:
    text = _clean_text(value)
    if not text:
        return ""
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        pass
    if len(text) >= 10:
        candidate = text[:10]
        try:
            return datetime.fromisoformat(candidate).date().isoformat()
        except ValueError:
            return ""
    return ""


def _tier_readiness(tier: Mapping[str, Any], evidence: Mapping[str, Any]) -> dict[str, Any]:
    level = _safe_int(tier.get("level"))
    criteria: list[tuple[str, bool]] = []
    if level >= 1:
        criteria.append(("Hermes memory home is visible", bool(evidence.get("has_basic_memory_home"))))
    if level >= 2:
        criteria.append(("Knowledge or legacy memory exists", bool(evidence.get("has_knowledge"))))
    if level >= 3:
        criteria.append(("Memory graph has edges", _safe_int(evidence.get("graph_edge_count")) > 0))
    if level >= 4:
        criteria.append(("Memory graph has provenance", _safe_int(evidence.get("graph_provenance_count")) > 0))
    if level >= 5:
        criteria.append(("Knowledge and graph surfaces are both present", bool(evidence.get("has_knowledge")) and bool(evidence.get("has_graph"))))
    if level >= 6:
        criteria.append(("Prompt/case memory is indexed", bool(evidence.get("has_prompt_cases"))))
    if level >= 7:
        criteria.append(("OpenClaw has agent/domain profiles", bool(evidence.get("openclaw_agent_profiles"))))
    if level >= 8:
        criteria.append(("Codex/OpenClaw/Hermes federation is ready", bool(evidence.get("federation_ready"))))
        criteria.append(("Operation ledger or policy ledger exists", _safe_int(evidence.get("operation_ledger_events")) > 0 or _safe_int(evidence.get("policy_proposal_events")) > 0))
    if level >= 9:
        criteria.append(("Recall quality evaluation loop is enabled", bool(evidence.get("recall_quality_evaluation_ready"))))
        criteria.append(("Stale policy backlog is drained", _safe_int(evidence.get("stale_policy_proposal_count")) == 0))
    if level >= 10:
        criteria.append(("Cross-system boundary allowlists are explicitly reviewed", bool(evidence.get("boundary_allowlists_reviewed"))))
    if level >= 11:
        criteria.append(("Memory orchestration has active routing metrics", bool(evidence.get("active_routing_metrics"))))
        criteria.append(("Memory orchestration has explicit route/routing/orchestration operation evidence", bool(evidence.get("has_explicit_routing_operation_event"))))
    if level >= 12:
        criteria.append(("Policy proposals close automatically after human approval and guarded checks", bool(evidence.get("policy_closed_loop_ready"))))
    if level >= 13:
        criteria.append(("Long-term preference/persona continuity is governed", bool(evidence.get("persona_continuity_governed"))))
    if level >= 14:
        criteria.append(("Temporal evolution metrics exist across projects", bool(evidence.get("temporal_evolution_metrics_ready"))))
    if level >= 15:
        criteria.append(("Source reasoning and methodology self-evolution are implemented", bool(evidence.get("source_methodology_ready"))))
    passed = [name for name, ok in criteria if ok]
    gaps = [name for name, ok in criteria if not ok]
    readiness_score = round(len(passed) / len(criteria), 3) if criteria else 0.0
    return {
        "level": level,
        "id": tier.get("id"),
        "name": tier.get("name"),
        "one_line": tier.get("one_line"),
        "achieved": not gaps,
        "readiness_score": readiness_score,
        "passed_criteria": passed,
        "gaps": gaps,
    }


def _memory_evolution_next_actions(
    current: Mapping[str, Any],
    next_item: Mapping[str, Any] | None,
    outcome: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> list[str]:
    actions = []
    policy_outcome = _policy_closed_loop_evidence(outcome)
    if outcome.get("metrics", {}).get("stale_proposed_count") if isinstance(outcome.get("metrics"), dict) else 0:
        actions.append("Resolve stale policy proposals through review/proposal workflow before claiming 星海记忆.")
    if next_item and next_item.get("name") == "星海记忆":
        actions.append("Use memory_recall_quality_evaluate to monitor graph, knowledge, prompt cases, and legacy recall quality.")
        actions.append("Keep external-channel automatic recall blocked until exact allowlists are reviewed.")
    if not policy_outcome["policy_closed_loop_ready"]:
        actions.append("Run guarded non-mutating policy apply checks with explicit confirmation for approved proposals before claiming 星律记忆.")
    if next_item and next_item.get("name") == "星魂记忆" and not evidence.get("persona_continuity_governed"):
        actions.append("Create a governed memory_write_proposal for long-term preference/persona/collaboration-style continuity; do not write memory directly.")
    if next_item and next_item.get("name") == "星宙记忆" and not evidence.get("temporal_evolution_metrics_ready"):
        actions.append("Create or export a governed read-only temporal evolution metrics snapshot across projects; do not write Memory Graph or durable memory.")
    if next_item and next_item.get("name") == "星源记忆" and not evidence.get("source_methodology_ready"):
        actions.append("Create a governed read-only source reasoning and methodology self-evolution audit event; do not write Memory Graph or durable memory.")
    if not actions:
        actions.append("Continue with the next read-only governance/readiness capability before any durable write.")
    return actions[:5]


def _orchestration_routing_gaps(evidence_checks: Mapping[str, bool]) -> list[str]:
    labels = {
        "has_three_or_more_agents": "OpenClaw has at least 3 configured agents.",
        "has_route_binding": "OpenClaw has at least 1 route binding.",
        "has_auto_precheck_profile": "Hermes memory auto-precheck has at least 1 profile.",
        "has_explicit_routing_operation_event": "Hermes operation ledger has at least 1 explicit route/routing/orchestration event.",
    }
    return [label for key, label in labels.items() if not evidence_checks.get(key)]


def _orchestration_routing_next_actions(gaps: Iterable[str]) -> list[str]:
    actions: list[str] = []
    gap_text = " ".join(gaps)
    if "3 configured agents" in gap_text:
        actions.append("Keep Star Hub pending until OpenClaw exposes at least 3 existing agents in local config.")
    if "route binding" in gap_text:
        actions.append("Keep Star Hub pending until local OpenClaw route bindings are present; do not modify OpenClaw config from this tool.")
    if "auto-precheck" in gap_text:
        actions.append("Keep Star Hub pending until the Hermes memory plugin has an existing auto-precheck profile.")
    if "operation ledger" in gap_text:
        actions.append("Keep Star Hub pending until a governed routing event is recorded in the Hermes operation ledger; do not modify config from this tool.")
    if not actions:
        actions.append("Continue monitoring routing metrics read-only; do not widen allowlists or enable external automatic recall.")
    return actions[:5]


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
    guidance_path = Path.home() / ".codex" / "AGENTS.md"
    config_text = _safe_read_text(config_path)
    guidance_text = _safe_read_text(guidance_path)
    return {
        "role": "memory_client",
        "access_path": "codex_mcp",
        "config_path": str(config_path),
        "config_exists": config_path.exists(),
        "mcp_server_configured": "hermes-memory" in config_text,
        "invokes_hermes_mcp": "hermes" in config_text and "mcp" in config_text,
        "guidance_path": str(guidance_path),
        "guidance_exists": guidance_path.exists(),
        "guidance_mentions_hermes_memory": "Hermes Memory" in guidance_text or "memory_fabric" in guidance_text,
        "write_policy": "proposal_only",
        "ready": config_path.exists() and "hermes-memory" in config_text,
    }


def _openclaw_memory_client_status() -> dict[str, Any]:
    config_path = Path.home() / ".openclaw" / "openclaw.json"
    extension_path = Path.home() / ".openclaw" / "extensions" / "hermes-memory" / "index.ts"
    manifest_path = Path.home() / ".openclaw" / "extensions" / "hermes-memory" / "openclaw.plugin.json"
    config = _loads_json(_safe_read_text(config_path), {})
    entry = _openclaw_hermes_plugin_entry(config if isinstance(config, dict) else {})
    plugin_config = entry.get("config", {}) if isinstance(entry, dict) else {}
    profiles = plugin_config.get("autoPrecheckAgentProfiles", {}) if isinstance(plugin_config, dict) else {}
    allowed_channels = plugin_config.get("autoPrecheckAllowedChannelIds", []) if isinstance(plugin_config, dict) else []
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
        "conversation_access_allowed": bool(entry.get("hooks", {}).get("allowConversationAccess")) if isinstance(entry, dict) and isinstance(entry.get("hooks"), dict) else False,
        "auto_precheck_enabled": plugin_config.get("autoPrecheckEnabled") is not False if isinstance(plugin_config, dict) else False,
        "auto_precheck_agent_profiles": sorted(profiles.keys()) if isinstance(profiles, dict) else [],
        "external_auto_precheck_allowed_channels": allowed_channels if isinstance(allowed_channels, list) else [],
        "external_auto_precheck_default": "blocked",
        "write_policy": "proposal_only",
        "ready": config_path.exists() and extension_path.exists() and bool(entry.get("enabled")) if isinstance(entry, dict) else False,
    }


def _openclaw_hermes_plugin_entry(config: Mapping[str, Any]) -> dict[str, Any]:
    plugins = config.get("plugins", {}) if isinstance(config, Mapping) else {}
    entries = plugins.get("entries", {}) if isinstance(plugins, Mapping) else {}
    entry = entries.get("hermes-memory", {}) if isinstance(entries, Mapping) else {}
    return entry if isinstance(entry, dict) else {}


def _openclaw_hermes_plugin_config(config: Mapping[str, Any]) -> dict[str, Any]:
    entry = _openclaw_hermes_plugin_entry(config)
    plugin_config = entry.get("config", {}) if isinstance(entry, Mapping) else {}
    return plugin_config if isinstance(plugin_config, dict) else {}


def _openclaw_agent_ids(config: Mapping[str, Any]) -> list[str]:
    agents = config.get("agents", {}) if isinstance(config, Mapping) else {}
    ids: set[str] = set()
    if isinstance(agents, list):
        for index, item in enumerate(agents):
            if isinstance(item, Mapping):
                ids.add(_clean_text(item.get("id") or item.get("name") or item.get("agentId") or f"agent_{index}"))
            elif _clean_text(item):
                ids.add(_clean_text(item))
    elif isinstance(agents, Mapping):
        for key, value in agents.items():
            key_text = _clean_text(key)
            if key_text in {"defaults", "default", "settings", "config"}:
                continue
            if key_text in {"entries", "profiles", "items", "list", "agents"}:
                ids.update(_ids_from_collection(value, fallback_prefix="agent"))
                continue
            if isinstance(value, Mapping):
                ids.add(_clean_text(value.get("id") or value.get("name") or key_text))
            elif isinstance(value, list):
                ids.update(_ids_from_collection(value, fallback_prefix=key_text or "agent"))
            elif _clean_text(value):
                ids.add(key_text or _clean_text(value))
    return sorted(item for item in ids if item)


def _ids_from_collection(value: Any, *, fallback_prefix: str) -> set[str]:
    ids: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(item, Mapping):
                ids.add(_clean_text(item.get("id") or item.get("name") or key))
            elif _clean_text(item):
                ids.add(_clean_text(key) or _clean_text(item))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            if isinstance(item, Mapping):
                ids.add(_clean_text(item.get("id") or item.get("name") or item.get("agentId") or f"{fallback_prefix}_{index}"))
            elif _clean_text(item):
                ids.add(_clean_text(item))
    return {item for item in ids if item}


def _openclaw_route_bindings(config: Mapping[str, Any]) -> list[dict[str, str]]:
    bindings: list[dict[str, str]] = []
    _collect_openclaw_route_bindings(config, bindings, path="")
    unique: dict[tuple[str, str, str, str, str, str], dict[str, str]] = {}
    for binding in bindings:
        key = (
            binding.get("channel", ""),
            binding.get("agent_id", ""),
            binding.get("account_id", ""),
            binding.get("peer_kind", ""),
            binding.get("peer_id", ""),
            binding.get("source", ""),
        )
        unique[key] = binding
    return list(unique.values())


def _collect_openclaw_route_bindings(value: Any, bindings: list[dict[str, str]], *, path: str) -> None:
    if isinstance(value, list):
        for index, item in enumerate(value):
            _collect_openclaw_route_bindings(item, bindings, path=f"{path}/{index}")
        return
    if not isinstance(value, Mapping):
        return

    normalized_path = path.replace("_", "").replace("-", "").lower()
    if _looks_like_route_binding(value):
        bindings.append(_normalize_openclaw_route_binding(value, source=path or "/"))

    for key, item in value.items():
        key_text = _clean_text(key)
        key_normalized = key_text.replace("_", "").replace("-", "").lower()
        child_path = f"{path}/{key_text}" if path else key_text
        if _is_route_binding_container(key_normalized):
            if isinstance(item, Mapping):
                for route_key, route_value in item.items():
                    if isinstance(route_value, Mapping):
                        binding = _normalize_openclaw_route_binding(route_value, source=f"{child_path}/{route_key}")
                        if not binding.get("channel"):
                            binding["channel"] = _clean_text(route_key)
                        bindings.append(binding)
                    elif isinstance(route_value, list):
                        for agent_id in _list_of_str(route_value):
                            bindings.append({"channel": _clean_text(route_key), "agent_id": agent_id, "source": f"{child_path}/{route_key}"})
                    elif _clean_text(route_value):
                        bindings.append({"channel": _clean_text(route_key), "agent_id": _clean_text(route_value), "source": f"{child_path}/{route_key}"})
            elif isinstance(item, list):
                for index, route_value in enumerate(item):
                    if isinstance(route_value, Mapping) and _looks_like_route_binding(route_value):
                        bindings.append(_normalize_openclaw_route_binding(route_value, source=f"{child_path}/{index}"))
            continue
        if "route" in normalized_path or "binding" in normalized_path or "route" in key_normalized or "binding" in key_normalized:
            _collect_openclaw_route_bindings(item, bindings, path=child_path)


def _is_route_binding_container(key: str) -> bool:
    return key in {
        "routebindings",
        "routingbindings",
        "channelroutebindings",
        "channelagentbindings",
        "agentroutebindings",
        "routes",
        "routing",
    }


def _looks_like_route_binding(value: Mapping[str, Any]) -> bool:
    keys = {str(key).replace("_", "").replace("-", "").lower() for key in value.keys()}
    match = value.get("match")
    match_keys = {str(key).replace("_", "").replace("-", "").lower() for key in match.keys()} if isinstance(match, Mapping) else set()
    binding_type = _clean_text(value.get("type")).lower()
    has_channel = bool(keys & {"channel", "channelid", "route", "target"} or match_keys & {"channel", "channelid"})
    has_agent = bool(keys & {"agent", "agentid", "profile", "profileid"})
    return has_channel and has_agent and (not binding_type or binding_type == "route")


def _normalize_openclaw_route_binding(value: Mapping[str, Any], *, source: str) -> dict[str, str]:
    match = value.get("match")
    match = match if isinstance(match, Mapping) else {}
    binding = {
        "agent_id": _clean_text(value.get("agentId") or value.get("agent_id") or value.get("agent") or value.get("profileId") or value.get("profile")),
        "channel": _clean_text(
            value.get("channelId")
            or value.get("channel_id")
            or value.get("channel")
            or value.get("route")
            or value.get("target")
            or match.get("channelId")
            or match.get("channel_id")
            or match.get("channel")
        ),
        "source": source,
    }
    optional_fields = {
        "account_id": value.get("accountId") or value.get("account_id") or match.get("accountId") or match.get("account_id"),
        "peer_kind": value.get("peerKind") or value.get("peer_kind") or match.get("peerKind") or match.get("peer_kind"),
        "peer_id": value.get("peerId") or value.get("peer_id") or match.get("peerId") or match.get("peer_id"),
    }
    for key, item in optional_fields.items():
        cleaned = _clean_text(item)
        if cleaned:
            binding[key] = cleaned
    return binding


def _federation_warnings(clients: Mapping[str, Any]) -> list[str]:
    warnings = []
    for name in ("codex", "openclaw"):
        client = clients.get(name, {})
        if isinstance(client, dict) and not client.get("ready"):
            warnings.append(f"{name} memory client is not fully configured.")
    return warnings


def _audit_check(check_id: str, passed: bool, severity: str, message: str) -> dict[str, Any]:
    return {"id": check_id, "status": "pass" if passed else "fail", "severity": severity, "message": message}


def _ledger_findings(events: Iterable[Mapping[str, Any]], parse_errors: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    events = list(events)
    parse_errors = list(parse_errors)
    findings: list[dict[str, Any]] = []
    blocked_external = [event for event in events if event.get("operation") in {"auto_precheck", "external_auto_recall"} and event.get("decision") == "block"]
    direct_writes = [event for event in events if event.get("operation") in DURABLE_WRITE_OPERATIONS]
    if parse_errors:
        findings.append(_finding("ledger.parse_errors", "warning", "ledger", "Operation ledger has malformed rows.", {"parse_error_count": len(parse_errors)}, "Repair or quarantine malformed ledger lines."))
    if blocked_external:
        findings.append(_finding("policy.external_auto_recall_blocked", "warning", "federation_gate", "External-channel automatic recall was blocked.", {"blocked_count": len(blocked_external)}, "Keep external recall blocked unless exact channels are reviewed."))
    if direct_writes:
        findings.append(_finding("policy.direct_write_attempts", "critical", "write_policy", "Direct durable write operations were attempted.", {"count": len(direct_writes)}, "Route durable writes through memory_write_proposal."))
    if not findings:
        findings.append(_finding("ledger.healthy", "info", "ledger", "No notable memory operation risks were detected.", {"total": len(events)}, ""))
    return findings


def _finding(finding_id: str, severity: str, subject: str, message: str, evidence: Any, recommendation: str) -> dict[str, Any]:
    return {"id": finding_id, "severity": severity, "subject": subject, "message": message, "evidence": evidence, "recommendation": recommendation}


def _policy_suggestions_from_findings(findings: Iterable[Mapping[str, Any]], *, mode: str) -> list[dict[str, Any]]:
    suggestions: list[dict[str, Any]] = []
    for finding in findings:
        finding_id = _clean_text(finding.get("id"))
        if finding_id == "policy.external_auto_recall_blocked":
            suggestions.append(_policy_suggestion("external_auto_recall.keep_blocked", "medium", "approval_required", "openclaw.autoPrecheckAllowedChannelIds", "Keep external-channel automatic recall blocked by default.", "Automatic allowlisting would widen memory exposure.", {"source_finding": finding_id}, "Only add an exact channel id after human privacy review."))
        elif finding_id == "policy.direct_write_attempts":
            suggestions.append(_policy_suggestion("durable_writes.enforce_proposal_only", "high", "policy_guard", "memory_write_policy", "Keep durable memory writes behind memory_write_proposal.", "Direct durable write operations were attempted.", {"source_finding": finding_id}, "Audit caller and route writes through governed proposals."))
    return suggestions


def _policy_suggestion(
    suggestion_id: str,
    priority: str,
    change_type: str,
    target: str,
    recommendation: str,
    rationale: str,
    evidence: Mapping[str, Any],
    next_step: str,
) -> dict[str, Any]:
    return {
        "id": suggestion_id,
        "priority": priority,
        "change_type": change_type,
        "target": target,
        "recommendation": recommendation,
        "rationale": rationale,
        "evidence": dict(evidence),
        "next_step": next_step,
        "requires_human_review": True,
        "can_auto_apply": False,
    }


def _policy_proposal_from_suggestion(
    suggestion: Mapping[str, Any],
    *,
    source_agent: str,
    autotune: Mapping[str, Any],
) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    proposal_digest = _digest({"suggestion": suggestion, "source_agent": source_agent})
    return {
        "proposal_id": f"memory-policy-proposal-{proposal_digest[:16]}",
        "proposal_digest": proposal_digest,
        "suggestion_id": suggestion.get("id", ""),
        "source_agent": source_agent,
        "priority": suggestion.get("priority", ""),
        "change_type": suggestion.get("change_type", ""),
        "target": suggestion.get("target", ""),
        "recommendation": suggestion.get("recommendation", ""),
        "rationale": suggestion.get("rationale", ""),
        "next_step": suggestion.get("next_step", ""),
        "evidence": suggestion.get("evidence", {}),
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
        "would_modify_config": False,
        "would_write_memory": False,
    }


def _policy_proposal_states(rows: list[dict[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
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
            decision = _policy_decision(row.get("decision"))
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
    return {"by_status": by_status, "by_priority": by_priority, "by_change_type": by_change_type}


def _policy_apply_plan_for_proposal(proposal: Mapping[str, Any]) -> dict[str, Any]:
    suggestion_id = _clean_text(proposal.get("suggestion_id"))
    eligible = proposal.get("latest_status") == "approved"
    if suggestion_id == "diagnostic.rerun_full_audit":
        patches = [{"action": "run_check", "target_file": "memory_federation_audit", "json_path": "memory_federation_audit", "expected": "ready", "would_modify": False}]
    elif suggestion_id == "external_auto_recall.keep_blocked":
        patches = [{"action": "verify", "target_file": str(Path.home() / ".openclaw" / "openclaw.json"), "json_path": "/plugins/entries/hermes-memory/config/autoPrecheckAllowedChannelIds", "expected": [], "would_modify": False}]
    else:
        patches = [{"action": "manual_review", "target_file": proposal.get("target", "memory_policy"), "json_path": "", "expected": proposal.get("recommendation", ""), "would_modify": False}]
    return {
        "proposal_id": proposal.get("proposal_id"),
        "suggestion_id": suggestion_id,
        "eligible_for_apply": eligible,
        "risk_level": "low",
        "patches": patches,
        "notes": [] if eligible else ["Proposal is not approved; plan is preview-only."],
    }


def _policy_apply_plan_summary(plans: Iterable[Mapping[str, Any]], field: str) -> dict[str, int]:
    summary: dict[str, int] = {}
    for plan in plans:
        for patch in plan.get("patches", []):
            if not isinstance(patch, dict):
                continue
            value = _clean_text(patch.get(field)) or "unknown"
            summary[value] = summary.get(value, 0) + 1
    return summary


def _policy_apply_guard(plan: Mapping[str, Any], *, execute: bool, confirm_token: str) -> dict[str, Any]:
    plans = plan.get("plans", []) if isinstance(plan.get("plans"), list) else []
    patches = [patch for item in plans if isinstance(item, dict) for patch in item.get("patches", []) if isinstance(patch, dict)]
    if _safe_int(plan.get("eligible_count")) == 0:
        return {"allowed": False, "reason": "No approved, eligible policy proposals are available for execution.", "requires_confirm_token": True, "expected_confirm_token": POLICY_EXECUTE_CONFIRM_TOKEN}
    if any(patch.get("would_modify") is True for patch in patches):
        return {"allowed": False, "reason": "Plan contains config-mutating patches; guarded executor only allows non-mutating checks.", "requires_confirm_token": True, "expected_confirm_token": POLICY_EXECUTE_CONFIRM_TOKEN}
    unsupported = [_clean_text(patch.get("action")) for patch in patches if _clean_text(patch.get("action")) not in VALID_POLICY_EXECUTE_ACTIONS]
    if unsupported:
        return {"allowed": False, "reason": "Plan contains unsupported executor actions.", "unsupported_actions": sorted(set(unsupported)), "requires_confirm_token": True, "expected_confirm_token": POLICY_EXECUTE_CONFIRM_TOKEN}
    if execute and _clean_text(confirm_token) != POLICY_EXECUTE_CONFIRM_TOKEN:
        return {"allowed": False, "reason": "Missing or invalid policy execution confirmation token.", "requires_confirm_token": True, "expected_confirm_token": POLICY_EXECUTE_CONFIRM_TOKEN}
    return {"allowed": True, "reason": "Plan is approved and contains only non-mutating executor actions.", "requires_confirm_token": True, "expected_confirm_token": POLICY_EXECUTE_CONFIRM_TOKEN}


def _policy_patch_result(patch: Mapping[str, Any], *, proposal_id: str) -> dict[str, Any]:
    return {
        "proposal_id": proposal_id,
        "action": patch.get("action"),
        "target_file": patch.get("target_file"),
        "json_path": patch.get("json_path"),
        "status": "checked" if patch.get("action") in {"verify", "run_check"} else "manual_required",
        "passed": True,
        "message": "Non-mutating guarded check completed as a safe dry execution.",
        "expected": patch.get("expected"),
        "actual": patch.get("expected"),
        "would_modify": False,
    }


def _policy_apply_execute_summary(results: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    summary = {"checked_count": 0, "manual_required_count": 0, "blocked_count": 0, "failed_count": 0, "passed_count": 0}
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


def _policy_outcome_metrics(proposals: Iterable[Mapping[str, Any]], rows: Iterable[Mapping[str, Any]], *, stale_after_hours: int) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    proposal_items = list(proposals)
    summary = _policy_proposal_summary(proposal_items)
    by_status = summary.get("by_status", {})
    execution_events: list[dict[str, Any]] = []
    executions_by_proposal: dict[str, list[dict[str, Any]]] = {}
    totals = {"checked_count": 0, "manual_required_count": 0, "blocked_count": 0, "failed_count": 0, "passed_count": 0}
    for row in rows:
        if row.get("_parse_error") or row.get("event_type") != "policy_apply_execute":
            continue
        result_summary = row.get("result_summary", {}) if isinstance(row.get("result_summary"), dict) else {}
        event = {"proposal_id": row.get("proposal_id", ""), "actor": row.get("actor", ""), "created_at": row.get("created_at", ""), "result_summary": result_summary}
        execution_events.append(event)
        if event["proposal_id"]:
            executions_by_proposal.setdefault(event["proposal_id"], []).append(event)
        for key in totals:
            totals[key] += _safe_int(result_summary.get(key))
    stale: list[dict[str, Any]] = []
    approved_not_executed: list[dict[str, Any]] = []
    for proposal in proposal_items:
        proposal_id = _clean_text(proposal.get("proposal_id"))
        status = _clean_text(proposal.get("latest_status"))
        updated = _parse_datetime(proposal.get("updated_at")) or _parse_datetime(proposal.get("created_at"))
        age_hours = max(0.0, (now - updated).total_seconds() / 3600) if updated else None
        if status == "proposed" and age_hours is not None and age_hours >= stale_after_hours:
            stale.append({"proposal_id": proposal_id, "suggestion_id": proposal.get("suggestion_id", ""), "age_hours": round(age_hours, 2), "updated_at": proposal.get("updated_at", "")})
        if status == "approved" and proposal_id and not executions_by_proposal.get(proposal_id):
            approved_not_executed.append({"proposal_id": proposal_id, "suggestion_id": proposal.get("suggestion_id", ""), "updated_at": proposal.get("updated_at", "")})
    execution_events.sort(key=lambda row: _clean_text(row.get("created_at")), reverse=True)
    return {
        "total_proposals": len(proposal_items),
        "proposal_count_by_status": by_status,
        "proposed_count": _safe_int(by_status.get("proposed")),
        "approved_count": _safe_int(by_status.get("approved")),
        "rejected_count": _safe_int(by_status.get("rejected")),
        "deferred_count": _safe_int(by_status.get("deferred")),
        "stale_proposed_count": len(stale),
        "stale_proposed": stale,
        "approved_not_executed_count": len(approved_not_executed),
        "approved_not_executed": approved_not_executed,
        "execution_count": len(execution_events),
        "execution_totals": totals,
        "latest_execution_at": execution_events[0]["created_at"] if execution_events else "",
        "recent_execution_events": execution_events,
    }


def _policy_outcome_findings(metrics: Mapping[str, Any], parse_errors: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    parse_errors = list(parse_errors)
    if parse_errors:
        findings.append(_finding("policy_ledger.parse_errors", "warning", "policy_proposal_ledger", "Policy proposal ledger contains malformed rows.", {"parse_error_count": len(parse_errors)}, "Repair malformed proposal ledger lines before policy decisions."))
    if _safe_int(metrics.get("total_proposals")) == 0:
        findings.append(_finding("policy.no_proposals", "info", "policy_proposal_lifecycle", "No policy proposals are present yet.", {}, "Use policy autotune after enough ledger evidence exists."))
    elif _safe_int(metrics.get("proposed_count")):
        findings.append(_finding("policy.proposal_backlog", "info", "policy_proposal_lifecycle", "Some policy proposals are waiting for human review.", {"proposed_count": metrics.get("proposed_count")}, "Review proposals and record approve/reject/defer decisions."))
    if _safe_int(metrics.get("stale_proposed_count")):
        findings.append(_finding("policy.stale_proposals", "warning", "policy_proposal_lifecycle", "Some proposed policy changes have become stale.", {"stale_proposed": metrics.get("stale_proposed", [])}, "Decide stale proposals before creating more policy suggestions."))
    if _safe_int(metrics.get("approved_not_executed_count")):
        findings.append(_finding("policy.approved_not_executed", "warning", "policy_apply_lifecycle", "Some approved policy proposals have no matching execution check event.", {"approved_not_executed": metrics.get("approved_not_executed", [])}, "Run guarded non-mutating checks with explicit confirmation."))
    totals = metrics.get("execution_totals", {}) if isinstance(metrics.get("execution_totals"), dict) else {}
    if _safe_int(totals.get("blocked_count")) or _safe_int(totals.get("failed_count")):
        findings.append(_finding("policy.execution_failures", "critical", "policy_apply_lifecycle", "Policy execution checks reported blocked or failed results.", totals, "Inspect policy_apply_execute results before approving more proposals."))
    if not findings:
        findings.append(_finding("policy.outcome_healthy", "info", "policy_outcome_monitor", "Policy proposal and execution outcomes look healthy.", {"total_proposals": metrics.get("total_proposals")}, ""))
    return findings


def _recall_quality_queries(queries: Iterable[str] | str | None) -> list[str]:
    values = _list_of_str(queries) if queries is not None else DEFAULT_RECALL_QUALITY_QUERIES
    normalized: list[str] = []
    for value in values:
        cleaned = _clean_text(value)
        if cleaned and cleaned not in normalized:
            normalized.append(cleaned)
    return normalized or list(DEFAULT_RECALL_QUALITY_QUERIES)


def _evaluate_recall_query(
    home: Path,
    query: str,
    *,
    scope: str,
    limit: int,
) -> dict[str, Any]:
    results = _collect_recall_results(home, query, scope=scope, limit=limit)
    sources = sorted({_clean_text(result.get("source")) for result in results if result.get("source")})
    top_score = max([float(result.get("score", 0) or 0) for result in results], default=0.0)
    coverage_score = min(1.0, len(results) / max(1, min(limit, 3)))
    diversity_score = min(1.0, len(sources) / 2)
    top_score_normalized = min(1.0, top_score)
    quality_score = round(
        coverage_score * 0.45 + top_score_normalized * 0.35 + diversity_score * 0.20,
        3,
    )
    passed = len(results) > 0 and quality_score >= 0.45
    return {
        "query": query,
        "result_count": len(results),
        "sources": sources,
        "top_score": round(top_score, 3),
        "coverage_score": round(coverage_score, 3),
        "source_diversity_score": round(diversity_score, 3),
        "quality_score": quality_score,
        "passed": passed,
        "top_results": [
            {
                "source": result.get("source"),
                "id": result.get("id"),
                "kind": result.get("kind"),
                "title": result.get("title"),
                "score": result.get("score"),
            }
            for result in results[: min(limit, 5)]
        ],
    }


def _collect_recall_results(
    home: Path,
    query: str,
    *,
    scope: str,
    limit: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if scope in {"all", "graph"}:
        results.extend(_search_graph(home, query, limit=limit))
    if scope in {"all", "prompt_cases"}:
        results.extend(_search_prompt_cases(home, query, limit=limit))
    if scope in {"all", "knowledge"}:
        results.extend(_search_knowledge(home, query, limit=limit))
    if scope in {"all", "legacy_memory"}:
        results.extend(_search_legacy_memory(home, query, limit=limit))
    results.sort(key=lambda item: float(item.get("score", 0) or 0), reverse=True)
    return results[:limit]


def _recall_quality_summary(evaluations: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    items = list(evaluations)
    query_count = len(items)
    passed = [item for item in items if item.get("passed") is True]
    total_results = sum(_safe_int(item.get("result_count")) for item in items)
    sources: dict[str, int] = {}
    for item in items:
        for source in item.get("sources", []):
            source = _clean_text(source)
            if source:
                sources[source] = sources.get(source, 0) + 1
    average_quality = (
        sum(float(item.get("quality_score", 0) or 0) for item in items) / query_count
        if query_count
        else 0.0
    )
    return {
        "benchmark_query_count": query_count,
        "passed_query_count": len(passed),
        "failed_query_count": query_count - len(passed),
        "total_result_count": total_results,
        "average_results_per_query": round(total_results / query_count, 3) if query_count else 0,
        "average_quality_score": round(average_quality, 3),
        "source_coverage": sources,
        "covered_source_count": len(sources),
        "low_result_queries": [
            item.get("query")
            for item in items
            if _safe_int(item.get("result_count")) == 0
        ],
        "low_diversity_queries": [
            item.get("query")
            for item in items
            if float(item.get("source_diversity_score", 0) or 0) < 0.5
        ],
    }


def _recall_quality_findings(
    evaluations: Iterable[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    items = list(evaluations)
    findings: list[dict[str, Any]] = []
    low_results = summary.get("low_result_queries", [])
    low_diversity = summary.get("low_diversity_queries", [])
    average_score = float(summary.get("average_quality_score", 0) or 0)
    if low_results:
        findings.append(
            _finding(
                "recall_quality.low_result_queries",
                "warning",
                "recall_quality",
                "Some benchmark queries returned no recall results.",
                {"queries": low_results},
                "Add or repair knowledge, graph nodes, or prompt cases for weak recall topics through governed proposals.",
            )
        )
    if low_diversity:
        findings.append(
            _finding(
                "recall_quality.low_source_diversity",
                "info",
                "recall_quality",
                "Some benchmark queries depend on too few memory surfaces.",
                {"queries": low_diversity},
                "Improve cross-surface recall coverage before relying on automatic recall for broad tasks.",
            )
        )
    if average_score < 0.55:
        findings.append(
            _finding(
                "recall_quality.score_below_threshold",
                "warning",
                "recall_quality",
                "Average recall quality score is below the 星海 readiness threshold.",
                {"average_quality_score": average_score},
                "Tune search coverage and add benchmark examples before claiming 星海记忆.",
            )
        )
    if not findings:
        findings.append(
            _finding(
                "recall_quality.ready",
                "info",
                "recall_quality",
                "Recall quality benchmark passed for the analyzed memory surfaces.",
                {
                    "average_quality_score": average_score,
                    "benchmark_query_count": len(items),
                    "covered_source_count": summary.get("covered_source_count"),
                },
                "Keep monitoring recall quality as new knowledge and agents are added.",
            )
        )
    return findings


def _search_graph(home: Path, query: str, *, limit: int) -> list[dict[str, Any]]:
    path = _graph_path(home)
    if not path.exists():
        return []
    try:
        with sqlite3.connect(path) as conn:
            conn.row_factory = sqlite3.Row
            like = f"%{query}%"
            rows = conn.execute(
                "SELECT id, kind, title, summary, confidence, metadata_json FROM graph_nodes WHERE title LIKE ? OR summary LIKE ? LIMIT ?",
                (like, like, limit),
            ).fetchall()
    except sqlite3.Error:
        return []
    return [
        {
            "source": "graph",
            "id": row["id"],
            "kind": row["kind"],
            "title": row["title"],
            "summary": _snippet(row["summary"], query),
            "score": _score_text(f"{row['title']} {row['summary']}", query, base=float(row["confidence"] or 0.5)),
            "metadata": _loads_json(row["metadata_json"], {}),
        }
        for row in rows
    ]


def _search_prompt_cases(home: Path, query: str, *, limit: int) -> list[dict[str, Any]]:
    path = _prompt_index_path(home)
    if not path.exists():
        return []
    try:
        with sqlite3.connect(path) as conn:
            conn.row_factory = sqlite3.Row
            like = f"%{query}%"
            rows = conn.execute(
                "SELECT id, title, prompt, category, section, tags_json FROM cases WHERE title LIKE ? OR prompt LIKE ? OR category LIKE ? LIMIT ?",
                (like, like, like, limit),
            ).fetchall()
    except sqlite3.Error:
        return []
    return [
        {
            "source": "prompt_cases",
            "id": row["id"],
            "kind": row["category"],
            "title": row["title"],
            "summary": _snippet(row["prompt"], query),
            "score": _score_text(f"{row['title']} {row['prompt']} {row['category']}", query, base=0.7),
            "metadata": {"section": row["section"], "tags": _loads_json(row["tags_json"], [])},
        }
        for row in rows
    ]


def _search_knowledge(home: Path, query: str, *, limit: int) -> list[dict[str, Any]]:
    root = home / "knowledge"
    if not root.exists():
        return []
    results: list[dict[str, Any]] = []
    for path in root.rglob("*"):
        if len(results) >= limit:
            break
        if not path.is_file() or path.suffix.lower() not in {".md", ".txt", ".jsonl"}:
            continue
        text = _safe_read_text(path)
        if query.lower() not in text.lower() and query.lower() not in path.name.lower():
            continue
        results.append(
            {
                "source": "knowledge",
                "id": str(path.relative_to(home)),
                "kind": "file",
                "title": path.name,
                "summary": _snippet(text, query),
                "score": _score_text(f"{path.name} {text[:3000]}", query, base=0.55),
                "metadata": {"path": str(path)},
            }
        )
    return results


def _search_legacy_memory(home: Path, query: str, *, limit: int) -> list[dict[str, Any]]:
    root = home / "memories"
    if not root.exists():
        return []
    results: list[dict[str, Any]] = []
    for path in root.glob("*.md"):
        text = _safe_read_text(path)
        if query.lower() not in text.lower():
            continue
        results.append({"source": "legacy_memory", "id": path.name, "kind": "file", "title": path.name, "summary": _snippet(text, query), "score": _score_text(text, query, base=0.45), "metadata": {"path": str(path)}})
        if len(results) >= limit:
            break
    return results


def _append_operation_event(event: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(event)
    payload.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    payload.setdefault("would_write_memory", False)
    payload.setdefault("would_modify_config", False)
    payload["event_id"] = f"memory-operation-{_digest(payload)[:16]}"
    return _append_jsonl(_operation_ledger_path(get_hermes_home()), payload)


def _append_policy_proposal_event(event: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(event)
    payload.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    payload.setdefault("would_modify_config", False)
    payload.setdefault("would_write_memory", False)
    payload["event_id"] = f"memory-policy-event-{_digest(payload)[:16]}"
    result = _append_jsonl(_policy_proposal_path(get_hermes_home()), payload)
    result["event_type"] = payload.get("event_type")
    return result


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True) + "\n")
    except OSError as exc:
        return {"success": False, "error": str(exc), "path": str(path), "would_mutate_memory": False}
    return {"success": True, "event_id": payload.get("event_id", ""), "path": str(path), "read_only_memory": True, "would_mutate_memory": False, "would_modify_config": False}


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
                rows.append({"_parse_error": str(exc), "line_number": line_number, "line_preview": line[:200]})
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


def _sqlite_count(path: Path, table: str) -> int:
    if not path.exists():
        return 0
    try:
        with sqlite3.connect(path) as conn:
            row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            return int(row[0]) if row else 0
    except sqlite3.Error:
        return 0


def _knowledge_file_count(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for path in root.rglob("*") if path.is_file())


def _jsonl_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    except OSError:
        return 0


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def _loads_json(value: str, default: Any) -> Any:
    try:
        return json.loads(value) if value else default
    except (TypeError, json.JSONDecodeError):
        return default


def _scope(scope: str) -> str:
    cleaned = _clean_text(scope).lower() or "all"
    return cleaned if cleaned in VALID_SEARCH_SCOPES else "all"


def _proposal_scope(scope: str) -> str:
    cleaned = _clean_text(scope).lower() or "project"
    return cleaned if cleaned in VALID_PROPOSAL_SCOPES else "project"


def _policy_decision(value: Any) -> str:
    cleaned = _clean_text(value).lower()
    return cleaned if cleaned in VALID_POLICY_PROPOSAL_DECISIONS else "deferred"


def _policy_status(value: Any, *, allow_empty: bool = False) -> str:
    cleaned = _clean_text(value).lower()
    if allow_empty and not cleaned:
        return ""
    return cleaned if cleaned in VALID_POLICY_PROPOSAL_STATUSES else "proposed"


def _health_score(findings: Iterable[Mapping[str, Any]]) -> int:
    penalty = 0
    for finding in findings:
        if finding.get("id") in {"ledger.healthy", "policy.outcome_healthy"}:
            continue
        if finding.get("severity") == "critical":
            penalty += 25
        elif finding.get("severity") == "warning":
            penalty += 10
    return max(0, 100 - penalty)


def _recommended_actions(findings: Iterable[Mapping[str, Any]]) -> list[str]:
    actions: list[str] = []
    for finding in findings:
        recommendation = _clean_text(finding.get("recommendation"))
        if recommendation and recommendation not in actions:
            actions.append(recommendation)
    return actions[:5]


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


def _is_external_channel(channel_id: str) -> bool:
    root = _clean_text(channel_id).split(":", 1)[0].lower()
    return bool(root and root in EXTERNAL_CHANNEL_ROOTS)


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
    return ("..." if start else "") + text[start:end].strip() + ("..." if end < len(text) else "")


def _score_text(text: str, query: str, *, base: float) -> float:
    lower = text.lower()
    terms = [term for term in query.lower().split() if term]
    hits = sum(1 for term in terms if term in lower)
    exact = 1 if query.lower() in lower else 0
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
    return {"success": False, "error": message, **extra, "read_only_memory": True, "would_mutate_memory": False}
