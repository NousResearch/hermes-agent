import agent.memory_fabric_bridge as memory_fabric_bridge
from agent.memory_fabric_bridge import (
    memory_boundary_allowlist_audit,
    MEMORY_EVOLUTION_TIERS,
    memory_bridge_status,
    memory_evolution_status,
    memory_federation_gate,
    memory_policy_outcome_monitor,
    memory_recall_quality_evaluate,
)


def test_memory_evolution_tiers_are_fixed():
    names = [tier["name"] for tier in MEMORY_EVOLUTION_TIERS]

    assert names == [
        "星火记忆",
        "星点记忆",
        "星链记忆",
        "星图记忆",
        "星河记忆",
        "星辰记忆",
        "星域记忆",
        "星穹记忆",
        "星海记忆",
        "星界记忆",
        "星枢记忆",
        "星律记忆",
        "星魂记忆",
        "星宙记忆",
        "星源记忆",
    ]


def test_memory_evolution_status_is_read_only():
    result = memory_evolution_status()

    assert result["success"] is True
    assert len(result["taxonomy"]) == 15
    assert result["current"]["level"] >= 1
    assert "recall_quality" in result
    assert result["policy"]["taxonomy_is_fixed"] is True
    assert result["policy"]["status_is_read_only"] is True
    assert result["read_only_memory"] is True
    assert result["would_mutate_memory"] is False
    assert result["would_modify_config"] is False


def test_memory_bridge_status_includes_graph_and_policy_surfaces():
    result = memory_bridge_status()

    assert result["success"] is True
    assert "graph" in result["surfaces"]
    assert "policy_proposals" in result["surfaces"]
    assert result["policy"]["writes_are_proposal_only"] is True


def test_memory_federation_gate_blocks_direct_writes():
    result = memory_federation_gate(
        client="codex",
        operation="direct_write",
        target_scope="memory",
    )

    assert result["success"] is True
    assert result["decision"] == "block"
    assert result["allowed"] is False


def test_memory_policy_outcome_monitor_is_read_only():
    result = memory_policy_outcome_monitor(limit=10, stale_after_hours=72)

    assert result["success"] is True
    assert result["policy"]["monitor_is_read_only"] is True
    assert result["read_only_memory"] is True
    assert result["would_modify_config"] is False


def test_memory_recall_quality_evaluate_is_read_only():
    result = memory_recall_quality_evaluate(queries="memory,policy", limit=3)

    assert result["success"] is True
    assert result["evaluation_type"] == "hermes_memory_recall_quality_evaluate"
    assert result["summary"]["benchmark_query_count"] == 2
    assert result["policy"]["evaluation_is_read_only"] is True
    assert result["policy"]["does_not_append_ledger_events"] is True
    assert result["read_only_memory"] is True
    assert result["would_mutate_memory"] is False


def _ready_federation_status():
    return {
        "ready": True,
        "clients": {
            "hermes": {"role": "primary_memory_owner"},
            "codex": {
                "role": "memory_client",
                "access_path": "codex_mcp",
                "write_policy": "proposal_only",
                "ready": True,
            },
            "openclaw": {
                "role": "memory_client",
                "access_path": "openclaw_plugin",
                "plugin_enabled": True,
                "conversation_access_allowed": True,
                "auto_precheck_enabled": True,
                "auto_precheck_agent_profiles": ["default"],
                "external_auto_precheck_allowed_channels": [],
                "external_auto_precheck_default": "blocked",
                "write_policy": "proposal_only",
                "ready": True,
            },
        },
        "policy": {
            "writes_are_proposal_only": True,
            "external_channel_auto_recall_requires_allowlist": True,
        },
    }


def _ready_federation_audit(*, log_limit=200):
    return {
        "ready": True,
        "checks": [
            {"id": "writes.proposal_only", "status": "pass", "severity": "critical"},
            {"id": "external.default_blocked", "status": "pass", "severity": "critical"},
        ],
    }


def _healthy_policy_outcome(*, limit=50, stale_after_hours=72):
    return {"health_score": 100, "risk_level": "low"}


def _healthy_ledger(*, limit=500, client="", operation=""):
    return {"health_score": 100, "risk_level": "low", "findings": []}


def test_memory_boundary_allowlist_audit_requires_manual_review_evidence(monkeypatch):
    monkeypatch.setattr(memory_fabric_bridge, "memory_federation_status", _ready_federation_status)
    monkeypatch.setattr(memory_fabric_bridge, "memory_federation_audit", _ready_federation_audit)
    monkeypatch.setattr(memory_fabric_bridge, "memory_policy_outcome_monitor", _healthy_policy_outcome)
    monkeypatch.setattr(memory_fabric_bridge, "memory_ledger_intelligence", _healthy_ledger)

    result = memory_boundary_allowlist_audit(log_limit=200)

    assert result["success"] is True
    assert result["audit_type"] == "hermes_memory_boundary_allowlist_audit"
    assert result["ready"] is False
    assert result["boundary_readiness_score"] < 100
    assert result["reviewed"] is False
    assert result["unreviewed_allowlists"][0]["id"] == "openclaw.external_auto_precheck_boundary_review"
    assert result["evidence"]["manual_boundary_review"]["complete"] is False
    assert result["evidence"]["external_auto_precheck_default_blocked"] is True
    assert result["evidence"]["external_auto_precheck_allowlist_empty"] is True
    assert any("formal policy proposal or manual review record" in action for action in result["recommended_next_actions"])
    assert result["policy"]["audit_is_read_only"] is True
    assert result["policy"]["does_not_modify_config"] is True
    assert result["policy"]["does_not_write_memory"] is True
    assert result["policy"]["does_not_write_graph"] is True
    assert result["policy"]["does_not_approve_allowlists"] is True
    assert result["policy"]["does_not_enable_external_recall"] is True
    assert result["would_mutate_memory"] is False
    assert result["would_modify_config"] is False
    assert result["would_write_graph"] is False


def test_memory_boundary_allowlist_audit_accepts_existing_policy_review_evidence(monkeypatch):
    monkeypatch.setattr(memory_fabric_bridge, "memory_federation_status", _ready_federation_status)
    monkeypatch.setattr(memory_fabric_bridge, "memory_federation_audit", _ready_federation_audit)
    monkeypatch.setattr(memory_fabric_bridge, "memory_policy_outcome_monitor", _healthy_policy_outcome)
    monkeypatch.setattr(memory_fabric_bridge, "memory_ledger_intelligence", _healthy_ledger)

    def policy_ledger(*, limit=500, status="", proposal_id=""):
        return {
            "exists": True,
            "proposals": [
                {
                    "proposal_id": "memory-policy-proposal-boundary-review",
                    "suggestion_id": "external_auto_recall.keep_blocked",
                    "target": "openclaw.autoPrecheckAllowedChannelIds",
                    "latest_status": "approved",
                    "decisions": [
                        {
                            "decision": "approved",
                            "reviewer": "human-reviewer",
                            "created_at": "2026-05-23T00:00:00+00:00",
                        }
                    ],
                }
            ],
        }

    monkeypatch.setattr(memory_fabric_bridge, "memory_policy_proposal_ledger", policy_ledger)

    result = memory_boundary_allowlist_audit(log_limit=200)

    assert result["ready"] is True
    assert result["reviewed"] is True
    assert result["unreviewed_allowlists"] == []
    assert result["boundary_readiness_score"] == 100
    assert result["evidence"]["manual_boundary_review"]["complete"] is True
    assert result["evidence"]["manual_boundary_review"]["source"] == "policy_proposal_ledger"
    assert result["reviewed_allowlists"][0]["review"] == "empty_allowlist_reviewed_with_manual_evidence"


def test_memory_evolution_does_not_claim_star_realm_without_boundary_review():
    result = memory_evolution_status()

    assert result["success"] is True
    assert result["evidence"]["boundary_allowlists_reviewed"] is False
    assert result["evidence"]["boundary_readiness_score"] < 100
