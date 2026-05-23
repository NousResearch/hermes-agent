import json

import agent.memory_fabric_bridge as memory_fabric_bridge
from agent.memory_fabric_bridge import (
    memory_boundary_allowlist_audit,
    MEMORY_EVOLUTION_TIERS,
    memory_bridge_status,
    memory_evolution_status,
    memory_federation_gate,
    memory_orchestration_routing_metrics,
    memory_policy_outcome_monitor,
    memory_recall_quality_evaluate,
)


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


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


def test_memory_orchestration_routing_metrics_ready_from_local_evidence(tmp_path, monkeypatch):
    openclaw = tmp_path / ".openclaw"
    openclaw.mkdir()
    (openclaw / "openclaw.json").write_text(
        """
        {
          "agents": {
            "entries": {
              "planner": {},
              "coder": {},
              "reviewer": {}
            }
          },
          "bindings": [
            {
              "type": "route",
              "agentId": "planner",
              "match": {
                "channel": "telegram",
                "accountId": "bot-main",
                "peerKind": "group",
                "peerId": "team"
              }
            },
            {
              "type": "route",
              "agentId": "coder",
              "match": {
                "channel": "slack",
                "accountId": "workspace-main",
                "peerKind": "channel",
                "peerId": "eng"
              }
            }
          ],
          "plugins": {
            "entries": {
              "hermes-memory": {
                "enabled": true,
                "config": {
                  "autoPrecheckAgentIds": ["planner", "coder"],
                  "autoPrecheckAgentProfiles": {
                    "default": {"scope": "project"}
                  }
                }
              }
            }
          }
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(tmp_path))

    def operation_ledger(*, limit=500, client="", operation="", decision="", event_type=""):
        return {
            "events": [
                {"client": "openclaw", "operation": "route_memory_request", "decision": "allow"},
                {"client": "hermes", "operation": "auto_precheck", "decision": "allow"},
                {"client": "codex", "event_type": "gate_decision", "operation": "search", "decision": "block"},
                {"client": "openclaw", "operation": "memory_search", "decision": "allow"},
            ]
        }

    monkeypatch.setattr(memory_fabric_bridge, "memory_operation_ledger", operation_ledger)

    result = memory_orchestration_routing_metrics()

    assert result["success"] is True
    assert result["metrics_type"] == "hermes_memory_orchestration_routing_metrics"
    assert result["agent_count"] == 3
    assert result["route_binding_count"] == 2
    assert result["routed_agent_count"] == 2
    assert result["routed_agent_ids"] == ["coder", "planner"]
    assert result["route_channel_counts"] == {"slack": 1, "telegram": 1}
    assert result["auto_precheck_profile_count"] == 1
    assert result["auto_precheck_agent_ids"] == ["planner", "coder"]
    assert result["operation_routing_event_count"] == 1
    assert result["has_explicit_routing_operation_event"] is True
    assert result["gate_decision_count"] == 1
    assert result["auto_precheck_operation_count"] == 1
    assert result["client_counts"] == {"codex": 1, "hermes": 1, "openclaw": 2}
    assert result["decision_counts"] == {"allow": 3, "block": 1}
    assert result["routing_readiness_score"] == 1.0
    assert result["ready"] is True
    assert result["active_routing_metrics"] is True
    assert result["policy"]["metrics_are_read_only"] is True
    assert result["policy"]["does_not_modify_openclaw_config"] is True
    assert result["policy"]["does_not_write_memory"] is True
    assert result["would_modify_config"] is False
    assert result["would_write_graph"] is False


def test_memory_orchestration_routing_metrics_requires_explicit_routing_event(tmp_path, monkeypatch):
    openclaw = tmp_path / ".openclaw"
    openclaw.mkdir()
    (openclaw / "openclaw.json").write_text(
        """
        {
          "agents": {
            "entries": {
              "planner": {},
              "coder": {},
              "reviewer": {}
            }
          },
          "bindings": [
            {
              "type": "route",
              "agentId": "planner",
              "match": {"channel": "telegram"}
            }
          ],
          "plugins": {
            "entries": {
              "hermes-memory": {
                "enabled": true,
                "config": {
                  "autoPrecheckAgentIds": ["planner"],
                  "autoPrecheckAgentProfiles": {
                    "default": {"scope": "project"}
                  }
                }
              }
            }
          }
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(tmp_path))

    def operation_ledger(*, limit=500, client="", operation="", decision="", event_type=""):
        return {
            "events": [
                {"client": "hermes", "operation": "auto_precheck", "decision": "allow"},
                {"client": "codex", "event_type": "gate_decision", "operation": "search", "decision": "block"},
            ]
        }

    monkeypatch.setattr(memory_fabric_bridge, "memory_operation_ledger", operation_ledger)

    result = memory_orchestration_routing_metrics()

    assert result["agent_count"] == 3
    assert result["route_binding_count"] == 1
    assert result["auto_precheck_profile_count"] == 1
    assert result["operation_routing_event_count"] == 0
    assert result["has_explicit_routing_operation_event"] is False
    assert result["gate_decision_count"] == 1
    assert result["auto_precheck_operation_count"] == 1
    assert result["client_counts"] == {"codex": 1, "hermes": 1}
    assert result["decision_counts"] == {"allow": 1, "block": 1}
    assert result["routing_readiness_score"] == 0.75
    assert result["ready"] is False
    assert result["active_routing_metrics"] is False
    assert "Hermes operation ledger has at least 1 explicit route/routing/orchestration event." in result["gaps"]
    assert result["recommended_next_actions"] == [
        "Keep Star Hub pending until a governed routing event is recorded in the Hermes operation ledger; do not modify config from this tool."
    ]


def test_openclaw_route_bindings_parse_top_level_bindings():
    bindings = memory_fabric_bridge._openclaw_route_bindings(
        {
            "bindings": [
                {
                    "type": "route",
                    "agentId": "telegram-agent",
                    "match": {
                        "channel": "telegram",
                        "accountId": "bot-main",
                        "peerKind": "group",
                        "peerId": "team",
                    },
                },
                {
                    "type": "tool",
                    "agentId": "ignored-agent",
                    "match": {"channel": "telegram"},
                },
            ]
        }
    )

    assert bindings == [
        {
            "agent_id": "telegram-agent",
            "channel": "telegram",
            "source": "bindings/0",
            "account_id": "bot-main",
            "peer_kind": "group",
            "peer_id": "team",
        }
    ]


def test_memory_orchestration_routing_metrics_reports_gaps(tmp_path, monkeypatch):
    (tmp_path / ".openclaw").mkdir()
    (tmp_path / ".openclaw" / "openclaw.json").write_text(
        '{"agents": {"entries": {"planner": {}}}, "plugins": {"entries": {"hermes-memory": {"config": {}}}}}',
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(memory_fabric_bridge, "memory_operation_ledger", lambda **kwargs: {"events": []})

    result = memory_orchestration_routing_metrics()

    assert result["ready"] is False
    assert result["active_routing_metrics"] is False
    assert result["routing_readiness_score"] == 0.0
    assert "OpenClaw has at least 3 configured agents." in result["gaps"]
    assert "OpenClaw has at least 1 route binding." in result["gaps"]
    assert "Hermes memory auto-precheck has at least 1 profile." in result["gaps"]
    assert "Hermes operation ledger has at least 1 explicit route/routing/orchestration event." in result["gaps"]


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


def _star_law_ready_policy_outcome(*, limit=50, stale_after_hours=72):
    return {
        "health_score": 100,
        "risk_level": "low",
        "metrics": {
            "stale_proposed_count": 0,
            "approved_not_executed_count": 0,
            "execution_count": 1,
            "execution_totals": {
                "checked_count": 1,
                "passed_count": 1,
                "blocked_count": 0,
                "failed_count": 0,
                "manual_required_count": 0,
            },
        },
    }


def _star_law_blocked_policy_outcome(*, limit=50, stale_after_hours=72):
    return {
        "health_score": 90,
        "risk_level": "medium",
        "metrics": {
            "stale_proposed_count": 0,
            "approved_not_executed_count": 1,
            "execution_count": 0,
            "execution_totals": {
                "checked_count": 0,
                "passed_count": 0,
                "blocked_count": 0,
                "failed_count": 0,
                "manual_required_count": 0,
            },
        },
    }


def _patch_star_law_prerequisites(monkeypatch, policy_outcome):
    monkeypatch.setattr(
        memory_fabric_bridge,
        "memory_bridge_status",
        lambda: {
            "hermes_home": "/tmp/hermes",
            "surfaces": {
                "graph": {"exists": True, "node_count": 3, "edge_count": 2, "provenance_count": 1},
                "gpt_image_prompt_cases": {"exists": True, "case_count": 1},
                "knowledge": {"exists": True, "file_count": 1},
                "operation_ledger": {"exists": True, "event_count": 3},
                "policy_proposals": {"exists": True, "event_count": 1},
            },
        },
    )
    monkeypatch.setattr(memory_fabric_bridge, "memory_federation_status", _ready_federation_status)
    monkeypatch.setattr(
        memory_fabric_bridge,
        "memory_boundary_allowlist_audit",
        lambda *, log_limit=200: {"ready": True, "boundary_readiness_score": 100},
    )
    monkeypatch.setattr(
        memory_fabric_bridge,
        "memory_orchestration_routing_metrics",
        lambda: {
            "ready": True,
            "active_routing_metrics": True,
            "routing_readiness_score": 1.0,
            "agent_count": 3,
            "route_binding_count": 1,
            "operation_routing_event_count": 1,
            "gate_decision_count": 1,
            "auto_precheck_operation_count": 1,
        },
    )
    monkeypatch.setattr(memory_fabric_bridge, "memory_policy_outcome_monitor", policy_outcome)
    monkeypatch.setattr(
        memory_fabric_bridge,
        "memory_recall_quality_evaluate",
        lambda *, limit=5: {
            "readiness": "ready",
            "quality_score": 1.0,
            "summary": {"passed_query_count": 5, "benchmark_query_count": 5},
        },
    )


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


def test_memory_evolution_uses_orchestration_routing_metrics_for_star_hub(monkeypatch):
    monkeypatch.setattr(
        memory_fabric_bridge,
        "memory_bridge_status",
        lambda: {
            "hermes_home": "/tmp/hermes",
            "surfaces": {
                "graph": {"exists": True, "node_count": 3, "edge_count": 2, "provenance_count": 1},
                "gpt_image_prompt_cases": {"exists": True, "case_count": 1},
                "knowledge": {"exists": True, "file_count": 1},
                "operation_ledger": {"exists": True, "event_count": 3},
                "policy_proposals": {"exists": True, "event_count": 1},
            },
        },
    )
    monkeypatch.setattr(memory_fabric_bridge, "memory_federation_status", _ready_federation_status)
    monkeypatch.setattr(
        memory_fabric_bridge,
        "memory_boundary_allowlist_audit",
        lambda *, log_limit=200: {"ready": True, "boundary_readiness_score": 100},
    )
    monkeypatch.setattr(
        memory_fabric_bridge,
        "memory_orchestration_routing_metrics",
        lambda: {
            "ready": True,
            "active_routing_metrics": True,
            "routing_readiness_score": 1.0,
            "agent_count": 3,
            "route_binding_count": 1,
            "operation_routing_event_count": 1,
            "gate_decision_count": 1,
            "auto_precheck_operation_count": 1,
        },
    )
    monkeypatch.setattr(memory_fabric_bridge, "memory_policy_outcome_monitor", _healthy_policy_outcome)
    monkeypatch.setattr(
        memory_fabric_bridge,
        "memory_recall_quality_evaluate",
        lambda *, limit=5: {
            "readiness": "ready",
            "quality_score": 1.0,
            "summary": {"passed_query_count": 5, "benchmark_query_count": 5},
        },
    )

    result = memory_evolution_status()
    star_hub = next(item for item in result["readiness"] if item["level"] == 11)

    assert result["evidence"]["active_routing_metrics"] is True
    assert result["evidence"]["has_explicit_routing_operation_event"] is True
    assert result["evidence"]["routing_readiness_score"] == 1.0
    assert result["orchestration_routing_metrics"]["ready"] is True
    assert star_hub["achieved"] is True
    assert "Memory orchestration has active routing metrics" in star_hub["passed_criteria"]
    assert "Memory orchestration has explicit route/routing/orchestration operation evidence" in star_hub["passed_criteria"]


def test_memory_evolution_does_not_claim_star_hub_without_explicit_routing_event(monkeypatch):
    monkeypatch.setattr(
        memory_fabric_bridge,
        "memory_bridge_status",
        lambda: {
            "hermes_home": "/tmp/hermes",
            "surfaces": {
                "graph": {"exists": True, "node_count": 3, "edge_count": 2, "provenance_count": 1},
                "gpt_image_prompt_cases": {"exists": True, "case_count": 1},
                "knowledge": {"exists": True, "file_count": 1},
                "operation_ledger": {"exists": True, "event_count": 3},
                "policy_proposals": {"exists": True, "event_count": 1},
            },
        },
    )
    monkeypatch.setattr(memory_fabric_bridge, "memory_federation_status", _ready_federation_status)
    monkeypatch.setattr(
        memory_fabric_bridge,
        "memory_boundary_allowlist_audit",
        lambda *, log_limit=200: {"ready": True, "boundary_readiness_score": 100},
    )
    monkeypatch.setattr(
        memory_fabric_bridge,
        "memory_orchestration_routing_metrics",
        lambda: {
            "ready": True,
            "active_routing_metrics": True,
            "routing_readiness_score": 1.0,
            "agent_count": 3,
            "route_binding_count": 1,
            "operation_routing_event_count": 0,
            "gate_decision_count": 1,
            "auto_precheck_operation_count": 1,
        },
    )
    monkeypatch.setattr(memory_fabric_bridge, "memory_policy_outcome_monitor", _healthy_policy_outcome)
    monkeypatch.setattr(
        memory_fabric_bridge,
        "memory_recall_quality_evaluate",
        lambda *, limit=5: {
            "readiness": "ready",
            "quality_score": 1.0,
            "summary": {"passed_query_count": 5, "benchmark_query_count": 5},
        },
    )

    result = memory_evolution_status()
    star_hub = next(item for item in result["readiness"] if item["level"] == 11)

    assert result["evidence"]["active_routing_metrics"] is True
    assert result["evidence"]["has_explicit_routing_operation_event"] is False
    assert star_hub["achieved"] is False
    assert "Memory orchestration has active routing metrics" in star_hub["passed_criteria"]
    assert "Memory orchestration has explicit route/routing/orchestration operation evidence" in star_hub["gaps"]


def test_memory_evolution_claims_star_law_after_guarded_policy_checks(monkeypatch):
    _patch_star_law_prerequisites(monkeypatch, _star_law_ready_policy_outcome)

    result = memory_evolution_status()
    star_law = next(item for item in result["readiness"] if item["level"] == 12)

    assert result["current"]["level"] == 12
    assert result["current"]["name"] == "星律记忆"
    assert result["evidence"]["policy_execution_count"] == 1
    assert result["evidence"]["policy_approved_not_executed_count"] == 0
    assert result["evidence"]["policy_execution_checked_count"] == 1
    assert result["evidence"]["policy_execution_passed_count"] == 1
    assert result["evidence"]["policy_execution_blocked_count"] == 0
    assert result["evidence"]["policy_execution_failed_count"] == 0
    assert result["evidence"]["policy_execution_manual_required_count"] == 0
    assert result["evidence"]["policy_closed_loop_ready"] is True
    assert star_law["achieved"] is True
    assert "Policy proposals close automatically after human approval and guarded checks" in star_law["passed_criteria"]
    assert not any("guarded non-mutating policy apply checks" in action for action in result["recommended_next_actions"])


def test_memory_evolution_keeps_star_soul_blocked_without_write_proposal(tmp_path, monkeypatch):
    monkeypatch.setattr(memory_fabric_bridge, "get_hermes_home", lambda: tmp_path)
    _patch_star_law_prerequisites(monkeypatch, _star_law_ready_policy_outcome)

    result = memory_evolution_status()
    star_soul = next(item for item in result["readiness"] if item["level"] == 13)

    assert result["current"]["level"] == 12
    assert result["next"]["level"] == 13
    assert result["evidence"]["write_proposal_count"] == 0
    assert result["evidence"]["write_proposal_operation_event_count"] == 0
    assert result["evidence"]["persona_continuity_write_proposal_count"] == 0
    assert result["evidence"]["persona_continuity_governed"] is False
    assert star_soul["achieved"] is False
    assert "Long-term preference/persona continuity is governed" in star_soul["gaps"]
    assert any("memory_write_proposal" in action and "do not write memory directly" in action for action in result["recommended_next_actions"])


def test_memory_evolution_keeps_star_soul_blocked_for_generic_write_proposal(tmp_path, monkeypatch):
    monkeypatch.setattr(memory_fabric_bridge, "get_hermes_home", lambda: tmp_path)
    _patch_star_law_prerequisites(monkeypatch, _star_law_ready_policy_outcome)
    _write_jsonl(
        memory_fabric_bridge._proposal_path(tmp_path),
        [
            {
                "proposal_id": "memory-write-proposal-generic",
                "target_scope": "project",
                "content": "Store deployment checklist notes.",
                "rationale": "Improve release handoff.",
                "tags": ["release"],
                "status": "proposed",
                "would_write_memory": False,
                "would_modify_graph": False,
            }
        ],
    )
    _write_jsonl(
        memory_fabric_bridge._operation_ledger_path(tmp_path),
        [
            {
                "event_type": "write_proposal_created",
                "operation": "write_proposal",
                "proposal_id": "memory-write-proposal-generic",
                "would_write_memory": False,
                "would_modify_config": False,
            }
        ],
    )

    result = memory_evolution_status()
    star_soul = next(item for item in result["readiness"] if item["level"] == 13)

    assert result["current"]["level"] == 12
    assert result["evidence"]["write_proposal_count"] == 1
    assert result["evidence"]["write_proposal_operation_event_count"] == 1
    assert result["evidence"]["persona_continuity_write_proposal_count"] == 0
    assert result["evidence"]["persona_continuity_governed"] is False
    assert star_soul["achieved"] is False


def test_memory_evolution_keeps_star_soul_blocked_for_mismatched_persona_proposal_event(tmp_path, monkeypatch):
    monkeypatch.setattr(memory_fabric_bridge, "get_hermes_home", lambda: tmp_path)
    _patch_star_law_prerequisites(monkeypatch, _star_law_ready_policy_outcome)
    _write_jsonl(
        memory_fabric_bridge._proposal_path(tmp_path),
        [
            {
                "proposal_id": "memory-write-proposal-star-soul",
                "source_agent": "codex",
                "target_scope": "user",
                "content": "Proposal for long-term collaboration style and preference continuity.",
                "rationale": "Govern 星魂记忆 persona continuity without writing memory directly.",
                "tags": ["persona", "preferences", "collaboration"],
                "status": "proposed",
                "would_write_memory": False,
                "would_modify_graph": False,
            }
        ],
    )
    _write_jsonl(
        memory_fabric_bridge._operation_ledger_path(tmp_path),
        [
            {
                "event_type": "write_proposal_created",
                "operation": "write_proposal",
                "proposal_id": "memory-write-proposal-other",
                "would_write_memory": False,
                "would_modify_config": False,
            }
        ],
    )

    result = memory_evolution_status()
    star_soul = next(item for item in result["readiness"] if item["level"] == 13)

    assert result["current"]["level"] == 12
    assert result["evidence"]["write_proposal_count"] == 1
    assert result["evidence"]["write_proposal_operation_event_count"] == 1
    assert result["evidence"]["persona_continuity_write_proposal_count"] == 1
    assert result["evidence"]["persona_continuity_governed_proposal_ids"] == []
    assert result["evidence"]["persona_continuity_governed_event_count"] == 0
    assert result["evidence"]["persona_continuity_governed"] is False
    assert star_soul["achieved"] is False
    assert "Long-term preference/persona continuity is governed" in star_soul["gaps"]


def test_memory_evolution_claims_star_soul_for_governed_persona_continuity_proposal(tmp_path, monkeypatch):
    monkeypatch.setattr(memory_fabric_bridge, "get_hermes_home", lambda: tmp_path)
    _patch_star_law_prerequisites(monkeypatch, _star_law_ready_policy_outcome)
    _write_jsonl(
        memory_fabric_bridge._proposal_path(tmp_path),
        [
            {
                "proposal_id": "memory-write-proposal-star-soul",
                "source_agent": "codex",
                "target_scope": "user",
                "content": "Proposal for long-term collaboration style and preference continuity.",
                "rationale": "Govern 星魂记忆 persona continuity without writing memory directly.",
                "tags": ["persona", "preferences", "collaboration"],
                "status": "proposed",
                "would_write_memory": False,
                "would_modify_graph": False,
            }
        ],
    )
    _write_jsonl(
        memory_fabric_bridge._operation_ledger_path(tmp_path),
        [
            {
                "event_type": "write_proposal_created",
                "operation": "write_proposal",
                "proposal_id": "memory-write-proposal-star-soul",
                "would_write_memory": False,
                "would_modify_config": False,
            }
        ],
    )

    result = memory_evolution_status()
    star_soul = next(item for item in result["readiness"] if item["level"] == 13)

    assert result["current"]["level"] == 13
    assert result["current"]["name"] == "星魂记忆"
    assert result["evidence"]["write_proposal_count"] == 1
    assert result["evidence"]["write_proposal_operation_event_count"] == 1
    assert result["evidence"]["persona_continuity_write_proposal_count"] == 1
    assert result["evidence"]["persona_continuity_governed_proposal_ids"] == ["memory-write-proposal-star-soul"]
    assert result["evidence"]["persona_continuity_governed_event_count"] == 1
    assert result["evidence"]["persona_continuity_governed"] is True
    assert star_soul["achieved"] is True
    assert "Long-term preference/persona continuity is governed" in star_soul["passed_criteria"]
    assert not any("memory_write_proposal" in action for action in result["recommended_next_actions"])
    assert result["read_only"] is True
    assert result["policy"]["status_is_read_only"] is True
    assert result["policy"]["does_not_modify_config"] is True
    assert result["policy"]["does_not_write_memory"] is True
    assert result["policy"]["persistent_changes_must_use_proposals"] is True
    assert result["would_mutate_memory"] is False
    assert result["would_modify_config"] is False


def test_memory_evolution_keeps_star_law_blocked_without_policy_execution(monkeypatch):
    _patch_star_law_prerequisites(monkeypatch, _star_law_blocked_policy_outcome)

    result = memory_evolution_status()
    star_law = next(item for item in result["readiness"] if item["level"] == 12)

    assert result["current"]["level"] == 11
    assert result["next"]["level"] == 12
    assert result["evidence"]["policy_execution_count"] == 0
    assert result["evidence"]["policy_approved_not_executed_count"] == 1
    assert result["evidence"]["policy_closed_loop_ready"] is False
    assert star_law["achieved"] is False
    assert "Policy proposals close automatically after human approval and guarded checks" in star_law["gaps"]
    assert any("guarded non-mutating policy apply checks" in action for action in result["recommended_next_actions"])


def test_memory_evolution_does_not_claim_star_realm_without_boundary_review():
    result = memory_evolution_status()

    assert result["success"] is True
    assert result["evidence"]["boundary_allowlists_reviewed"] is False
    assert result["evidence"]["boundary_readiness_score"] < 100
