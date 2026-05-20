from agent.memory_evidence_repair_planner import (
    ACTION_ATTACH_PROVENANCE,
    ACTION_REFRESH_OBSERVATION,
    build_evidence_repair_plan,
    empty_evidence_repair_plan,
)


def test_plan_from_policy_gate_filters_requires_provenance():
    plan = build_evidence_repair_plan(
        gate={
            "provider_results": [
                {
                    "provider": "builtin",
                    "action": "require_provenance_before_reuse",
                    "effect": "filtered",
                    "reason": "Recall candidate lacks explicit provenance required by policy.",
                },
                {
                    "provider": "graph",
                    "action": "allow_memory_recall",
                    "effect": "allowed",
                    "reason": "No restriction.",
                },
            ]
        }
    )

    payload = plan.to_dict()
    assert payload["read_only"] is True
    assert payload["summary"]["repair_count"] == 1
    assert payload["summary"]["blocked_count"] == 1
    assert payload["repairs"][0]["provider"] == "builtin"
    assert payload["repairs"][0]["action"] == ACTION_ATTACH_PROVENANCE
    assert "source_url" in payload["repairs"][0]["required_evidence"]


def test_plan_from_diagnostics_and_audit():
    plan = build_evidence_repair_plan(
        diagnostics={
            "issues": [
                {
                    "provider": "graph",
                    "code": "refresh_stale_recall",
                    "reason": "Recall may be stale.",
                    "evidence": "staleness=0.7",
                }
            ]
        },
        audit={
            "entries": [
                {
                    "provider": "builtin",
                    "decision": "injected",
                    "recommendation": "needs_evidence",
                    "dimensions": {"source_strength": 0.2, "evidence_strength": 0.3},
                    "content_preview": "Remember a weakly sourced fact.",
                }
            ]
        },
    )

    payload = plan.to_dict()
    actions = {repair["action"] for repair in payload["repairs"]}
    assert ACTION_ATTACH_PROVENANCE in actions
    assert ACTION_REFRESH_OBSERVATION in actions
    assert payload["summary"]["by_provider"] == {"builtin": 1, "graph": 1}


def test_empty_plan_is_read_only():
    payload = empty_evidence_repair_plan().to_dict()

    assert payload["read_only"] is True
    assert payload["summary"]["repair_count"] == 0
    assert payload["repairs"] == []
