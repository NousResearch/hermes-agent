from agent.memory_evidence_repair_approval import (
    CANDIDATE_TYPE,
    PENDING_VALUE,
    STATUS_NEEDS_CONFIRMATION,
    build_evidence_repair_approval_candidates,
    empty_evidence_repair_approval_candidates,
)


def test_builds_approval_candidate_from_repair_plan():
    report = build_evidence_repair_approval_candidates(
        plan={
            "repairs": [
                {
                    "provider": "builtin",
                    "priority": "high",
                    "action": "attach_provenance",
                    "source": "policy_gate",
                    "reason": "Missing explicit provenance.",
                    "content_preview": "Weak memory content.",
                    "required_evidence": [
                        "source_url",
                        "observed_at",
                        "verification_signal",
                    ],
                }
            ],
            "read_only": True,
        }
    )

    payload = report.to_dict()
    candidate = payload["candidates"][0]
    assert payload["read_only"] is True
    assert payload["read_only_memory"] is True
    assert payload["summary"]["candidate_count"] == 1
    assert payload["summary"]["requires_user_confirmation"] is True
    assert candidate["id"].startswith("repair-")
    assert candidate["candidate_type"] == CANDIDATE_TYPE
    assert candidate["status"] == STATUS_NEEDS_CONFIRMATION
    assert candidate["provider"] == "builtin"
    assert candidate["required_evidence"] == [
        "source_url",
        "observed_at",
        "verification_signal",
    ]
    assert candidate["proposed_evidence_patch"] == {
        "source_url": PENDING_VALUE,
        "observed_at": PENDING_VALUE,
        "verification_signal": PENDING_VALUE,
    }


def test_candidate_id_is_stable_and_proposed_evidence_fills_patch():
    plan = {
        "repairs": [
            {
                "provider": "graph",
                "priority": "medium",
                "action": "refresh_observation",
                "source": "diagnostics",
                "reason": "Recall may be stale.",
                "required_evidence": ["observed_at", "freshness_check"],
            }
        ]
    }

    first = build_evidence_repair_approval_candidates(
        plan=plan,
        proposed_evidence={
            "observed_at": "2026-05-11T00:00:00Z",
            "freshness_check": "confirmed_current",
        },
    ).to_dict()
    second = build_evidence_repair_approval_candidates(
        plan=plan,
        proposed_evidence={
            "observed_at": "2026-05-11T00:00:00Z",
            "freshness_check": "confirmed_current",
        },
    ).to_dict()

    first_candidate = first["candidates"][0]
    second_candidate = second["candidates"][0]
    assert first_candidate["id"] == second_candidate["id"]
    assert first_candidate["proposed_evidence_patch"] == {
        "observed_at": "2026-05-11T00:00:00Z",
        "freshness_check": "confirmed_current",
    }
    assert first["summary"]["by_provider"] == {"graph": 1}
    assert first["summary"]["by_repair_action"] == {"refresh_observation": 1}


def test_empty_approval_report_is_read_only():
    payload = empty_evidence_repair_approval_candidates().to_dict()

    assert payload["read_only"] is True
    assert payload["read_only_memory"] is True
    assert payload["summary"]["candidate_count"] == 0
    assert payload["summary"]["requires_user_confirmation"] is False
    assert payload["candidates"] == []
