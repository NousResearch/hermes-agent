from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path


def _loads(value: str) -> dict:
    return json.loads(value)


def _objective(objective_id: str, *, targets: list[str] | None = None) -> dict:
    return {
        "objective_id": objective_id,
        "capability": f"Demonstrate {objective_id} under an observable rubric.",
        "success_criteria": ["The result is correct and independently evaluated."],
        "evidence_targets": targets or ["execution"],
        "source_anchors": [],
    }


def _project(
    *,
    project_id: str = "learning-runtime",
    deadline: str = "2026-12-31",
    objectives: list[dict] | None = None,
) -> dict:
    return {
        "schema_version": "study_project.v2",
        "project_id": project_id,
        "title": "Learning Runtime",
        "domain": "engineering",
        "timezone": "Asia/Shanghai",
        "phase": "implementation",
        "domain_pack": "engineering.v1",
        "workspace_type": "engineering-repo",
        "artifact_policy": "source-and-command",
        "deadline": deadline,
        "tracks": [{"id": "runtime", "label": "Runtime"}],
        "objectives": objectives or [_objective("trace-request")],
        "prompt_policy": {
            "base_max_chars": 2000,
            "intent_max_chars": 2500,
            "domain_max_chars": 2000,
            "project_summary_max_chars": 1200,
            "total_max_chars": 6000,
            "updates_apply": "next_session",
        },
        "created_at": "2026-07-01T09:00:00+08:00",
        "updated_at": "2026-07-01T09:00:00+08:00",
    }


def _attempt(
    attempt_id: str,
    *,
    objective_id: str,
    occurred_at: str,
    assistance: str,
    score: float = 1.0,
) -> dict:
    return {
        "attempt_id": attempt_id,
        "occurred_at": occurred_at,
        "objective_ids": [objective_id],
        "transfer_level": "execution",
        "score": score,
        "result": "correct" if score >= 0.8 else "partial",
        "assistance": {"level": assistance, "hints_used": 0},
        "evaluator": {"kind": "program", "confidence": 0.9},
        "concepts": [],
        "diagnoses": [],
    }


def _build(project: dict, attempts: list[dict], as_of: str) -> dict:
    from plugins.study_os.interventions import InterventionOrchestrator, parse_as_of
    from plugins.study_os.learning import _diagnosis

    return InterventionOrchestrator(
        project=project,
        diagnosis_builder=_diagnosis,
    ).build(
        attempts=attempts,
        as_of=parse_as_of(as_of),
    )


def test_queue_selects_one_current_gap_per_objective_and_skips_fresh_independent():
    project = _project(
        objectives=[
            _objective("supported-objective"),
            _objective("independent-objective"),
        ]
    )
    attempts = [
        _attempt(
            "att-supported",
            objective_id="supported-objective",
            occurred_at="2026-07-12T10:00:00+08:00",
            assistance="guided",
        ),
        _attempt(
            "att-independent",
            objective_id="independent-objective",
            occurred_at="2026-07-12T11:00:00+08:00",
            assistance="independent",
        ),
    ]

    result = _build(project, attempts, "2026-07-13T10:00:00+08:00")

    assert [item["objective_id"] for item in result["queue"]["items"]] == [
        "supported-objective"
    ]
    selected = result["queue"]["items"][0]
    assert selected["kind"] == "independence_probe"
    assert selected["reason_factors"]["verification_status"] == "supported"
    assert selected["recommended_activity"]["assistance_level"] == "independent"


def test_nearer_deadline_increases_priority_for_the_same_evidence_gap():
    far = _project(project_id="deadline-far", deadline="2027-12-31")
    near = _project(project_id="deadline-near", deadline="2026-07-20")

    far_item = _build(far, [], "2026-07-13T10:00:00+08:00")["queue"]["items"][0]
    near_item = _build(near, [], "2026-07-13T10:00:00+08:00")["queue"]["items"][0]

    assert far_item["kind"] == near_item["kind"] == "evidence_probe"
    assert near_item["priority_score"] > far_item["priority_score"]
    assert near_item["reason_factors"]["deadline_band"] == "critical"


def test_stale_independent_evidence_becomes_a_retention_probe():
    project = _project(deadline="2027-12-31")
    attempts = [
        _attempt(
            "att-independent",
            objective_id="trace-request",
            occurred_at="2026-05-01T10:00:00+08:00",
            assistance="independent",
        )
    ]

    fresh = _build(project, attempts, "2026-05-10T10:00:00+08:00")
    stale = _build(project, attempts, "2026-07-13T10:00:00+08:00")

    assert fresh["queue"]["items"] == []
    assert stale["queue"]["items"][0]["kind"] == "retention_probe"
    assert stale["queue"]["items"][0]["reason_factors"]["evidence_age_band"] == "stale"


def test_proposal_identity_is_stable_while_semantic_priority_bands_are_unchanged():
    project = _project(deadline="2027-12-31")

    first = _build(project, [], "2026-07-13T10:00:00+08:00")["proposal"]
    second = _build(project, [], "2026-07-14T10:00:00+08:00")["proposal"]

    assert first["proposal_id"] == second["proposal_id"]
    assert first["generation_fingerprint"] == second["generation_fingerprint"]
    assert first["created_at"] != second["created_at"]


def test_legacy_exam_project_gets_an_honest_project_level_intervention():
    project = {
        "schema_version": "study_project.v1",
        "project_id": "kaoyan-2027",
        "title": "2027 考研",
        "domain_pack": "kaoyan.v1",
        "exam_date": "2027-12-20",
    }

    result = _build(project, [], "2026-07-13T10:00:00+08:00")

    assert result["queue"]["items"][0]["objective_id"] == "project-readiness"
    assert result["queue"]["items"][0]["evidence_dimension"] == "recall"


def test_propose_save_and_accept_is_auditable_and_never_implicitly_saves_schedule(
    tmp_path: Path,
):
    from plugins.study_os.learning import handle_study_activity, handle_study_coach
    from plugins.study_os.tools import handle_study_project

    vault = tmp_path / "vault"
    vault.mkdir()
    project = _project()
    initialized = _loads(
        handle_study_project({"action": "init", "vault_path": str(vault), **project})
    )
    assert initialized["ok"] is True

    proposed = _loads(
        handle_study_coach({
            "action": "propose_plan",
            "scope": "project",
            "vault_path": str(vault),
            "project_id": project["project_id"],
            "data": {"as_of": "2026-07-13T10:00:00+08:00"},
        })
    )
    proposal = proposed["data"]["proposal"]
    proposal_root = (
        vault / ".StudyOS" / "projects" / project["project_id"] / "plan-proposals"
    )
    assert proposed["ok"] is True
    assert proposal_root.exists() is False

    tampered = deepcopy(proposal)
    tampered["items"][0]["capability"] = "A different capability"
    tampered_result = _loads(
        handle_study_activity({
            "resource": "plan_proposal",
            "action": "save",
            "vault_path": str(vault),
            "project_id": project["project_id"],
            "data": {"proposal": tampered},
        })
    )
    assert tampered_result["error"]["code"] == "PROPOSAL_FINGERPRINT_MISMATCH"

    smuggled_decision = deepcopy(proposal)
    smuggled_decision["status"] = "accepted"
    smuggled_decision["decision"] = {
        "outcome": "accepted",
        "decided_at": "2026-07-13T10:05:00+08:00",
    }
    smuggled_result = _loads(
        handle_study_activity(
            {
                "resource": "plan_proposal",
                "action": "save",
                "vault_path": str(vault),
                "project_id": project["project_id"],
                "data": {"proposal": smuggled_decision},
            },
            session_id="cron_weekly-study_20260713_100000",
        )
    )
    assert smuggled_result["error"]["code"] == "INVALID_PROPOSAL_TRANSITION"

    save_args = {
        "resource": "plan_proposal",
        "action": "save",
        "vault_path": str(vault),
        "project_id": project["project_id"],
        "data": {"proposal": proposal},
    }
    saved = _loads(handle_study_activity(save_args))
    saved_again = _loads(handle_study_activity(save_args))

    assert saved["data"]["created"] is True
    assert saved_again["data"]["created"] is False
    assert len(list(proposal_root.glob("*.json"))) == 1

    cron_decision = _loads(
        handle_study_activity(
            {
                "resource": "plan_proposal",
                "action": "accept",
                "vault_path": str(vault),
                "project_id": project["project_id"],
                "data": {"proposal_id": proposal["proposal_id"]},
            },
            session_id="cron_weekly-study_20260713_100000",
        )
    )
    assert cron_decision["error"]["code"] == "CRON_PROPOSAL_ONLY"

    accepted = _loads(
        handle_study_activity({
            "resource": "plan_proposal",
            "action": "accept",
            "vault_path": str(vault),
            "project_id": project["project_id"],
            "data": {
                "proposal_id": proposal["proposal_id"],
                "decided_at": "2026-07-13T10:10:00+08:00",
                "decision_note": "Apply this after choosing a free slot.",
            },
        })
    )

    schedule_root = (
        vault / ".StudyOS" / "projects" / project["project_id"] / "schedules"
    )
    assert accepted["ok"] is True
    assert accepted["data"]["proposal"]["status"] == "accepted"
    assert accepted["data"]["schedule_mutated"] is False
    assert not schedule_root.exists()


def test_cron_cannot_save_a_schedule_through_the_study_interface(tmp_path: Path):
    from plugins.study_os.learning import handle_study_activity

    result = _loads(
        handle_study_activity(
            {
                "resource": "schedule",
                "action": "save",
                "vault_path": str(tmp_path),
                "data": {},
            },
            session_id="cron_daily-study_20260713_100000",
        )
    )

    assert result["error"]["code"] == "CRON_PROPOSAL_ONLY"


def test_registered_model_interface_carries_proactive_actions_and_cron_policy(
    tmp_path: Path,
    monkeypatch,
):
    from hermes_cli import plugins as plugins_mod
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
    from plugins import study_os
    from tools.registry import registry

    manager = PluginManager()
    monkeypatch.setattr(plugins_mod, "_plugin_manager", manager)
    manifest = PluginManifest(
        name="study_os",
        version="0.1.0",
        description="study",
        source="bundled",
    )
    try:
        study_os.register(PluginContext(manifest, manager))
        activity_schema = registry.get_entry("study_activity").schema
        coach_schema = registry.get_entry("study_coach").schema
        resources = activity_schema["parameters"]["properties"]["resource"]["enum"]
        actions = coach_schema["parameters"]["properties"]["action"]["enum"]
        blocked = _loads(
            registry.dispatch(
                "study_activity",
                {
                    "resource": "schedule",
                    "action": "save",
                    "vault_path": str(tmp_path),
                    "data": {},
                },
                session_id="cron_registered_20260713_100000",
            )
        )

        assert "plan_proposal" in resources
        assert {"prioritize", "propose_plan"}.issubset(actions)
        assert blocked["error"]["code"] == "CRON_PROPOSAL_ONLY"
    finally:
        registry.deregister("study_activity")
        registry.deregister("study_coach")


def test_plan_proposal_validator_enforces_relational_invariants():
    from plugins.study_os.schemas import validate_plan_proposal

    proposal = _build(
        _project(deadline="2027-12-31"),
        [],
        "2026-07-13T10:00:00+08:00",
    )["proposal"]
    invalid = deepcopy(proposal)
    invalid["items"][0]["priority_band"] = "high"
    invalid["schedule_change"]["state"] = "applied"

    ok, errors = validate_plan_proposal(invalid)

    assert ok is False
    assert "items[0].priority_band must match priority_score" in errors
    assert "schedule_change.state must be not_applied" in errors
