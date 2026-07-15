from __future__ import annotations

import json
from pathlib import Path

import pytest


def _project(project_id: str, title: str) -> dict:
    return {
        "schema_version": "study_project.v1",
        "project_id": project_id,
        "title": title,
        "domain": "kaoyan",
        "exam_type": "考研",
        "exam_date": "2027-12-20",
        "timezone": "Asia/Shanghai",
        "phase": "foundation",
        "domain_pack": "kaoyan.v1",
        "subjects": [{"id": "math", "label": "数学", "target_score": 120}],
        "prompt_policy": {
            "base_max_chars": 2000,
            "intent_max_chars": 2500,
            "domain_max_chars": 2000,
            "project_summary_max_chars": 1200,
            "total_max_chars": 6000,
            "updates_apply": "next_session",
        },
        "created_at": "2026-07-01T00:00:00+08:00",
        "updated_at": "2026-07-01T00:00:00+08:00",
    }


def _write_project(vault: Path, project_id: str, title: str) -> None:
    project_dir = vault / ".StudyOS" / "projects" / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "manifest.json").write_text(
        json.dumps(_project(project_id, title), ensure_ascii=False),
        encoding="utf-8",
    )


def test_workspace_prefers_explicit_then_profile_config_over_legacy_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    from plugins.study_os.workspace import StudyWorkspace

    explicit = tmp_path / "explicit"
    configured = tmp_path / "configured"
    legacy = tmp_path / "legacy"
    for path in (explicit, configured, legacy):
        path.mkdir()
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(legacy))

    from_config = StudyWorkspace.open(
        config={"study_os": {"vault_path": str(configured)}}
    )
    from_explicit = StudyWorkspace.open(
        str(explicit),
        config={"study_os": {"vault_path": str(configured)}},
    )

    assert from_config.vault == configured.resolve()
    assert from_config.source == "config"
    assert from_explicit.vault == explicit.resolve()
    assert from_explicit.source == "explicit"


def test_workspace_active_project_is_shared_with_legacy_agent_path(tmp_path: Path):
    from plugins.study_os import tools as legacy
    from plugins.study_os.workspace import StudyWorkspace

    vault = tmp_path / "vault"
    vault.mkdir()
    _write_project(vault, "kaoyan-2027", "考研")
    _write_project(vault, "research-agents", "Agent Research")
    workspace = StudyWorkspace.open(str(vault), config={})

    selected = workspace.select_project("research-agents")

    assert selected["project_id"] == "research-agents"
    assert workspace.active_project_id() == "research-agents"
    assert legacy._resolve_project_id(vault) == "research-agents"
    assert [project["project_id"] for project in workspace.list_projects()] == [
        "kaoyan-2027",
        "research-agents",
    ]


def test_workspace_rejects_unknown_project_without_changing_active_pointer(
    tmp_path: Path,
):
    from plugins.study_os.workspace import StudyWorkspace

    vault = tmp_path / "vault"
    vault.mkdir()
    _write_project(vault, "kaoyan-2027", "考研")
    workspace = StudyWorkspace.open(str(vault), config={})
    workspace.select_project("kaoyan-2027")

    with pytest.raises(FileNotFoundError, match="Project not found"):
        workspace.select_project("missing-project")

    assert workspace.active_project_id() == "kaoyan-2027"


def test_workspace_discovers_long_term_plans_and_reports_invalid_session_files(
    tmp_path: Path,
):
    from plugins.study_os.workspace import StudyWorkspace

    vault = tmp_path / "vault"
    vault.mkdir()
    _write_project(vault, "kaoyan-2027", "考研")
    workspace = StudyWorkspace.open(str(vault), config={})
    schedule_dir = workspace.projects_root / "kaoyan-2027" / "schedules"
    schedule_dir.mkdir()
    long_term = {
        "schema_version": "study_schedule.v1",
        "schedule_id": "summer-roadmap",
        "project_id": "kaoyan-2027",
        "title": "暑期长期路线图",
        "timezone": "Asia/Shanghai",
        "range": {"start": "2026-07-16", "end": "2026-10-21"},
        "phases": [
            {
                "id": "foundation",
                "title": "专题覆盖",
                "start": "2026-07-16",
                "end": "2026-08-31",
                "goal": "完成核心专题覆盖",
            }
        ],
        "events": [],
    }
    invalid = {
        **long_term,
        "schedule_id": "invalid-roadmap",
        "events": [
            {
                "id": "phase-as-event",
                "title": "专题覆盖",
                "subject_id": "math",
                "type": "learning",
                "start": "2026-07-16T08:00:00+08:00",
                "end": "2026-08-31T20:00:00+08:00",
                "duration_minutes": 3600,
                "goals": ["完成核心专题覆盖"],
                "status": "planned",
            }
        ],
    }
    (schedule_dir / "summer-roadmap.json").write_text(
        json.dumps(long_term, ensure_ascii=False),
        encoding="utf-8",
    )
    (schedule_dir / "invalid-roadmap.json").write_text(
        json.dumps(invalid, ensure_ascii=False),
        encoding="utf-8",
    )

    catalog = workspace.discover_schedules("kaoyan-2027")

    assert [entry.schedule["schedule_id"] for entry in catalog.schedules] == [
        "summer-roadmap"
    ]
    assert [issue.schedule_id for issue in catalog.invalid_schedules] == [
        "invalid-roadmap"
    ]
    assert "events[0].duration_minutes must be an integer from 1 to 720" in (
        catalog.invalid_schedules[0].errors
    )


def test_overview_derives_today_and_evidence_semantics_from_immutable_records(
    tmp_path: Path,
):
    from plugins.study_os.interventions import InterventionOrchestrator
    from plugins.study_os.learning import _diagnosis
    from plugins.study_os.overview import build_study_overview
    from plugins.study_os.workspace import StudyWorkspace

    vault = tmp_path / "vault"
    vault.mkdir()
    _write_project(vault, "kaoyan-2027", "考研")
    workspace = StudyWorkspace.open(str(vault), config={})
    workspace.select_project("kaoyan-2027")
    project_dir = workspace.projects_root / "kaoyan-2027"
    schedule_dir = project_dir / "schedules"
    schedule_dir.mkdir()
    schedule = {
        "schema_version": "study_schedule.v1",
        "schedule_id": "daily-plan",
        "project_id": "kaoyan-2027",
        "title": "每日计划",
        "timezone": "Asia/Shanghai",
        "range": {"start": "2026-07-01", "end": "2026-07-31"},
        "phases": [],
        "events": [
            {
                "id": "evt-math-review",
                "title": "数学复习",
                "subject_id": "math",
                "type": "review",
                "start": "2026-07-13T09:00:00+08:00",
                "end": "2026-07-13T09:30:00+08:00",
                "duration_minutes": 30,
                "goals": ["回忆定义"],
                "status": "planned",
            }
        ],
    }
    (schedule_dir / "daily-plan.json").write_text(
        json.dumps(schedule, ensure_ascii=False),
        encoding="utf-8",
    )
    review_note = vault / "Math" / "examples" / "derivative.md"
    review_note.parent.mkdir(parents=True)
    review_note.write_text(
        "---\ntype: example\nreview_level: 1\nreview_count: 1\n"
        "next_review_at: 2026-07-13\n---\n# 导数复习\n",
        encoding="utf-8",
    )
    attempts_dir = project_dir / "activity"
    attempts_dir.mkdir()
    attempt = {
        "schema_version": "study_attempt.v1",
        "attempt_id": "att-self-review",
        "project_id": "kaoyan-2027",
        "item_id": "Math/examples/derivative.md",
        "occurred_at": "2026-07-13T10:00:00+08:00",
        "response": "按定义计算",
        "result": "correct",
        "score": 1.0,
        "hints_used": 0,
        "self_confidence": 4,
        "evaluator": {"kind": "self", "id": "desktop-review"},
        "assistance": {"level": "independent", "hints_used": 0},
        "transfer_level": "execution",
        "concepts": ["导数"],
        "patterns": [],
        "objective_ids": [],
        "diagnoses": [],
        "activity_kind": "review",
    }
    (attempts_dir / "attempts-2026-07.jsonl").write_text(
        json.dumps(attempt, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    proposal = InterventionOrchestrator(
        project=_project("kaoyan-2027", "考研"),
        diagnosis_builder=_diagnosis,
    ).build(
        attempts=[attempt],
        as_of=__import__("datetime").datetime.fromisoformat(
            "2026-07-13T12:00:00+08:00"
        ),
    )["proposal"]
    proposals_dir = project_dir / "plan-proposals"
    proposals_dir.mkdir()
    (proposals_dir / f"{proposal['proposal_id']}.json").write_text(
        json.dumps(proposal, ensure_ascii=False),
        encoding="utf-8",
    )

    overview = build_study_overview(
        workspace,
        as_of="2026-07-13T12:00:00+08:00",
    )

    assert [event["id"] for event in overview["today_events"]] == [
        "evt-math-review"
    ]
    assert overview["completed_today"] == 1
    assert overview["due_reviews"]["scope"] == "vault"
    assert overview["due_reviews"]["count"] == 1
    assert overview["due_reviews"]["items"][0]["path"] == (
        "Math/examples/derivative.md"
    )
    assert overview["evidence"]["attempt_count"] == 1
    assert overview["evidence"]["independently_verified_count"] == 0
    assert overview["evidence"]["dimensions"]["execution"][
        "verification_status"
    ] == "supported"
    assert overview["evidence"]["evaluator_provenance"] == {"self": 1}
    assert len(overview["pending_plan_proposals"]) == 1


def test_review_stats_describe_spacing_coverage_not_mastery(tmp_path: Path):
    from plugins.study_os.tools import _build_review_stats

    vault = tmp_path / "vault"
    example = vault / "Math" / "examples" / "limit.md"
    example.parent.mkdir(parents=True)
    example.write_text(
        "---\ntype: example\nreview_level: 5\nreview_count: 0\n---\n# 极限\n",
        encoding="utf-8",
    )

    stats = _build_review_stats(vault)

    assert stats["spacing_coverage_pct"] == 0.0
    assert stats["reviewed_examples"] == 0
    assert "mastered" not in stats
