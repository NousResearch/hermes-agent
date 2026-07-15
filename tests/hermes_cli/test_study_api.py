from __future__ import annotations

import asyncio
import json
from pathlib import Path

import yaml
import pytest
from fastapi import HTTPException

from hermes_cli import web_server


def _project() -> dict:
    return {
        "schema_version": "study_project.v1",
        "project_id": "kaoyan-2027",
        "title": "2027 考研学习计划",
        "domain": "kaoyan",
        "exam_type": "考研",
        "exam_date": "2027-12-20",
        "timezone": "Asia/Shanghai",
        "phase": "foundation",
        "domain_pack": "kaoyan.v1",
        "subjects": [
            {"id": "math", "label": "数学", "target_score": 120},
            {"id": "english", "label": "英语一", "target_score": 75},
            {"id": "politics", "label": "政治", "target_score": 75},
        ],
        "prompt_policy": {
            "base_max_chars": 2000,
            "intent_max_chars": 2500,
            "domain_max_chars": 2000,
            "project_summary_max_chars": 1200,
            "total_max_chars": 6000,
            "updates_apply": "next_session",
        },
        "created_at": "2026-06-28T00:00:00+08:00",
        "updated_at": "2026-06-28T00:00:00+08:00",
    }


def _schedule() -> dict:
    return {
        "schema_version": "study_schedule.v1",
        "schedule_id": "kaoyan-2027-master-plan",
        "project_id": "kaoyan-2027",
        "title": "2027 考研数学基础阶段计划",
        "timezone": "Asia/Shanghai",
        "range": {"start": "2026-07-01", "end": "2026-07-31"},
        "phases": [
            {
                "id": "foundation",
                "title": "基础阶段",
                "start": "2026-07-01",
                "end": "2026-09-30",
                "goal": "完成核心考点覆盖",
            }
        ],
        "events": [
            {
                "id": "evt-20260701-math-derivative",
                "title": "数学：导数定义整理",
                "subject_id": "math",
                "type": "learning",
                "start": "2026-07-01T19:00:00+08:00",
                "end": "2026-07-01T21:00:00+08:00",
                "duration_minutes": 120,
                "goals": ["整理导数定义例题"],
                "source_curriculum": "一元函数微分学",
                "status": "planned",
            }
        ],
    }


def _write_fixture_vault(vault: Path) -> None:
    project_dir = vault / ".StudyOS" / "projects" / "kaoyan-2027"
    schedule_dir = project_dir / "schedules"
    schedule_dir.mkdir(parents=True)
    (project_dir / "manifest.json").write_text(json.dumps(_project(), ensure_ascii=False), encoding="utf-8")
    (schedule_dir / "kaoyan-2027-master-plan.json").write_text(
        json.dumps(_schedule(), ensure_ascii=False),
        encoding="utf-8",
    )
    (project_dir.parent / "active.json").write_text(
        json.dumps({"project_id": "kaoyan-2027"}),
        encoding="utf-8",
    )


def _write_due_example(path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\n"
        f"title: {title}\n"
        "type: example\n"
        "review_level: 0\n"
        "review_count: 0\n"
        "tags:\n"
        "  - 408\n"
        "---\n\n"
        f"# {title}\n",
        encoding="utf-8",
    )


def test_study_projects_unconfigured_vault(monkeypatch, tmp_path: Path):
    missing = tmp_path / "missing"
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(missing))
    response = asyncio.run(web_server.list_study_projects())
    assert response == {
        "projects": [],
        "configured": False,
        "message": "StudyOS vault not configured",
    }


def test_study_api_lists_and_reads_schedule(monkeypatch, tmp_path: Path):
    vault = tmp_path / "vault"
    vault.mkdir()
    _write_fixture_vault(vault)
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(vault))
    projects = asyncio.run(web_server.list_study_projects())
    schedules = asyncio.run(web_server.list_study_schedules("kaoyan-2027"))
    schedule = asyncio.run(
        web_server.get_study_schedule("kaoyan-2027", "kaoyan-2027-master-plan")
    )

    assert projects["projects"][0]["project_id"] == "kaoyan-2027"
    assert projects["active_project_id"] == "kaoyan-2027"
    assert schedules["schedules"] == [
        {
            "schedule_id": "kaoyan-2027-master-plan",
            "project_id": "kaoyan-2027",
            "title": "2027 考研数学基础阶段计划",
            "timezone": "Asia/Shanghai",
            "range": {"start": "2026-07-01", "end": "2026-07-31"},
            "phase_count": 1,
            "event_count": 1,
        }
    ]
    assert schedules["invalid_schedules"] == []
    assert schedule["events"][0]["title"] == "数学：导数定义整理"


def test_study_api_reports_invalid_schedule_instead_of_hiding_it(
    monkeypatch,
    tmp_path: Path,
):
    vault = tmp_path / "vault"
    vault.mkdir()
    _write_fixture_vault(vault)
    schedule_path = (
        vault
        / ".StudyOS"
        / "projects"
        / "kaoyan-2027"
        / "schedules"
        / "kaoyan-2027-master-plan.json"
    )
    invalid = _schedule()
    invalid["events"][0].update(
        {
            "start": "2026-07-16T08:00:00+08:00",
            "end": "2026-07-21T20:00:00+08:00",
            "duration_minutes": 3600,
        }
    )
    schedule_path.write_text(json.dumps(invalid, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(vault))

    response = asyncio.run(web_server.list_study_schedules("kaoyan-2027"))

    assert response["schedules"] == []
    assert response["invalid_schedules"] == [
        {
            "schedule_id": "kaoyan-2027-master-plan",
            "path": ".StudyOS/projects/kaoyan-2027/schedules/kaoyan-2027-master-plan.json",
            "errors": [
                "events[0].duration_minutes must be an integer from 1 to 720",
                "events[0] spans more than 720 minutes; use phases for long-term ranges "
                "and events only for concrete study sessions",
            ],
        }
    ]


def test_study_review_due_discovers_examples_in_subject_folders(monkeypatch, tmp_path: Path):
    vault = tmp_path / "408"
    vault.mkdir()
    _write_due_example(vault / "OS" / "examples" / "process.md", "进程")
    _write_due_example(vault / "计组" / "examples" / "cache.md", "Cache")
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(vault))
    response = asyncio.run(web_server.get_study_due_reviews())

    assert response["count"] == 2
    assert {item["path"] for item in response["due"]} == {
        "OS/examples/process.md",
        "计组/examples/cache.md",
    }
    assert {item["subject"] for item in response["due"]} == {"OS", "计组"}
    assert response["subjects"] == ["OS", "计组"]


def test_study_api_rejects_path_traversal(monkeypatch, tmp_path: Path):
    vault = tmp_path / "vault"
    vault.mkdir()
    _write_fixture_vault(vault)
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(vault))
    with pytest.raises(HTTPException) as raised:
        asyncio.run(web_server.get_study_project("../escape"))
    assert raised.value.status_code in {400, 404}
    assert not (tmp_path / "escape").exists()


def test_study_api_missing_schedule_returns_404(monkeypatch, tmp_path: Path):
    vault = tmp_path / "vault"
    vault.mkdir()
    _write_fixture_vault(vault)
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(vault))
    with pytest.raises(HTTPException) as raised:
        asyncio.run(
            web_server.get_study_schedule(
                "kaoyan-2027", "kaoyan-2027-missing"
            )
        )
    assert raised.value.status_code == 404


def test_study_review_runner_endpoints_share_one_submission_contract(monkeypatch, tmp_path: Path):
    vault = tmp_path / "vault"
    vault.mkdir()
    _write_fixture_vault(vault)
    note = vault / "math" / "examples" / "derivative.md"
    note.parent.mkdir(parents=True)
    note.write_text(
        "---\ntitle: 导数\ntype: example\nreview_level: 1\nreview_count: 0\n"
        "concepts: [导数]\n---\n\n# 导数\n\n求导。\n\n## 答案\n\n使用定义。\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(vault))

    detail = asyncio.run(
        web_server.post_study_review_detail(
            web_server.StudyReviewDetailRequest(note="math/examples/derivative.md")
        )
    )
    submitted = asyncio.run(
        web_server.post_study_review_attempt(
            web_server.StudyReviewSubmissionRequest(
                project_id="kaoyan-2027",
                note="math/examples/derivative.md",
                response="按定义求导",
                result="correct",
                duration_seconds=42,
                self_confidence=4,
            )
        )
    )

    assert detail["prompt_markdown"].endswith("求导。")
    assert detail["answer_markdown"].startswith("## 答案")
    assert submitted["attempt"]["response"] == "按定义求导"
    assert submitted["review"]["review_level"] == {"old": 1, "new": 2}


def test_study_settings_persist_profile_vault_and_opt_in_toolset(monkeypatch, tmp_path: Path):
    home = tmp_path / "hermes"
    vault = tmp_path / "vault"
    home.mkdir()
    vault.mkdir()
    _write_fixture_vault(vault)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("OBSIDIAN_VAULT_PATH", raising=False)

    response = asyncio.run(
        web_server.put_study_settings(
            web_server.StudySettingsUpdate(vault_path=str(vault))
        )
    )
    loaded = asyncio.run(web_server.get_study_settings())

    assert response["configured"] is True
    assert response["study_toolset_enabled"] is True
    assert response["requires_new_session"] is True
    assert loaded["vault_path"] == str(vault.resolve())
    config = yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))
    assert config["study_os"]["vault_path"] == str(vault.resolve())
    assert "study" in config["platform_toolsets"]["cli"]


def test_study_overview_and_plan_proposal_decision_share_active_project(
    monkeypatch,
    tmp_path: Path,
):
    from plugins.study_os.learning import handle_study_activity, handle_study_coach

    vault = tmp_path / "vault"
    vault.mkdir()
    _write_fixture_vault(vault)
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(vault))
    proposed = json.loads(
        handle_study_coach(
            {
                "action": "propose_plan",
                "scope": "project",
                "vault_path": str(vault),
                "project_id": "kaoyan-2027",
                "data": {"as_of": "2026-07-13T12:00:00+08:00"},
            }
        )
    )["data"]["proposal"]
    saved = json.loads(
        handle_study_activity(
            {
                "resource": "plan_proposal",
                "action": "save",
                "vault_path": str(vault),
                "project_id": "kaoyan-2027",
                "data": {"proposal": proposed},
            }
        )
    )
    assert saved["ok"] is True
    schedule_before = (
        vault
        / ".StudyOS"
        / "projects"
        / "kaoyan-2027"
        / "schedules"
        / "kaoyan-2027-master-plan.json"
    ).read_text(encoding="utf-8")

    overview = asyncio.run(
        web_server.get_study_overview(as_of="2026-07-13T12:00:00+08:00")
    )
    decided = asyncio.run(
        web_server.put_study_plan_proposal_decision(
            "kaoyan-2027",
            proposed["proposal_id"],
            web_server.StudyPlanProposalDecision(action="accept"),
        )
    )
    after = asyncio.run(
        web_server.get_study_overview(as_of="2026-07-13T12:00:00+08:00")
    )

    assert overview["active_project_id"] == "kaoyan-2027"
    assert [item["proposal_id"] for item in overview["pending_plan_proposals"]] == [
        proposed["proposal_id"]
    ]
    assert decided["proposal"]["status"] == "accepted"
    assert decided["schedule_mutated"] is False
    assert after["pending_plan_proposals"] == []
    assert schedule_before == (
        vault
        / ".StudyOS"
        / "projects"
        / "kaoyan-2027"
        / "schedules"
        / "kaoyan-2027-master-plan.json"
    ).read_text(encoding="utf-8")
