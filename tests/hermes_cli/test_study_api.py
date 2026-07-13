from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi.testclient import TestClient

from hermes_cli import web_server


def _client():
    prev_auth = getattr(web_server.app.state, "auth_required", None)
    prev_host = getattr(web_server.app.state, "bound_host", None)
    web_server.app.state.auth_required = False
    web_server.app.state.bound_host = None
    client = TestClient(web_server.app)
    return client, prev_auth, prev_host


def _restore(prev_auth, prev_host) -> None:
    if prev_auth is None:
        if hasattr(web_server.app.state, "auth_required"):
            delattr(web_server.app.state, "auth_required")
    else:
        web_server.app.state.auth_required = prev_auth
    if prev_host is None:
        if hasattr(web_server.app.state, "bound_host"):
            delattr(web_server.app.state, "bound_host")
    else:
        web_server.app.state.bound_host = prev_host


def _get(client: TestClient, path: str):
    return client.get(path, headers={"X-Hermes-Session-Token": web_server._SESSION_TOKEN})


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
    client, pa, ph = _client()
    try:
        response = _get(client, "/api/study/projects")
        assert response.status_code == 200
        assert response.json() == {
            "projects": [],
            "configured": False,
            "message": "StudyOS vault not configured",
        }
    finally:
        _restore(pa, ph)
        client.close()


def test_study_api_lists_and_reads_schedule(monkeypatch, tmp_path: Path):
    vault = tmp_path / "vault"
    vault.mkdir()
    _write_fixture_vault(vault)
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(vault))
    client, pa, ph = _client()
    try:
        projects = _get(client, "/api/study/projects")
        schedules = _get(client, "/api/study/projects/kaoyan-2027/schedules")
        schedule = _get(client, "/api/study/projects/kaoyan-2027/schedules/kaoyan-2027-master-plan")

        assert projects.status_code == 200
        assert projects.json()["projects"][0]["project_id"] == "kaoyan-2027"
        assert schedules.status_code == 200
        assert schedules.json()["schedules"] == [
            {
                "schedule_id": "kaoyan-2027-master-plan",
                "project_id": "kaoyan-2027",
                "title": "2027 考研数学基础阶段计划",
                "timezone": "Asia/Shanghai",
                "range": {"start": "2026-07-01", "end": "2026-07-31"},
                "event_count": 1,
            }
        ]
        assert schedule.status_code == 200
        assert schedule.json()["events"][0]["title"] == "数学：导数定义整理"
    finally:
        _restore(pa, ph)
        client.close()


def test_study_review_due_discovers_examples_in_subject_folders(monkeypatch, tmp_path: Path):
    vault = tmp_path / "408"
    vault.mkdir()
    _write_due_example(vault / "OS" / "examples" / "process.md", "进程")
    _write_due_example(vault / "计组" / "examples" / "cache.md", "Cache")
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(vault))
    client, pa, ph = _client()
    try:
        response = _get(client, "/api/study/review/due")

        assert response.status_code == 200
        assert response.json()["count"] == 2
        assert {item["path"] for item in response.json()["due"]} == {
            "OS/examples/process.md",
            "计组/examples/cache.md",
        }
        assert {item["subject"] for item in response.json()["due"]} == {"OS", "计组"}
        assert response.json()["subjects"] == ["OS", "计组"]
    finally:
        _restore(pa, ph)
        client.close()


def test_study_api_rejects_path_traversal(monkeypatch, tmp_path: Path):
    vault = tmp_path / "vault"
    vault.mkdir()
    _write_fixture_vault(vault)
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(vault))
    client, pa, ph = _client()
    try:
        response = _get(client, "/api/study/projects/..%2Fescape")
        assert response.status_code in {400, 404}
        assert not (tmp_path / "escape").exists()
    finally:
        _restore(pa, ph)
        client.close()


def test_study_api_missing_schedule_returns_404(monkeypatch, tmp_path: Path):
    vault = tmp_path / "vault"
    vault.mkdir()
    _write_fixture_vault(vault)
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(vault))
    client, pa, ph = _client()
    try:
        response = _get(client, "/api/study/projects/kaoyan-2027/schedules/kaoyan-2027-missing")
        assert response.status_code == 404
    finally:
        _restore(pa, ph)
        client.close()


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
