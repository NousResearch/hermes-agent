from __future__ import annotations

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
