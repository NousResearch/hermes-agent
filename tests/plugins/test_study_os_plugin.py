from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    root = tmp_path / "vault"
    (root / "OS" / "Box").mkdir(parents=True)
    (root / "OS" / "examples").mkdir(parents=True)
    (root / "Math" / "Box" / "题型").mkdir(parents=True)
    (root / "OS" / "Box" / "进程创建.md").write_text(
        """---
type: concept
aliases:
  - 作业接纳
tags:
  - OS
  - 进程与线程
---
# 进程创建

进程创建会建立 [[进程控制块]]，并进入就绪队列。
""",
        encoding="utf-8",
    )
    (root / "OS" / "examples" / "OS-0043.md").write_text(
        """---
type: example
id: OS-0043
difficulty: 2
review_level: 2
status: 可复习
tags:
  - OS
patterns:
  - "[[题型：高级调度与进程接纳辨析]]"
concepts:
  - "[[处理机调度层次]]"
  - "[[进程创建]]"
---
# OS-0043 进程从创建态转为就绪态

## 题型特征
核心是区分高级调度和低级调度。
""",
        encoding="utf-8",
    )
    (root / "Math" / "Box" / "题型" / "题型：泰勒展开.md").write_text(
        """---
type: pattern
tags: [数学, 极限]
concepts: ["[[泰勒展开]]"]
---
# 题型：泰勒展开

看到无穷小阶数匹配时考虑 [[泰勒公式]]。
""",
        encoding="utf-8",
    )
    return root


def _loads(result: str) -> dict:
    return json.loads(result)


def _valid_study_project() -> dict:
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


def _valid_study_schedule() -> dict:
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


def test_study_project_schema_accepts_kaoyan_manifest():
    from plugins.study_os.schemas import validate_study_project

    project = _valid_study_project()
    project["unknown_future_field"] = {"kept": True}

    ok, data_or_errors = validate_study_project(project)

    assert ok is True
    assert data_or_errors is project
    assert data_or_errors["unknown_future_field"] == {"kept": True}


@pytest.mark.parametrize("project_id", ["../bad", "Kaoyan", "xy"])
def test_study_project_schema_rejects_invalid_project_ids(project_id: str):
    from plugins.study_os.schemas import validate_study_project

    project = _valid_study_project()
    project["project_id"] = project_id

    ok, errors = validate_study_project(project)

    assert ok is False
    assert any("project_id must match" in error for error in errors)


def test_study_schedule_schema_accepts_kaoyan_schedule():
    from plugins.study_os.schemas import validate_study_schedule

    schedule = _valid_study_schedule()
    schedule["unknown_future_field"] = "kept"

    ok, data_or_errors = validate_study_schedule(schedule, project=_valid_study_project())

    assert ok is True
    assert data_or_errors is schedule
    assert data_or_errors["unknown_future_field"] == "kept"


def test_study_schedule_schema_rejects_datetime_without_timezone():
    from plugins.study_os.schemas import validate_study_schedule

    schedule = _valid_study_schedule()
    schedule["events"][0]["start"] = "2026-07-01T19:00:00"

    ok, errors = validate_study_schedule(schedule, project=_valid_study_project())

    assert ok is False
    assert "events[0].start must include timezone offset" in errors


def test_study_schedule_schema_rejects_mismatched_cross_midnight_duration():
    from plugins.study_os.schemas import validate_study_schedule

    schedule = _valid_study_schedule()
    schedule["events"][0]["start"] = "2026-07-01T23:30:00+08:00"
    schedule["events"][0]["end"] = "2026-07-02T00:15:00+08:00"
    schedule["events"][0]["duration_minutes"] = 120

    ok, errors = validate_study_schedule(schedule, project=_valid_study_project())

    assert ok is False
    assert "events[0].duration_minutes does not match start/end" in errors


def test_study_schedule_schema_rejects_unknown_project_subject():
    from plugins.study_os.schemas import validate_study_schedule

    schedule = _valid_study_schedule()
    schedule["events"][0]["subject_id"] = "physics"

    ok, errors = validate_study_schedule(schedule, project=_valid_study_project())

    assert ok is False
    assert "events[0].subject_id must exist in project subjects" in errors


def test_study_project_init_and_schedule_save(vault: Path):
    from plugins.study_os.tools import handle_study_project, handle_study_schedule

    init = _loads(handle_study_project({"vault_path": str(vault), "action": "init"}))
    status = _loads(handle_study_project({"vault_path": str(vault), "action": "status"}))
    schedule = _valid_study_schedule()
    save = _loads(handle_study_schedule({"vault_path": str(vault), "action": "save", "data": schedule}))
    read = _loads(
        handle_study_schedule(
            {
                "vault_path": str(vault),
                "action": "read",
                "project_id": "kaoyan-2027",
                "schedule_id": "kaoyan-2027-master-plan",
            }
        )
    )

    assert init["ok"] is True
    assert init["data"]["path"] == ".StudyOS/projects/kaoyan-2027/manifest.json"
    assert init["data"]["active_path"] == ".StudyOS/projects/active.json"
    assert status["ok"] is True
    assert status["data"]["project"]["project_id"] == "kaoyan-2027"
    assert save["ok"] is True
    assert save["data"]["path"] == ".StudyOS/projects/kaoyan-2027/schedules/kaoyan-2027-master-plan.json"
    assert read["ok"] is True
    assert read["data"]["schedule"]["events"][0]["title"] == "数学：导数定义整理"
    assert (vault / ".StudyOS" / "projects" / "kaoyan-2027" / "manifest.json").exists()
    assert (vault / ".StudyOS" / "projects" / "active.json").exists()
    assert (vault / ".StudyOS" / "projects" / "kaoyan-2027" / "schedules" / "kaoyan-2027-master-plan.json").exists()


def test_study_schedule_read_returns_not_found(vault: Path):
    from plugins.study_os.tools import handle_study_project, handle_study_schedule

    init = _loads(handle_study_project({"vault_path": str(vault), "action": "init"}))
    missing = _loads(
        handle_study_schedule(
            {
                "vault_path": str(vault),
                "action": "read",
                "project_id": "kaoyan-2027",
                "schedule_id": "kaoyan-2027-missing",
            }
        )
    )

    assert init["ok"] is True
    assert missing["ok"] is False
    assert missing["error"]["code"] == "SCHEDULE_NOT_FOUND"


def test_study_project_and_prompt_context_reject_invalid_inputs(vault: Path):
    from plugins.study_os.tools import handle_study_project, handle_study_prompt_context

    traversal = _loads(handle_study_project({"vault_path": str(vault), "action": "init", "project_id": "../escape"}))
    invalid_intent = _loads(
        handle_study_prompt_context(
            {
                "vault_path": str(vault),
                "intent": "motivation-hype",
                "project_id": "kaoyan-2027",
            }
        )
    )

    assert traversal["ok"] is False
    assert traversal["error"]["code"] == "VALIDATION_FAILED"
    assert invalid_intent["ok"] is False
    assert invalid_intent["error"]["code"] == "INVALID_INTENT"
    assert not (vault.parent / "escape").exists()


def test_study_prompt_context_truncates_project_summary(vault: Path):
    from plugins.study_os.tools import handle_study_project, handle_study_prompt_context

    init = _loads(
        handle_study_project(
            {
                "vault_path": str(vault),
                "action": "init",
                "project_id": "general-2027",
                "domain": "general",
                "domain_pack": "general.v1",
            }
        )
    )
    summary = _loads(
        handle_study_project(
            {
                "vault_path": str(vault),
                "action": "update_prompt_summary",
                "project_id": "general-2027",
                "summary": "x" * 2000,
            }
        )
    )
    context = _loads(
        handle_study_prompt_context(
            {
                "vault_path": str(vault),
                "intent": "reviewing",
                "project_id": "general-2027",
            }
        )
    )

    assert init["ok"] is True
    assert summary["ok"] is True
    assert summary["warnings"] == ["summary truncated to 1200 characters"]
    (vault / ".StudyOS" / "projects" / "general-2027" / "prompt_summary.md").write_text("y" * 2000, encoding="utf-8")
    context = _loads(
        handle_study_prompt_context(
            {
                "vault_path": str(vault),
                "intent": "reviewing",
                "project_id": "general-2027",
            }
        )
    )
    assert context["ok"] is True
    assert context["warnings"] == ["project_summary truncated to 1200 characters"]
    fragments = {fragment["kind"]: fragment for fragment in context["data"]["fragments"]}
    assert fragments["project_summary"]["char_count"] == 1200


def test_study_os_registers_modular_skills(monkeypatch):
    from hermes_cli import plugins as plugins_mod
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
    from plugins import study_os
    from tools.registry import registry

    manager = PluginManager()
    monkeypatch.setattr(plugins_mod, "_plugin_manager", manager)
    manifest = PluginManifest(name="study_os", version="0.1.0", description="study", source="bundled")
    ctx = PluginContext(manifest, manager)

    try:
        study_os.register(ctx)
        for name in (
            "study-os",
            "study-plan",
            "study-organize",
            "study-review",
            "study-assessment",
            "study-kaoyan",
        ):
            assert manager.find_plugin_skill(f"study_os:{name}") is not None
    finally:
        for name in (
            "study_list_notes",
            "study_read_note",
            "study_extract_concepts",
            "study_log_error",
            "study_create_review_task",
            "study_generate_weekly_report",
            "study_export_anki_candidates",
            "study_due_reviews",
            "study_record_review",
            "study_sync_memory",
            "study_concept_graph",
            "study_review_stats",
            "study_learning_queue",
            "study_log_session",
            "study_update_concept_state",
            "study_import_plan",
            "study_plan_progress",
            "study_create_curriculum",
            "study_list_curricula",
            "study_project",
            "study_schedule",
            "study_prompt_context",
        ):
            registry.deregister(name)


def test_study_os_skill_descriptions_and_budgets(monkeypatch):
    from hermes_cli import plugins as plugins_mod
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
    from plugins import study_os
    from tools.registry import registry

    manager = PluginManager()
    monkeypatch.setattr(plugins_mod, "_plugin_manager", manager)
    manifest = PluginManifest(name="study_os", version="0.1.0", description="study", source="bundled")
    ctx = PluginContext(manifest, manager)

    try:
        study_os.register(ctx)
        expected = {
            "study-os": ("Route StudyOS learning workflows.", 6000),
            "study-plan": ("Plan StudyOS projects and schedules.", 9000),
            "study-organize": ("Organize problems into StudyOS notes.", 9000),
            "study-review": ("Run StudyOS spaced repetition reviews.", 9000),
            "study-assessment": ("Analyze StudyOS exams and mistakes.", 9000),
            "study-kaoyan": ("Guide 考研 learning with StudyOS.", 9000),
        }
        all_text = ""
        for name, (description, max_chars) in expected.items():
            entry = manager._plugin_skills[f"study_os:{name}"]
            assert entry["description"] == description
            assert len(description) <= 60
            assert description.endswith(".")
            assert description.count(".") == 1
            body = Path(entry["path"]).read_text(encoding="utf-8")
            all_text += body
            assert len(body) <= max_chars, name
            assert "study_prompt_context" in body
            assert "mutate system prompts" in body
        for term in ("艾宾浩斯", "整理", "错题", "weekly", "curriculum", "考研"):
            assert term in all_text
    finally:
        for name in (
            "study_list_notes",
            "study_read_note",
            "study_extract_concepts",
            "study_log_error",
            "study_create_review_task",
            "study_generate_weekly_report",
            "study_export_anki_candidates",
            "study_due_reviews",
            "study_record_review",
            "study_sync_memory",
            "study_concept_graph",
            "study_review_stats",
            "study_learning_queue",
            "study_log_session",
            "study_update_concept_state",
            "study_import_plan",
            "study_plan_progress",
            "study_create_curriculum",
            "study_list_curricula",
            "study_project",
            "study_schedule",
            "study_prompt_context",
        ):
            registry.deregister(name)


def test_study_toolset_is_opt_in():
    from toolsets import TOOLSETS, _HERMES_CORE_TOOLS

    expected_tools = [
        "study_list_notes",
        "study_read_note",
        "study_extract_concepts",
        "study_log_error",
        "study_create_review_task",
        "study_generate_weekly_report",
        "study_export_anki_candidates",
        "study_due_reviews",
        "study_record_review",
        "study_sync_memory",
        "study_concept_graph",
        "study_review_stats",
        "study_learning_queue",
        "study_log_session",
        "study_update_concept_state",
        "study_import_plan",
        "study_plan_progress",
        "study_create_curriculum",
        "study_list_curricula",
        "study_project",
        "study_schedule",
        "study_prompt_context",
    ]

    assert TOOLSETS["study"]["tools"] == expected_tools
    for tool in expected_tools:
        assert tool not in _HERMES_CORE_TOOLS


def test_list_notes_reads_obsidian_frontmatter(vault: Path):
    from plugins.study_os.tools import handle_study_list_notes

    result = _loads(
        handle_study_list_notes(
            {
                "vault_path": str(vault),
                "folder": "OS",
                "tag": "OS",
                "limit": 10,
            }
        )
    )

    assert result["ok"] is True
    assert result["data"]["count"] == 2
    paths = {note["path"] for note in result["data"]["notes"]}
    assert "OS/Box/进程创建.md" in paths
    assert "OS/examples/OS-0043.md" in paths


def test_read_note_resolves_title_alias_and_extracts_links(vault: Path):
    from plugins.study_os.tools import handle_study_read_note

    result = _loads(
        handle_study_read_note(
            {
                "vault_path": str(vault),
                "note": "作业接纳",
                "include_body": True,
            }
        )
    )

    assert result["ok"] is True
    note = result["data"]["note"]
    assert note["path"] == "OS/Box/进程创建.md"
    assert note["title"] == "进程创建"
    assert note["layer"] == "concept"
    assert "进程控制块" in note["wikilinks"]
    assert "body" in note


def test_extract_concepts_uses_frontmatter_and_candidates(vault: Path):
    from plugins.study_os.tools import handle_study_extract_concepts

    result = _loads(
        handle_study_extract_concepts(
            {
                "vault_path": str(vault),
                "notes": ["OS/examples/OS-0043.md", "Math/Box/题型/题型：泰勒展开.md"],
            }
        )
    )

    assert result["ok"] is True
    concepts = dict(result["data"]["concepts"])
    patterns = dict(result["data"]["patterns"])
    assert concepts["进程创建"] == 1
    assert concepts["泰勒展开"] == 1
    assert patterns["题型：高级调度与进程接纳辨析"] == 1


def test_write_tools_only_create_studyos_files(vault: Path):
    from plugins.study_os.tools import (
        handle_study_create_review_task,
        handle_study_generate_weekly_report,
        handle_study_log_error,
    )

    err = _loads(
        handle_study_log_error(
            {
                "vault_path": str(vault),
                "title": "创建态到就绪态调度混淆",
                "source_note": "OS/examples/OS-0043.md",
                "subject": "OS",
                "concepts": ["进程创建"],
                "patterns": ["题型：高级调度与进程接纳辨析"],
                "cause": "concept_confusion",
                "severity": "high",
                "next_action": "二刷高级调度与低级调度区别",
                "detail": "把作业接纳误判为 CPU 分配。",
                "occurred_on": "2026-06-22",
            }
        )
    )
    task = _loads(
        handle_study_create_review_task(
            {
                "vault_path": str(vault),
                "title": "二刷进程调度层次",
                "source_note": "OS/examples/OS-0043.md",
                "due_date": "2026-06-23",
                "priority": "high",
                "concepts": ["进程创建"],
                "review_level": 2,
            }
        )
    )
    report = _loads(
        handle_study_generate_weekly_report(
            {
                "vault_path": str(vault),
                "start_date": "2026-06-22",
                "end_date": "2026-06-28",
            }
        )
    )

    assert err["ok"] is True
    assert task["ok"] is True
    assert report["ok"] is True
    assert err["data"]["path"].startswith(".StudyOS/errors/")
    assert task["data"]["path"] == ".StudyOS/review_tasks.md"
    assert report["data"]["path"] == ".StudyOS/reports/2026-W26.md"
    assert (vault / ".StudyOS" / "errors" / "2026-06.md").exists()
    assert (vault / ".StudyOS" / "review_tasks.md").exists()
    assert "concept_confusion: 1" in (vault / ".StudyOS" / "reports" / "2026-W26.md").read_text(encoding="utf-8")
    assert not (vault / "errors").exists()


def test_export_anki_candidates_writes_candidates_under_studyos(vault: Path):
    from plugins.study_os.tools import handle_study_export_anki_candidates

    result = _loads(
        handle_study_export_anki_candidates(
            {
                "vault_path": str(vault),
                "folder": "OS",
                "query": "进程",
                "limit": 2,
                "include_errors": False,
            }
        )
    )

    assert result["ok"] is True
    path = vault / result["data"]["path"]
    assert path.as_posix().endswith(".StudyOS/anki_candidates/" + path.name)
    text = path.read_text(encoding="utf-8")
    assert "START" in text
    assert "Tags: StudyOS Obsidian" in text


def test_resolve_vault_path_uses_obsidian_env(vault: Path, monkeypatch):
    from plugins.study_os.tools import handle_study_list_notes

    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(vault))
    result = _loads(handle_study_list_notes({"folder": "OS", "limit": 1}))

    assert result["ok"] is True
    assert result["data"]["vault_path"] == str(vault)


def test_plugin_registers_tools_and_skill(monkeypatch):
    from hermes_cli import plugins as plugins_mod
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
    from plugins import study_os
    from tools.registry import registry

    manager = PluginManager()
    monkeypatch.setattr(plugins_mod, "_plugin_manager", manager)
    manifest = PluginManifest(name="study_os", version="0.1.0", description="study", source="bundled")
    ctx = PluginContext(manifest, manager)

    try:
        study_os.register(ctx)
        assert registry.get_toolset_for_tool("study_list_notes") == "study"
        assert registry.get_toolset_for_tool("study_export_anki_candidates") == "study"
        assert registry.get_toolset_for_tool("study_due_reviews") == "study"
        assert registry.get_toolset_for_tool("study_record_review") == "study"
        assert registry.get_toolset_for_tool("study_sync_memory") == "study"
        assert registry.get_toolset_for_tool("study_concept_graph") == "study"
        assert registry.get_toolset_for_tool("study_review_stats") == "study"
        assert registry.get_toolset_for_tool("study_learning_queue") == "study"
        assert registry.get_toolset_for_tool("study_log_session") == "study"
        assert registry.get_toolset_for_tool("study_update_concept_state") == "study"
        assert registry.get_toolset_for_tool("study_import_plan") == "study"
        assert registry.get_toolset_for_tool("study_plan_progress") == "study"
        assert registry.get_toolset_for_tool("study_create_curriculum") == "study"
        assert registry.get_toolset_for_tool("study_list_curricula") == "study"
        assert registry.get_toolset_for_tool("study_project") == "study"
        assert registry.get_toolset_for_tool("study_schedule") == "study"
        assert registry.get_toolset_for_tool("study_prompt_context") == "study"
        assert manager.find_plugin_skill("study_os:study-os") is not None
    finally:
        for name in (
            "study_list_notes",
            "study_read_note",
            "study_extract_concepts",
            "study_log_error",
            "study_create_review_task",
            "study_generate_weekly_report",
            "study_export_anki_candidates",
            "study_due_reviews",
            "study_record_review",
            "study_sync_memory",
            "study_concept_graph",
            "study_review_stats",
            "study_learning_queue",
            "study_log_session",
            "study_update_concept_state",
            "study_import_plan",
            "study_plan_progress",
            "study_create_curriculum",
            "study_list_curricula",
            "study_project",
            "study_schedule",
            "study_prompt_context",
        ):
            registry.deregister(name)
