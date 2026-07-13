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


def test_due_reviews_discovers_examples_in_subject_folders(vault: Path):
    from plugins.study_os.tools import handle_study_due_reviews

    math_example = vault / "Math" / "examples" / "limit.md"
    math_example.parent.mkdir(parents=True)
    math_example.write_text(
        "---\ntype: example\nreview_level: 0\n---\n# 跨课程优先项\n",
        encoding="utf-8",
    )
    result = _loads(handle_study_due_reviews({"vault_path": str(vault), "limit": 1}))

    assert result["ok"] is True
    assert result["data"]["count"] == 1
    assert result["data"]["due"][0]["path"] == "Math/examples/limit.md"
    assert result["data"]["due"][0]["subject"] == "Math"
    assert result["data"]["subjects"] == ["Math", "OS"]


def test_due_reviews_supports_explicit_and_composable_review_selectors(vault: Path):
    from plugins.study_os.tools import handle_study_due_reviews

    selected = vault / "Math" / "examples" / "limit.md"
    selected.parent.mkdir(parents=True)
    selected.write_text(
        "---\n"
        "type: example\n"
        "difficulty: hard\n"
        "review_level: 4\n"
        "review_count: 3\n"
        "next_review_at: 2099-01-01\n"
        "tags: [math, calculus]\n"
        "concepts: [Taylor expansion]\n"
        "---\n# Taylor drill\n",
        encoding="utf-8",
    )
    result = _loads(
        handle_study_due_reviews(
            {
                "vault_path": str(vault),
                "notes": ["Math/examples/limit.md"],
                "tags": ["math", "calculus"],
                "concepts": ["taylor"],
                "difficulties": ["hard"],
                "min_review_level": 3,
                "review_state": "all",
                "match": "all",
                "sort": "title",
            }
        )
    )

    assert result["ok"] is True
    assert [item["path"] for item in result["data"]["due"]] == ["Math/examples/limit.md"]
    assert result["data"]["selection"] == {"review_state": "all", "sort": "title", "match": "all"}


def test_due_reviews_default_scope_remains_due_only(vault: Path):
    from plugins.study_os.tools import handle_study_due_reviews

    future = vault / "OS" / "examples" / "future.md"
    future.write_text(
        "---\ntype: example\nreview_level: 0\nnext_review_at: 2099-01-01\n---\n# Not due\n",
        encoding="utf-8",
    )

    default_result = _loads(handle_study_due_reviews({"vault_path": str(vault)}))
    all_result = _loads(handle_study_due_reviews({"vault_path": str(vault), "review_state": "all"}))

    assert "OS/examples/future.md" not in {item["path"] for item in default_result["data"]["due"]}
    assert "OS/examples/future.md" in {item["path"] for item in all_result["data"]["due"]}


def test_review_submit_records_one_atomic_attempt_and_spacing_update(vault: Path):
    from plugins.study_os.learning import handle_study_activity
    from plugins.study_os.tools import handle_study_project

    init = _loads(handle_study_project({"vault_path": str(vault), "action": "init"}))
    submitted = _loads(
        handle_study_activity(
            {
                "resource": "review",
                "action": "submit",
                "vault_path": str(vault),
                "project_id": init["data"]["project"]["project_id"],
                "data": {
                    "note": "OS/examples/OS-0043.md",
                    "response": "高级调度决定作业接纳，低级调度负责进程切换。",
                    "result": "correct",
                    "duration_seconds": 30,
                    "self_confidence": 4,
                },
            }
        )
    )

    assert submitted["ok"] is True
    assert submitted["data"]["attempt"]["item_id"] == "OS/examples/OS-0043.md"
    assert submitted["data"]["review"]["review_count"] == {"old": 0, "new": 1}
    assert list((vault / ".StudyOS" / "projects" / "general-learning" / "activity").glob("attempts-*.jsonl"))


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
    schedule["project_id"] = "general-learning"
    schedule["schedule_id"] = "general-learning-master-plan"
    schedule["events"] = []
    save = _loads(handle_study_schedule({"vault_path": str(vault), "action": "save", "data": schedule}))
    read = _loads(
        handle_study_schedule(
            {
                "vault_path": str(vault),
                "action": "read",
                "project_id": "general-learning",
                "schedule_id": "general-learning-master-plan",
            }
        )
    )

    assert init["ok"] is True
    assert init["data"]["path"] == ".StudyOS/projects/general-learning/manifest.json"
    assert init["data"]["active_path"] == ".StudyOS/projects/active.json"
    assert init["data"]["project"]["domain_pack"] == "general.v1"
    assert status["ok"] is True
    assert status["data"]["project"]["project_id"] == "general-learning"
    assert save["ok"] is True
    assert save["data"]["path"] == ".StudyOS/projects/general-learning/schedules/general-learning-master-plan.json"
    assert read["ok"] is True
    assert read["data"]["schedule"]["events"] == []
    assert (vault / ".StudyOS" / "projects" / "general-learning" / "manifest.json").exists()
    assert (vault / ".StudyOS" / "projects" / "active.json").exists()
    assert (vault / ".StudyOS" / "projects" / "general-learning" / "schedules" / "general-learning-master-plan.json").exists()


def test_study_project_init_keeps_kaoyan_explicit(vault: Path):
    from plugins.study_os.tools import handle_study_project, handle_study_schedule

    init = _loads(handle_study_project({"vault_path": str(vault), "action": "init", "domain_pack": "kaoyan.v1"}))
    template = _loads(handle_study_schedule({"vault_path": str(vault), "action": "template", "project_id": "kaoyan-2027"}))

    assert init["ok"] is True
    assert init["data"]["project"]["project_id"] == "kaoyan-2027"
    assert init["data"]["project"]["domain_pack"] == "kaoyan.v1"
    assert init["data"]["project"]["workspace_type"] == "exam-vault"
    assert template["ok"] is True
    assert template["data"]["schedule"]["events"][0]["subject_id"] == "math"


def test_study_project_engineering_prompt_context(vault: Path):
    from plugins.study_os.tools import handle_study_project, handle_study_prompt_context, handle_study_schedule

    init = _loads(
        handle_study_project(
            {
                "vault_path": str(vault),
                "action": "init",
                "project_id": "ai-infra",
                "title": "AI Infra Learning",
                "domain": "ai-infra",
                "domain_pack": "engineering.v1",
                "workspace_type": "hybrid",
                "subjects": [{"id": "ai-infra", "label": "AI Infra"}],
            }
        )
    )
    context = _loads(
        handle_study_prompt_context(
            {
                "vault_path": str(vault),
                "intent": "planning",
                "project_id": "ai-infra",
            }
        )
    )
    template = _loads(handle_study_schedule({"vault_path": str(vault), "action": "template", "project_id": "ai-infra"}))

    assert init["ok"] is True
    assert init["data"]["project"]["domain_pack"] == "engineering.v1"
    assert init["data"]["project"]["workspace_type"] == "hybrid"
    assert context["ok"] is True
    fragments = {fragment["kind"]: fragment["content"] for fragment in context["data"]["fragments"]}
    assert "engineering and skill learning" in fragments["domain"]
    assert "hybrid" in fragments["domain"]
    assert template["ok"] is True
    assert template["data"]["schedule"]["events"][0]["subject_id"] == "ai-infra"
    assert "Scout one concept" in template["data"]["schedule"]["events"][0]["title"]


def test_study_decision_creates_learning_decision_record(vault: Path):
    from plugins.study_os.tools import handle_study_decision, handle_study_project

    init = _loads(
        handle_study_project(
            {
                "vault_path": str(vault),
                "action": "init",
                "project_id": "ai-infra",
                "domain_pack": "engineering.v1",
                "subjects": [{"id": "ai-infra", "label": "AI Infra"}],
            }
        )
    )
    created = _loads(
        handle_study_decision(
            {
                "vault_path": str(vault),
                "project_id": "ai-infra",
                "action": "create",
                "title": "Use hybrid AI Infra workspace",
                "decision": "Keep source exploration in infra-learning and reusable concepts in AIInfra.",
                "context": "The user wants StudyOS to support engineering learning without copying exam-vault behavior.",
                "options_considered": ["all notes in repo", "heavy StudyOS vault", "hybrid workspace"],
                "linked_concepts": ["KV Cache", "PagedAttention"],
                "linked_sources": ["/home/puji/infra-learning"],
                "linked_sessions": ["grill:2026-07-01"],
            }
        )
    )
    listed = _loads(handle_study_decision({"vault_path": str(vault), "project_id": "ai-infra", "action": "list"}))
    read = _loads(
        handle_study_decision(
            {
                "vault_path": str(vault),
                "project_id": "ai-infra",
                "action": "read",
                "decision_id": created["data"]["decision"]["decision_id"],
            }
        )
    )

    assert init["ok"] is True
    assert created["ok"] is True
    assert created["data"]["path"].startswith(".StudyOS/projects/ai-infra/decisions/")
    assert (vault / created["data"]["path"]).exists()
    assert listed["data"]["decisions"][0]["title"] == "Use hybrid AI Infra workspace"
    assert read["ok"] is True
    assert "schema_version: learning_decision_record.v1" in read["data"]["content"]
    assert "KV Cache" in read["data"]["content"]


def test_study_learning_record_creates_learning_record(vault: Path):
    from plugins.study_os.tools import handle_study_learning_record, handle_study_project, handle_study_prompt_context

    init = _loads(
        handle_study_project(
            {
                "vault_path": str(vault),
                "action": "init",
                "project_id": "ai-infra",
                "domain_pack": "engineering.v1",
                "subjects": [{"id": "ai-infra", "label": "AI Infra"}],
            }
        )
    )
    context = _loads(
        handle_study_prompt_context(
            {
                "vault_path": str(vault),
                "project_id": "ai-infra",
                "intent": "teaching",
            }
        )
    )
    created = _loads(
        handle_study_learning_record(
            {
                "vault_path": str(vault),
                "project_id": "ai-infra",
                "action": "create",
                "title": "Understands prefill versus decode",
                "summary": "The user can explain why LLM serving separates prompt prefill from token decode.",
                "evidence": "They correctly compared long-prompt startup cost with per-token decode cost.",
                "implications": "Future lessons can discuss continuous batching without re-teaching this split.",
                "linked_concepts": ["Prefill", "Decode"],
                "linked_sources": ["/home/puji/infra-learning/vllm"],
            }
        )
    )
    listed = _loads(handle_study_learning_record({"vault_path": str(vault), "project_id": "ai-infra", "action": "list"}))
    read = _loads(
        handle_study_learning_record(
            {
                "vault_path": str(vault),
                "project_id": "ai-infra",
                "action": "read",
                "record_id": created["data"]["record"]["record_id"],
            }
        )
    )

    assert init["ok"] is True
    assert context["ok"] is True
    assert any("study-teach" in fragment["source"] for fragment in context["data"]["fragments"])
    assert created["ok"] is True
    assert created["data"]["path"].startswith(".StudyOS/projects/ai-infra/learning-records/")
    assert listed["data"]["records"][0]["title"] == "Understands prefill versus decode"
    assert read["ok"] is True
    assert "schema_version: learning_record.v1" in read["data"]["content"]
    assert "continuous batching" in read["data"]["content"]


def test_study_learning_record_requires_evidence(vault: Path):
    from plugins.study_os.tools import handle_study_learning_record, handle_study_project

    init = _loads(handle_study_project({"vault_path": str(vault), "action": "init"}))
    created = _loads(
        handle_study_learning_record(
            {
                "vault_path": str(vault),
                "action": "create",
                "title": "Covered prefill",
                "summary": "The session mentioned prefill.",
            }
        )
    )

    assert init["ok"] is True
    assert created["ok"] is False
    assert created["error"]["code"] == "VALIDATION_FAILED"
    assert not (vault / ".StudyOS" / "projects" / "general-learning" / "learning-records").exists()


def test_study_lesson_creates_visual_lesson_artifact(vault: Path):
    from plugins.study_os.tools import handle_study_lesson, handle_study_project

    init = _loads(handle_study_project({"vault_path": str(vault), "action": "init"}))
    html = "<!doctype html><html><head><title>Prefill vs Decode</title></head><body><h1>Prefill vs Decode</h1></body></html>"
    created = _loads(
        handle_study_lesson(
            {
                "vault_path": str(vault),
                "action": "create",
                "title": "Prefill vs Decode",
                "rationale": "The split is easiest to understand as a request timeline.",
                "html": html,
                "linked_concepts": ["Prefill", "Decode"],
                "linked_sources": ["/home/puji/infra-learning/vllm"],
            }
        )
    )
    listed = _loads(handle_study_lesson({"vault_path": str(vault), "action": "list"}))
    read = _loads(
        handle_study_lesson(
            {
                "vault_path": str(vault),
                "action": "read",
                "lesson_id": created["data"]["lesson"]["lesson_id"],
            }
        )
    )

    assert init["ok"] is True
    assert created["ok"] is True
    assert created["data"]["path"].startswith(".StudyOS/projects/general-learning/lessons/")
    assert created["data"]["metadata_path"].endswith(".json")
    assert (vault / created["data"]["path"]).exists()
    assert listed["data"]["lessons"][0]["title"] == "Prefill vs Decode"
    assert read["ok"] is True
    assert read["data"]["metadata"]["schema_version"] == "visual_lesson.v1"
    assert "Prefill vs Decode" in read["data"]["html"]


def test_study_lesson_requires_complete_html(vault: Path):
    from plugins.study_os.tools import handle_study_lesson, handle_study_project

    init = _loads(handle_study_project({"vault_path": str(vault), "action": "init"}))
    created = _loads(
        handle_study_lesson(
            {
                "vault_path": str(vault),
                "action": "create",
                "title": "Incomplete",
                "rationale": "Visual layout needed.",
                "html": "<section>not complete</section>",
            }
        )
    )

    assert init["ok"] is True
    assert created["ok"] is False
    assert created["error"]["code"] == "VALIDATION_FAILED"


def test_study_schedule_read_returns_not_found(vault: Path):
    from plugins.study_os.tools import handle_study_project, handle_study_schedule

    init = _loads(handle_study_project({"vault_path": str(vault), "action": "init", "domain_pack": "kaoyan.v1"}))
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


def test_study_activity_loads_all_workflow_contexts_within_budget(vault: Path):
    from plugins.study_os.learning import handle_study_activity

    initialized = _loads(
        handle_study_activity(
            {
                "resource": "project",
                "action": "init",
                "vault_path": str(vault),
                "data": {"project_id": "general-2027"},
            }
        )
    )
    assert initialized["ok"] is True

    for intent in ("planning", "schedule_adjustment", "organizing", "reviewing", "teaching", "assessment", "error_analysis"):
        context = _loads(
            handle_study_activity(
                {
                    "resource": "prompt_context",
                    "action": "load",
                    "vault_path": str(vault),
                    "project_id": "general-2027",
                    "data": {"intent": intent},
                }
            )
        )
        assert context["ok"] is True, intent
        assert {fragment["kind"] for fragment in context["data"]["fragments"]} == {"base", "intent"}
        assert context["data"]["total_char_count"] <= 6000

    for project_id, domain_pack in (("kaoyan-2027", "kaoyan.v1"), ("ai-infra", "engineering.v1")):
        project = _loads(
            handle_study_activity(
                {
                    "resource": "project",
                    "action": "init",
                    "vault_path": str(vault),
                    "data": {"project_id": project_id, "domain_pack": domain_pack},
                }
            )
        )
        context = _loads(
            handle_study_activity(
                {
                    "resource": "prompt_context",
                    "action": "load",
                    "vault_path": str(vault),
                    "project_id": project_id,
                    "data": {"intent": "reviewing"},
                }
            )
        )

        assert project["ok"] is True
        assert context["ok"] is True
        assert {fragment["kind"] for fragment in context["data"]["fragments"]} == {"base", "intent", "domain"}
        assert context["data"]["total_char_count"] <= 6000


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
            "study-teach",
            "study-lesson",
            "study-assessment",
            "study-kaoyan",
            "study-engineering",
            "study-grill",
        ):
            assert manager.find_plugin_skill(f"study_os:{name}") is not None
    finally:
        for name in ("study_activity", "study_coach"):
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
            "study-teach": ("Teach through StudyOS learning records.", 9000),
            "study-lesson": ("Create visual StudyOS lesson artifacts.", 9000),
            "study-assessment": ("Analyze StudyOS exams and mistakes.", 9000),
            "study-kaoyan": ("Guide 考研 learning with StudyOS.", 9000),
            "study-engineering": ("Guide engineering and skill learning with StudyOS.", 9000),
            "study-grill": ("Bridge grilling sessions into StudyOS decisions.", 9000),
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
            assert "study_activity" in body
            assert "mutate system prompts" in body
        for term in ("艾宾浩斯", "整理", "错题", "weekly", "curriculum", "考研", "engineering", "LearningDecisionRecord", "LearningRecord", "VisualLesson"):
            assert term in all_text
    finally:
        for name in ("study_activity", "study_coach"):
            registry.deregister(name)


def test_study_toolset_is_opt_in():
    from toolsets import TOOLSETS, _HERMES_CORE_TOOLS

    expected_tools = ["study_activity", "study_coach"]

    assert TOOLSETS["study"]["tools"] == expected_tools
    for tool in expected_tools:
        assert tool not in _HERMES_CORE_TOOLS


def test_study_activity_records_and_queries_immutable_attempts(vault: Path):
    from plugins.study_os.learning import handle_study_activity

    initialized = _loads(
        handle_study_activity(
            {
                "resource": "project",
                "action": "init",
                "vault_path": str(vault),
                "data": {"project_id": "calculus-2027"},
            }
        )
    )
    recorded = _loads(
        handle_study_activity(
            {
                "resource": "attempt",
                "action": "record",
                "vault_path": str(vault),
                "project_id": "calculus-2027",
                "data": {
                    "attempt_id": "att-derivative-001",
                    "item_id": "derivative-sign-01",
                    "occurred_at": "2026-07-12T10:00:00+08:00",
                    "response": "Divided by an expression without checking its sign.",
                    "result": "incorrect",
                    "score": 0.2,
                    "self_confidence": 5,
                    "transfer_level": "execution",
                    "concepts": ["函数单调性"],
                    "patterns": ["含参导数符号判断"],
                    "diagnoses": [
                        {
                            "kind": "condition_missed",
                            "concept": "函数单调性",
                            "evidence": "The sign of the divisor was never established.",
                        }
                    ],
                },
            }
        )
    )
    listed = _loads(
        handle_study_activity(
            {
                "resource": "attempt",
                "action": "list",
                "vault_path": str(vault),
                "project_id": "calculus-2027",
                "data": {"concept": "函数单调性"},
            }
        )
    )
    duplicate = _loads(
        handle_study_activity(
            {
                "resource": "attempt",
                "action": "record",
                "vault_path": str(vault),
                "project_id": "calculus-2027",
                "data": recorded["data"]["attempt"],
            }
        )
    )

    assert initialized["ok"] is True
    assert recorded["ok"] is True
    assert recorded["data"]["path"].endswith("activity/attempts-2026-07.jsonl")
    assert listed["data"]["count"] == 1
    assert listed["data"]["attempts"][0]["response"].startswith("Divided")
    assert duplicate["ok"] is False
    assert duplicate["error"]["code"] == "ATTEMPT_EXISTS"


def test_study_coach_uses_evidence_for_summary_recommendation_and_pattern_proposal(vault: Path):
    from plugins.study_os.learning import handle_study_activity, handle_study_coach

    init = _loads(
        handle_study_activity(
            {"resource": "project", "action": "init", "vault_path": str(vault), "data": {"project_id": "calculus-2027"}}
        )
    )
    assert init["ok"] is True
    for index, confidence in enumerate((4, 5), start=1):
        result = _loads(
            handle_study_activity(
                {
                    "resource": "attempt",
                    "action": "record",
                    "vault_path": str(vault),
                    "project_id": "calculus-2027",
                    "data": {
                        "attempt_id": f"att-condition-{index:03d}",
                        "item_id": f"item-{index}",
                        "occurred_at": f"2026-07-1{index}T10:00:00+08:00",
                        "response": "Applied the routine before checking the condition.",
                        "result": "incorrect",
                        "score": 0.0,
                        "self_confidence": confidence,
                        "transfer_level": "execution",
                        "concepts": ["函数单调性"],
                        "patterns": ["含参导数符号判断"],
                        "diagnoses": [
                            {
                                "kind": "condition_missed",
                                "concept": "函数单调性",
                                "evidence": "The required sign condition was omitted.",
                            }
                        ],
                    },
                }
            )
        )
        assert result["ok"] is True

    summary = _loads(
        handle_study_coach(
            {"action": "summarize", "scope": "week", "vault_path": str(vault), "project_id": "calculus-2027"}
        )
    )
    recommendation = _loads(
        handle_study_coach(
            {"action": "recommend", "vault_path": str(vault), "project_id": "calculus-2027"}
        )
    )
    proposed = _loads(
        handle_study_coach(
            {"action": "propose_pattern", "vault_path": str(vault), "project_id": "calculus-2027"}
        )
    )
    probe = _loads(
        handle_study_coach(
            {"action": "generate_probe", "vault_path": str(vault), "project_id": "calculus-2027"}
        )
    )

    assert summary["data"]["summary"]["attempt_count"] == 2
    assert summary["data"]["summary"]["evidence_attempt_ids"] == ["att-condition-001", "att-condition-002"]
    assert "transfer" in summary["data"]["summary"]["unverified"]
    assert summary["data"]["summary"]["attempt_count"] == 2
    assert recommendation["data"]["diagnosis"]["mastery_dimensions"]["execution"]["attempt_count"] == 2
    interventions = {item["intervention"] for item in recommendation["data"]["recommendations"]}
    assert {"misconception_probe", "calibration_check", "near_transfer_probe"} <= interventions
    proposal = proposed["data"]["proposals"][0]
    assert proposal["status"] == "candidate"
    assert proposal["evidence_attempt_ids"] == ["att-condition-001", "att-condition-002"]
    assert probe["data"]["probe_blueprint"]["target_concept"] == "函数单调性"
    assert probe["data"]["probe_blueprint"]["response_policy"].startswith("ask for")
    assert not (vault / ".StudyOS" / "projects" / "calculus-2027" / "pattern-proposals").exists()

    saved = _loads(
        handle_study_activity(
            {
                "resource": "pattern_proposal",
                "action": "save",
                "vault_path": str(vault),
                "project_id": "calculus-2027",
                "data": {"proposal": proposal},
            }
        )
    )
    assert saved["ok"] is True
    assert saved["data"]["path"].endswith(f"pattern-proposals/{proposal['proposal_id']}.json")


def test_review_runner_reads_hidden_answer_and_submits_one_compound_result(vault: Path):
    from plugins.study_os.learning import handle_study_review_detail, handle_study_review_submission
    from plugins.study_os.tools import handle_study_project

    initialized = _loads(handle_study_project({"vault_path": str(vault), "action": "init", "project_id": "calculus-2027"}))
    assert initialized["ok"] is True
    note = vault / "math" / "examples" / "derivative.md"
    note.parent.mkdir(parents=True)
    note.write_text(
        "---\n"
        "title: 导数符号判断\n"
        "type: example\n"
        "review_level: 2\n"
        "review_count: 1\n"
        "concepts: [函数单调性]\n"
        "patterns: [含参导数符号判断]\n"
        "---\n\n"
        "# 导数符号判断\n\n求函数的单调区间。\n\n## 解析\n\n先判断参数符号。\n",
        encoding="utf-8",
    )

    detail = _loads(handle_study_review_detail({"vault_path": str(vault), "note": "math/examples/derivative.md"}))
    submitted = _loads(
        handle_study_review_submission(
            {
                "vault_path": str(vault),
                "project_id": "calculus-2027",
                "note": "math/examples/derivative.md",
                "attempt_id": "att-review-runner-001",
                "occurred_at": "2026-07-12T10:00:00+08:00",
                "response": "先求导，再按参数符号分类讨论。",
                "result": "correct",
                "duration_seconds": 93,
                "self_confidence": 4,
                "transfer_level": "execution",
                "diagnoses": [],
            }
        )
    )

    assert detail["ok"] is True
    assert detail["data"]["prompt_markdown"].endswith("求函数的单调区间。")
    assert "先判断参数符号" not in detail["data"]["prompt_markdown"]
    assert detail["data"]["answer_markdown"].startswith("## 解析")
    assert submitted["ok"] is True
    assert submitted["data"]["attempt"]["item_id"] == "math/examples/derivative.md"
    assert submitted["data"]["attempt"]["self_confidence"] == 4
    assert submitted["data"]["review"]["review_level"] == {"old": 2, "new": 3}
    assert submitted["data"]["completed_today_increment"] == 1
    updated = note.read_text(encoding="utf-8")
    assert "review_level: 3" in updated
    assert "review_count: 2" in updated


def test_review_runner_rolls_back_attempt_when_review_update_fails(vault: Path, monkeypatch):
    from plugins.study_os import learning
    from plugins.study_os.tools import handle_study_project

    initialized = _loads(handle_study_project({"vault_path": str(vault), "action": "init", "project_id": "calculus-2027"}))
    assert initialized["ok"] is True
    note = vault / "examples" / "rollback.md"
    note.parent.mkdir(parents=True)
    original = "---\ntitle: Rollback\ntype: example\nreview_level: 0\n---\n\n# Rollback\n"
    note.write_text(original, encoding="utf-8")
    monkeypatch.setattr(
        learning.legacy,
        "handle_study_record_review",
        lambda _args: json.dumps({"ok": False, "error": {"message": "disk write failed"}, "warnings": []}),
    )

    result = _loads(
        learning.handle_study_review_submission(
            {
                "vault_path": str(vault),
                "project_id": "calculus-2027",
                "note": "examples/rollback.md",
                "attempt_id": "att-rollback-001",
                "response": "answer",
                "result": "incorrect",
                "duration_seconds": 12,
                "self_confidence": 3,
            }
        )
    )

    assert result["ok"] is False
    assert result["error"]["code"] == "REVIEW_SUBMISSION_FAILED"
    assert note.read_text(encoding="utf-8") == original
    activity = vault / ".StudyOS" / "projects" / "calculus-2027" / "activity"
    assert not list(activity.glob("attempts-*.jsonl"))


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
        assert registry.get_toolset_for_tool("study_activity") == "study"
        assert registry.get_toolset_for_tool("study_coach") == "study"
        assert manager.find_plugin_skill("study_os:study-os") is not None
    finally:
        for name in ("study_activity", "study_coach"):
            registry.deregister(name)
