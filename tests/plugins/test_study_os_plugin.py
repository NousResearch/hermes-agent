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
        ):
            registry.deregister(name)
