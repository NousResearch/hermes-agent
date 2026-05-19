import json
from pathlib import Path

from agent.context_hygiene import audit_context_hygiene


def test_context_hygiene_separates_layers_without_raw_content(tmp_path):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    cwd = tmp_path / "private-project"
    cwd.mkdir()

    (hermes_home / "SOUL.md").write_text("# Identity\nHermes is Carson's operator.\n")
    (cwd / "AGENTS.md").write_text("# Project rules\nUse scripts/run_tests.sh.\n")
    (hermes_home / "MEMORY.md").write_text(
        "§\nFixed PR #123 in /home/alice/private-project; 45 tests passed.\n"
    )
    (hermes_home / "USER.md").write_text("§\nUser prefers concise status.\n")
    skill_dir = hermes_home / "skills" / "local" / "bad-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: bad-skill\n---\n# Skill\n")
    (hermes_home / "sessions").mkdir()
    (hermes_home / "sessions" / "session_1.json").write_text("{}")
    (hermes_home / "harness").mkdir()
    (hermes_home / "harness" / "turn-traces.jsonl").write_text('{"trace_id":"turn_1"}\n')

    summary = audit_context_hygiene(hermes_home=hermes_home, cwd=cwd)

    assert summary["content_policy"] == "metadata_only"
    assert set(summary["layers"]) == {
        "soul",
        "project_context",
        "skills",
        "memory",
        "sessions_traces",
    }
    assert summary["layers"]["soul"]["present"] is True
    assert summary["layers"]["project_context"]["present"] is True
    assert summary["layers"]["skills"]["skill_count"] == 1
    assert summary["layers"]["skills"]["invalid_frontmatter_count"] == 1
    assert summary["layers"]["memory"]["task_progress_hits"] >= 1
    assert summary["layers"]["sessions_traces"]["session_count"] == 1
    assert summary["layers"]["sessions_traces"]["turn_trace_count"] == 1

    issue_codes = {issue["code"] for issue in summary["issues"]}
    assert "memory_contains_task_progress" in issue_codes
    assert "skill_frontmatter_incomplete" in issue_codes

    raw = json.dumps(summary, sort_keys=True)
    assert "Fixed PR #123" not in raw
    assert "private-project" not in raw
    assert "Use scripts/run_tests.sh" not in raw
    assert "Carson's operator" not in raw


def test_context_hygiene_missing_sources_are_structural(tmp_path):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    cwd = tmp_path / "project"
    cwd.mkdir()

    summary = audit_context_hygiene(hermes_home=hermes_home, cwd=cwd)

    assert summary["layers"]["soul"]["present"] is False
    assert summary["layers"]["project_context"]["present"] is False
    assert summary["layers"]["memory"]["memory_files_present"] == []
    assert {issue["code"] for issue in summary["issues"]} >= {
        "soul_missing",
        "project_context_missing",
    }


def test_context_hygiene_defaults_to_terminal_or_process_cwd(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    cwd = tmp_path / "project"
    cwd.mkdir()
    (cwd / "HERMES.md").write_text("# Private project context\nDo not leak this line.\n")

    monkeypatch.chdir(cwd)
    process_summary = audit_context_hygiene(hermes_home=hermes_home)
    assert process_summary["layers"]["project_context"]["present"] is True
    assert process_summary["layers"]["project_context"]["file_count"] == 1

    other = tmp_path / "other"
    other.mkdir()
    (other / "agents.md").write_text("# Lowercase agents context\n")
    monkeypatch.setenv("TERMINAL_CWD", str(other))
    terminal_summary = audit_context_hygiene(hermes_home=hermes_home)
    assert terminal_summary["layers"]["project_context"]["present"] is True
    assert terminal_summary["layers"]["project_context"]["file_count"] == 1

    raw = json.dumps(terminal_summary, sort_keys=True)
    assert "Lowercase agents context" not in raw
    assert "other" not in raw


def test_context_hygiene_project_context_matches_prompt_builder_boundaries(tmp_path):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    parent = tmp_path / "parent"
    repo = parent / "repo"
    sub = repo / "sub"
    sub.mkdir(parents=True)
    (repo / ".git").mkdir()
    (parent / "HERMES.md").write_text("# Parent context outside git root\n")

    outside_summary = audit_context_hygiene(hermes_home=hermes_home, cwd=sub)
    assert outside_summary["layers"]["project_context"]["present"] is False

    (repo / "HERMES.md").write_text("# Repo HERMES context\n")
    (sub / "HERMES.md").write_text("   \n")
    empty_nearest_summary = audit_context_hygiene(hermes_home=hermes_home, cwd=sub)
    assert empty_nearest_summary["layers"]["project_context"]["present"] is False

    (repo / "HERMES.md").write_text("   \n")
    empty_summary = audit_context_hygiene(hermes_home=hermes_home, cwd=repo)
    assert empty_summary["layers"]["project_context"]["present"] is False

    (repo / "AGENTS.md").write_text("# Repo context after empty HERMES fallback\n")
    fallback_summary = audit_context_hygiene(hermes_home=hermes_home, cwd=repo)
    assert fallback_summary["layers"]["project_context"]["present"] is True
    assert fallback_summary["layers"]["project_context"]["file_count"] == 1

    raw = json.dumps(fallback_summary, sort_keys=True)
    assert "Parent context outside git root" not in raw
    assert "Repo context after empty HERMES fallback" not in raw
