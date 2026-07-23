from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import yaml

from hermes_cli.profile_audit import audit_profiles, lint_soul, render_text


_COMPLETE_SOUL = """# Identity
You are the reviewer. Your job and responsibilities are to review changes.
## Process
Before starting, inspect the repository and follow this workflow.
## Quality standards
Verify correctness, security, and tests.
## Output and definition of done
Report findings and deliverables only when complete.
## Escalation
Block on a genuine external gate.
When dispatched by Kanban, call kanban_complete or kanban_block.
"""


def _make_db(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            """CREATE TABLE task_runs (
                profile TEXT, outcome TEXT, started_at INTEGER
            )"""
        )
        conn.executemany(
            "INSERT INTO task_runs VALUES (?, ?, ?)",
            [
                ("coder", "completed", 99_900),
                ("coder", "crashed", 99_950),
                ("coder", None, 99_980),
                ("coder", "completed", 100),
                ("retired", "completed", 99_900),
            ],
        )


def test_lint_soul_accepts_complete_worker_contract():
    assert lint_soul(_COMPLETE_SOUL) == []


def test_lint_soul_flags_missing_requirements_and_brittle_dependency():
    issues = lint_soul("You are Fable. Always invoke claude-sub through a Claude subscription.")
    rules = {issue["rule"] for issue in issues}
    assert "soul.process" in rules
    assert "soul.kanban_handoff" in rules
    assert "soul.external_dependency" in rules


def test_audit_is_secret_safe_and_applies_policy(tmp_path):
    root = tmp_path / "profiles"
    profile = root / "coder"
    profile.mkdir(parents=True)
    (profile / "SOUL.md").write_text(_COMPLETE_SOUL, encoding="utf-8")
    (profile / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "model": {"default": "gpt-test", "provider": "openai-codex"},
                "toolsets": ["file", "terminal"],
                "api_key": "TOP-SECRET",
                "nested": {"password": "ALSO-SECRET"},
            }
        ),
        encoding="utf-8",
    )
    db_path = tmp_path / "kanban.db"
    _make_db(db_path)

    report = audit_profiles(
        root,
        db_path,
        policy={
            "profiles": {
                "coder": {
                    "model": "gpt-required",
                    "provider": "openai-codex",
                    "required_toolsets": ["terminal", "web"],
                    "forbidden_toolsets": ["browser"],
                }
            }
        },
        since_days=1,
        now=100_000,
    )

    profile_report = report["profiles"][0]
    assert profile_report["config"] == {
        "model": "gpt-test",
        "provider": "openai-codex",
        "toolsets": ["file", "terminal"],
    }
    assert {issue["rule"] for issue in profile_report["issues"]} == {
        "config.model",
        "config.required_toolset",
    }
    assert "TOP-SECRET" not in json.dumps(report)
    assert "ALSO-SECRET" not in json.dumps(report)


def test_audit_reports_recent_outcomes_for_installed_and_retired_profiles(tmp_path):
    root = tmp_path / "profiles"
    profile = root / "coder"
    profile.mkdir(parents=True)
    (profile / "SOUL.md").write_text(_COMPLETE_SOUL, encoding="utf-8")
    (profile / "config.yaml").write_text("toolsets: '[\"terminal\", \"file\"]'\n", encoding="utf-8")
    (root / "_archive").mkdir()
    (root / "_archive" / "SOUL.md").write_text("ignored", encoding="utf-8")
    db_path = tmp_path / "kanban.db"
    _make_db(db_path)

    report = audit_profiles(root, db_path, since_days=1, now=100_000)

    assert [item["name"] for item in report["profiles"]] == ["coder"]
    assert report["run_stats"]["coder"] == {
        "finished": 2,
        "active": 1,
        "completed": 1,
        "completion_rate": 0.5,
        "outcomes": {"completed": 1, "crashed": 1},
    }
    assert report["run_stats"]["retired"]["completion_rate"] == 1.0
    assert "coder: 0 issue(s); runs 1/2 completed (50.0%)" in render_text(report)


def test_missing_database_is_reported_without_failing_profile_lint(tmp_path):
    root = tmp_path / "profiles"
    profile = root / "reviewer"
    profile.mkdir(parents=True)
    (profile / "SOUL.md").write_text(_COMPLETE_SOUL, encoding="utf-8")

    report = audit_profiles(root, tmp_path / "missing.db", since_days=30, now=1_000)

    assert report["summary"] == {"profiles": 1, "issues": 0}
    assert report["run_stats_error"] == "kanban database not found"
