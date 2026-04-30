"""Tests for the curator per-run report writer (run.json + REPORT.md).

Reports live under ``~/.hermes/logs/curator/{YYYYMMDD-HHMMSS}/`` alongside
the standard log dir, not inside the user's ``skills/`` data directory.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest


@pytest.fixture
def curator_env(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with a skills/ dir + reset curator module state."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "skills").mkdir()
    (home / "logs").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    import importlib
    import hermes_constants
    importlib.reload(hermes_constants)
    from agent import curator
    importlib.reload(curator)
    from tools import skill_usage
    importlib.reload(skill_usage)
    yield {"home": home, "curator": curator, "skill_usage": skill_usage}


def _make_llm_meta(**overrides):
    base = {
        "final": "short summary of the pass",
        "summary": "short summary",
        "model": "test-model",
        "provider": "test-provider",
        "tool_calls": [],
        "error": None,
    }
    base.update(overrides)
    return base


def test_reports_root_is_under_logs_not_skills(curator_env):
    """Reports live in logs/curator/, not skills/ — operational telemetry
    belongs with the logs, not with user-authored skill data."""
    curator = curator_env["curator"]
    root = curator._reports_root()
    home = curator_env["home"]
    # Must be under logs/
    assert root == home / "logs" / "curator"
    # Must NOT be under skills/
    assert "skills" not in root.parts


def test_write_run_report_creates_both_files(curator_env):
    """Each run writes both a run.json (machine) and a REPORT.md (human)."""
    curator = curator_env["curator"]
    start = datetime.now(timezone.utc)

    run_dir = curator._write_run_report(
        started_at=start,
        elapsed_seconds=12.345,
        auto_counts={"checked": 5, "marked_stale": 1, "archived": 0, "reactivated": 0},
        auto_summary="1 marked stale",
        before_report=[],
        before_names=set(),
        after_report=[],
        llm_meta=_make_llm_meta(),
    )
    assert run_dir is not None
    assert run_dir.is_dir()
    assert (run_dir / "run.json").exists()
    assert (run_dir / "REPORT.md").exists()

    # The directory name is a timestamp under logs/curator/
    assert run_dir.parent == curator._reports_root()


def test_run_json_has_expected_shape(curator_env):
    """run.json must carry the machine-readable fields downstream tooling needs."""
    curator = curator_env["curator"]
    start = datetime.now(timezone.utc)

    before_report = [
        {"name": "old-thing", "state": "active", "pinned": False},
        {"name": "keeper", "state": "active", "pinned": True},
    ]
    after_report = [
        {"name": "keeper", "state": "active", "pinned": True},
        {"name": "new-umbrella", "state": "active", "pinned": False},
    ]

    run_dir = curator._write_run_report(
        started_at=start,
        elapsed_seconds=42.0,
        auto_counts={"checked": 2, "marked_stale": 0, "archived": 0, "reactivated": 0},
        auto_summary="no changes",
        before_report=before_report,
        before_names={r["name"] for r in before_report},
        after_report=after_report,
        llm_meta=_make_llm_meta(
            final="I consolidated the whole universe.",
            tool_calls=[
                {"name": "skills_list", "arguments": "{}"},
                {"name": "skill_manage", "arguments": '{"action":"create"}'},
                {"name": "terminal", "arguments": "mv ..."},
            ],
        ),
    )
    payload = json.loads((run_dir / "run.json").read_text())

    # top-level shape
    for k in (
        "started_at", "duration_seconds", "model", "provider",
        "auto_transitions", "counts", "tool_call_counts",
        "archived", "added", "state_transitions",
        "llm_final", "llm_summary", "llm_error", "tool_calls",
    ):
        assert k in payload, f"missing key: {k}"

    # Diff logic
    assert payload["archived"] == ["old-thing"]
    assert payload["added"] == ["new-umbrella"]
    # Counts reflect the diff
    assert payload["counts"]["before"] == 2
    assert payload["counts"]["after"] == 2
    assert payload["counts"]["archived_this_run"] == 1
    assert payload["counts"]["added_this_run"] == 1
    # Tool call counts are aggregated
    assert payload["tool_call_counts"]["skills_list"] == 1
    assert payload["tool_call_counts"]["skill_manage"] == 1
    assert payload["tool_call_counts"]["terminal"] == 1
    assert payload["counts"]["tool_calls_total"] == 3


def test_report_md_is_human_readable(curator_env):
    """REPORT.md should be a valid markdown doc with the key sections visible."""
    curator = curator_env["curator"]
    start = datetime.now(timezone.utc)

    run_dir = curator._write_run_report(
        started_at=start,
        elapsed_seconds=75.0,
        auto_counts={"checked": 10, "marked_stale": 2, "archived": 1, "reactivated": 0},
        auto_summary="2 marked stale, 1 archived",
        before_report=[{"name": "foo", "state": "active", "pinned": False}],
        before_names={"foo"},
        after_report=[{"name": "foo-umbrella", "state": "active", "pinned": False}],
        llm_meta=_make_llm_meta(
            final="Consolidated foo-like skills into foo-umbrella.",
            model="claude-opus-4.7",
            provider="openrouter",
            tool_calls=[
                # Evidence that `foo` was absorbed into `foo-umbrella`:
                # write_file under foo-umbrella referencing foo.
                {
                    "name": "skill_manage",
                    "arguments": json.dumps({
                        "action": "write_file",
                        "name": "foo-umbrella",
                        "file_path": "references/foo.md",
                        "file_content": "# foo\nContent absorbed from the old foo skill.\n",
                    }),
                },
            ],
        ),
    )
    md = (run_dir / "REPORT.md").read_text()

    # Structural checks
    assert "# Curator run" in md
    assert "Auto-transitions" in md
    assert "LLM consolidation pass" in md
    assert "Recovery" in md

    # The model / provider we passed in show up
    assert "claude-opus-4.7" in md
    assert "openrouter" in md

    # The consolidated/added lists are present with clear language
    assert "Consolidated into umbrella skills" in md
    assert "`foo`" in md
    assert "merged into" in md
    assert "`foo-umbrella`" in md
    assert "New skills this run" in md

    # The full LLM final response is included verbatim (no 240-char truncation)
    assert "Consolidated foo-like skills into foo-umbrella." in md


def test_same_second_reruns_get_unique_dirs(curator_env):
    """If the curator somehow runs twice in the same second, the second
    report still gets its own directory rather than overwriting the first."""
    curator = curator_env["curator"]
    start = datetime(2026, 4, 29, 5, 33, 34, tzinfo=timezone.utc)

    kwargs = dict(
        started_at=start,
        elapsed_seconds=1.0,
        auto_counts={"checked": 0, "marked_stale": 0, "archived": 0, "reactivated": 0},
        auto_summary="no changes",
        before_report=[],
        before_names=set(),
        after_report=[],
        llm_meta=_make_llm_meta(),
    )
    a = curator._write_run_report(**kwargs)
    b = curator._write_run_report(**kwargs)
    assert a != b
    assert a is not None and b is not None
    # Second dir has a numeric disambiguator suffix
    assert b.name.startswith(a.name)


def test_report_captures_llm_error_and_continues(curator_env):
    """If the LLM pass recorded an error, the report still writes and
    surfaces the error prominently."""
    curator = curator_env["curator"]
    run_dir = curator._write_run_report(
        started_at=datetime.now(timezone.utc),
        elapsed_seconds=2.0,
        auto_counts={"checked": 0, "marked_stale": 0, "archived": 0, "reactivated": 0},
        auto_summary="no changes",
        before_report=[],
        before_names=set(),
        after_report=[],
        llm_meta=_make_llm_meta(
            error="HTTP 400: No models provided",
            final="",
            summary="error",
        ),
    )
    md = (run_dir / "REPORT.md").read_text()
    assert "HTTP 400" in md
    payload = json.loads((run_dir / "run.json").read_text())
    assert payload["llm_error"] == "HTTP 400: No models provided"


def test_state_transitions_captured_in_report(curator_env):
    """When a skill moves active → stale or stale → archived between
    before/after snapshots, the report records it."""
    curator = curator_env["curator"]
    start = datetime.now(timezone.utc)

    before = [{"name": "getting-old", "state": "active", "pinned": False}]
    after = [{"name": "getting-old", "state": "stale", "pinned": False}]

    run_dir = curator._write_run_report(
        started_at=start,
        elapsed_seconds=1.0,
        auto_counts={"checked": 1, "marked_stale": 1, "archived": 0, "reactivated": 0},
        auto_summary="1 marked stale",
        before_report=before,
        before_names={r["name"] for r in before},
        after_report=after,
        llm_meta=_make_llm_meta(),
    )
    payload = json.loads((run_dir / "run.json").read_text())
    assert payload["state_transitions"] == [
        {"name": "getting-old", "from": "active", "to": "stale"}
    ]
    md = (run_dir / "REPORT.md").read_text()
    assert "State transitions" in md
    assert "getting-old" in md
    assert "active → stale" in md
