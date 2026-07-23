"""Tests for built-in Hermes memory hygiene/audit helpers."""

from __future__ import annotations

import json
from pathlib import Path

from hermes_cli.memory_hygiene import (
    audit_memory,
    classify_entry,
    lint_entry,
    refresh_metadata,
)


def _write_memories(home: Path, memory: str = "", user: str = "") -> None:
    mem_dir = home / "memories"
    mem_dir.mkdir(parents=True)
    (mem_dir / "MEMORY.md").write_text(memory, encoding="utf-8")
    (mem_dir / "USER.md").write_text(user, encoding="utf-8")


def test_classify_entry_recognizes_common_memory_categories():
    assert classify_entry("Comms: no Discord; default delivery to Desktop/local") == "communication"
    assert classify_entry("Ori: app.oripilates.com uses Supabase") == "project"
    assert classify_entry("Hermes/dev: Mac mini /Users/ngoclawd") == "environment"
    assert classify_entry("Travel: Japan trains favor carry-on suitcases") == "travel"


def test_lint_entry_flags_task_logs_and_long_entries():
    issues = lint_entry(
        "2026-07-07 — fixed bug, shipped PR #123, commit abcdef1, session_id 20260707_x "
        + "x" * 260,
        target="memory",
    )
    codes = {issue["code"] for issue in issues}
    assert "task_log" in codes
    assert "artifact_id" in codes
    assert "too_long" in codes


def test_audit_memory_reports_usage_entries_and_issues(tmp_path):
    home = tmp_path / ".hermes"
    _write_memories(
        home,
        memory="Hermes/dev: Mac mini\n§\n2026-07-07 — shipped PR #123",
        user="Style: concise\n§\nDiscord is the primary chat surface",
    )

    report = audit_memory(home, write_metadata=False)

    assert report["stores"]["memory"]["entry_count"] == 2
    assert report["stores"]["user"]["entry_count"] == 2
    issue_codes = {
        issue["code"]
        for store in report["stores"].values()
        for entry in store["entries"]
        for issue in entry["issues"]
    }
    assert "task_log" in issue_codes
    assert "stale_discord" in issue_codes
    assert report["summary"]["total_issues"] >= 2


def test_refresh_metadata_preserves_first_seen_and_adds_category(tmp_path):
    home = tmp_path / ".hermes"
    _write_memories(home, memory="Hermes/dev: Mac mini", user="Style: concise")
    meta_path = home / "memories" / "metadata.json"
    meta_path.write_text(
        json.dumps({"entries": {"old": {"first_seen": "2026-01-01T00:00:00Z"}}}),
        encoding="utf-8",
    )

    entries = [
        {"target": "memory", "index": 1, "content": "Hermes/dev: Mac mini"},
        {"target": "user", "index": 1, "content": "Style: concise"},
    ]
    metadata = refresh_metadata(home, entries, now="2026-07-07T00:00:00Z")

    assert len(metadata["entries"]) == 2
    categories = {record["category"] for record in metadata["entries"].values()}
    assert categories == {"environment", "preference"}
    assert metadata["updated_at"] == "2026-07-07T00:00:00Z"
    assert meta_path.exists()
