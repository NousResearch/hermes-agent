"""WATCHDOG.yml discovery: search path, merge precedence, validation, round-trip."""

from __future__ import annotations

import os
from pathlib import Path

from agent.adviser_config import (
    discover_advisers,
    discover_watchdog_files,
    expand_at_imports,
    load_watchdog_config_file,
    save_watchdog_config_file,
    serialize_watchdog_config,
    slugify_adviser_name,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_user_and_project_merge_leaf_wins(tmp_path):
    home = tmp_path / "home"
    project = tmp_path / "project" / "sub"
    project.mkdir(parents=True)
    _write(home / "WATCHDOG.yml", """
instructions: user baseline
advisors:
  - name: Reviewer
    model: user-model
    tools: [read, grep]
""")
    _write(project / "WATCHDOG.yml", """
instructions: leaf baseline
advisors:
  - name: Reviewer
    model: leaf-model
""")

    found = discover_advisers(str(project), str(home))

    assert [a.name for a in found.advisers] == ["Reviewer"]
    entry = found.advisers[0]
    assert entry.model == "leaf-model"
    # A more specific entry replaces the whole entry, so the user tools do not leak through.
    assert entry.tools is None
    assert found.shared_instructions == "user baseline\n\nleaf baseline"


def test_hidden_project_dir_uses_hermes_dir(tmp_path):
    home = tmp_path / "home"
    project = tmp_path / "repo"
    project.mkdir()
    _write(project / ".hermes" / "WATCHDOG.yml", "advisors:\n  - name: Hidden Reviewer\n")

    found = discover_advisers(str(project), str(home))

    assert [a.slug for a in found.advisers] == ["hidden-reviewer"]


def test_bad_entry_is_dropped_and_healthy_ones_load(tmp_path):
    home = tmp_path / "home"
    project = tmp_path / "project"
    project.mkdir()
    _write(project / "WATCHDOG.yml", """
advisors:
  - name: Good
    reviewMode: turn
  - tools: [read]
  - name: Bad Mode
    reviewMode: sometimes
  - name: Bad Interval
    reviewInterval: 0
""")

    found = discover_advisers(str(project), str(home))

    assert [a.name for a in found.advisers] == ["Good", "Bad Mode", "Bad Interval"]
    assert found.advisers[1].review_mode is None
    assert found.advisers[2].review_interval is None
    assert any("name is required" in warning for warning in found.warnings)
    assert any("reviewMode" in warning for warning in found.warnings)
    assert any("reviewInterval" in warning for warning in found.warnings)


def test_unknown_tools_are_aliased_and_empty_list_is_kept(tmp_path):
    home = tmp_path / "home"
    project = tmp_path / "project"
    project.mkdir()
    _write(project / "WATCHDOG.yml", """
advisors:
  - name: Reader
    tools: [read, grep, glob, read]
  - name: Silent
    tools: []
""")

    found = discover_advisers(str(project), str(home))

    assert found.advisers[0].tools == ["read_file", "search_files"]
    assert found.advisers[1].tools == []


def test_watchdog_md_blocks_follow_the_same_search_path(tmp_path):
    home = tmp_path / "home"
    project = tmp_path / "project" / "sub"
    project.mkdir(parents=True)
    _write(home / "WATCHDOG.md", "user attention")
    _write(project / "WATCHDOG.md", "leaf attention")

    text = discover_watchdog_files(str(project), str(home))

    assert text.index("user attention") < text.index("leaf attention")
    assert text.count("<attention>") == 2


def test_at_imports_expand_nested_files_and_leave_missing_targets(tmp_path):
    base = tmp_path / "WATCHDOG.yml"
    _write(tmp_path / "chunk.md", "imported body @missing.md")
    text = expand_at_imports("start @chunk.md end", str(base))

    assert "imported body" in text
    assert "@missing.md" in text


def test_serialize_round_trips_through_load(tmp_path):
    doc = {
        "instructions": "shared rules",
        "maxNotesPerUpdate": 3,
        "advisors": [
            {"name": "Security Reviewer", "model": "anthropic/claude", "tools": ["read_file"],
             "reviewMode": "agent-end", "reviewInterval": 2, "syncBacklog": "strict",
             "instructions": "Focus on auth.\nCite the rule.", "enabled": True,
             "maxNotesPerUpdate": 2},
            {"name": "Silent", "tools": []},
        ],
    }
    path = tmp_path / "WATCHDOG.yml"
    save_watchdog_config_file(str(path), doc)

    loaded = load_watchdog_config_file(str(path))

    assert loaded["instructions"] == "shared rules"
    assert loaded["maxNotesPerUpdate"] == 3
    first, second = loaded["advisors"]
    assert first["name"] == "Security Reviewer"
    assert first["tools"] == ["read_file"]
    assert first["instructions"] == "Focus on auth.\nCite the rule."
    assert first["reviewInterval"] == 2
    assert second["tools"] == []
    assert serialize_watchdog_config(loaded) == path.read_text(encoding="utf-8")


def test_empty_document_deletes_the_file(tmp_path):
    path = tmp_path / "WATCHDOG.yml"
    _write(path, "advisors:\n  - name: X\n")

    save_watchdog_config_file(str(path), {"advisors": []})

    assert not os.path.exists(path)


def test_slug_falls_back_when_nothing_survives():
    assert slugify_adviser_name("Security Reviewer!") == "security-reviewer"
    assert slugify_adviser_name("!!!") == "adviser"
