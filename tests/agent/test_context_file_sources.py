#!/usr/bin/env python3
"""Per-file context source manifest (`list_context_file_sources`).

Read-only mirror of ``build_context_files_prompt`` discovery: one entry per
candidate context/instruction file with size, token estimate, and whether it
was loaded, truncated, or shadowed by a higher-priority context type.

Inspired by: GitHub Copilot CLI 1.0.81 — "Show each user instruction file
separately in /instructions".
"""

import pytest

from agent.prompt_builder import list_context_file_sources


@pytest.fixture()
def project(tmp_path):
    (tmp_path / ".git").mkdir()
    return tmp_path


def _by_label(sources):
    return {s["label"]: s for s in sources}


class TestDiscovery:
    def test_agents_md_loaded(self, project):
        (project / "AGENTS.md").write_text("# rules\n" * 10)
        sources = list_context_file_sources(cwd=str(project))
        entry = _by_label(sources)["AGENTS.md"]
        assert entry["loaded"] is True
        assert entry["status"] == "loaded"
        assert entry["chars"] > 0
        assert entry["est_tokens"] == (entry["chars"] + 3) // 4

    def test_empty_project_returns_empty(self, tmp_path, tmp_path_factory):
        (tmp_path / ".git").mkdir()
        empty_home = tmp_path_factory.mktemp("empty_home")
        assert (
            list_context_file_sources(cwd=str(tmp_path), home_override=empty_home)
            == []
        )

    def test_priority_shadowing_hermes_md_over_agents_md(self, project):
        (project / ".hermes.md").write_text("hermes rules")
        (project / "AGENTS.md").write_text("agents rules")
        entries = _by_label(list_context_file_sources(cwd=str(project)))
        assert entries[".hermes.md"]["status"] == "loaded"
        assert entries["AGENTS.md"]["status"] == "shadowed"
        assert entries["AGENTS.md"]["loaded"] is False

    def test_claude_md_shadowed_by_agents_md(self, project):
        (project / "AGENTS.md").write_text("agents rules")
        (project / "CLAUDE.md").write_text("claude rules")
        entries = _by_label(list_context_file_sources(cwd=str(project)))
        assert entries["AGENTS.md"]["status"] == "loaded"
        assert entries["CLAUDE.md"]["status"] == "shadowed"

    def test_cursorrules_listed(self, project):
        (project / ".cursorrules").write_text("cursor rules")
        rules_dir = project / ".cursor" / "rules"
        rules_dir.mkdir(parents=True)
        (rules_dir / "a.mdc").write_text("rule a")
        entries = _by_label(list_context_file_sources(cwd=str(project)))
        assert entries[".cursorrules"]["status"] == "loaded"
        assert entries[".cursor/rules/a.mdc"]["status"] == "loaded"

    def test_agents_override_wins_per_directory(self, project):
        (project / "AGENTS.md").write_text("committed")
        (project / "AGENTS.override.md").write_text("personal override")
        labels = [s["label"] for s in list_context_file_sources(cwd=str(project))]
        assert "AGENTS.override.md" in labels
        assert "AGENTS.md" not in labels  # first name wins per directory

    def test_directory_chain_lists_both_agents_files(self, project):
        (project / "AGENTS.md").write_text("root rules")
        sub = project / "pkg"
        sub.mkdir()
        (sub / "AGENTS.md").write_text("pkg rules")
        sources = list_context_file_sources(cwd=str(sub))
        agents_entries = [s for s in sources if "AGENTS.md" in s["label"]]
        assert len(agents_entries) == 2
        assert all(s["loaded"] for s in agents_entries)

    def test_truncated_status_when_over_cap(self, project, monkeypatch):
        import agent.prompt_builder as pb

        monkeypatch.setattr(pb, "_get_context_file_max_chars", lambda *_a: 10)
        (project / "AGENTS.md").write_text("x" * 100)
        entries = _by_label(list_context_file_sources(cwd=str(project)))
        assert entries["AGENTS.md"]["status"] == "truncated"
        assert entries["AGENTS.md"]["loaded"] is True

    def test_soul_md_from_home_override(self, project, tmp_path_factory):
        home = tmp_path_factory.mktemp("hermes_home")
        (home / "SOUL.md").write_text("identity")
        entries = _by_label(
            list_context_file_sources(cwd=str(project), home_override=home)
        )
        assert entries["SOUL.md"]["status"] == "loaded"

    def test_read_only_no_side_effects(self, project):
        (project / "AGENTS.md").write_text("rules")
        before = (project / "AGENTS.md").read_text()
        list_context_file_sources(cwd=str(project))
        assert (project / "AGENTS.md").read_text() == before

    def test_accepts_path_object(self, project):
        (project / "AGENTS.md").write_text("rules")
        sources = list_context_file_sources(cwd=project)
        assert _by_label(sources)["AGENTS.md"]["loaded"] is True
