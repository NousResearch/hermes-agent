"""Tests for RepoKnowledge / AGENTS.md detection."""

import pytest
from pathlib import Path


class TestDetectGuidanceFiles:
    def test_no_agents_md(self, tmp_path):
        from hermes_cli.code.repo_knowledge import detect_guidance_files
        result = detect_guidance_files(tmp_path)
        assert result["agents_md"] is None
        assert result["claude_md"] is None
        assert result["guidance_docs"] == []

    def test_detects_agents_md(self, tmp_path):
        from hermes_cli.code.repo_knowledge import detect_guidance_files
        (tmp_path / "AGENTS.md").write_text("# Agents\n")
        result = detect_guidance_files(tmp_path)
        assert result["agents_md"] is not None
        assert result["agents_md"]["name"] == "AGENTS.md"

    def test_detects_claude_md(self, tmp_path):
        from hermes_cli.code.repo_knowledge import detect_guidance_files
        (tmp_path / "CLAUDE.md").write_text("# Claude\n")
        result = detect_guidance_files(tmp_path)
        assert result["claude_md"] is not None

    def test_detects_guidance_docs(self, tmp_path):
        from hermes_cli.code.repo_knowledge import detect_guidance_files
        arch_dir = tmp_path / "docs" / "architecture"
        arch_dir.mkdir(parents=True)
        (arch_dir / "overview.md").write_text("# Overview\n")
        (arch_dir / "database.md").write_text("# DB\n")

        result = detect_guidance_files(tmp_path)
        assert len(result["guidance_docs"]) == 2
        names = {Path(d["path"]).name for d in result["guidance_docs"]}
        assert "overview.md" in names

    def test_detects_nested_agents_md(self, tmp_path):
        from hermes_cli.code.repo_knowledge import detect_guidance_files
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "AGENTS.md").write_text("# Nested AGENTS\n")

        result = detect_guidance_files(tmp_path)
        assert len(result["nested_agents_md"]) == 1

    def test_repo_root_in_result(self, tmp_path):
        from hermes_cli.code.repo_knowledge import detect_guidance_files
        result = detect_guidance_files(tmp_path)
        assert result["repo_root"] == str(tmp_path)


class TestReadAgentsMd:
    def test_read_existing_agents_md(self, tmp_path):
        from hermes_cli.code.repo_knowledge import read_agents_md
        (tmp_path / "AGENTS.md").write_text("# Project\n\nBuild with `make`.\n")
        content = read_agents_md(tmp_path)
        assert content is not None
        assert "Build with" in content

    def test_returns_none_when_missing(self, tmp_path):
        from hermes_cli.code.repo_knowledge import read_agents_md
        assert read_agents_md(tmp_path) is None


class TestBootstrapAgentsMd:
    def test_creates_agents_md_when_absent(self, tmp_path):
        from hermes_cli.code.repo_knowledge import bootstrap_agents_md
        result = bootstrap_agents_md(tmp_path, project_summary="A test project.")
        assert result["created"] is True
        assert (tmp_path / "AGENTS.md").exists()
        content = (tmp_path / "AGENTS.md").read_text()
        assert "A test project." in content

    def test_does_not_overwrite_existing(self, tmp_path):
        from hermes_cli.code.repo_knowledge import bootstrap_agents_md
        existing = "# My existing AGENTS.md\nDo not overwrite me.\n"
        (tmp_path / "AGENTS.md").write_text(existing)

        result = bootstrap_agents_md(tmp_path)
        assert result["created"] is False
        assert "not overwriting" in result["reason"]
        # Content unchanged
        assert (tmp_path / "AGENTS.md").read_text() == existing

    def test_created_file_has_required_sections(self, tmp_path):
        from hermes_cli.code.repo_knowledge import bootstrap_agents_md
        bootstrap_agents_md(tmp_path)
        content = (tmp_path / "AGENTS.md").read_text()
        assert "# AGENTS.md" in content
        assert "## Project summary" in content

    def test_nonexistent_repo_root_returns_created_false(self, tmp_path):
        from hermes_cli.code.repo_knowledge import bootstrap_agents_md
        result = bootstrap_agents_md(tmp_path / "does_not_exist")
        assert result["created"] is False

    def test_references_existing_guidance_docs(self, tmp_path):
        from hermes_cli.code.repo_knowledge import bootstrap_agents_md
        arch_dir = tmp_path / "docs" / "architecture"
        arch_dir.mkdir(parents=True)
        (arch_dir / "overview.md").write_text("# Overview")

        bootstrap_agents_md(tmp_path)
        content = (tmp_path / "AGENTS.md").read_text()
        assert "overview.md" in content


class TestRepoKnowledgeService:
    def test_detect(self, tmp_path):
        from hermes_cli.code.repo_knowledge import RepoKnowledgeService
        (tmp_path / "AGENTS.md").write_text("# Agents\n")
        svc = RepoKnowledgeService()
        result = svc.detect(tmp_path)
        assert result["agents_md"] is not None

    def test_bootstrap_creates_file(self, tmp_path):
        from hermes_cli.code.repo_knowledge import RepoKnowledgeService
        svc = RepoKnowledgeService()
        result = svc.bootstrap(tmp_path, project_summary="My project.")
        assert result["created"] is True

    def test_bootstrap_does_not_overwrite(self, tmp_path):
        from hermes_cli.code.repo_knowledge import RepoKnowledgeService
        (tmp_path / "AGENTS.md").write_text("keep me\n")
        svc = RepoKnowledgeService()
        result = svc.bootstrap(tmp_path)
        assert result["created"] is False
