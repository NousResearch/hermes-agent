"""Tests for RepoKnowledgeService."""

from hermes_cli.code.repo_knowledge import RepoKnowledgeService, detect_repo_guidance


def test_detect_agents_md_and_guidance_docs(tmp_path):
    (tmp_path / "AGENTS.md").write_text("# Agents\n", encoding="utf-8")
    arch_dir = tmp_path / "docs" / "architecture"
    arch_dir.mkdir(parents=True)
    (arch_dir / "overview.md").write_text("# Overview\n", encoding="utf-8")

    manifest = detect_repo_guidance(tmp_path)
    assert manifest["agents_md"] is not None
    assert len(manifest["guidance_docs"]) == 1


def test_bootstrap_does_not_overwrite_existing_agents_md(tmp_path):
    existing = "# Existing\nDo not overwrite\n"
    (tmp_path / "AGENTS.md").write_text(existing, encoding="utf-8")
    service = RepoKnowledgeService()
    result = service.bootstrap(tmp_path, project_summary="ignored")
    assert result["created"] is False
    assert "already exists" in result["reason"]
    assert (tmp_path / "AGENTS.md").read_text(encoding="utf-8") == existing


def test_bootstrap_creates_agents_md_when_missing(tmp_path):
    service = RepoKnowledgeService()
    result = service.bootstrap(tmp_path, project_summary="Project summary")
    assert result["created"] is True
    content = (tmp_path / "AGENTS.md").read_text(encoding="utf-8")
    assert "Project summary" in content


def test_service_read_agents_md(tmp_path):
    (tmp_path / "AGENTS.md").write_text("# A\n\nB\n", encoding="utf-8")
    service = RepoKnowledgeService()
    content = service.read_agents_md(tmp_path)
    assert content is not None
    assert "B" in content
