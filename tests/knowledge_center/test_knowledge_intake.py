"""Tests for Knowledge Intake routing."""

import json
from pathlib import Path
from unittest.mock import patch

from agent.knowledge_intake import (
    classify_knowledge,
    find_merge_candidates,
    list_intake_notes,
    sync_obsidian_maps,
    write_intake_note,
)
from tools.knowledge_intake import capture_knowledge


def _make_vault(tmp_path: Path) -> Path:
    vault = tmp_path / "vault"
    (vault / "projects").mkdir(parents=True)
    (vault / "roles").mkdir()
    (vault / "domains" / "frontend").mkdir(parents=True)
    (vault / "domains" / "backend").mkdir(parents=True)
    (vault / "domains" / "frontend" / "README.md").write_text(
        "---\ntitle: Frontend\n---\n# Frontend\n",
        encoding="utf-8",
    )
    (vault / "projects" / "proj-a.md").write_text(
        "---\ntitle: Project A\nproject_slug: proj-a\ndomain: [frontend, backend]\n---\n"
        "| Stack | `node/next` |\n",
        encoding="utf-8",
    )
    (vault / "roles" / "knowledge.md").write_text(
        "---\ntitle: knowledge\n---\n# knowledge\n\n## Default Skills\n\n- [[project-context-pack]]\n",
        encoding="utf-8",
    )
    (vault / "roles" / "README.md").write_text("# Roles\n", encoding="utf-8")
    (vault / "MOC.md").write_text(
        "---\ntitle: MOC\n---\n# MOC\n\n## Entry Points\n\n- [[projects/README|Projects]]\n",
        encoding="utf-8",
    )
    return vault


def _make_skills(tmp_path: Path) -> Path:
    root = tmp_path / "skills"
    skill_dir = root / "software-development" / "project-context-pack"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: project-context-pack\n"
        "description: Build project context packs.\n"
        "category: software-development\n"
        "---\n"
        "# Project Context Pack\n",
        encoding="utf-8",
    )
    return root


def test_classify_skill_candidate(tmp_path: Path) -> None:
    vault = _make_vault(tmp_path)
    skills = _make_skills(tmp_path)
    result = classify_knowledge(
        title="Reusable SKILL.md workflow",
        content=(
            "When to use: create a repeatable project context pack procedure "
            "with prerequisites, pitfalls, and verification steps."
        ),
        source_project="proj-a",
        vault_path=vault,
        skills_root=skills,
    )
    assert result.destination == "skill_candidate"
    assert result.confidence >= 0.5
    assert any(item["kind"] == "skill" for item in result.related_files)


def test_classify_agent_candidate(tmp_path: Path) -> None:
    vault = _make_vault(tmp_path)
    skills = _make_skills(tmp_path)
    result = classify_knowledge(
        title="New research agent profile",
        content="Create an agent role that owns routing, handoff, delegate behavior, and responsibilities.",
        vault_path=vault,
        skills_root=skills,
    )
    assert result.destination == "agent_candidate"
    assert any("agent_candidate" in reason for reason in result.rationale)


def test_classify_domain_knowledge(tmp_path: Path) -> None:
    vault = _make_vault(tmp_path)
    skills = _make_skills(tmp_path)
    result = classify_knowledge(
        title="React architecture pattern",
        content="Reusable frontend pattern for React component architecture and integration.",
        source_project="proj-a",
        vault_path=vault,
        skills_root=skills,
    )
    assert result.destination == "domain_knowledge"
    assert "frontend" in result.domains


def test_classify_workspace_knowledge(tmp_path: Path) -> None:
    vault = _make_vault(tmp_path)
    skills = _make_skills(tmp_path)
    (vault / "workspace").mkdir()
    (vault / "workspace" / "strategy.md").write_text(
        "---\ntitle: Strategy\n---\n# Strategy\nPortfolio roadmap and project cluster notes.\n",
        encoding="utf-8",
    )
    result = classify_knowledge(
        title="Workspace portfolio roadmap",
        content=(
            "ภาพรวมโปรเจกต์ทั้งหมด: define SaaS portfolio strategy, project clusters, "
            "dependencies, roadmap priorities, and decision principles."
        ),
        vault_path=vault,
        skills_root=skills,
    )
    assert result.destination == "workspace_knowledge"
    assert result.confidence >= 0.5
    assert any(item["kind"] == "workspace" for item in result.related_files)


def test_write_and_list_intake_note(tmp_path: Path) -> None:
    vault = _make_vault(tmp_path)
    result = classify_knowledge(
        title="Rollback playbook",
        content="Runbook with rollback gates, incident recovery, escalation, and verification.",
        vault_path=vault,
        skills_root=_make_skills(tmp_path),
    )
    note_path = write_intake_note(result, "content body", vault_path=vault)
    assert note_path.exists()
    text = note_path.read_text(encoding="utf-8")
    assert "intake_type:" in text
    notes = list_intake_notes(destination=result.destination, vault_path=vault)
    assert len(notes) == 1
    assert notes[0]["path"] == str(note_path)


def test_source_url_written_to_intake_note(tmp_path: Path) -> None:
    vault = _make_vault(tmp_path)
    result = classify_knowledge(
        title="Facebook SaaS case study",
        content="Portfolio strategy and SaaS pricing lessons.",
        source_url="https://www.facebook.com/example/posts/123",
        source_title="SaaS ARR Case Study",
        preferred_destination="workspace_knowledge",
        vault_path=vault,
        skills_root=_make_skills(tmp_path),
    )
    note_path = write_intake_note(
        result,
        "Portfolio strategy and SaaS pricing lessons.",
        source_url="https://www.facebook.com/example/posts/123",
        source_title="SaaS ARR Case Study",
        vault_path=vault,
    )
    text = note_path.read_text(encoding="utf-8")
    assert "source:" in text
    assert "url: https://www.facebook.com/example/posts/123" in text
    assert "content_hash:" in text


def test_merge_candidates_same_source_url(tmp_path: Path) -> None:
    vault = _make_vault(tmp_path)
    workspace = vault / "workspace"
    workspace.mkdir()
    existing = workspace / "strategy.md"
    existing.write_text(
        "---\ntitle: Strategy\nsource:\n  url: https://www.facebook.com/example/posts/123\n---\n"
        "# Strategy\nClaude Code SaaS ARR pricing moat portfolio strategy.\n",
        encoding="utf-8",
    )
    result = classify_knowledge(
        title="Claude Code SaaS ARR case study",
        content="SaaS ARR pricing moat portfolio strategy.",
        source_url="https://www.facebook.com/example/posts/123",
        preferred_destination="workspace_knowledge",
        vault_path=vault,
        skills_root=_make_skills(tmp_path),
    )
    candidates = find_merge_candidates(
        result,
        "SaaS ARR pricing moat portfolio strategy.",
        source_url="https://www.facebook.com/example/posts/123",
        vault_path=vault,
    )
    assert candidates
    assert candidates[0]["path"] == str(existing)
    assert candidates[0]["recommended_merge"] is True


def test_sync_obsidian_maps(tmp_path: Path) -> None:
    vault = _make_vault(tmp_path)
    skills = _make_skills(tmp_path)
    result = sync_obsidian_maps(vault_path=vault, skills_root=skills, write_runtime_db=False)
    assert result["skill_cards"]
    assert result["agent_cards"]
    assert (vault / "skills" / "README.md").exists()
    assert (vault / "agents" / "README.md").exists()
    assert (vault / "workspace" / "README.md").exists()
    assert (vault / "relations" / "index.md").exists()
    assert "Knowledge Intake" in (vault / "MOC.md").read_text(encoding="utf-8")


def test_capture_knowledge_tool_classify_with_patches(tmp_path: Path) -> None:
    vault = _make_vault(tmp_path)
    skills = _make_skills(tmp_path)
    with patch("agent.knowledge_intake.resolve_vault_path", return_value=vault):
        with patch("agent.knowledge_intake.resolve_skills_root", return_value=skills):
            result = json.loads(capture_knowledge(
                action="classify",
                title="Agent routing profile",
                content="Agent role profile for delegate routing and handoff responsibilities.",
                source_url="https://example.com/agent-routing",
            ))
    assert result["success"] is True
    assert result["classification"]["destination"] == "agent_candidate"
