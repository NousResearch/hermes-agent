"""E2E test: Full knowledge flow — work → create → detect relevance → promote → consume."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.knowledge_domains import DomainRelevanceMatcher
from agent.knowledge_relevance import KnowledgeRelevanceEngine
from agent.knowledge_preferences import KnowledgePreferenceManager
from tools.knowledge_promote import promote_knowledge
from tools.knowledge_review import review_knowledge, _get_vault_path


@pytest.fixture
def full_e2e_env(tmp_path: Path) -> dict:
    """Create a complete test environment."""
    vault = tmp_path / "vault"
    vault.mkdir()
    domains_dir = vault / "domains"
    domains_dir.mkdir()
    (domains_dir / "frontend").mkdir()
    (domains_dir / "backend").mkdir()
    projects_dir = vault / "projects"
    projects_dir.mkdir()
    (vault / "review-queue").mkdir()
    (vault / "sources").mkdir()
    (vault / "knowledge").mkdir()
    (vault / "lessons").mkdir()
    (vault / "patterns").mkdir()
    (vault / "playbooks").mkdir()
    (vault / "skills").mkdir()

    # Create project notes
    for slug, stack, domains in [
        ("proj-a", "node/next", ["frontend", "backend"]),
        ("proj-b", "node/vite", ["frontend", "backend"]),
    ]:
        note = projects_dir / f"{slug}.md"
        note.write_text(
            f"---\ntitle: {slug}\nproject_slug: {slug}\n---\n"
            f"| Stack | `{stack}` |\n"
            f"\ndomain: [{', '.join(domains)}]\n"
        )

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()

    return {"vault": vault, "hermes_home": hermes_home}


def test_e2e_full_knowledge_flow(full_e2e_env: dict) -> None:
    """Test the complete knowledge promotion flow."""
    vault = full_e2e_env["vault"]
    hermes_home = full_e2e_env["hermes_home"]

    # Step 1: Agent works on Project A, creates knowledge note (simulated)
    knowledge_content = "React component pattern with hooks for state management and useEffect cleanup"

    # Step 2: Relevance engine detects cross-project match with Project B
    engine = KnowledgeRelevanceEngine(vault_path=vault)
    assert engine.is_cross_project_relevant(knowledge_content, "proj-a") is True
    matches = engine.find_matching_projects(knowledge_content, "proj-a")
    assert "proj-b" in matches

    # Step 3: No existing preference → add to review queue
    with patch("tools.knowledge_review._get_vault_path", return_value=vault):
        with patch("tools.knowledge_promote._resolve_vault_path", return_value=vault):
            result_str = review_knowledge(
                action="add",
                title="React Hooks Pattern",
                content=knowledge_content,
                source_project="proj-a",
                target_domain="frontend",
                summary="Useful pattern for state management",
            )
            add_result = json.loads(result_str)
            assert add_result["success"] is True
            knowledge_id = add_result["knowledge_id"]

    # Step 4: User approves → promoted to domain KB
    with patch("tools.knowledge_review._get_vault_path", return_value=vault):
        with patch("tools.knowledge_promote._resolve_vault_path", return_value=vault):
            result_str = review_knowledge(action="approve", knowledge_id=knowledge_id)
            approve_result = json.loads(result_str)
            assert approve_result["success"] is True
            assert "promote_result" in approve_result
            assert approve_result["promote_result"]["success"] is True

    # Verify the note was actually created in the domain directory
    note_path = Path(approve_result["promote_result"]["note_path"])
    assert note_path.exists()
    content = note_path.read_text(encoding="utf-8")
    assert "React Hooks Pattern" in content
    assert "origin_project: proj-a" in content

    # Step 5: Agent works on Project B, loads domain KB
    matcher = DomainRelevanceMatcher(vault_path=vault)
    domains = matcher.classify("proj-b")
    assert "frontend" in domains
    notes = matcher.get_domain_notes(domains)
    assert len(notes) >= 1  # The promoted note

    # Step 6: Verify knowledge was found
    found = False
    for note in notes:
        if "react" in note.name.lower() or "hooks" in note.name.lower():
            found = True
            break
    assert found, "Promoted knowledge not found in Project B's domain load"
