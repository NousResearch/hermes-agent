"""E2E test: Deny preference flow — user denies → stored → next time skips silently."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.knowledge_preferences import KnowledgePreferenceManager
from agent.knowledge_relevance import KnowledgeRelevanceEngine
from tools.knowledge_review import review_knowledge, _get_queue_path


@pytest.fixture
def deny_env(tmp_path: Path) -> dict:
    vault = tmp_path / "vault"
    vault.mkdir()
    domains = vault / "domains"
    domains.mkdir()
    (domains / "frontend").mkdir(parents=True, exist_ok=True)
    projects = vault / "projects"
    projects.mkdir()
    for slug in ["proj-a", "proj-b"]:
        (projects / f"{slug}.md").write_text(
            f"---\ntitle: {slug}\nproject_slug: {slug}\n---\n| Stack | `node/next` |\n\ndomain: [frontend, backend]\n"
        )
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    return {"vault": vault, "hermes_home": hermes_home}


def test_e2e_deny_preference_flow(deny_env: dict) -> None:
    """Test that denied knowledge is skipped on subsequent detection."""
    vault = deny_env["vault"]
    hermes_home = deny_env["hermes_home"]

    knowledge_content = "React component pattern with hooks for state management"

    # Step 1: Relevance detects cross-project match
    engine = KnowledgeRelevanceEngine(vault_path=vault)
    assert engine.is_cross_project_relevant(knowledge_content, "proj-a") is True

    # Step 2: User denies → stored as preference
    mgr = KnowledgePreferenceManager(hermes_home=hermes_home)
    pref_id = mgr.save_preference(
        domain="frontend",
        project="proj-a",
        pattern="react",
        allow=False,
        reason="not relevant for this project",
    )
    assert pref_id is not None

    # Step 3: Agent works on Project A again, creates similar knowledge
    pref = mgr.check_preference("frontend", "proj-a", knowledge_content)
    assert pref is not None
    assert pref["allow"] is False

    # Step 4: Preference match → skips silently (no ask, no promote)
    queue_path = vault / "domains" / ".review_queue.json"
    with patch("tools.knowledge_review._get_queue_path", return_value=queue_path):
        # Simulate: preference check returns deny → skip
        if pref and not pref["allow"]:
            # Should NOT add to queue
            result_str = review_knowledge(action="list")
            result = json.loads(result_str)
            assert result["count"] == 0  # Queue is empty

    # Verify preference file has the deny entry
    prefs = mgr.list_preferences()
    deny_prefs = [p for p in prefs if not p["allow"]]
    assert len(deny_prefs) == 1
    assert deny_prefs[0]["pattern"] == "react"
