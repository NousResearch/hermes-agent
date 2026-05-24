"""E2E test: Domain-level deny — 'never for domain=frontend' skips all frontend knowledge."""

from pathlib import Path

import pytest

from agent.knowledge_preferences import KnowledgePreferenceManager
from agent.knowledge_relevance import KnowledgeRelevanceEngine


@pytest.fixture
def domain_deny_env(tmp_path: Path) -> dict:
    vault = tmp_path / "vault"
    vault.mkdir()
    domains = vault / "domains"
    domains.mkdir()
    (domains / "frontend").mkdir(parents=True, exist_ok=True)
    (domains / "backend").mkdir(parents=True, exist_ok=True)
    projects = vault / "projects"
    projects.mkdir()
    for slug, stack, domains in [
        ("proj-a", "node/next", ["frontend", "backend"]),
        ("proj-b", "node/vite", ["frontend", "backend"]),
    ]:
        (projects / f"{slug}.md").write_text(
            f"---\ntitle: {slug}\nproject_slug: {slug}\n---\n| Stack | `{stack}` |\n\ndomain: [{', '.join(domains)}]\n"
        )
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    return {"vault": vault, "hermes_home": hermes_home}


def test_e2e_domain_deny(domain_deny_env: dict) -> None:
    """Test that domain-level deny skips all knowledge from that domain."""
    hermes_home = domain_deny_env["hermes_home"]
    vault = domain_deny_env["vault"]

    # Step 1: User sets 'never for domain=frontend'
    mgr = KnowledgePreferenceManager(hermes_home=hermes_home)
    mgr.save_preference(
        domain="frontend",
        project="*",  # Any project
        pattern="react",
        allow=False,
        reason="not interested in frontend patterns",
    )

    # Step 2: Any frontend knowledge from any project skips automatically
    content_a = "React component with hooks and state management"
    content_b = "React router configuration for SPA"

    pref_a = mgr.check_preference("frontend", "proj-a", content_a)
    assert pref_a is not None
    assert pref_a["allow"] is False

    pref_b = mgr.check_preference("frontend", "proj-b", content_b)
    assert pref_b is not None
    assert pref_b["allow"] is False

    # Verify both are denied (same domain, same pattern)
    assert pref_a["domain"] == "frontend"
    assert pref_b["domain"] == "frontend"
