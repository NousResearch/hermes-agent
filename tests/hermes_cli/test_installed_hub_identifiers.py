"""Installed state for hub skills: the ``installed`` map that ``/api/skills/hub/sources``,
``/search`` and ``/official`` return, which Desktop catalog cards and the dashboard's hub
search check with ``installed[identifier]``.

ClawHub records ``@owner/slug`` in the lock, while a catalog card installs
``clawhub/<slug>`` and the hermes-index / hub search list the bare slug."""

import asyncio

from hermes_cli.web_routers.skills import search_skills_hub
from hermes_cli.web_server_profiles import _installed_hub_identifiers
from tools.skills_hub import HubLockFile
from tools.skills_hub_models import SkillMeta


def _lock(*rows):
    """Write ``(source, identifier, name)`` rows to the active profile's hub lock."""
    HubLockFile().save({"version": 1, "installed": {
        name: {"source": source, "identifier": identifier, "trust_level": "community",
               "scan_verdict": "safe", "install_path": name, "files": ["SKILL.md"]}
        for source, identifier, name in rows}})


def test_clawhub_install_answers_to_the_card_and_search_ids():
    _lock(("clawhub", "@tmstudio667-commits/agent-git-oracle", "agent-git-oracle"),
          ("clawhub", "legacy-slug", "legacy-slug"))

    installed = _installed_hub_identifiers()

    record = installed["@tmstudio667-commits/agent-git-oracle"]
    assert record["name"] == "agent-git-oracle"
    assert installed["clawhub/agent-git-oracle"] == record
    assert installed["agent-git-oracle"] == record
    # A lock row from before owners were recorded keeps its bare id and gains the card id.
    assert installed["clawhub/legacy-slug"] == installed["legacy-slug"]


def test_other_sources_keep_their_exact_ids_only():
    _lock(("skills.sh", "skills-sh/browserbase/skills/browser", "browser"),
          ("official", "official/research/arxiv", "arxiv"),
          ("github", "anthropics/skills/skills/pdf", "pdf"))

    assert set(_installed_hub_identifiers()) == {
        "skills-sh/browserbase/skills/browser", "official/research/arxiv",
        "anthropics/skills/skills/pdf"}


def test_an_exact_lock_id_beats_a_clawhub_alias():
    _lock(("hermes-index", "foo", "foo"), ("clawhub", "@alice/foo", "foo-fork"))

    installed = _installed_hub_identifiers()

    assert installed["foo"]["name"] == "foo"
    assert installed["clawhub/foo"]["name"] == "foo-fork"


def test_hub_search_marks_an_installed_clawhub_result(monkeypatch):
    # Index-backed results carry ClawHub's bare slug; the dashboard checks installed[identifier].
    hit = SkillMeta(name="Agent Git Oracle", description="", source="clawhub",
                    identifier="agent-git-oracle", trust_level="community")
    monkeypatch.setattr("tools.skills_hub_search.create_source_router", list)
    monkeypatch.setattr("tools.skills_hub_search.parallel_search_sources",
                        lambda sources, **kw: ([hit], {"clawhub": 1}, []))
    _lock(("clawhub", "@tmstudio667-commits/agent-git-oracle", "agent-git-oracle"))

    body = asyncio.run(search_skills_hub(q="agent git oracle"))

    assert [r["identifier"] for r in body["results"]] == ["agent-git-oracle"]
    assert body["installed"]["agent-git-oracle"]["name"] == "agent-git-oracle"
