"""Installed state for hub skills: the ``installed`` map that ``/api/skills/hub/sources``,
``/search`` and ``/official`` return. Desktop catalog cards look themselves up by the id
they install; the dashboard looks a hub hit up by its id and, for ClawHub, its card id.

ClawHub records ``@owner/slug`` in the lock, a catalog card installs ``clawhub/<slug>``,
and the hermes-index / hub search list the bare slug."""

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


def test_clawhub_install_answers_to_the_card_id():
    _lock(("clawhub", "@tmstudio667-commits/agent-git-oracle", "agent-git-oracle"),
          ("clawhub", "legacy-slug", "legacy-slug"))

    installed = _installed_hub_identifiers()

    record = installed["@tmstudio667-commits/agent-git-oracle"]
    assert record["name"] == "agent-git-oracle"
    assert installed["clawhub/agent-git-oracle"] == record
    # A lock row from before owners were recorded keeps its bare id and gains the card id.
    assert installed["clawhub/legacy-slug"] == installed["legacy-slug"]


def test_a_clawhub_install_never_answers_for_another_sources_bare_id():
    # Only the source-qualified card id is added: a bare `foo` from any other source
    # must not read installed because @alice/foo is.
    _lock(("clawhub", "@alice/foo", "foo-fork"))

    assert set(_installed_hub_identifiers()) == {"@alice/foo", "clawhub/foo"}


def test_other_sources_keep_their_exact_ids_only():
    _lock(("skills.sh", "skills-sh/browserbase/skills/browser", "browser"),
          ("official", "official/research/arxiv", "arxiv"),
          ("github", "anthropics/skills/skills/pdf", "pdf"))

    assert set(_installed_hub_identifiers()) == {
        "skills-sh/browserbase/skills/browser", "official/research/arxiv",
        "anthropics/skills/skills/pdf"}


def test_an_exact_lock_id_beats_a_clawhub_alias():
    _lock(("clawhub", "@alice/foo", "foo-fork"), ("clawhub", "clawhub/foo", "foo"))

    assert _installed_hub_identifiers()["clawhub/foo"]["name"] == "foo"


def test_a_malformed_clawhub_row_does_not_hide_the_other_installs():
    # A hand-edited lock can hold a non-string identifier. It used to reach the
    # ClawHub parser, raise, and leave the whole map empty.
    HubLockFile().save({"version": 1, "installed": {
        "broken": {"source": "clawhub", "identifier": 123, "install_path": "broken"},
        "arxiv": {"source": "official", "identifier": "official/research/arxiv",
                  "install_path": "arxiv"}}})

    assert _installed_hub_identifiers()["official/research/arxiv"]["name"] == "arxiv"


def test_hub_search_returns_the_card_id_a_clawhub_hit_is_found_under(monkeypatch):
    # Index-backed results carry ClawHub's bare slug. The map answers to clawhub/<slug>,
    # which the dashboard derives from the hit's source (web/src/lib/skill-hub-installed.ts).
    hit = SkillMeta(name="Agent Git Oracle", description="", source="clawhub",
                    identifier="agent-git-oracle", trust_level="community")
    monkeypatch.setattr("tools.skills_hub_search.create_source_router", list)
    monkeypatch.setattr("tools.skills_hub_search.parallel_search_sources",
                        lambda sources, **kw: ([hit], {"clawhub": 1}, []))
    _lock(("clawhub", "@tmstudio667-commits/agent-git-oracle", "agent-git-oracle"))

    body = asyncio.run(search_skills_hub(q="agent git oracle"))

    assert [(r["source"], r["identifier"]) for r in body["results"]] == [("clawhub", "agent-git-oracle")]
    assert body["installed"]["clawhub/agent-git-oracle"]["name"] == "agent-git-oracle"
    assert "agent-git-oracle" not in body["installed"]
