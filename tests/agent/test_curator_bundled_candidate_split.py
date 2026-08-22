"""End-to-end coverage for the bundled candidate/archival split.

``curator.prune_builtins`` makes bundled built-ins eligible for *archival*.
Archival is decided by ``apply_automatic_transitions()``, which calls
``skill_usage.archive_skill()`` directly. The LLM consolidation pass is a
different thing: it builds umbrellas and patches skill bodies, and the bundled
policy is archive-only (``agent/curator.py`` hard rule #1).

Listing built-ins as consolidation candidates therefore invites writes that
``_background_review_write_guard`` unconditionally refuses. Each refusal burns
a tool call, and enough of them trip the tool-loop guard and abort the whole
pass -- so agent-created skills that *were* consolidatable never get processed.

These tests pin the split on a real temporary ``HERMES_HOME``:
  * the LLM candidate list never contains a bundled skill, either flag value;
  * the deterministic archival walk still archives a stale bundled skill when
    ``prune_builtins`` is on, and leaves it alone when off.
"""

import json
from datetime import datetime, timedelta, timezone

import pytest


def _write_skill(root, name, category=None):
    d = root / "skills" / category / name if category else root / "skills" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: fixture skill {name}\n---\n\n# {name}\n",
        encoding="utf-8",
    )
    return d


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A throwaway HERMES_HOME with one bundled and one agent-created skill."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import skill_usage
    from agent import curator

    _write_skill(tmp_path, "bundled-fixture", category="research")
    _write_skill(tmp_path, "agent-fixture")

    # Mark bundled-fixture as shipped-with-Hermes.
    # Manifest format is "<name>:<hash>" per line in `.bundled_manifest`.
    (tmp_path / "skills" / ".bundled_manifest").write_text(
        "bundled-fixture:deadbeef\n", encoding="utf-8"
    )

    old = datetime.now(timezone.utc) - timedelta(days=200)
    (tmp_path / "skills" / ".usage.json").write_text(
        json.dumps(
            {
                "bundled-fixture": {
                    "created_by": None,
                    "state": "active",
                    "last_used_at": old.isoformat(),
                    "use_count": 1,
                },
                "agent-fixture": {
                    "created_by": "agent",
                    "state": "active",
                    "last_used_at": old.isoformat(),
                    "use_count": 1,
                },
            }
        ),
        encoding="utf-8",
    )

    for mod in (skill_usage, curator):
        for attr in ("_usage_cache", "_BUNDLED_CACHE", "_HUB_CACHE"):
            if hasattr(mod, attr):
                setattr(mod, attr, None)

    return tmp_path, skill_usage, curator


@pytest.mark.parametrize("prune_builtins", [True, False])
def test_llm_candidate_list_never_lists_bundled(home, monkeypatch, prune_builtins):
    """The consolidation candidate list is agent-created only."""
    _, skill_usage, curator = home
    monkeypatch.setattr(
        skill_usage, "_prune_builtins_enabled", lambda: prune_builtins
    )

    text = curator._render_candidate_list()

    assert "bundled-fixture" not in text, (
        "bundled skill leaked into the LLM consolidation candidate list "
        f"(prune_builtins={prune_builtins})"
    )


def test_bundled_still_archived_by_deterministic_pass(home, monkeypatch):
    """prune_builtins=True must still archive a long-idle built-in."""
    root, skill_usage, curator = home
    monkeypatch.setattr(skill_usage, "_prune_builtins_enabled", lambda: True)

    rows = skill_usage.agent_created_report()
    names = {r["name"] for r in rows}

    assert "bundled-fixture" in names, (
        "prune_builtins=True must keep built-ins eligible for the "
        "deterministic archival walk"
    )


def test_bundled_not_eligible_when_prune_builtins_off(home, monkeypatch):
    """With the flag off, built-ins are outside the curator entirely."""
    _, skill_usage, curator = home
    monkeypatch.setattr(skill_usage, "_prune_builtins_enabled", lambda: False)

    names = {r["name"] for r in skill_usage.agent_created_report()}

    assert "bundled-fixture" not in names
    assert "agent-fixture" in names
