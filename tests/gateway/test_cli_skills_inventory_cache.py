"""The owner re-reads the skill tree for a classic-CLI session only when it changed or the entry aged out."""

import pytest

from gateway import session_local_inventory as inventory
from tools import skills_tool


@pytest.fixture
def scans(monkeypatch):
    """Count catalog scans; the scan signature is whatever the test sets in ``state['signature']``."""
    state = {"signature": ("tree-1",), "calls": 0, "skills": [{"name": "plan", "category": "dev"},
                                                             {"name": "ship", "category": "dev"},
                                                             {"name": "poem", "category": None}]}

    def find_all_skills(**_kw):
        state["calls"] += 1
        return [dict(s) for s in state["skills"]]

    monkeypatch.setattr(inventory, "_cli_skills_cache", None)
    monkeypatch.setattr(skills_tool, "skills_scan_signature", lambda **_kw: state["signature"])
    monkeypatch.setattr(skills_tool, "find_all_skills", find_all_skills)
    return state


def test_an_unchanged_tree_is_scanned_once_across_sessions(scans):
    first = inventory._cli_skills()
    second = inventory._cli_skills()
    assert scans["calls"] == 1
    assert first == second == {"dev": ["plan", "ship"], "general": ["poem"]}


def test_a_caller_cannot_poison_the_next_sessions_list(scans):
    inventory._cli_skills()["dev"].append("injected")
    assert inventory._cli_skills() == {"dev": ["plan", "ship"], "general": ["poem"]}


def test_a_changed_tree_is_scanned_again(scans):
    inventory._cli_skills()
    scans["signature"] = ("tree-2",)
    scans["skills"].append({"name": "audit", "category": "dev"})
    assert inventory._cli_skills()["dev"] == ["plan", "ship", "audit"]
    assert scans["calls"] == 2


def test_an_old_entry_is_scanned_again_even_if_the_signature_is_unchanged(scans, monkeypatch):
    inventory._cli_skills()
    monkeypatch.setattr(inventory, "_CLI_SKILLS_TTL_SECONDS", 0.0)  # an in-place SKILL.md edit is invisible to the signature
    inventory._cli_skills()
    assert scans["calls"] == 2
