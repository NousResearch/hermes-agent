"""A card must not be dispatched with a skill its assignee cannot load.

`Unknown skill(s): X` is raised INSIDE the spawned worker at agent init
(`cli.py:8976`, `hermes_cli/oneshot.py:80`) and exits 1 — after the card is
claimed, the workspace resolved and the PID forked. The dispatcher sees only
`exit_code 1`, so the card burns its whole retry budget on two identical
crashes with no diagnosis on the board. Six cards were lost that way:
t_7f4ea155, t_0ec6abcf, t_867e31c9, t_03e5426e, t_8c40251d, t_34e858c9.

The decisive detail, and why root's skills are not a fallback: a profile sees
ONLY its own `skills/` directory.
"""

from __future__ import annotations

import os

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / ".hermes"
    (h / "skills" / "github" / "github-code-review").mkdir(parents=True)
    (h / "skills" / "github" / "github-code-review" / "SKILL.md").write_text("x")
    (h / "skills" / "devops" / "sdlc-review").mkdir(parents=True)
    (h / "skills" / "devops" / "sdlc-review" / "SKILL.md").write_text("x")
    # rodge has sdlc-review; switch has NOTHING (the live 2026-09-05 shape).
    (h / "profiles" / "rodge" / "skills" / "devops" / "sdlc-review").mkdir(parents=True)
    (h / "profiles" / "rodge" / "skills" / "devops" / "sdlc-review" / "SKILL.md").write_text("x")
    # switch has a populated tree that simply lacks sdlc-review — this matters:
    # an EMPTY tree is "cannot tell" and fails open, a populated one is an answer.
    (h / "profiles" / "switch" / "skills" / "core" / "clarify").mkdir(parents=True)
    (h / "profiles" / "switch" / "skills" / "core" / "clarify" / "SKILL.md").write_text("x")
    monkeypatch.setattr(os.path, "expanduser",
                        lambda p: str(h) if p == "~/.hermes" else os.path.expanduser(p))
    return h


def test_present_skill_is_not_missing(home):
    assert kb.missing_skills_for("rodge", ["sdlc-review"]) == []


def test_absent_skill_is_reported(home):
    """The live case: the review lane injects sdlc-review; switch lacks it."""
    assert kb.missing_skills_for("switch", ["sdlc-review"]) == ["sdlc-review"]


def test_root_skills_are_NOT_a_fallback_for_a_profile(home):
    """github-code-review exists at root only. switch still cannot load it."""
    assert kb.missing_skills_for("switch", ["github-code-review"]) == ["github-code-review"]


def test_default_and_root_resolve_to_the_root_skills_dir(home):
    for name in ("default", "root", None, ""):
        assert kb.missing_skills_for(name, ["github-code-review"]) == []


def test_only_the_missing_names_are_returned(home):
    got = kb.missing_skills_for("rodge", ["sdlc-review", "nope-1", "nope-2"])
    assert got == ["nope-1", "nope-2"]


def test_symlinked_skill_counts_as_present(home):
    src = home / "skills" / "github" / "github-code-review"
    dst = home / "profiles" / "rodge" / "skills" / "github-code-review"
    dst.symlink_to(src, target_is_directory=True)
    assert kb.missing_skills_for("rodge", ["github-code-review"]) == []


# --- fail-open: a dispatch gate must never invent a blocker -----------------

def test_no_skills_requested_is_allowed(home):
    assert kb.missing_skills_for("switch", []) == []
    assert kb.missing_skills_for("switch", None) == []


def test_profile_without_a_skills_dir_fails_open(home):
    assert kb.missing_skills_for("nonexistent-profile", ["anything"]) == []


def test_empty_skills_tree_fails_open(home):
    """An EMPTY tree is 'cannot tell', not 'has nothing'.

    A half-provisioned profile must not have every card blocked; a populated
    tree that lacks the name is the only case we are willing to call.
    """
    empty = home / "profiles" / "karl" / "skills"
    empty.mkdir(parents=True)
    assert kb.missing_skills_for("karl", ["sdlc-review"]) == []


def test_walk_failure_fails_open(home, monkeypatch):
    def boom(*a, **k):
        raise OSError("unreadable")
    monkeypatch.setattr(os, "walk", boom)
    assert kb.missing_skills_for("switch", ["sdlc-review"]) == []
