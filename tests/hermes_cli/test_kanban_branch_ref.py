"""Branch refs handed to ``git worktree add`` must always be refs git accepts.

A card title can produce a branch name git refuses (``..`` anywhere, a trailing
``.``, a ``.lock`` suffix). ``branch_name_for`` no longer emits those, but a card
created earlier — or one carrying a hand-written ``branch_name`` — would still
fail on every retry, so the worktree layer sanitises what it is given.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from hermes_cli.kanban_db_workspace import _sanitize_branch_ref


def _git_accepts(ref: str) -> bool:
    if shutil.which("git") is None:  # pragma: no cover - git is a hard dependency elsewhere
        pytest.skip("git not installed")
    return (
        subprocess.run(
            ["git", "check-ref-format", "--branch", ref], capture_output=True, check=False
        ).returncode
        == 0
    )


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("jomform/t_x-word_budget.", "jomform/t_x-word_budget"),
        ("jomform/t_x-go-test-.-...-runs", "jomform/t_x-go-test-runs"),
        ("jomform/t_x-fix.lock", "jomform/t_x-fix-lock"),
        ("jomform/t_x-has spaces", "jomform/t_x-has-spaces"),
        # A name git already accepts must survive untouched.
        ("jomform/t_abc123-fix-the-thing", "jomform/t_abc123-fix-the-thing"),
    ],
)
def test_sanitize_branch_ref_yields_a_ref_git_accepts(raw, expected):
    assert _sanitize_branch_ref(raw) == expected
    assert _git_accepts(expected)


@pytest.mark.parametrize("raw", ["", "  ", "..", "./.", "..."])
def test_sanitize_branch_ref_never_returns_an_empty_or_dot_ref(raw):
    ref = _sanitize_branch_ref(raw)
    assert ref
    assert _git_accepts(ref)
