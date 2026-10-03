"""Regression for issue #106672.

The bundled `requesting-code-review` skill must stay read-only unless the
user explicitly authorizes mutation: a review/verify/inspect/report request
must not reach file edits, blanket staging, or a commit.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL = REPO_ROOT / "skills" / "software-development" / "requesting-code-review" / "SKILL.md"


def _text() -> str:
    return SKILL.read_text(encoding="utf-8")


def test_no_blanket_stage():
    """Never stage the whole working tree; stage only reviewed paths."""
    assert "git add -A" not in _text(), "bare `git add -A` can capture unrelated changes"


def test_commit_requires_explicit_authorization():
    """The commit step must be gated on the user explicitly requesting it."""
    text = _text().lower()
    assert "explicit" in text and "commit" in text
    # The gate must be unconditional, not advisory.
    assert "only" in text or "must" in text or "do not" in text


def test_review_only_default():
    """Review/verify/inspect requests are read-only by default."""
    text = _text().lower()
    assert "read-only" in text or "read only" in text
