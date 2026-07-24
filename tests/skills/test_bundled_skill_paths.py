"""Regression tests for developer-specific paths in shipped skill docs."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEVELOPER_ROOT = "/home/bb/hermes-agent"
SEARCH_ROOTS = (
    REPO_ROOT / "skills",
    REPO_ROOT / "website" / "docs" / "user-guide" / "skills",
    REPO_ROOT
    / "website"
    / "i18n"
    / "zh-Hans"
    / "docusaurus-plugin-content-docs"
    / "current"
    / "user-guide"
    / "skills",
)


def test_shipped_skill_docs_do_not_expose_developer_checkout() -> None:
    offenders = [
        path.relative_to(REPO_ROOT)
        for root in SEARCH_ROOTS
        for path in root.rglob("*.md")
        if DEVELOPER_ROOT in path.read_text(encoding="utf-8")
    ]

    assert not offenders, f"developer checkout leaked into: {offenders}"
