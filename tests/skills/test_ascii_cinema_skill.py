"""Hermetic contract tests for the bundled ascii-cinema skill."""

from __future__ import annotations

import ast
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO_ROOT / "skills" / "creative" / "ascii-cinema"
SKILL_PATH = SKILL_DIR / "SKILL.md"


def _skill_text() -> str:
    return SKILL_PATH.read_text(encoding="utf-8")


def _frontmatter_value(key: str) -> str:
    match = re.search(rf"^{re.escape(key)}:\s*(.+)$", _skill_text(), re.MULTILINE)
    assert match, f"missing frontmatter key: {key}"
    return match.group(1).strip().strip('"')


def test_description_matches_hardline_listing_contract() -> None:
    description = _frontmatter_value("description")
    assert len(description) <= 60
    assert description.endswith(".")


def test_author_credits_human_and_github_handle_first() -> None:
    author = _frontmatter_value("author")
    assert author.startswith("Mustafa Sarac (@mrsarac)")
    assert author.endswith("Hermes Agent")


def test_skill_uses_required_modern_section_order() -> None:
    text = _skill_text()
    required = (
        "# ASCII Cinema Skill",
        "## When to Use",
        "## Prerequisites",
        "## How to Run",
        "## Quick Reference",
        "## Procedure",
        "## Pitfalls",
        "## Verification",
    )
    positions = [text.index(heading) for heading in required]
    assert positions == sorted(positions)


def test_contract_suite_stays_hermetic() -> None:
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    imported_roots = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    prohibited = {
        "aiohttp",
        "httpx",
        "requests",
        "socket",
        "subprocess",
        "urllib",
    }
    assert imported_roots.isdisjoint(prohibited)


def test_frontmatter_is_portable_and_categorized() -> None:
    text = _skill_text()
    assert text.startswith("---\n")
    assert "platforms: [linux, macos, windows]" in text
    assert "category: creative" in text
    assert "related_skills: [ascii-art, ascii-video, claude-design, excalidraw]" in text


def test_how_to_run_names_native_hermes_tools() -> None:
    text = _skill_text()
    required_tools = (
        "`read_file`",
        "`search_files`",
        "`write_file`",
        "`patch`",
        "`terminal`",
        "`browser_navigate`",
        "`browser_console`",
        "`browser_vision`",
    )
    for tool in required_tools:
        assert tool in text
    assert not re.search(r"`(?:grep|cat|head|tail|sed|awk|find|ls)`", text)


def test_scene_grammar_reference_is_real_and_linked() -> None:
    reference = SKILL_DIR / "references" / "scene-grammar.md"
    assert reference.is_file()
    assert "references/scene-grammar.md" in _skill_text()
    grammar = reference.read_text(encoding="utf-8")
    assert "Meaning layer" in grammar
    assert "Motion layer" in grammar
    assert "Background layer" in grammar


def test_scope_stays_distinct_from_neighboring_ascii_skills() -> None:
    text = _skill_text()
    assert "banner, logo, or single frame" in text
    assert "MP4, GIF, or an image sequence" in text
    assert "interactive HTML storytelling" in text
    assert "use `ascii-art`" in text
    assert "use `ascii-video`" in text


def test_readme_points_to_the_contract_and_reference() -> None:
    readme = (SKILL_DIR / "README.md").read_text(encoding="utf-8")
    assert "self-contained" in readme
    assert "references/scene-grammar.md" in readme
    assert "tests/skills/test_ascii_cinema_skill.py" in readme
    assert "reduced motion" in readme.lower()
    assert "narrow screens" in readme.lower()


def test_scene_grammar_covers_accessible_responsive_motion() -> None:
    skill = _skill_text()
    grammar = (SKILL_DIR / "references" / "scene-grammar.md").read_text(encoding="utf-8")
    assert 'aria-atomic="true"' in skill
    assert "Reduced motion" in grammar
    assert "Narrow screens" in grammar
    assert "Keyboard" in grammar
    assert "Timer ownership" in grammar
    assert "cancelAnimationFrame" in grammar


def test_artifact_contract_covers_runtime_and_responsive_risks() -> None:
    text = _skill_text().lower()
    required = (
        "inline css",
        "inline javascript",
        "no build step",
        "no external asset",
        "prefers-reduced-motion",
        "complete glyph",
        "at most one active timer",
    )
    for phrase in required:
        assert phrase in text


def test_verification_covers_static_and_real_browser_failures() -> None:
    text = _skill_text()
    required = (
        "node --check",
        "duplicate element IDs",
        "`browser_console`",
        "`browser_vision`",
        "desktop, tablet, mobile, and narrow-mobile",
        "console, page, or network errors",
    )
    for phrase in required:
        assert phrase in text


def test_skill_package_contains_no_obvious_sensitive_markers() -> None:
    package_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            SKILL_PATH,
            SKILL_DIR / "README.md",
            SKILL_DIR / "references" / "scene-grammar.md",
        )
    )
    forbidden_patterns = (
        r"(?:/Users/|/home/|[A-Za-z]:\\Users\\)",
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
        r"(?i)\b(?:api[_-]?key|access[_-]?token|password)\s*[:=]\s*[^\s<]{8,}",
        r"(?i)\bBearer\s+[A-Za-z0-9._-]{8,}",
        r"\b(?:sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,})\b",
    )
    for pattern in forbidden_patterns:
        assert not re.search(pattern, package_text)
