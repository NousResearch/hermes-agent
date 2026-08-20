"""
Smoke tests for the longbridge optional skill.

This skill is a pure prose wrapper around the third-party `longbridge` CLI/SDKs —
there's no bundled script to unit-test and no way to hit the live brokerage API in
CI (needs a real, authenticated Longbridge account). Instead these tests verify:
  - SKILL.md frontmatter conforms to the hardline format
  - the modern section order is present
  - referenced doc files exist and internal relative links resolve
  - state-changing SDK calls (submit/replace/cancel order) carry an explicit
    confirm-before-executing warning
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

SKILL_DIR = Path(__file__).resolve().parents[2] / "optional-skills" / "finance" / "longbridge"


@pytest.fixture(scope="module")
def frontmatter() -> dict:
    src = (SKILL_DIR / "SKILL.md").read_text()
    m = re.search(r"^---\n(.*?)\n---", src, re.DOTALL)
    assert m, "SKILL.md missing YAML frontmatter"
    return yaml.safe_load(m.group(1))


@pytest.fixture(scope="module")
def skill_body() -> str:
    return (SKILL_DIR / "SKILL.md").read_text()


def test_skill_dir_exists() -> None:
    assert SKILL_DIR.is_dir(), f"missing skill dir: {SKILL_DIR}"


def test_skill_md_present() -> None:
    assert (SKILL_DIR / "SKILL.md").is_file()


def test_description_under_60_chars(frontmatter) -> None:
    desc = frontmatter["description"]
    assert len(desc) <= 60, f"description is {len(desc)} chars (hardline <=60): {desc!r}"
    assert desc.endswith("."), f"description must end with a period: {desc!r}"


def test_name_matches_dir(frontmatter) -> None:
    assert frontmatter["name"] == "longbridge"


def test_author_credits_human_contributor(frontmatter) -> None:
    author = frontmatter["author"]
    assert "Hogan" in author, f"author should credit the human contributor first: {author!r}"


def test_license_mit(frontmatter) -> None:
    assert frontmatter["license"] == "MIT"


def test_requires_setup_flagged(frontmatter) -> None:
    # Longbridge needs OAuth login before any live-data call works — the loader
    # surfaces this to the user via metadata.hermes.requires_setup.
    assert frontmatter["metadata"]["hermes"]["requires_setup"] is True


@pytest.mark.parametrize(
    "heading",
    [
        "## When to Use",
        "## Prerequisites",
        "## How to Run",
        "## Quick Reference",
        "## Procedure",
        "## Pitfalls",
        "## Verification",
    ],
)
def test_modern_section_order_headings_present(skill_body, heading: str) -> None:
    assert heading in skill_body, f"missing hardline section: {heading}"


def test_modern_sections_appear_in_order(skill_body) -> None:
    required = [
        "## When to Use",
        "## Prerequisites",
        "## How to Run",
        "## Quick Reference",
        "## Procedure",
        "## Pitfalls",
        "## Verification",
    ]
    positions = [skill_body.index(h) for h in required]
    assert positions == sorted(positions), "hardline sections are out of order"


REFERENCE_FILES = [
    "references/setup.md",
    "references/mcp.md",
    "references/llm.md",
    "references/cli/overview.md",
    "references/python-sdk/overview.md",
    "references/python-sdk/quote-context.md",
    "references/python-sdk/content-context.md",
    "references/python-sdk/trade-context.md",
    "references/python-sdk/types.md",
    "references/rust-sdk/overview.md",
    "references/rust-sdk/quote-context.md",
    "references/rust-sdk/trade-context.md",
    "references/rust-sdk/content.md",
    "references/rust-sdk/types.md",
]


@pytest.mark.parametrize("rel_path", REFERENCE_FILES)
def test_referenced_files_exist(rel_path: str) -> None:
    assert (SKILL_DIR / rel_path).is_file(), f"SKILL.md links to missing file: {rel_path}"


def _extract_markdown_links(text: str) -> list[str]:
    return re.findall(r"\[[^\]]+\]\(([^)]+)\)", text)


@pytest.mark.parametrize(
    "doc_path",
    [
        "SKILL.md",
        "references/setup.md",
        "references/mcp.md",
        "references/rust-sdk/content.md",
    ],
)
def test_relative_links_resolve(doc_path: str) -> None:
    """Every relative (non-http) link in these docs must resolve to a real file,
    relative to the *linking file's own directory* — not the skill root. This is
    exactly the bug class flagged in review (references/setup.md wrongly linked
    to references/references/mcp.md by writing "references/mcp.md" from within
    references/)."""
    full_path = SKILL_DIR / doc_path
    text = full_path.read_text()
    # The "Related Skills" table intentionally links to sibling skills (e.g.
    # ../longbridge-quote) from the wider longbridge/skills family that aren't
    # vendored into this optional skill — those are "defer to it if installed"
    # pointers, not files this skill ships. Only check links above that table.
    text = text.split("## Related Skills")[0]
    for link in _extract_markdown_links(text):
        if link.startswith(("http://", "https://", "#")):
            continue
        resolved = (full_path.parent / link).resolve()
        assert resolved.is_file(), (
            f"{doc_path} links to {link!r}, which resolves to {resolved} (missing)"
        )


def test_setup_md_links_to_sibling_mcp_not_nested() -> None:
    text = (SKILL_DIR / "references" / "setup.md").read_text()
    assert "references/mcp.md" not in text, (
        "setup.md is itself inside references/ — linking to 'references/mcp.md' "
        "resolves to a nonexistent nested path; use 'mcp.md'"
    )
    assert "(mcp.md)" in text


def test_rust_content_doc_agrees_python_has_content_context() -> None:
    rust_content = (SKILL_DIR / "references" / "rust-sdk" / "content.md").read_text()
    python_content = (SKILL_DIR / "references" / "python-sdk" / "content-context.md").read_text()

    assert "ContentContext" in python_content
    assert "does not expose a `ContentContext`" not in rust_content, (
        "rust-sdk/content.md must not claim Python lacks ContentContext — "
        "the Python SDK does expose it (see python-sdk/content-context.md)"
    )
    assert "does" in rust_content and "ContentContext" in rust_content


@pytest.mark.parametrize(
    "doc_path",
    [
        "references/python-sdk/trade-context.md",
        "references/rust-sdk/trade-context.md",
    ],
)
def test_state_changing_order_calls_have_confirmation_guard(doc_path: str) -> None:
    text = (SKILL_DIR / doc_path).read_text()

    submit_idx = text.index("submit_order")
    preceding = text[:submit_idx]
    # The nearest confirmation warning before the first submit_order call must be
    # closer than the nearest preceding section heading (i.e. it lives directly
    # above this call, not just somewhere earlier in the doc).
    last_heading = preceding.rfind("## Submit Order")
    assert last_heading != -1, f"{doc_path} missing '## Submit Order' heading"
    section = text[last_heading:submit_idx]
    assert "confirm" in section.lower(), (
        f"{doc_path}: no confirmation warning between '## Submit Order' and the "
        f"first submit_order call"
    )

    replace_heading_idx = text.find("## Replace / Cancel Order")
    assert replace_heading_idx != -1, f"{doc_path} missing '## Replace / Cancel Order' heading"
    cancel_idx = text.index("cancel_order", replace_heading_idx)
    section = text[replace_heading_idx:cancel_idx]
    assert "confirm" in section.lower(), (
        f"{doc_path}: no confirmation warning between '## Replace / Cancel Order' "
        f"and the cancel_order call"
    )


def test_pitfalls_mentions_order_confirmation(skill_body) -> None:
    pitfalls_idx = skill_body.index("## Pitfalls")
    verification_idx = skill_body.index("## Verification")
    pitfalls_section = skill_body[pitfalls_idx:verification_idx]
    assert "confirm" in pitfalls_section.lower()
