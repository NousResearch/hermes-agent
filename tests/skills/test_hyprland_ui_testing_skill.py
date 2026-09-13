"""Contract tests for the optional Hyprland UI testing skill."""

import re
from pathlib import Path


SKILL_MD = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "linux-system-admin"
    / "hyprland-ui-testing"
    / "SKILL.md"
)


def _text() -> str:
    return SKILL_MD.read_text(encoding="utf-8")


def _frontmatter_value(name: str) -> str:
    match = re.search(rf"^{re.escape(name)}:\s*(.+)$", _text(), re.MULTILINE)
    assert match, f"missing {name} frontmatter"
    return match.group(1).strip().strip('"')


def test_frontmatter_is_linux_only_and_discoverable():
    assert _frontmatter_value("name") == "hyprland-ui-testing"
    description = _frontmatter_value("description")
    assert len(description) <= 60
    assert description.endswith(".")
    assert _frontmatter_value("platforms") == "[linux]"
    assert "requires_toolsets: [terminal]" in _text()


def test_modern_sections_are_present_in_order():
    sections = [
        "## When to Use",
        "## Prerequisites",
        "## How to Run",
        "## Quick Reference",
        "## Procedure",
        "## Pitfalls",
        "## Verification",
    ]
    positions = [_text().index(section) for section in sections]
    assert positions == sorted(positions)


def test_rendering_and_compositor_proof_are_not_conflated():
    body = _text()
    assert "Broadway proves GTK rendering only" in body
    assert "Use a Hyprland headless output" in body
    assert "Never call a Broadway screenshot proof of Hyprland placement" in body


def test_workspace_isolation_safety_invariants_are_explicit():
    body = _text()
    assert "Never launch the app on the normal display and move it afterward" in body
    assert "Keep the test window unpinned" in body
    assert "pinning intentionally makes a window visible across workspaces" in body
    assert "active workspace and focused client match the recorded baseline" in body


def test_cleanup_covers_every_temporary_resource():
    body = _text()
    assert "process(action=\"kill\", session_id=...)" in body
    assert "hyprctl output remove HERMES_UI_TEST" in body
    assert "Remove temporary runtime rules" in body
    assert "no test client remains" in body
