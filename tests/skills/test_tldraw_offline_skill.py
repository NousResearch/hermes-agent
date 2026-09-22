"""Tests for the tldraw-offline optional skill.

Structural + internal-consistency checks only (stdlib + pytest, no network).
The skill's runtime claims were validated live against the real tldraw offline
app (headless) and its bundled script-context.d.ts; scripts/validate_shapes.mjs
re-checks the shape schema against the tldraw SDK.
"""

import re
from pathlib import Path

import pytest

SKILL_DIR = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "creative"
    / "tldraw-offline"
)
SKILL_MD = SKILL_DIR / "SKILL.md"
MAIN_JS = SKILL_DIR / "scripts" / "main.js"


@pytest.fixture(scope="module")
def skill_text() -> str:
    return SKILL_MD.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def main_js() -> str:
    return MAIN_JS.read_text(encoding="utf-8")


def test_skill_file_exists():
    assert SKILL_MD.is_file(), f"missing {SKILL_MD}"


def test_frontmatter_present(skill_text: str):
    assert skill_text.startswith("---\n"), "SKILL.md must open with YAML frontmatter"
    assert skill_text.count("---") >= 2, "frontmatter must be delimited by two '---'"




def test_required_sections_present(skill_text: str):
    for heading in (
        "## When to Use",
        "## Prerequisites",
        "## How to Run",
        "## Quick Reference",
        "## Procedure",
        "## Pitfalls",
        "## Verification",
    ):
        assert heading in skill_text, f"missing section: {heading}"




def test_counter_example_is_interactive_and_safe():
    """The counter.js example must show the verified interactive-UI pattern:
    ctx contract, pointer_down handling, and REQUIRED signal-based cleanup
    (whose absence causes the double-fire bug found live)."""
    counter = (SKILL_DIR / "scripts" / "counter.js").read_text(encoding="utf-8")
    assert "export default function ({ editor, helpers, signal })" in counter
    assert "pointer_down" in counter
    assert "editor.on('event'" in counter
    # the cleanup that prevents the click-doubling leak
    assert "signal.addEventListener('abort'" in counter
    assert "editor.off('event'" in counter
    # state kept in meta, rendered as label
    assert "meta" in counter and "count" in counter










def test_uses_richtext_not_bare_string(skill_text: str):
    assert "toRichText" in skill_text
    assert "richText" in skill_text




def test_main_js_matches_verified_contract(main_js: str):
    # main.js must use the real contract learned from the running app.
    assert "export default function ({ editor, helpers, signal })" in main_js
    # primitives imported from tldraw, not used as globals
    assert "from 'tldraw'" in main_js
    assert "createShapeId" in main_js and "toRichText" in main_js
    # idempotent furniture
    assert "createShapeIfMissing" in main_js
    # batched writes
    assert "editor.run(" in main_js
    # reactive + REQUIRED signal cleanup
    assert "editor.store.listen" in main_js
    assert "signal.addEventListener('abort'" in main_js
    # script-owned writes kept out of undo
    assert "history: 'ignore'" in main_js




def test_platforms_declared(skill_text: str):
    m = re.search(r"^platforms: (.*)$", skill_text, re.MULTILINE)
    assert m, "platforms field required (cross-platform desktop app)"
    for os_name in ("linux", "macos", "windows"):
        assert os_name in m.group(1)
