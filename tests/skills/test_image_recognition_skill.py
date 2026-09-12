"""Contract checks for the optional image-recognition skill asset.

Reads only the SKILL.md and script assets — no network, no Hermes runtime.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_DIR = (
    REPO_ROOT
    / "optional-skills"
    / "autonomous-ai-agents"
    / "image-recognition"
)
SKILL_PATH = SKILL_DIR / "SKILL.md"
SCRIPT_PATH = SKILL_DIR / "scripts" / "check_vision_path.py"

REQUIRED_SECTIONS = [
    "## When to Use",
    "## Prerequisites",
    "## How to Run",
    "## Quick Reference",
    "## Procedure",
    "## Pitfalls",
    "## Verification",
]

NATIVE_TOOLS = [
    "`vision_analyze`",
    "`terminal`",
    "`read_file`",
]


def _frontmatter_and_body():
    content = SKILL_PATH.read_text(encoding="utf-8")
    assert content.startswith("---"), "SKILL.md must open with frontmatter"
    m = re.search(r"\n---\s*\n", content[3:])
    assert m, "frontmatter must close with ---"
    fm_text = content[3 : m.start() + 3]
    body = content[m.end() + 3 :]
    fm = {}
    for line in fm_text.splitlines():
        km = re.match(r"^(\w[\w-]*):\s*(.*)$", line)
        if km:
            fm[km.group(1)] = km.group(2).strip().strip('"')
    return fm, body


def test_skill_file_exists():
    assert SKILL_PATH.is_file()
    assert SCRIPT_PATH.is_file()


def test_frontmatter_required_fields():
    fm, _ = _frontmatter_and_body()
    for field in ("name", "description", "version", "author", "license", "platforms"):
        assert field in fm, f"missing frontmatter field: {field}"
    assert fm["name"] == "image-recognition"


def test_author_credits_human_first():
    fm, _ = _frontmatter_and_body()
    author = fm["author"]
    assert author.startswith("moqiecuican"), "human author must come first"
    assert ", Hermes Agent" in author, "tool credit must come second"


def test_description_hardline():
    fm, _ = _frontmatter_and_body()
    desc = fm["description"]
    assert len(desc) <= 60, f"description is {len(desc)} chars; hardline is 60"
    assert desc.endswith("."), "description must end with a period"
    assert desc.count(".") == 1, "description must be one sentence"


def test_required_sections_present_in_order():
    _, body = _frontmatter_and_body()
    positions = []
    for section in REQUIRED_SECTIONS:
        idx = body.find(section)
        assert idx != -1, f"missing section: {section}"
        positions.append(idx)
    assert positions == sorted(positions), "sections out of required order"


def test_body_size_within_bundled_norms():
    content = SKILL_PATH.read_text(encoding="utf-8")
    lines = content.count("\n") + 1
    assert 80 <= lines <= 320, f"SKILL.md is {lines} lines; expected ~100-320"


def test_references_native_hermes_tools():
    _, body = _frontmatter_and_body()
    for tool in NATIVE_TOOLS:
        assert tool in body, f"body must reference native tool {tool}"


def test_no_machine_local_paths():
    content = SKILL_PATH.read_text(encoding="utf-8")
    assert "/home/" not in content
    assert "C:\\Users" not in content


def test_script_scope_declaration():
    """The checker must state it targets Hermes Agent only."""
    content = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "Hermes Agent" in content
    assert "will not work with other agents" in content


def test_script_is_read_only():
    """No network / subprocess / filesystem mutation in the checker."""
    content = SCRIPT_PATH.read_text(encoding="utf-8")
    for forbidden in (
        "subprocess",
        "requests",
        "socket",
        "httpx.post",
        "open(",
        "os.remove",
        "shutil",
    ):
        assert forbidden not in content, f"checker must stay read-only: {forbidden}"


def test_script_portable_hermes_home():
    """HERMES_HOME env override + ~/.hermes fallback, no hardcoded user paths."""
    content = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "HERMES_HOME" in content
    assert "expanduser" in content
    assert "/home/miracle" not in content
