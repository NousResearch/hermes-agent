from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_MD = (
    REPO_ROOT / "optional-skills" / "blockchain" / "graph-advocate" / "SKILL.md"
)


def _frontmatter(text: str) -> dict:
    """Parse the top-level YAML frontmatter (the block between the first two
    '---' fences). Nested/indented keys are ignored on purpose."""
    assert text.startswith("---"), "SKILL.md must start with YAML frontmatter"
    _, block, _ = text.split("---", 2)
    fields: dict[str, str] = {}
    for line in block.splitlines():
        if not line or line.startswith((" ", "\t")):
            continue
        if ":" in line:
            key, _, value = line.partition(":")
            fields[key.strip()] = value.strip()
    return fields


def test_frontmatter_name_and_description_meet_standards():
    fm = _frontmatter(SKILL_MD.read_text(encoding="utf-8"))
    assert fm["name"] == "graph-advocate"
    desc = fm["description"]
    assert len(desc) <= 60, f"description is {len(desc)} chars (max 60)"
    assert desc.endswith("."), "description must end with a period"


def test_required_sections_present():
    text = SKILL_MD.read_text(encoding="utf-8")
    for heading in (
        "## When to Use",
        "## Prerequisites",
        "## How to Run",
        "## Quick Reference",
        "## Procedure",
        "## Pitfalls",
        "## Verification",
    ):
        assert heading in text, f"missing required section: {heading}"


def test_documented_workflow_surface():
    text = SKILL_MD.read_text(encoding="utf-8")
    # Free routing surface documented
    assert "https://graphadvocate.com" in text
    for endpoint in ("/route", "/chat", "/quota"):
        assert endpoint in text, f"missing documented endpoint: {endpoint}"
    # Paid endpoints settle over x402 on Base
    for token in ("x402", "402", "USDC"):
        assert token in text, f"missing payment detail: {token}"


def test_skill_is_instruction_only():
    text = SKILL_MD.read_text(encoding="utf-8").lower()
    # This skill must not ship or execute local code; it calls a hosted API.
    assert "instruction-only" in text
