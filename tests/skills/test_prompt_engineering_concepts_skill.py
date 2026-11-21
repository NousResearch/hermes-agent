"""Guards the prompt-engineering-concepts skill's contract.

The skill is a pure reference: its value is that the 20 techniques and their
selection cues are all present and that the transcription reference ships with
it. If a future edit drops a technique row or unlinks the reference, the skill
silently degrades into a partial list — this test fails instead.
"""

from __future__ import annotations

from pathlib import Path

from agent.skill_utils import SKILL_PROMPT_DESC_LIMIT, parse_frontmatter

SKILL_DIR = (
    Path(__file__).resolve().parents[2]
    / "optional-skills/software-development/prompt-engineering-concepts"
)
SKILL_MD = SKILL_DIR / "SKILL.md"
REFERENCE = SKILL_DIR / "references/20-concepts-cheatsheet.md"

# Canonical technique names as they appear in the Quick Reference table.
TECHNIQUES = [
    "Zero-shot",
    "Few-shot",
    "Chain-of-thought",
    "Role prompting",
    "Instruction prompting",
    "Output formatting",
    "Temperature control",
    "Prompt chaining",
    "Self-consistency",
    "RAG",
    "Prompt templates",
    "Guardrails",
    "Dynamic prompting",
    "ReAct",
    "Tree-of-thought",
    "Prompt compression",
    "Multi-modal",
    "JSON mode",
    "Prompt evaluation",
    "Prompt optimization",
]


def _frontmatter() -> tuple[dict, str]:
    content = SKILL_MD.read_text(encoding="utf-8")
    fm, body = parse_frontmatter(content)
    return fm, body


class TestPromptEngineeringConceptsSkill:
    def test_frontmatter_fields(self):
        fm, _ = _frontmatter()
        for field in (
            "name",
            "description",
            "version",
            "author",
            "license",
            "platforms",
        ):
            assert fm.get(field), f"missing frontmatter field: {field}"
        assert fm["name"] == SKILL_DIR.name
        tags = (fm.get("metadata") or {}).get("hermes", {}).get("tags")
        assert tags, "metadata.hermes.tags is required"

    def test_description_fits_the_index(self):
        fm, _ = _frontmatter()
        description = str(fm["description"])
        assert len(description) <= SKILL_PROMPT_DESC_LIMIT
        assert description.endswith(".")
        assert "prompt engineering" not in description.lower(), (
            "description must not repeat the skill name"
        )

    def test_author_credits_the_human_first(self):
        fm, _ = _frontmatter()
        assert str(fm["author"]).startswith("Joerg Peetz"), (
            "author must credit the human contributor first"
        )

    def test_all_twenty_techniques_are_listed(self):
        _, body = _frontmatter()
        missing = [name for name in TECHNIQUES if f"| {name} |" not in body]
        assert not missing, f"Quick Reference is missing technique rows: {missing}"

    def test_pitfalls_and_verification_sections_present(self):
        _, body = _frontmatter()
        assert "## Pitfalls" in body
        assert "## Verification" in body
        # Eight numbered pitfalls, so a rewrite that silently halves them fails.
        pitfalls = body.split("## Pitfalls", 1)[1].split("## Verification", 1)[0]
        assert (
            len(
                [
                    line
                    for line in pitfalls.splitlines()
                    if line[:3].strip().rstrip(".").isdigit()
                ]
            )
            == 8
        )

    def test_reference_is_shipped_and_linked(self):
        assert REFERENCE.is_file(), "references/20-concepts-cheatsheet.md is missing"
        text = REFERENCE.read_text(encoding="utf-8")
        assert len(text) > 1000, "reference transcription looks truncated"
        # The poster capitalises technique names in Title Case ("Zero-Shot
        # Prompting", "Tree-of-Thought"), so match case-insensitively.
        lowered = text.lower()
        missing = [name for name in TECHNIQUES if name.lower() not in lowered]
        assert not missing, f"reference is missing techniques: {missing}"
        _, body = _frontmatter()
        assert "references/20-concepts-cheatsheet.md" in body, (
            "SKILL.md must point at the reference file"
        )

    def test_no_machine_local_paths(self):
        content = SKILL_MD.read_text(encoding="utf-8")
        for needle in ("/Users/", "/home/", "~/workspace", "C:\\Users"):
            assert needle not in content, f"machine-local path leaked: {needle}"
