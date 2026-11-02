"""Contract tests for the in-repo `commandcode` skill (skills/autonomous-ai-agents/commandcode).

These assert the skill's authoring standards (AGENTS.md "Skill authoring standards")
and that its factual claims about Hermes tooling match the real implementation.
No network access; stdlib + pytest only.
"""

import re
from pathlib import Path

import pytest

SKILL_PATH = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "autonomous-ai-agents"
    / "commandcode"
    / "SKILL.md"
)

# Section order required by AGENTS.md, "Skill authoring standards" rule 5.
REQUIRED_SECTIONS = [
    "## When to Use",
    "## Prerequisites",
    "## How to Run",
    "## Quick Reference",
    "## Procedure",
    "## Pitfalls",
    "## Verification",
]

# Primitives that are not available on Windows. A skill declaring `windows`
# support must not rely on them (AGENTS.md rule 3).
POSIX_ONLY_TOKENS = ["mktemp", "/tmp/", "<(", "$(", "chmod +x"]

MARKETING_WORDS = ["powerful", "comprehensive", "seamless", "advanced", "robust", "cutting-edge"]


@pytest.fixture(scope="module")
def content() -> str:
    return SKILL_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def frontmatter(content: str) -> dict:
    assert content.startswith("---"), "SKILL.md must start with '---'"
    match = re.search(r"\n---\s*\n", content[3:])
    assert match, "frontmatter must close with a '---' line"
    raw = content[3 : match.start() + 3]
    fields = {}
    for line in raw.splitlines():
        if not line.strip() or line.startswith((" ", "\t", "#")):
            continue
        key, _, value = line.partition(":")
        fields[key.strip()] = value.strip().strip('"').strip("'")
    return fields


class TestFrontmatter:
    def test_file_is_where_the_loader_expects_it(self):
        assert SKILL_PATH.is_file(), f"missing {SKILL_PATH}"

    def test_name_matches_directory(self, frontmatter):
        assert frontmatter["name"] == "commandcode"

    def test_description_is_one_short_sentence(self, frontmatter):
        description = frontmatter["description"]
        assert len(description) <= 60, f"description is {len(description)} chars: {description!r}"
        assert description.endswith("."), "description must end with a period"

    def test_description_does_not_repeat_the_skill_name(self, frontmatter):
        assert "commandcode" not in frontmatter["description"].lower()

    def test_description_has_no_marketing_words(self, frontmatter):
        lowered = frontmatter["description"].lower()
        found = [word for word in MARKETING_WORDS if word in lowered]
        assert not found, f"marketing words in description: {found}"

    def test_author_credits_the_human_contributor_first(self, frontmatter):
        assert frontmatter["author"].lower().startswith("james drake")

    def test_platforms_declared(self, frontmatter):
        assert frontmatter.get("platforms"), "platforms must be declared"


class TestSections:
    def test_required_sections_present_in_order(self, content):
        positions = [content.find(section) for section in REQUIRED_SECTIONS]
        missing = [
            section for section, pos in zip(REQUIRED_SECTIONS, positions) if pos == -1
        ]
        assert not missing, f"missing sections: {missing}"
        assert positions == sorted(positions), (
            "sections are out of order: "
            f"{[s for _, s in sorted(zip(positions, REQUIRED_SECTIONS))]}"
        )

    def test_prerequisites_is_not_duplicated(self, content):
        assert content.count("## Prerequisites") == 1

    def test_opens_with_an_intro_before_the_first_section(self, content):
        body = content.split("---\n", 2)[-1]
        intro = body.split("## ", 1)[0]
        # A title line plus at least two sentences of framing.
        assert intro.count(".") >= 2, "skill must open with a short factual intro"


class TestPlatformClaims:
    def test_windows_claim_is_not_contradicted_by_posix_only_examples(self, content, frontmatter):
        if "windows" not in frontmatter.get("platforms", ""):
            pytest.skip("skill does not claim Windows support")
        found = [token for token in POSIX_ONLY_TOKENS if token in content]
        assert not found, (
            f"declares Windows support but uses POSIX-only primitives: {found}"
        )


class TestSubmitGuidanceMatchesHermes:
    """`process(action='submit')` writes data plus a newline, so it does act as Enter.

    Source of truth: tools/process_registry.py::submit_stdin. The skill previously
    claimed the Enter key is never sent, which is wrong and misleads the model into
    avoiding a working mechanism.
    """

    def test_does_not_claim_submit_fails_to_send_enter(self, content):
        lowered = content.lower()
        for wrong in (
            "enter key isn't sent",
            "enter key is not sent",
            "isn't sent through the pty",
            "never rely on",
        ):
            assert wrong not in lowered, f"stale claim about submit(): {wrong!r}"

    def test_documents_submit_and_its_newline_behaviour(self, content):
        assert 'process(action="submit"' in content
        assert "newline" in content.lower()

    def test_submit_stdin_really_appends_a_newline(self):
        """Behavioural check against the implementation, not its source text."""
        from tools import process_registry

        registry = process_registry.ProcessRegistry()
        written = []

        class FakeSession:
            exited = False
            _pty = False

            def write_stdin(self, data):
                written.append(data)

        registry.get = lambda session_id: FakeSession()  # type: ignore[assignment]
        registry.write_stdin = lambda session_id, data: written.append(data)  # type: ignore[assignment]

        registry.submit_stdin("session-x", "yes")

        assert written == ["yes\n"], written


class TestNativeToolSurface:
    def test_prose_names_native_hermes_tools(self, content):
        for tool in ("terminal", "process"):
            assert f"`{tool}`" in content, f"skill should name the native `{tool}` tool"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
