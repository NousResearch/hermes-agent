"""Tests for optional-skills/devops/uteke/SKILL.md compliance."""
import re
import pathlib
import unittest

SKILL_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "devops"
    / "uteke"
    / "SKILL.md"
)


def _read():
    return SKILL_PATH.read_text()


class TestUtekeSkillFrontmatter(unittest.TestCase):
    def test_description_max_60_chars_and_ends_with_period(self):
        text = _read()
        m = re.search(r"^description: (.*)$", text, re.MULTILINE)
        self.assertIsNotNone(m, "description field missing")
        desc = m.group(1).strip().strip('"')
        self.assertLessEqual(len(desc), 60, f"description too long ({len(desc)} chars): {desc}")
        self.assertTrue(desc.endswith("."), f"description must end with period: {desc}")

    def test_author_credits_human_contributor(self):
        text = _read()
        m = re.search(r"^author: (.*)$", text, re.MULTILINE)
        self.assertIsNotNone(m, "author field missing")
        author = m.group(1).strip().strip('"')
        self.assertNotIn("codecoradev", author.lower(), "author should be human, not org handle")
        self.assertNotIn("Hermes Agent", author, "author should credit human, not bot")

    def test_name_field_present(self):
        text = _read()
        m = re.search(r"^name:\s+uteke\s*$", text, re.MULTILINE)
        self.assertIsNotNone(m, "name field missing or incorrect")

    def test_version_field_present(self):
        text = _read()
        m = re.search(r"^version:", text, re.MULTILINE)
        self.assertIsNotNone(m, "version field missing")

    def test_license_apache2(self):
        text = _read()
        self.assertIn("license: Apache-2.0", text)

    def test_platforms_gated(self):
        text = _read()
        self.assertIn("platforms:", text)
        self.assertIn("linux", text)
        self.assertIn("macos", text)

    def test_prerequisites_with_commands(self):
        text = _read()
        self.assertIn("prerequisites:", text)
        self.assertIn("commands:", text)

    def test_no_marketing_words(self):
        text = _read().lower()
        for word in ("powerful", "comprehensive", "seamless", "advanced"):
            self.assertNotIn(word, text, f"marketing word '{word}' found")

    def test_tags_present(self):
        text = _read()
        self.assertIn("tags:", text)
        for tag in ("memory", "semantic-search", "knowledge-graph"):
            self.assertIn(tag, text, f"expected tag '{tag}' missing")


class TestUtekeSkillBody(unittest.TestCase):
    def test_section_order(self):
        text = _read()
        sections = re.findall(r"^## (.+)$", text, re.MULTILINE)
        expected = [
            "When to Use",
            "Prerequisites",
            "How to Run",
            "Quick Reference",
            "Procedure",
            "Pitfalls",
            "Verification",
        ]
        for sec in expected:
            self.assertIn(sec, sections, f"missing section: {sec}")
        for i in range(1, len(expected)):
            self.assertGreater(
                sections.index(expected[i]),
                sections.index(expected[i - 1]),
                f"'{expected[i]}' appears before '{expected[i-1]}'",
            )

    def test_mode_c_hook_outputs_json_context(self):
        text = _read()
        self.assertIn('{"context"', text, "Mode C handler must output {\"context\": ...} JSON")

    def test_mode_c_hook_reads_stdin(self):
        text = _read()
        self.assertIn("sys.stdin.read()", text, "handler must read JSON from stdin")

    def test_no_bare_uteke_hook_command(self):
        text = _read()
        # The broken command from original PR
        self.assertNotIn(
            '"uteke hook recall',
            text,
            "bare 'uteke hook recall' command does not work with Hermes hooks",
        )

    def test_pitfalls_section_exists(self):
        text = _read()
        self.assertIn("## Pitfalls", text)

    def test_verification_section_exists(self):
        text = _read()
        self.assertIn("## Verification", text)

    def test_verification_includes_round_trip(self):
        text = _read()
        self.assertIn("uteke remember", text)
        self.assertIn("uteke recall", text)

    def test_nested_json_output_documented(self):
        text = _read()
        self.assertIn('"memory"', text, "nested JSON output under 'memory' key must be documented")

    def test_namespace_silo_warning(self):
        text = _read()
        self.assertIn("silo", text.lower(), "namespace isolation warning should be present")


if __name__ == "__main__":
    unittest.main()
