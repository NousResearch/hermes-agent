"""Tests for optional-skills/devops/setup-wizard-generator (template integrity)."""

import re
import subprocess
from pathlib import Path

import pytest
import yaml

SKILL_DIR = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "devops"
    / "setup-wizard-generator"
)
SKILL_MD = SKILL_DIR / "SKILL.md"
TEMPLATE = SKILL_DIR / "templates" / "template.sh"


class TestFrontmatter:
    def _fm(self):
        text = SKILL_MD.read_text(encoding="utf-8")
        m = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
        assert m, "SKILL.md missing YAML frontmatter"
        return yaml.safe_load(m.group(1))

    def test_name_matches_directory(self):
        assert self._fm()["name"] == "setup-wizard-generator"

    def test_description_length(self):
        assert len(self._fm()["description"]) <= 60

    def test_license_and_platforms(self):
        fm = self._fm()
        assert fm["license"] == "MIT"
        assert "linux" in fm["platforms"]


class TestTemplate:
    def test_template_exists(self):
        assert TEMPLATE.is_file()

    def test_bash_syntax(self):
        proc = subprocess.run(
            ["bash", "-n", str(TEMPLATE)], capture_output=True, text=True
        )
        assert proc.returncode == 0, proc.stderr

    def test_stages_marker_present(self):
        text = TEMPLATE.read_text(encoding="utf-8")
        assert "STAGES" in text, "authoring marker missing"

    def test_library_helpers_defined(self):
        text = TEMPLATE.read_text(encoding="utf-8")
        for helper in (
            "stage()",
            "say()",
            "step()",
            "open_url()",
            "write_env()",
            "set_secret()",
            "finish()",
        ):
            assert helper in text, f"missing library helper {helper}"

    def test_ask_secret_defined(self):
        # secret entry must exist (hidden input path)
        assert "ask_secret" in TEMPLATE.read_text(encoding="utf-8")

    def test_total_stages_variable(self):
        assert re.search(
            r"^TOTAL_STAGES=", TEMPLATE.read_text(encoding="utf-8"), re.MULTILINE
        )


class TestSkillBody:
    def test_references_template_path(self):
        assert "templates/template.sh" in SKILL_MD.read_text(encoding="utf-8")

    def test_no_upstream_harness_residue(self):
        text = SKILL_MD.read_text(encoding="utf-8").lower()
        for token in ("claude", "/wizard", "slash command"):
            assert token not in text, f"upstream residue: {token}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
