"""Hermetic contract tests for the fleet-collab skill scripts.

Validates the script-level contracts documented in SKILL.md:
- fleet_dispatch.sh: FLEET_MAP parsing, rhermes usage, failure -> block (not complete)
- fleet_create.sh: arg transport via tempfile, while-read (no mapfile), IS_HUB routing
- SKILL.md frontmatter: description <= 60 chars, platforms gate

No network calls, no live SSH — all external commands are mocked.
Run via: scripts/run_tests.sh tests/skills/test_fleet_collab_skill.py -q
"""

import os
import re
import subprocess
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

SKILL_DIR = Path(__file__).resolve().parents[2] / "optional-skills" / "devops" / "fleet-collab"
DISPATCH = SKILL_DIR / "scripts" / "fleet_dispatch.sh"
CREATE = SKILL_DIR / "scripts" / "fleet_create.sh"
SKILL_MD = SKILL_DIR / "SKILL.md"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class TestSkillFrontmatter(unittest.TestCase):
    """SKILL.md frontmatter meets the skill authoring standard."""

    def setUp(self):
        self.content = _read(SKILL_MD)

    def _frontmatter(self):
        m = re.search(r"^---\s*\n(.*?)\n---\s*\n", self.content, re.DOTALL)
        self.assertIsNotNone(m, "SKILL.md must have YAML frontmatter delimiters")
        return m.group(1)

    def test_description_le_60_chars(self):
        fm = self._frontmatter()
        m = re.search(r'^description:\s*"?(.*?)"?\s*$', fm, re.MULTILINE)
        self.assertIsNotNone(m, "frontmatter must have a description field")
        desc = m.group(1).strip()
        self.assertLessEqual(
            len(desc), 60,
            f"description is {len(desc)} chars, must be <= 60: {desc!r}",
        )

    def test_platforms_linux_only(self):
        """fleet_dispatch.sh uses GNU timeout; macOS not supported without coreutils."""
        fm = self._frontmatter()
        m = re.search(r"^platforms:\s*\[(.*?)\]", fm, re.MULTILINE)
        self.assertIsNotNone(m, "frontmatter must declare platforms")
        platforms = [p.strip() for p in m.group(1).split(",")]
        self.assertIn("linux", platforms)
        self.assertNotIn("macos", platforms, "macOS lacks GNU timeout by default")


class TestDispatchScriptContract(unittest.TestCase):
    """fleet_dispatch.sh must honor the FLEET_MAP rhermes field and not
    mark failed dispatches as done."""

    def setUp(self):
        self.src = _read(DISPATCH)

    def test_uses_rhermes_variable(self):
        """The SSH command must invoke $rhermes, not a hardcoded path."""
        # The remote execution line should reference $rhermes (the parsed 4th FLEET_MAP field)
        self.assertIn("$rhermes", self.src,
                      "dispatch must use the $rhermes variable from FLEET_MAP, not a hardcoded path")
        # Must NOT hardcode $HOME/.local/bin/hermes in the ssh exec line
        ssh_lines = [l for l in self.src.splitlines() if "ssh " in l and "chat" in l]
        for line in ssh_lines:
            self.assertNotIn('"$HOME/.local/bin/hermes"', line,
                             "ssh exec line must use $rhermes, not hardcoded $HOME/.local/bin/hermes")

    def test_failure_blocks_not_completes(self):
        """On remote failure, the script must block the task, not complete it."""
        # The else branch (failure) must call kanban block, not kanban complete
        lines = self.src.splitlines()
        in_else = False
        else_has_block = False
        else_has_complete = False
        for line in lines:
            stripped = line.strip()
            if stripped == "else":
                in_else = True
                continue
            if in_else:
                if stripped == "fi":
                    in_else = False
                    # Check this else block
                    if else_has_block and not else_has_complete:
                        break  # found a correct else
                    else_has_block = False
                    else_has_complete = False
                if "kanban block" in stripped:
                    else_has_block = True
                if "kanban complete" in stripped:
                    else_has_complete = True
        self.assertTrue(
            else_has_block,
            "failure branch must call 'kanban block' to keep the card retryable"
        )

    def test_success_branch_completes(self):
        """On success, the script must comment + complete."""
        self.assertIn("kanban comment", self.src)
        self.assertIn("kanban complete", self.src)

    def test_no_mapfile_usage(self):
        """mapfile is bash 4+ only; macOS default bash is 3.2."""
        self.assertNotIn("mapfile", self.src, "use while-read instead of mapfile for portability")


class TestCreateScriptContract(unittest.TestCase):
    """fleet_create.sh must transport args without mapfile (bash 3.2 compatible)."""

    def setUp(self):
        self.src = _read(CREATE)

    def test_no_mapfile_usage(self):
        """mapfile is bash 4+ only; macOS default bash is 3.2."""
        self.assertNotIn("mapfile", self.src,
                         "fleet_create.sh must use while-read, not mapfile (bash 3.2 compat)")

    def test_uses_while_read_for_args(self):
        """Args must be read via while-read loop, not mapfile."""
        self.assertIn("while IFS= read", self.src,
                      "must use 'while IFS= read' loop for portable arg reading")

    def test_tempfile_transport(self):
        """Args must be transported via tempfile + scp (avoids shell quote nesting)."""
        self.assertIn("tmpargs", self.src)
        self.assertIn("scp", self.src)

    def test_is_hub_routing(self):
        """Script must branch on IS_HUB to decide local vs remote execution."""
        self.assertIn("IS_HUB", self.src)
        self.assertIn("$HUB_HERMES", self.src)


class TestScriptSyntax(unittest.TestCase):
    """Scripts must pass bash -n syntax check."""

    def test_dispatch_syntax(self):
        result = subprocess.run(
            ["bash", "-n", str(DISPATCH)],
            capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(result.returncode, 0,
                         f"fleet_dispatch.sh syntax error:\n{result.stderr}")

    def test_create_syntax(self):
        result = subprocess.run(
            ["bash", "-n", str(CREATE)],
            capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(result.returncode, 0,
                         f"fleet_create.sh syntax error:\n{result.stderr}")


if __name__ == "__main__":
    unittest.main()
