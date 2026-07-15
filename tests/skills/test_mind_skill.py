"""Tests for optional-skills/autonomous-ai-agents/mind/SKILL.md.

The skill wraps an external single-file tool, so these tests enforce the
skill contract itself, hermetically: frontmatter standards, the modern
section order, and internal consistency between the declared version, the
pinned install URL, and the integrity checksum (a drifting pin is exactly
the failure mode a wrapper skill can silently develop).
"""
import re
from pathlib import Path

SKILL = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "autonomous-ai-agents"
    / "mind"
    / "SKILL.md"
)
TEXT = SKILL.read_text(encoding="utf-8")
FRONT = TEXT.split("---")[1]


def _front_value(key):
    m = re.search(r"^%s: (.+)$" % key, FRONT, re.MULTILINE)
    return m.group(1).strip() if m else None


class TestFrontmatter:
    def test_description_within_60_chars_one_sentence(self):
        desc = _front_value("description")
        assert desc is not None
        assert len(desc) <= 60, len(desc)
        assert desc.endswith(".")
        assert desc.count(".") == 1

    def test_description_has_no_marketing_words(self):
        desc = _front_value("description").lower()
        for banned in ("powerful", "comprehensive", "seamless", "advanced"):
            assert banned not in desc

    def test_platforms_declared(self):
        assert _front_value("platforms") == "[linux, macos, windows]"

    def test_prerequisite_commands_declared(self):
        assert "commands: [python3, curl]" in FRONT

    def test_pseudonymous_author_only(self):
        assert _front_value("author") == "Da7_Tech"


class TestSectionOrder:
    def test_modern_section_order(self):
        wanted = ["## When to Use", "## Prerequisites", "## How to Run",
                  "## Quick Reference", "## Procedure", "## Pitfalls",
                  "## Verification"]
        positions = [TEXT.find(h) for h in wanted]
        assert all(p != -1 for p in positions), positions
        assert positions == sorted(positions), "sections out of order"

    def test_title_is_skill_form(self):
        assert "\n# mind Skill\n" in TEXT


class TestInstallPinConsistency:
    def test_version_matches_pinned_url(self):
        version = _front_value("version")
        pinned = re.findall(r"raw\.githubusercontent\.com/Da7-Tech/mind/v([\d.]+)/mind\.py", TEXT)
        assert pinned, "install must pin a release tag, not a branch"
        assert all(v == version for v in pinned), (version, pinned)

    def test_no_install_from_moving_branch(self):
        assert "/mind/main/mind.py" not in TEXT

    def test_integrity_checksum_present_and_wellformed(self):
        # cross-platform form: a python3 hashlib assert (`shasum` does not
        # exist on Windows, which this skill declares as a platform)
        shas = re.findall(r"h=='([0-9a-f]{64})'", TEXT)
        assert shas, "install must include a sha256 integrity check"
        assert len(set(shas)) == 1, "install/verification checksums differ"
        assert len(shas) >= 2, "the Verification block must integrity-check too"
        assert "shasum" not in TEXT, "shasum is not available on Windows"

    def test_terminal_tool_framing(self):
        assert "`terminal`" in TEXT

    def test_no_banned_heredoc(self):
        # HARDLINE standard #2: `cat <<EOF` / `echo > file` must be write_file
        assert "cat >" not in TEXT and "<<'EOF'" not in TEXT and "<<EOF" not in TEXT

    def test_write_file_framing_for_script_creation(self):
        # the nightly-automation step must frame file creation via write_file
        assert "write_file" in TEXT


class TestProfileAndPlatformSafety:
    def test_active_profile_scripts_directory(self):
        assert 'HERMES_ROOT="${HERMES_HOME:-$HOME/.hermes}"' in TEXT
        assert "$HERMES_ROOT/scripts/mind_dream.sh" in TEXT
        assert "~/.hermes/scripts/mind_dream.sh" not in TEXT

    def test_cron_keeps_relative_script_name(self):
        assert "--script mind_dream.sh" in TEXT
        assert "--script ~/.hermes/" not in TEXT
        assert "--script $HERMES_ROOT/" not in TEXT

    def test_native_windows_uses_task_scheduler(self):
        assert "```powershell" in TEXT
        assert "Windows Task Scheduler" in TEXT
        assert "Register-ScheduledTask" in TEXT
        assert "-WorkingDirectory $Project" in TEXT

    def test_windows_profile_default_is_documented(self):
        assert "HERMES_HOME" in TEXT
        assert "%LOCALAPPDATA%/hermes" in TEXT
