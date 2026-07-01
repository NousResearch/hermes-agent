"""Invariants for the bundled `kilo` skill (skills/autonomous-ai-agents/kilo).

Pins the Hermes skill-authoring standards that a reviewer can't eyeball at a
glance: the description length/terminator contract (standard #1), required
frontmatter, valid platform tokens, the cross-link to the parent fork
(`opencode`), and that the prose reaches for native Hermes tools
(`terminal`/`process`) rather than raw shell utilities as headline tools
(standard #2). Parses frontmatter with regex (stdlib only) per the standard's
own validation snippet.
"""

import re
from pathlib import Path

import pytest

SKILL_MD = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "autonomous-ai-agents"
    / "kilo"
    / "SKILL.md"
)

VALID_PLATFORMS = {"linux", "macos", "windows"}


def _read() -> str:
    return SKILL_MD.read_text(encoding="utf-8")


def _fm_value(text: str, key: str) -> str | None:
    """Return the raw value of a top-level `key: value` frontmatter line."""
    m = re.search(rf"^{key}:\s*(.+?)\s*$", text, re.MULTILINE)
    return m.group(1) if m else None


class TestFrontmatter:
    def test_skill_md_exists(self):
        assert SKILL_MD.is_file(), f"missing SKILL.md at {SKILL_MD}"

    def test_name_is_kilo(self):
        assert _fm_value(_read(), "name") == "kilo"

    def test_description_length_and_terminator(self):
        raw = _fm_value(_read(), "description")
        assert raw is not None, "missing description frontmatter"
        desc = raw.strip("\"'")
        # Standard #1: <= 60 chars, one sentence, ends with a period.
        assert len(desc) <= 60, f"description too long: {len(desc)} chars: {desc!r}"
        assert desc.endswith("."), f"description must end with '.': {desc!r}"

    def test_version_is_semver(self):
        version = _fm_value(_read(), "version")
        assert version is not None, "missing version frontmatter"
        assert re.match(r"^\d+\.\d+\.\d+$", version), f"bad version: {version!r}"

    def test_license_is_mit(self):
        assert _fm_value(_read(), "license") == "MIT"

    def test_author_present_and_nonempty(self):
        author = _fm_value(_read(), "author")
        assert author is not None, "missing author frontmatter"
        assert author.strip(), "author is blank"

    def test_platforms_valid(self):
        raw = _fm_value(_read(), "platforms")
        assert raw is not None, "missing platforms frontmatter"
        tokens = [t.strip() for t in raw.strip("[]").split(",") if t.strip()]
        assert tokens, "no platform tokens"
        for t in tokens:
            assert t in VALID_PLATFORMS, f"invalid platform token: {t!r}"

    def test_related_skills_links_parent_fork(self):
        text = _read()
        m = re.search(r"^\s*related_skills:\s*(.+?)\s*$", text, re.MULTILINE)
        assert m is not None, "missing related_skills frontmatter"
        assert "opencode" in m.group(1), "must cross-link the opencode parent fork"


class TestBodyUsesNativeTools:
    """Prose must reference native Hermes tools, not raw shell utilities as the
    headline interaction surface (standard #2)."""

    def test_uses_terminal_tool(self):
        assert "terminal(command=" in _read()

    def test_uses_process_tool(self):
        assert "process(action=" in _read()


class TestScopeDecision:
    """The serve/daemon/attach client-server surface is intentionally out of
    scope for this skill (decision recorded in the plan). Guard against it
    creeping back in as a recommended path."""

    def test_does_not_recommend_serve_daemon_attach(self):
        body = _read()
        for needle in ("kilo serve", "kilo daemon", "--attach"):
            assert needle not in body, f"out-of-scope command leaked into skill: {needle!r}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
