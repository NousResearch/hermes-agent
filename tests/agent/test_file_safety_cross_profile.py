"""Tests for the cross-Hermes-profile write guard in agent/file_safety.

The guard fires when a tool tries to write into another Hermes profile's
skills/plugins/cron/memories directory. It's a soft guard — defense in
depth, NOT a security boundary — but it prevents the agent from silently
corrupting a profile that belongs to a different session.

Reference: May 2026 incident — a hermes-security profile session
accidentally edited skills under both ~/.hermes/profiles/hermes-security/skills/
AND ~/.hermes/skills/ (the default profile's skills), realizing only
afterwards that the second path belonged to a different profile.
"""
from __future__ import annotations

from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers — set up a fake Hermes root with two profiles, monkeypatch the
# resolver helpers so the classifier sees the test layout.
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_hermes(tmp_path, monkeypatch):
    """Build a fake Hermes layout:

        <tmp>/
          skills/foo/SKILL.md           # default profile
          plugins/foo/__init__.py
          cron/<state>
          memories/MEMORY.md
          profiles/
            hermes-security/
              skills/foo/SKILL.md       # named profile
              plugins/...
            coder/
              skills/foo/SKILL.md       # another named profile
    """
    root = tmp_path / "fake-hermes"
    (root / "skills" / "foo").mkdir(parents=True)
    (root / "skills" / "foo" / "SKILL.md").write_text("# default skill\n")
    (root / "plugins" / "foo").mkdir(parents=True)
    (root / "memories").mkdir(parents=True)
    (root / "cron").mkdir(parents=True)

    sec_home = root / "profiles" / "hermes-security"
    (sec_home / "skills" / "foo").mkdir(parents=True)
    (sec_home / "skills" / "foo" / "SKILL.md").write_text("# sec skill\n")
    (sec_home / "plugins").mkdir(parents=True)

    coder_home = root / "profiles" / "coder"
    (coder_home / "skills" / "foo").mkdir(parents=True)
    (coder_home / "skills" / "foo" / "SKILL.md").write_text("# coder skill\n")

    # Monkeypatch the resolver functions used by file_safety so each test
    # can choose which profile is "active".
    import hermes_constants
    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda: root)

    # The reloads below ensure get_cross_profile_warning/classify see the patched root.
    import agent.file_safety as fs
    monkeypatch.setattr(fs, "_hermes_root_path", lambda: root)

    return {
        "root": root,
        "default_home": root,
        "security_home": sec_home,
        "coder_home": coder_home,
    }


def _set_active_home(monkeypatch, hermes_home: Path):
    """Point file_safety._hermes_home_path at a specific profile dir."""
    import agent.file_safety as fs
    monkeypatch.setattr(fs, "_hermes_home_path", lambda: hermes_home)


# ---------------------------------------------------------------------------
# _resolve_active_profile_name
# ---------------------------------------------------------------------------


class TestResolveActiveProfileName:
    def test_default_when_home_is_root(self, fake_hermes, monkeypatch):
        _set_active_home(monkeypatch, fake_hermes["default_home"])
        from agent.file_safety import _resolve_active_profile_name
        assert _resolve_active_profile_name() == "default"


    def test_falls_back_to_default_on_resolution_failure(self, fake_hermes, monkeypatch):
        """If HERMES_HOME resolution raises, return 'default' rather than crashing the tool."""
        import agent.file_safety as fs

        def _boom():
            raise RuntimeError("simulated")

        monkeypatch.setattr(fs, "_hermes_home_path", _boom)
        # Should not raise — falls back to "default"
        assert fs._resolve_active_profile_name() == "default"


# ---------------------------------------------------------------------------
# classify_cross_profile_target
# ---------------------------------------------------------------------------


class TestClassifyCrossProfileTarget:

    def test_security_writing_default_skill(self, fake_hermes, monkeypatch):
        """The exact incident from May 2026."""
        _set_active_home(monkeypatch, fake_hermes["security_home"])
        from agent.file_safety import classify_cross_profile_target
        result = classify_cross_profile_target(
            str(fake_hermes["default_home"] / "skills" / "foo" / "SKILL.md")
        )
        assert result is not None
        assert result["active_profile"] == "hermes-security"
        assert result["target_profile"] == "default"
        assert result["area"] == "skills"

    def test_default_writing_security_skill(self, fake_hermes, monkeypatch):
        """Inverse direction — default-profile session reaching into a named profile."""
        _set_active_home(monkeypatch, fake_hermes["default_home"])
        from agent.file_safety import classify_cross_profile_target
        result = classify_cross_profile_target(
            str(fake_hermes["security_home"] / "skills" / "foo" / "SKILL.md")
        )
        assert result is not None
        assert result["active_profile"] == "default"
        assert result["target_profile"] == "hermes-security"


    @pytest.mark.parametrize("area", ["skills", "plugins", "cron", "memories"])
    def test_all_profile_scoped_areas_classified(self, fake_hermes, monkeypatch, area):
        _set_active_home(monkeypatch, fake_hermes["security_home"])
        from agent.file_safety import classify_cross_profile_target
        target = fake_hermes["default_home"] / area / "foo.txt"
        result = classify_cross_profile_target(str(target))
        assert result is not None
        assert result["area"] == area




# ---------------------------------------------------------------------------
# Early-return regression tests (PR #48784)
# ---------------------------------------------------------------------------
# The fix in d7e30dcd2 added a path-normalization early return before the
# name comparison. When active_profile is a named profile and the resolved
# target path lives under the active profile's home directory, the guard
# returns None immediately — even if the name inference would have matched
# a different profile. These tests verify the early return is correct and
# does not accidentally suppress genuine cross-profile warnings.
# ---------------------------------------------------------------------------


class TestCrossProfileGuardEarlyReturn:
    """Regression: the path-normalization early return must not suppress
    genuine cross-profile warnings, and must correctly identify in-profile
    writes when paths resolve through symlinks or relative prefixes."""

    def test_same_profile_path_returns_none_through_early_return(
        self, fake_hermes, monkeypatch
    ):
        """When active_profile is a named profile and the target path
        resolves inside the active profile's home, the early return fires
        and returns None — even when the path doesn't go through the
        name-comparison path."""
        import agent.file_safety as fs

        _set_active_home(monkeypatch, fake_hermes["security_home"])
        # Target inside the active profile's own home directory
        result = fs.classify_cross_profile_target(
            str(fake_hermes["security_home"] / "skills" / "foo" / "SKILL.md")
        )
        assert result is None

    def test_other_profile_target_is_still_blocked(
        self, fake_hermes, monkeypatch
    ):
        """A genuine path into another profile's scoped area must still
        be flagged — the early return must not be too broad."""
        _set_active_home(monkeypatch, fake_hermes["security_home"])
        from agent.file_safety import classify_cross_profile_target

        result = classify_cross_profile_target(
            str(fake_hermes["coder_home"] / "skills" / "foo" / "SKILL.md")
        )
        assert result is not None
        assert result["target_profile"] == "coder"

    def test_symlink_to_other_profile_does_not_early_return(
        self, fake_hermes, monkeypatch, tmp_path
    ):
        """A symlink pointing to a different profile's area must still
        be flagged — the early return uses the resolved path and must
        detect that the resolved path is not inside the active profile."""
        import agent.file_safety as fs

        _set_active_home(monkeypatch, fake_hermes["security_home"])

        # Symlink from tmp_path to the coder profile's skills
        symlink = tmp_path / "link-to-coder-skills"
        symlink.symlink_to(fake_hermes["coder_home"] / "skills")

        result = fs.classify_cross_profile_target(
            str(symlink / "some-skill.md")
        )
        assert result is not None
        assert result["target_profile"] == "coder"

    def test_default_profile_skips_early_return_when_writing_named_profile(
        self, fake_hermes, monkeypatch
    ):
        """When active_profile is 'default', the early return is skipped
        (active_profile != 'default' is False), so the name comparison
        must still detect cross-profile writes correctly."""
        _set_active_home(monkeypatch, fake_hermes["default_home"])
        from agent.file_safety import classify_cross_profile_target

        result = classify_cross_profile_target(
            str(fake_hermes["security_home"] / "skills" / "foo" / "SKILL.md")
        )
        assert result is not None
        assert result["active_profile"] == "default"
        assert result["target_profile"] == "hermes-security"


# ---------------------------------------------------------------------------
# get_cross_profile_warning
# ---------------------------------------------------------------------------


class TestGetCrossProfileWarning:
    def test_in_profile_returns_none(self, fake_hermes, monkeypatch):
        _set_active_home(monkeypatch, fake_hermes["security_home"])
        from agent.file_safety import get_cross_profile_warning
        assert get_cross_profile_warning(
            str(fake_hermes["security_home"] / "skills" / "foo" / "SKILL.md")
        ) is None

    def test_cross_profile_warning_names_both_profiles(self, fake_hermes, monkeypatch):
        _set_active_home(monkeypatch, fake_hermes["security_home"])
        from agent.file_safety import get_cross_profile_warning
        warn = get_cross_profile_warning(
            str(fake_hermes["default_home"] / "skills" / "foo" / "SKILL.md")
        )
        assert warn is not None
        # Must name BOTH profiles so the model knows which is which.
        assert "default" in warn
        assert "hermes-security" in warn
        # Must name the bypass kwarg.
        assert "cross_profile=True" in warn
        # Must reference the area.
        assert "skills" in warn

    def test_warning_is_defense_in_depth_not_boundary(self, fake_hermes, monkeypatch):
        _set_active_home(monkeypatch, fake_hermes["security_home"])
        from agent.file_safety import get_cross_profile_warning
        warn = get_cross_profile_warning(
            str(fake_hermes["default_home"] / "skills" / "foo" / "SKILL.md")
        )
        # Must self-document as defense-in-depth so future reviewers
        # don't promote it to a hard block.
        assert "not a security boundary" in warn.lower()
