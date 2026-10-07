"""Tests for the active-profile resolver in agent/file_safety."""
from __future__ import annotations

from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers — set up a fake Rabbit root with two profiles, monkeypatch the
# resolver helpers so the classifier sees the test layout.
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_rabbit(tmp_path, monkeypatch):
    """Build a fake Rabbit layout:

        <tmp>/
          skills/foo/SKILL.md           # default profile
          plugins/foo/__init__.py
          cron/<state>
          memories/MEMORY.md
          profiles/
            rabbit-security/
              skills/foo/SKILL.md       # named profile
              plugins/...
            coder/
              skills/foo/SKILL.md       # another named profile
    """
    root = tmp_path / "fake-rabbit"
    (root / "skills" / "foo").mkdir(parents=True)
    (root / "skills" / "foo" / "SKILL.md").write_text("# default skill\n")
    (root / "plugins" / "foo").mkdir(parents=True)
    (root / "memories").mkdir(parents=True)
    (root / "cron").mkdir(parents=True)

    sec_home = root / "profiles" / "rabbit-security"
    (sec_home / "skills" / "foo").mkdir(parents=True)
    (sec_home / "skills" / "foo" / "SKILL.md").write_text("# sec skill\n")
    (sec_home / "plugins").mkdir(parents=True)

    coder_home = root / "profiles" / "coder"
    (coder_home / "skills" / "foo").mkdir(parents=True)
    (coder_home / "skills" / "foo" / "SKILL.md").write_text("# coder skill\n")

    # Monkeypatch the resolver functions used by file_safety so each test
    # can choose which profile is "active".
    import rabbit_constants
    monkeypatch.setattr(rabbit_constants, "get_default_rabbit_root", lambda: root)

    import agent.file_safety as fs
    monkeypatch.setattr(fs, "_rabbit_root_path", lambda: root)

    return {
        "root": root,
        "default_home": root,
        "security_home": sec_home,
        "coder_home": coder_home,
    }


def _set_active_home(monkeypatch, rabbit_home: Path):
    """Point file_safety._rabbit_home_path at a specific profile dir."""
    import agent.file_safety as fs
    monkeypatch.setattr(fs, "_rabbit_home_path", lambda: rabbit_home)


# ---------------------------------------------------------------------------
# _resolve_active_profile_name
# ---------------------------------------------------------------------------


class TestResolveActiveProfileName:
    def test_default_when_home_is_root(self, fake_rabbit, monkeypatch):
        _set_active_home(monkeypatch, fake_rabbit["default_home"])
        from agent.file_safety import _resolve_active_profile_name
        assert _resolve_active_profile_name() == "default"


    def test_falls_back_to_default_on_resolution_failure(self, fake_rabbit, monkeypatch):
        """If RABBIT_HOME resolution raises, return 'default' rather than crashing the tool."""
        import agent.file_safety as fs

        def _boom():
            raise RuntimeError("simulated")

        monkeypatch.setattr(fs, "_rabbit_home_path", _boom)
        # Should not raise — falls back to "default"
        assert fs._resolve_active_profile_name() == "default"
