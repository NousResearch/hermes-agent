"""Tests for the active-profile resolver in agent/file_safety."""
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
# R2 (security review L-1): scope the credential/protected rules to the ROOT.
# _hermes_dirs() returned {active home, root} and nothing else, so a SIBLING
# profile's .env / auth.json / state.db / skills/.hub were unguarded on both
# the read and the write path — the exact control the primary fix left open.
# ---------------------------------------------------------------------------


class TestCrossProfileScopeIsGuarded:
    @pytest.fixture(autouse=True)
    def _no_safe_root(self, monkeypatch):
        """HERMES_WRITE_SAFE_ROOT would mask the assertion under test."""
        import agent.file_safety as fs
        monkeypatch.setattr(fs, "get_safe_write_roots", lambda: set())

    def _active(self, monkeypatch, fake_hermes):
        _set_active_home(monkeypatch, fake_hermes["security_home"])
        import agent.file_safety as fs
        return fs

    def test_sibling_env_is_read_and_write_denied(self, fake_hermes, monkeypatch):
        fs = self._active(monkeypatch, fake_hermes)
        sibling = fake_hermes["coder_home"] / ".env"
        assert fs.get_read_block_error(str(sibling)), "sibling .env read was allowed"
        assert fs.is_write_denied(str(sibling)), "sibling .env write was allowed"
        # The active profile's own copy and the root's keep their existing posture.
        for own in (fake_hermes["security_home"] / ".env", fake_hermes["root"] / ".env"):
            assert fs.get_read_block_error(str(own))
            assert fs.is_write_denied(str(own))

    def test_sibling_control_files_keep_their_documented_posture(self, fake_hermes, monkeypatch):
        """auth.json is read-denied but deliberately NOT write-denied (#45947);
        config.yaml and SOUL.md are owned by other guards, not this layer."""
        fs = self._active(monkeypatch, fake_hermes)
        auth = fake_hermes["coder_home"] / "auth.json"
        assert fs.get_read_block_error(str(auth))
        assert not fs.is_write_denied(str(auth))
        for other in ("config.yaml", "SOUL.md"):
            path = fake_hermes["coder_home"] / other
            assert not fs.get_read_block_error(str(path)), other
            assert not fs.is_write_denied(str(path)), other

    def test_sibling_session_state_and_token_dirs_are_guarded(self, fake_hermes, monkeypatch):
        fs = self._active(monkeypatch, fake_hermes)
        coder = fake_hermes["coder_home"]
        for rel in ("state.db", "sessions" + "/x.json"):
            assert fs.is_write_denied(str(coder / rel)), rel
        tokens = coder / "mcp-tokens" / "x.json"
        assert fs.get_read_block_error(str(tokens))
        assert fs.is_write_denied(str(tokens))

    def test_project_env_under_the_root_stays_writable(self, fake_hermes, monkeypatch):
        """<root>/kanban/**, <root>/hooks/** and a profile's HOME subtree hold PROJECT
        .env files: documented as writable-but-read-denied, so the new profile-wide
        write deny must not swallow them."""
        fs = self._active(monkeypatch, fake_hermes)
        root = fake_hermes["root"]
        for rel in (
            ("kanban", "workspaces", "t1", ".env"),
            ("hooks", ".env"),
            ("profiles", "coder", "home", "project", ".env"),
        ):
            path = str(root.joinpath(*rel))
            assert not fs.is_write_denied(path), path
            assert fs.get_read_block_error(path), path
