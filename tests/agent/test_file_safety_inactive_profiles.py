"""Tests for inactive-profile credential write blocking in file_safety.

Regression for https://github.com/NousResearch/hermes-agent/issues/37617 —
``is_write_denied`` previously only covered the active profile's credential
files (auth.json, mcp-tokens/, etc.) and the global root view.  Credential
files under *inactive* profile directories (e.g.
``~/.hermes/profiles/<name>/auth.json``) were left unprotected, allowing a
prompt-injected write_file to overwrite them without any audit or approval
gate.

These tests verify that ``is_write_denied`` returns True for credential paths
under any profile — active or inactive — while leaving non-credential profile
files writable.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture()
def fake_root(tmp_path, monkeypatch):
    """Set up a fake Hermes root with two profiles (active + inactive)."""
    import agent.file_safety as fs

    root = tmp_path / "hermes"
    root.mkdir()

    # Active profile (default)
    active = root / "default"
    active.mkdir()

    # Inactive profile
    inactive = root / "profiles" / "hermes-security"
    inactive.mkdir(parents=True)

    # Another inactive profile
    inactive2 = root / "profiles" / "work"
    inactive2.mkdir(parents=True)

    monkeypatch.setattr(fs, "_hermes_home_path", lambda: active)
    monkeypatch.setattr(fs, "_hermes_root_path", lambda: root)
    return root, active, inactive, inactive2


def _touch(parent: Path, *parts: str) -> Path:
    """Create a file (with parents) so realpath() resolves it."""
    p = parent.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("dummy", encoding="utf-8")
    return p


class TestInactiveProfileCredentialBlocking:
    """Credential files under inactive profiles must be write-denied."""

    def test_inactive_profile_auth_json_denied(self, fake_root):
        """auth.json under an inactive profile is blocked."""
        from agent.file_safety import is_write_denied

        root, active, inactive, inactive2 = fake_root
        auth = _touch(inactive, "auth.json")
        assert is_write_denied(str(auth)) is True

    def test_inactive_profile_mcp_tokens_denied(self, fake_root):
        """mcp-tokens/ under an inactive profile is blocked."""
        from agent.file_safety import is_write_denied

        root, active, inactive, inactive2 = fake_root
        token = _touch(inactive, "mcp-tokens", "openai.json")
        assert is_write_denied(str(token)) is True

    def test_inactive_profile_mcp_tokens_dir_denied(self, fake_root):
        """The mcp-tokens directory itself under an inactive profile is blocked."""
        from agent.file_safety import is_write_denied

        root, active, inactive, inactive2 = fake_root
        mcp_dir = inactive / "mcp-tokens"
        mcp_dir.mkdir(parents=True, exist_ok=True)
        assert is_write_denied(str(mcp_dir)) is True

    def test_second_inactive_profile_auth_denied(self, fake_root):
        """auth.json under a second inactive profile is also blocked."""
        from agent.file_safety import is_write_denied

        root, active, inactive, inactive2 = fake_root
        auth = _touch(inactive2, "auth.json")
        assert is_write_denied(str(auth)) is True

    def test_inactive_profile_config_yaml_denied(self, fake_root):
        """config.yaml under an inactive profile is blocked."""
        from agent.file_safety import is_write_denied

        root, active, inactive, inactive2 = fake_root
        config = _touch(inactive, "config.yaml")
        assert is_write_denied(str(config)) is True

    def test_inactive_profile_pairing_denied(self, fake_root):
        """pairing/ under an inactive profile is blocked."""
        from agent.file_safety import is_write_denied

        root, active, inactive, inactive2 = fake_root
        pairing = _touch(inactive, "pairing", "device.json")
        assert is_write_denied(str(pairing)) is True


class TestInactiveProfileNonCredentialAllowed:
    """Non-credential files under inactive profiles should remain writable."""

    def test_inactive_profile_skills_allowed(self, fake_root):
        """skills/ under an inactive profile is NOT blocked by is_write_denied."""
        from agent.file_safety import is_write_denied

        root, active, inactive, inactive2 = fake_root
        skill = _touch(inactive, "skills", "my-skill", "SKILL.md")
        # Skills are not credential files — should not be write-denied
        # (they may be cross-profile-warned, but not hard-blocked)
        assert is_write_denied(str(skill)) is False

    def test_inactive_profile_cron_allowed(self, fake_root):
        """cron/ under an inactive profile is NOT blocked."""
        from agent.file_safety import is_write_denied

        root, active, inactive, inactive2 = fake_root
        cron = _touch(inactive, "cron", "jobs.yaml")
        assert is_write_denied(str(cron)) is False

    def test_inactive_profile_memories_allowed(self, fake_root):
        """memories/ under an inactive profile is NOT blocked."""
        from agent.file_safety import is_write_denied

        root, active, inactive, inactive2 = fake_root
        mem = _touch(inactive, "memories", "notes.md")
        assert is_write_denied(str(mem)) is False


class TestActiveProfileStillProtected:
    """Active profile credential protection must still work after the fix."""

    def test_active_profile_auth_json_denied(self, fake_root):
        """auth.json under the active profile is still blocked."""
        from agent.file_safety import is_write_denied

        root, active, inactive, inactive2 = fake_root
        auth = _touch(active, "auth.json")
        assert is_write_denied(str(auth)) is True

    def test_active_profile_mcp_tokens_denied(self, fake_root):
        """mcp-tokens/ under the active profile is still blocked."""
        from agent.file_safety import is_write_denied

        root, active, inactive, inactive2 = fake_root
        token = _touch(active, "mcp-tokens", "openai.json")
        assert is_write_denied(str(token)) is True

    def test_root_auth_json_denied(self, fake_root):
        """auth.json at the root level is still blocked."""
        from agent.file_safety import is_write_denied

        root, active, inactive, inactive2 = fake_root
        auth = _touch(root, "auth.json")
        assert is_write_denied(str(auth)) is True
