"""G2 (S-0429-01) — _is_read_denied blocks cross-user artemis reads.

Closes audit M-3: prompt injection via uploaded resume / pasted text could
have asked the agent to ``read_file('~/.hermes/artemis/<B>/strategy.json')``.
The static deny list in ``_is_read_denied`` covered ssh / aws / etc but had
no per-user sandbox.

When ``HERMES_SESSION_USER_ID`` is set, deny any path under
``~/.hermes/artemis/<other_user>/`` and any path under ``~/.hermes/memories/``
(global file by design — never injectable).

When env is unset (CLI / test harness), fall through to existing static
deny list. This preserves dev ergonomics: someone running the CLI to
inspect another user's strategy for debugging is not the threat model.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "artemis" / "U0ALICE").mkdir(parents=True)
    (tmp_path / "artemis" / "U0ALICE" / "profile.json").write_text("{}")
    (tmp_path / "artemis" / "U0BOB").mkdir(parents=True)
    (tmp_path / "artemis" / "U0BOB" / "profile.json").write_text("{}")
    (tmp_path / "artemis" / "U0BOB" / "strategy.json").write_text("{}")
    (tmp_path / "artemis" / "U0BOB" / "inbox").mkdir()
    (tmp_path / "artemis" / "U0BOB" / "inbox" / "secret.md").write_text("x")
    (tmp_path / "memories").mkdir()
    (tmp_path / "memories" / "USER.md").write_text("")
    (tmp_path / "skills" / "test-skill").mkdir(parents=True)
    (tmp_path / "skills" / "test-skill" / "SKILL.md").write_text("# skill")
    return tmp_path


class TestSessionBoundUser:
    def test_blocks_sibling_user_profile(self, hermes_home, monkeypatch):
        monkeypatch.setenv("HERMES_SESSION_USER_ID", "U0ALICE")
        from tools.file_operations import _is_read_denied

        assert _is_read_denied(
            str(hermes_home / "artemis" / "U0BOB" / "profile.json")
        )

    def test_blocks_sibling_user_strategy(self, hermes_home, monkeypatch):
        monkeypatch.setenv("HERMES_SESSION_USER_ID", "U0ALICE")
        from tools.file_operations import _is_read_denied

        assert _is_read_denied(
            str(hermes_home / "artemis" / "U0BOB" / "strategy.json")
        )

    def test_blocks_sibling_user_inbox(self, hermes_home, monkeypatch):
        monkeypatch.setenv("HERMES_SESSION_USER_ID", "U0ALICE")
        from tools.file_operations import _is_read_denied

        assert _is_read_denied(
            str(hermes_home / "artemis" / "U0BOB" / "inbox" / "secret.md")
        )

    def test_allows_own_user_profile(self, hermes_home, monkeypatch):
        monkeypatch.setenv("HERMES_SESSION_USER_ID", "U0ALICE")
        from tools.file_operations import _is_read_denied

        assert not _is_read_denied(
            str(hermes_home / "artemis" / "U0ALICE" / "profile.json")
        )

    def test_blocks_memories_unconditionally(self, hermes_home, monkeypatch):
        monkeypatch.setenv("HERMES_SESSION_USER_ID", "U0ALICE")
        from tools.file_operations import _is_read_denied

        assert _is_read_denied(str(hermes_home / "memories" / "USER.md"))
        # Even subpaths under memories/ — defense in depth in case someone
        # later writes per-user files there.
        assert _is_read_denied(str(hermes_home / "memories" / "anything.md"))

    def test_allows_skills_dir(self, hermes_home, monkeypatch):
        """Hermes hub skills are read freely by every session — they're
        not user-scoped data."""
        monkeypatch.setenv("HERMES_SESSION_USER_ID", "U0ALICE")
        from tools.file_operations import _is_read_denied

        assert not _is_read_denied(
            str(hermes_home / "skills" / "test-skill" / "SKILL.md")
        )


class TestNoSessionBound:
    """When env is unset (CLI debug, test harness without monkeypatch),
    keep existing static deny list only. Don't add the dynamic check —
    legitimate dev workflows like ``hermes`` CLI inspecting another
    user's strategy for debugging shouldn't be blocked."""

    def test_falls_through_to_static_deny(self, hermes_home, monkeypatch):
        monkeypatch.delenv("HERMES_SESSION_USER_ID", raising=False)
        from tools.file_operations import _is_read_denied

        # Sibling-user paths allowed when no session is bound
        assert not _is_read_denied(
            str(hermes_home / "artemis" / "U0BOB" / "profile.json")
        )
        # memories/ also allowed in CLI mode (no user identity to defend)
        assert not _is_read_denied(str(hermes_home / "memories" / "USER.md"))
        # Static deny list still works
        assert _is_read_denied(str(Path.home() / ".ssh" / "id_ed25519"))


class TestSymlinkBypass:
    """Symlink-based bypass: an attacker plants a symlink in their own
    user dir pointing at a sibling's file, then asks the agent to read
    the symlink. realpath-based check must resolve the link before the
    cross-user comparison."""

    def test_symlink_to_sibling_user_resolved_and_blocked(
        self, hermes_home, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("HERMES_SESSION_USER_ID", "U0ALICE")
        from tools.file_operations import _is_read_denied

        link = tmp_path / "evil-link"
        link.symlink_to(hermes_home / "artemis" / "U0BOB" / "profile.json")
        assert _is_read_denied(str(link))


class TestStaticDenyStillWorks:
    """G2 layers on top of the existing deny list — make sure adding
    the cross-user rule didn't break the static credential checks."""

    def test_ssh_key_still_blocked(self, hermes_home, monkeypatch):
        monkeypatch.setenv("HERMES_SESSION_USER_ID", "U0ALICE")
        from tools.file_operations import _is_read_denied

        assert _is_read_denied(str(Path.home() / ".ssh" / "id_ed25519"))

    def test_aws_dir_still_blocked(self, hermes_home, monkeypatch):
        monkeypatch.setenv("HERMES_SESSION_USER_ID", "U0ALICE")
        from tools.file_operations import _is_read_denied

        assert _is_read_denied(str(Path.home() / ".aws" / "credentials"))
