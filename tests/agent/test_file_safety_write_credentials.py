"""Secret stores under HERMES_HOME are write-denied; control files stay writable (#110464).

``get_read_block_error`` refuses every credential store. The write side is deliberately
narrower — #45947 freed ``auth.json`` / ``config.yaml`` / ``webhook_subscriptions.json`` so
the user can ask to edit them — but the secret *material* it meant to keep blocked had
drifted: ``auth/google_oauth.json``, the plaintext Bitwarden cache, ``vault/`` and
``browser-profile/`` were writable through ``write_file`` / ``patch``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import agent.file_safety as fs
from hermes_constants import get_real_home


def _real_home() -> str:
    """The OS user's real home, however the platform spells it (``pwd`` is POSIX-only)."""
    return str(get_real_home() or Path.home())

SECRET_STORES = (
    "auth/google_oauth.json", "cache/bws_cache.json", "vault/vault.key", "browser-profile/Default/Cookies",
)
WRITABLE_CONTROL_FILES = ("auth.json", "config.yaml", "webhook_subscriptions.json")


@pytest.fixture()
def hermes_layout(tmp_path, monkeypatch):
    """Profile HERMES_HOME plus a distinct global root, both patched."""
    root = tmp_path / "hermes_root"
    profile = root / "profiles" / "coder"
    profile.mkdir(parents=True)
    monkeypatch.setattr(fs, "_hermes_home_path", lambda: profile)
    monkeypatch.setattr(fs, "_hermes_root_path", lambda: root)
    return root, profile


def _touch(base: Path, rel: str) -> Path:
    p = base / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("dummy", encoding="utf-8")
    return p


def test_read_denied_secret_stores_are_write_denied_on_profile_and_root(hermes_layout):
    root, profile = hermes_layout
    for base in (profile, root):
        for rel in SECRET_STORES:
            path = _touch(base, rel)
            assert fs.get_read_block_error(str(path)), f"fixture drift: not read-denied: {path}"
            assert fs.is_write_denied(str(path)), f"write allowed: {path}"


def test_control_files_and_lookalikes_outside_home_stay_writable(hermes_layout, tmp_path):
    root, profile = hermes_layout
    for base in (profile, root):
        for rel in WRITABLE_CONTROL_FILES:
            assert fs.is_write_denied(str(_touch(base, rel))) is False, f"#45947 regression: {rel}"
    assert fs.is_write_denied(str(_touch(tmp_path / "myproject", "cache/bws_cache.json"))) is False
    assert fs.is_write_denied(str(_touch(tmp_path / "myproject", "vault/vault.key"))) is False


class TestProfileHomeProcessHome:
    """With the process HOME pinned to ``{HERMES_HOME}/home`` (TERMINAL_HOME_MODE=profile,
    containers, spawned workers) the write guards must still cover every home a write can
    land in: the OS user's real home, the profile home and ``~name`` accounts."""

    @pytest.fixture()
    def profile_home_env(self, tmp_path, monkeypatch):
        profile = tmp_path / "profile"
        (profile / "home").mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(profile))
        monkeypatch.setenv("HOME", str(profile / "home"))
        monkeypatch.setattr(fs, "_hermes_home_path", lambda: profile)
        monkeypatch.setattr(fs, "_hermes_root_path", lambda: profile.parent)
        return profile

    def test_every_home_is_guarded(self, profile_home_env):
        real_home = Path(_real_home())
        for rel in (".aws/credentials", ".ssh/id_ed25519", ".netrc", ".config/gh/hosts.yml"):
            assert fs.is_write_denied(str(real_home / rel)), rel
            assert fs.is_write_denied(str(profile_home_env / "home" / rel)), rel
        assert fs.is_write_denied("~/.aws/credentials")
        # ``~/.ssh/config`` stays approval-gated (not hard-denied) on the real home too.
        assert fs.is_write_approval_required(str(real_home / ".ssh" / "config"))
        assert fs.is_write_denied(str(real_home / ".ssh" / "config")) is False

    @pytest.mark.skipif(not os.path.isdir(os.path.expanduser("~root")), reason="no ~root account")
    def test_named_account_home_is_guarded(self, profile_home_env):
        assert fs.is_write_denied("~root/.ssh/authorized_keys")

    def test_benign_paths_stay_writable(self, profile_home_env, tmp_path):
        real_home = Path(_real_home())
        for benign in (tmp_path / "scratch" / "notes.txt", real_home / "projects" / "notes.md"):
            assert fs.is_write_denied(str(benign)) is False, benign
            assert fs.is_write_approval_required(str(benign)) is False, benign

    def test_unknown_account_does_not_synthesize_a_guard_home(self, profile_home_env):
        """``ntpath.expanduser`` resolves no account -- it fabricates a sibling of the current
        user's home -- so on Windows every ``~anything`` used to become a guard home built out of
        caller-supplied text, and unrelated paths under it were denied."""
        missing = "~nosuchuser-hopefully"
        assert not os.path.isdir(os.path.expanduser(missing)), "fixture assumption: no such account"
        assert os.path.realpath(os.path.expanduser(missing)) not in fs._guard_homes(f"{missing}/x")
        assert fs.is_write_denied(f"{missing}/.ssh/authorized_keys") is False
