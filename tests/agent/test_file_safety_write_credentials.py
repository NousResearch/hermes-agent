"""Secret stores under HERMES_HOME are write-denied; control files stay writable (#110464).

``get_read_block_error`` refuses every credential store. The write side is deliberately
narrower — #45947 freed ``auth.json`` / ``config.yaml`` / ``webhook_subscriptions.json`` so
the user can ask to edit them — but the secret *material* it meant to keep blocked had
drifted: ``auth/google_oauth.json``, the plaintext Bitwarden cache, ``vault/`` and
``browser-profile/`` were writable through ``write_file`` / ``patch``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import agent.file_safety as fs

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
    """The write guard must cover the OS user's real home even when the process HOME is
    pinned to ``{HERMES_HOME}/home`` (TERMINAL_HOME_MODE=profile / container / spawned
    worker). Anchoring the deny and approval lists on ``expanduser("~")`` alone left the
    real home's credential paths writable via absolute paths — while file tools resolve
    ``~`` through ``get_subprocess_home()`` and can land a ``~``-spelled write on the real
    home too."""

    @pytest.fixture()
    def profile_home_env(self, tmp_path, monkeypatch):
        profile = tmp_path / "profile"
        (profile / "home").mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(profile))
        monkeypatch.setenv("HOME", str(profile / "home"))
        monkeypatch.setattr(fs, "_hermes_home_path", lambda: profile)
        monkeypatch.setattr(fs, "_hermes_root_path", lambda: profile.parent)
        return profile

    def test_real_home_credentials_denied(self, profile_home_env):
        import pwd

        real_home = Path(pwd.getpwuid(__import__("os").getuid()).pw_dir)
        for rel in (".aws/credentials", ".ssh/id_ed25519", ".netrc", ".config/gh/hosts.yml"):
            assert fs.is_write_denied(str(real_home / rel)), rel

    def test_real_home_ssh_config_still_approval_gated(self, profile_home_env):
        import pwd

        real_home = Path(pwd.getpwuid(__import__("os").getuid()).pw_dir)
        assert fs.is_write_approval_required(str(real_home / ".ssh" / "config"))

    def test_profile_home_credentials_also_denied(self, profile_home_env):
        """A pinned-HOME process still guards the profile home children see as ``~``."""
        for rel in (".aws/credentials", ".ssh/id_rsa"):
            assert fs.is_write_denied(str(profile_home_env / "home" / rel)), rel

    def test_tilde_spelling_still_gated(self, profile_home_env):
        assert fs.is_write_denied("~/.aws/credentials")
        assert fs.is_write_approval_required("~/.ssh/config")

    def test_benign_path_unaffected(self, profile_home_env, tmp_path):
        benign = tmp_path / "scratch" / "notes.txt"
        assert fs.is_write_denied(str(benign)) is False
        assert fs.is_write_approval_required(str(benign)) is False
