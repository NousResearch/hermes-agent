"""Secret stores under HERMES_HOME must be write-denied, control files must not.

``is_read_denied`` refuses every credential store. The write side is
deliberately narrower: #45947 freed ``auth.json``, ``config.yaml`` and
``webhook_subscriptions.json`` on the grounds that containment belongs in
Docker/remote backends and OS permissions rather than an expanding denylist.

What #45947 kept blocked is secret *material*, and that list had drifted:
``auth/google_oauth.json`` (an OAuth token store), the plaintext Bitwarden
cache, ``vault/`` (key + ciphertext side by side) and ``browser-profile/``
(copied cookies / Login Data) were writable through ``write_file`` / ``patch``.

These tests pin both halves so neither can drift again.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import agent.file_safety as fs

# Secret material: read-denied AND write-denied.
SECRET_FILES = (
    ".env",
    ".anthropic_oauth.json",
    "auth/google_oauth.json",
    "cache/bws_cache.json",
    "cache/bws_cache.enc.json",
)

# Read-denied control files that #45947 deliberately left writable.
WRITABLE_CONTROL_FILES = ("auth.json", "auth.lock", "config.yaml", "webhook_subscriptions.json")


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


@pytest.mark.parametrize("name", SECRET_FILES)
def test_secret_files_are_write_denied(hermes_layout, name):
    root, profile = hermes_layout
    for base in (profile, root):
        path = _touch(base, name)
        assert fs.is_write_denied(str(path)), f"write allowed: {path}"


@pytest.mark.parametrize("sub", fs._WRITE_DENIED_SECRET_DIRS)
def test_secret_dirs_are_write_denied(hermes_layout, sub):
    root, profile = hermes_layout
    for base in (profile, root):
        path = _touch(base, f"{sub}/inside.bin")
        assert fs.is_write_denied(str(path)), f"write allowed: {path}"


@pytest.mark.parametrize("name", WRITABLE_CONTROL_FILES)
def test_control_files_stay_writable(hermes_layout, name):
    """#45947 freed these on purpose; re-blocking them is a policy regression."""
    root, profile = hermes_layout
    for base in (profile, root):
        path = _touch(base, name)
        assert fs.is_write_denied(str(path)) is False, f"write denied: {path}"


def test_every_write_denied_secret_is_read_denied(hermes_layout):
    """Write denies are a subset of read denies — never a superset."""
    _, profile = hermes_layout
    for name in SECRET_FILES:
        if name == "cache/bws_cache.enc.json":
            continue  # encrypted sibling: write-only extra, plaintext is the read-denied one
        path = _touch(profile, name)
        assert fs.get_read_block_error(str(path)) is not None, f"not read-denied: {name}"


def test_nested_secret_basename_stays_writable(hermes_layout):
    _, profile = hermes_layout
    path = _touch(profile, "skills/my-skill/.env.example")
    assert fs.is_write_denied(str(path)) is False


def test_secret_outside_hermes_home_is_not_denied_by_this_rule(hermes_layout, tmp_path):
    project = tmp_path / "myproject"
    path = _touch(project, "cache/bws_cache.json")
    assert fs.is_write_denied(str(path)) is False


def test_write_file_does_not_replace_vault_key(hermes_layout):
    _, profile = hermes_layout
    from tools.environments.local import LocalEnvironment
    from tools.file_operations import ShellFileOperations

    target = _touch(profile, "vault/vault.key")
    target.write_text("not-a-real-key\n", encoding="utf-8")
    ops = ShellFileOperations(LocalEnvironment(cwd=str(profile)), cwd=str(profile))
    res = ops.write_file(str(target), "stolen\n")
    assert res.error is not None
    assert "protected system/credential file" in res.error
    assert target.read_text(encoding="utf-8") == "not-a-real-key\n"


def test_write_file_does_not_replace_google_oauth(hermes_layout):
    _, profile = hermes_layout
    from tools.environments.local import LocalEnvironment
    from tools.file_operations import ShellFileOperations

    target = _touch(profile, "auth/google_oauth.json")
    target.write_text('{"ok": true}\n', encoding="utf-8")
    ops = ShellFileOperations(LocalEnvironment(cwd=str(profile)), cwd=str(profile))
    res = ops.write_file(str(target), '{"pwned": true}\n')
    assert res.error is not None
    assert "protected system/credential file" in res.error
    assert target.read_text(encoding="utf-8") == '{"ok": true}\n'
