"""Write denylist must cover the same HERMES_HOME credential stores as reads.

Reads already refuse auth.json, webhook HMAC secrets, google_oauth.json,
the Bitwarden plaintext cache, vault/, and browser-profile/. Writes only
blocked .env, the Anthropic PKCE store, and bws_cache.enc.json, so
write_file/patch could replace the rest.

The invariant: every name in ``_CREDENTIAL_FILE_NAMES`` and every directory
in ``_READ_DENIED_DIRS`` is write-denied under both the active home and the
global root. Nested same-basename files (skill mocks) stay writable.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import agent.file_safety as fs


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


def test_every_read_credential_file_is_write_denied(hermes_layout):
    root, profile = hermes_layout
    for name in fs._CREDENTIAL_FILE_NAMES:
        for base in (profile, root):
            path = _touch(base, name)
            assert fs.is_write_denied(str(path)), f"write allowed: {path}"


def test_every_read_denied_dir_is_write_denied(hermes_layout):
    root, profile = hermes_layout
    for sub, _, _ in fs._READ_DENIED_DIRS:
        for base in (profile, root):
            path = _touch(base, f"{sub}/inside.bin")
            assert fs.is_write_denied(str(path)), f"write allowed: {path}"


def test_encrypted_bitwarden_cache_stays_write_denied(hermes_layout):
    root, profile = hermes_layout
    for base in (profile, root):
        path = _touch(base, "cache/bws_cache.enc.json")
        assert fs.is_write_denied(str(path))


def test_config_yaml_stays_writable(hermes_layout):
    _, profile = hermes_layout
    path = _touch(profile, "config.yaml")
    assert fs.is_write_denied(str(path)) is False


def test_nested_auth_json_stays_writable(hermes_layout):
    _, profile = hermes_layout
    path = _touch(profile, "skills/my-skill/auth.json")
    assert fs.is_write_denied(str(path)) is False


def test_auth_json_outside_hermes_home_stays_writable(hermes_layout, tmp_path):
    project = tmp_path / "myproject"
    path = _touch(project, "auth.json")
    assert fs.is_write_denied(str(path)) is False


def test_write_file_does_not_replace_auth_json(hermes_layout):
    _, profile = hermes_layout
    from tools.environments.local import LocalEnvironment
    from tools.file_operations import ShellFileOperations

    target = _touch(profile, "auth.json")
    target.write_text("{\"ok\": true}\n", encoding="utf-8")
    ops = ShellFileOperations(
        LocalEnvironment(cwd=str(profile)), cwd=str(profile)
    )
    res = ops.write_file(str(target), "{\"pwned\": true}\n")
    assert res.error is not None
    assert "protected system/credential file" in res.error
    assert target.read_text(encoding="utf-8") == "{\"ok\": true}\n"


def test_write_file_does_not_replace_vault_key(hermes_layout):
    _, profile = hermes_layout
    from tools.environments.local import LocalEnvironment
    from tools.file_operations import ShellFileOperations

    target = _touch(profile, "vault/vault.key")
    target.write_text("not-a-real-key\n", encoding="utf-8")
    ops = ShellFileOperations(
        LocalEnvironment(cwd=str(profile)), cwd=str(profile)
    )
    res = ops.write_file(str(target), "stolen\n")
    assert res.error is not None
    assert "protected system/credential file" in res.error
    assert target.read_text(encoding="utf-8") == "not-a-real-key\n"
