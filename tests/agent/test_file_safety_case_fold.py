"""Deny lists match case variants of the paths they guard.

``resolve()`` / ``realpath()`` keep the spelling the caller typed, and a case-insensitive
filesystem (default macOS APFS, Windows, case-folding mounts) opens ``AUTH.JSON`` as
``auth.json``, so a case-sensitive comparison let a re-cased path through every read and write
deny list. Allow lists (``HERMES_WRITE_SAFE_ROOT``) keep the exact comparison.
"""

from __future__ import annotations

import os
import sys

import pytest

import agent.file_safety as fs
import agent.runtime_self_protection as rsp
import tools.file_tools_write_guards as wg


@pytest.fixture()
def guarded_layout(tmp_path, monkeypatch):
    """Profile HERMES_HOME, a distinct global root and a tmp user home, all patched."""
    root = tmp_path / "hermes_root"
    profile = root / "profiles" / "coder"
    home = tmp_path / "home"
    profile.mkdir(parents=True)
    home.mkdir()
    monkeypatch.setattr(fs, "_hermes_home_path", lambda: profile)
    monkeypatch.setattr(fs, "_hermes_root_path", lambda: root)
    monkeypatch.setattr(fs, "_guard_homes", lambda path="": {os.path.realpath(home)})
    return root, profile, home


@pytest.fixture()
def fake_venv(tmp_path, monkeypatch):
    """The running interpreter's venv, under a symlink-free tmp path (the runtime guard realpaths it)."""
    venv = os.path.join(os.path.realpath(tmp_path), "agent", "venv")
    os.makedirs(os.path.join(venv, "bin"))
    monkeypatch.setattr(sys, "executable", os.path.join(venv, "bin", "python"))
    monkeypatch.setattr(sys, "prefix", venv)
    rsp._protected_snapshot.cache_clear()
    yield venv
    rsp._protected_snapshot.cache_clear()


def test_deny_lists_match_case_variants_and_allow_lists_stay_exact(guarded_layout, fake_venv, tmp_path, monkeypatch):
    root, profile, home = guarded_layout

    for base in (profile, root):
        for rel in ("AUTH.JSON", ".ENV", "MCP-Tokens/server.json", "Vault/vault.key", "SKILLS/.HUB/index.json"):
            assert fs.get_read_block_error(str(base / rel)), f"read allowed: {base / rel}"
        for rel in (".ENV", "Vault/vault.key", "State.db"):
            assert fs.is_write_denied(str(base / rel)), f"write allowed: {base / rel}"
    for rel in (".SSH/authorized_keys", ".Netrc", ".AWS/credentials", ".SSH/Config"):
        assert fs.is_write_denied(str(home / rel)), f"write allowed: ~/{rel}"
    assert fs.is_write_denied(os.path.join(fake_venv.swapcase(), "pyvenv.cfg")), "runtime venv writable"
    monkeypatch.setattr(wg, "_hermes_config_resolved", os.path.realpath(profile / "config.yaml"))
    monkeypatch.setattr(wg, "_hermes_config_resolved_loaded", True)
    assert wg._check_sensitive_path(str(profile / "Config.YAML")), "config.yaml writable"

    # A same-named file below a store is not the store, in any case.
    assert fs.get_read_block_error(str(profile / "notes" / "AUTH.JSON")) is None

    # The safe-root allow list keeps the exact spelling: a re-cased path is outside it.
    safe_root = tmp_path / "Workspace"  # not created, so no host canonicalizes the case
    monkeypatch.setenv("HERMES_WRITE_SAFE_ROOT", str(safe_root))
    assert not fs.is_write_denied(str(safe_root / "out.txt"))
    assert fs.is_write_denied(str(tmp_path / "workspace" / "out.txt"))


@pytest.mark.platforms("macos")
def test_recased_stores_are_refused_on_a_case_insensitive_volume(tmp_path, monkeypatch):
    """End to end on a real case-insensitive volume, where ``AUTH.JSON`` IS ``auth.json``: a skill
    cannot mount it into a sandbox, and a ``MEDIA:`` directive cannot upload it to a chat."""
    import tools.credential_files as cf
    from gateway.platforms.base import validate_media_delivery_path

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "auth.json").write_text('{"providers": {}}', encoding="utf-8")
    if not (hermes_home / "AUTH.JSON").exists():
        pytest.skip("volume is case-sensitive")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("HERMES_MEDIA_DELIVERY_STRICT", raising=False)
    monkeypatch.setattr(cf, "_config_files", {})
    cf.clear_credential_files()
    try:
        assert cf.register_credential_file("AUTH.JSON") is False
        assert cf.get_credential_file_mounts() == []
    finally:
        cf.clear_credential_files()
    assert validate_media_delivery_path(str(hermes_home / "AUTH.JSON")) is None
