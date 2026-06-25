"""Tests for ``hermes auth sync`` — credential drift detection and fix.

These tests exercise the real ``_discover_profile_envs`` /
``_read_env_file`` / ``_upsert_env_key`` path against a temp
``HERMES_HOME`` with a main ``.env`` and multiple profile ``.env``
files, verifying both the dry-run report and the ``--fix`` behaviour.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest


@pytest.fixture
def sync_env(tmp_path, monkeypatch):
    """Set up a HERMES_HOME with a main .env and two profile .env files."""
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True)
    profiles_root = hermes_home / "profiles"

    # Main .env — source of truth
    (hermes_home / ".env").write_text(
        "OPENROUTER_API_KEY=sk-or-fresh-abcd1234\n"
        "DEEPSEEK_API_KEY=sk-ds-fresh-5678\n"
        "# Comment line\n"
    )

    # Profile: worker-1 — stale OPENROUTER key
    p1 = profiles_root / "worker-1"
    p1.mkdir(parents=True)
    (p1 / ".env").write_text(
        "OPENROUTER_API_KEY=sk-or-stale-0000\n"
        "DEEPSEEK_API_KEY=sk-ds-fresh-5678\n"
    )

    # Profile: worker-2 — missing OPENROUTER, stale DEEPSEEK
    p2 = profiles_root / "worker-2"
    p2.mkdir(parents=True)
    (p2 / ".env").write_text(
        "DEEPSEEK_API_KEY=sk-ds-stale-0000\n"
    )

    # Profile: worker-3 — fully in sync
    p3 = profiles_root / "worker-3"
    p3.mkdir(parents=True)
    (p3 / ".env").write_text(
        "OPENROUTER_API_KEY=sk-or-fresh-abcd1234\n"
        "DEEPSEEK_API_KEY=sk-ds-fresh-5678\n"
    )

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    # Patch _get_default_hermes_home and _get_profiles_root to use our temp
    from hermes_cli.profiles import _get_profiles_root, _get_default_hermes_home
    monkeypatch.setattr(
        "hermes_cli.profiles._get_default_hermes_home", lambda: hermes_home
    )
    monkeypatch.setattr(
        "hermes_cli.profiles._get_profiles_root", lambda: hermes_home / "profiles"
    )

    return {
        "hermes_home": hermes_home,
        "main_env": hermes_home / ".env",
        "profiles_root": profiles_root,
        "profiles": {
            "worker-1": p1 / ".env",
            "worker-2": p2 / ".env",
            "worker-3": p3 / ".env",
        },
    }


def test_sync_dry_run_reports_mismatches(capsys, sync_env):
    """Dry-run should report mismatches without fixing them."""
    from hermes_cli.auth_commands import auth_sync_command

    auth_sync_command(SimpleNamespace(fix=False))
    out = capsys.readouterr().out

    assert "OPENROUTER_API_KEY" in out
    assert "DEEPSEEK_API_KEY" in out
    assert "worker-1" in out
    assert "worker-2" in out
    # worker-3 is in sync — should NOT appear
    assert "worker-3" not in out
    # Should mention the --fix hint
    assert "--fix" in out
    # Should NOT have fixed anything
    assert "fixed" not in out.lower() or "mismatch(es) found" in out

    # Verify files were NOT modified
    w1_env = sync_env["profiles"]["worker-1"]
    assert "stale" in w1_env.read_text()


def test_sync_fix_updates_stale_profiles(capsys, sync_env):
    """--fix should update stale profile .env files to match main .env."""
    from hermes_cli.auth_commands import auth_sync_command

    auth_sync_command(SimpleNamespace(fix=True))
    out = capsys.readouterr().out

    assert "fixed" in out

    # worker-1: OPENROUTER should be updated to fresh key
    w1_env = sync_env["profiles"]["worker-1"]
    w1_text = w1_env.read_text()
    assert "sk-or-fresh-abcd1234" in w1_text
    assert "stale" not in w1_text

    # worker-2: both keys should now be present and fresh
    w2_env = sync_env["profiles"]["worker-2"]
    w2_text = w2_env.read_text()
    assert "sk-or-fresh-abcd1234" in w2_text
    assert "sk-ds-fresh-5678" in w2_text
    assert "stale" not in w2_text

    # worker-3: should be unchanged (was already in sync)
    w3_env = sync_env["profiles"]["worker-3"]
    w3_text = w3_env.read_text()
    assert "sk-or-fresh-abcd1234" in w3_text


def test_sync_after_fix_reports_clean(capsys, sync_env):
    """After --fix, a subsequent dry-run should report all in sync."""
    from hermes_cli.auth_commands import auth_sync_command

    # Fix
    auth_sync_command(SimpleNamespace(fix=True))
    capsys.readouterr()  # clear

    # Dry-run
    auth_sync_command(SimpleNamespace(fix=False))
    out = capsys.readouterr().out

    assert "All profile .env files are in sync" in out


def test_sync_preserves_file_permissions(capsys, sync_env):
    """--fix should preserve the original file permissions."""
    from hermes_cli.auth_commands import auth_sync_command
    import stat

    w1_env = sync_env["profiles"]["worker-1"]
    os.chmod(str(w1_env), 0o600)
    original_mode = stat.S_IMODE(w1_env.stat().st_mode)

    auth_sync_command(SimpleNamespace(fix=True))

    new_mode = stat.S_IMODE(w1_env.stat().st_mode)
    assert new_mode == original_mode


def test_sync_no_profiles(tmp_path, monkeypatch, capsys):
    """Should handle gracefully when no .env files exist."""
    hermes_home = tmp_path / "empty"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(
        "hermes_cli.profiles._get_default_hermes_home", lambda: hermes_home
    )
    monkeypatch.setattr(
        "hermes_cli.profiles._get_profiles_root", lambda: hermes_home / "profiles"
    )

    from hermes_cli.auth_commands import auth_sync_command

    auth_sync_command(SimpleNamespace(fix=False))
    out = capsys.readouterr().out

    assert "No .env files found" in out


def test_sync_empty_main_env(tmp_path, monkeypatch, capsys):
    """Should handle gracefully when main .env is empty."""
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / ".env").write_text("# only comments\n")

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(
        "hermes_cli.profiles._get_default_hermes_home", lambda: hermes_home
    )
    monkeypatch.setattr(
        "hermes_cli.profiles._get_profiles_root", lambda: hermes_home / "profiles"
    )

    from hermes_cli.auth_commands import auth_sync_command

    auth_sync_command(SimpleNamespace(fix=False))
    out = capsys.readouterr().out

    assert "empty or missing" in out


def test_mask_key():
    """Test key masking for display."""
    from hermes_cli.auth_commands import _mask_key

    assert _mask_key("") == "(empty)"
    assert _mask_key("short") == "sh…rt"
    assert _mask_key("sk-or-fresh-abcd1234") == "sk-o…1234"
    assert _mask_key("exactly12ch") == "ex…ch"
