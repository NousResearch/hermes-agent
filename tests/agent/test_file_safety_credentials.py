"""Tests for HERMES_HOME credential-file read blocking in file_safety.

Regression for https://github.com/NousResearch/hermes-agent/issues/17656 —
``read_file`` was previously only sandboxed against ``HERMES_HOME`` itself,
which left ``auth.json`` and ``.anthropic_oauth.json`` (plaintext provider
keys + OAuth tokens) readable by the agent. A prompt-injection reaching
``read_file`` could exfiltrate active credentials.

These tests verify that ``get_read_block_error`` returns a denial message
for the credential stores while leaving arbitrary ``HERMES_HOME`` files
readable, and that the existing ``skills/.hub`` deny still applies.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture()
def fake_home(tmp_path, monkeypatch):
    """Point ``_hermes_home_path()`` at a tmp dir for isolated checks."""
    import agent.file_safety as fs

    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setattr(fs, "_hermes_home_path", lambda: home)
    return home


def _create(home: Path, rel: str | Path) -> Path:
    """Create the file (with parents) so realpath() resolves it."""
    p = home / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("dummy", encoding="utf-8")
    return p


def test_auth_json_blocked(fake_home):
    from agent.file_safety import get_read_block_error

    auth = _create(fake_home, "auth.json")
    err = get_read_block_error(str(auth))
    assert err is not None
    assert "credential store" in err
    assert "auth.json" in err


def test_auth_lock_blocked(fake_home):
    from agent.file_safety import get_read_block_error

    lock = _create(fake_home, "auth.lock")
    err = get_read_block_error(str(lock))
    assert err is not None
    assert "credential store" in err


def test_anthropic_oauth_json_blocked(fake_home):
    from agent.file_safety import get_read_block_error

    oauth = _create(fake_home, ".anthropic_oauth.json")
    err = get_read_block_error(str(oauth))
    assert err is not None
    assert "credential store" in err


def test_arbitrary_hermes_home_file_not_blocked(fake_home):
    """Non-credential files inside HERMES_HOME stay readable."""
    from agent.file_safety import get_read_block_error

    safe = _create(fake_home, "session_log.txt")
    assert get_read_block_error(str(safe)) is None


def test_subdirectory_named_auth_json_not_blocked(fake_home):
    """Only the top-level auth.json is the credential store; a file with the
    same name in a subdirectory (e.g., a skill mock) must remain readable."""
    from agent.file_safety import get_read_block_error

    nested = _create(fake_home, Path("skills") / "my-skill" / "auth.json")
    assert get_read_block_error(str(nested)) is None


def test_skills_hub_block_still_applies(fake_home):
    """Regression guard: the original skills/.hub deny must keep working."""
    from agent.file_safety import get_read_block_error

    hub_file = _create(fake_home, "skills/.hub/manifest.json")
    err = get_read_block_error(str(hub_file))
    assert err is not None
    assert "internal Hermes cache file" in err


def test_path_traversal_resolves_to_blocked(fake_home, tmp_path):
    """A path that traverses through a sibling dir back into HERMES_HOME's
    auth.json must still be caught — the check resolves through realpath."""
    from agent.file_safety import get_read_block_error

    _create(fake_home, "auth.json")
    sibling = tmp_path / "elsewhere"
    sibling.mkdir()
    traversal = sibling / ".." / "hermes_home" / "auth.json"
    err = get_read_block_error(str(traversal))
    assert err is not None
    assert "credential store" in err


def test_symlink_to_auth_json_blocked(fake_home, tmp_path):
    """A symlink pointing at HERMES_HOME/auth.json from outside the home
    must be blocked — readlink-resolution catches the indirection."""
    from agent.file_safety import get_read_block_error

    target = _create(fake_home, "auth.json")
    link = tmp_path / "shim.json"
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported on this platform/filesystem")
    err = get_read_block_error(str(link))
    assert err is not None
    assert "credential store" in err


def test_read_file_tool_blocks_relative_path_under_terminal_cwd(
    fake_home, tmp_path, monkeypatch
):
    """Bypass guard: a relative path like ``"auth.json"`` resolved by
    ``read_file_tool`` against ``TERMINAL_CWD == HERMES_HOME`` must still
    be blocked, even though ``get_read_block_error``'s own ``resolve()``
    is anchored at the (different) Python process cwd.
    """
    import json

    import tools.file_tools as ft

    _create(fake_home, "auth.json")
    # Force the file_tools resolver to anchor relative paths at HERMES_HOME
    # while the Python process cwd remains tmp_path (a different directory).
    monkeypatch.setenv("TERMINAL_CWD", str(fake_home))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        ft, "_get_live_tracking_cwd", lambda task_id="default": None
    )

    out = json.loads(ft.read_file_tool("auth.json"))
    assert "error" in out
    assert "credential store" in out["error"]
