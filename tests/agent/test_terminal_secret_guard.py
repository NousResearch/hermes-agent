"""Tests for the terminal-tool secret-read guard in file_safety.

``get_read_block_error`` blocks ``read_file`` from returning credential
stores (auth.json, .env, mcp-tokens/, ...), but its own docstring admits the
``terminal`` tool can trivially reach the same files with ``cat``/``grep``/
``source`` — nothing cross-checked terminal commands against that denylist.
``get_terminal_secret_access_error`` closes the literal-path/bare-basename
case of that gap by tokenizing the command and deferring to
``get_read_block_error`` for the actual verdict, so both guards share one
source of truth.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture()
def fake_home(tmp_path, monkeypatch):
    import agent.file_safety as fs

    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setattr(fs, "_hermes_home_path", lambda: home)
    return home


def _create(home: Path, rel: str | Path) -> Path:
    p = home / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("dummy", encoding="utf-8")
    return p


def test_absolute_path_to_env_file_blocked(fake_home):
    """.env under HERMES_HOME matches the credential-store denylist entry
    (checked before the anywhere-on-disk env-basename check), so either
    denial wording is a correct outcome here — only the block itself, and
    that the path is echoed, are asserted."""
    from agent.file_safety import get_terminal_secret_access_error

    env = _create(fake_home, ".env")
    err = get_terminal_secret_access_error(f'grep "TOKEN=" {env} | head -3')
    assert err is not None
    assert str(env) in err


def test_project_local_env_file_outside_hermes_home_blocked(fake_home, tmp_path):
    """.env anywhere on disk (not just under HERMES_HOME) is blocked too —
    exercises the anywhere-on-disk basename branch specifically."""
    from agent.file_safety import get_terminal_secret_access_error

    project = tmp_path / "someproject"
    project.mkdir()
    env = project / ".env"
    env.write_text("API_KEY=x", encoding="utf-8")
    err = get_terminal_secret_access_error(f"cat {env}")
    assert err is not None
    assert "secret-bearing environment file" in err


def test_source_command_blocked(fake_home):
    from agent.file_safety import get_terminal_secret_access_error

    env = _create(fake_home, ".env")
    err = get_terminal_secret_access_error(f"source {env} && echo done")
    assert err is not None


def test_bare_basename_in_cwd_blocked(fake_home):
    """``cat .env`` with no path separator — resolved against ``cwd``."""
    from agent.file_safety import get_terminal_secret_access_error

    _create(fake_home, ".env")
    err = get_terminal_secret_access_error("cat .env", cwd=str(fake_home))
    assert err is not None


def test_auth_json_path_blocked(fake_home):
    from agent.file_safety import get_terminal_secret_access_error

    auth = _create(fake_home, "auth.json")
    err = get_terminal_secret_access_error(f"cat {auth}")
    assert err is not None
    assert "credential store" in err


def test_tilde_path_blocked(fake_home, monkeypatch):
    """``~/.hermes/.env``-style tokens must expand before resolution."""
    from agent.file_safety import get_terminal_secret_access_error

    _create(fake_home, ".env")
    monkeypatch.setenv("HOME", str(fake_home.parent))
    fake_home.parent.joinpath(".hermes").symlink_to(fake_home, target_is_directory=True)
    err = get_terminal_secret_access_error("cat ~/.hermes/.env")
    assert err is not None


def test_unrelated_command_not_blocked(fake_home):
    from agent.file_safety import get_terminal_secret_access_error

    assert get_terminal_secret_access_error("ls -la /tmp") is None
    assert get_terminal_secret_access_error("grep TOKEN config.yaml") is None


def test_media_delivery_path_not_blocked(fake_home):
    """Regression guard: ordinary cache-dir file operations (the guest-media
    delivery flow) must not be caught by the secret-file heuristic."""
    from agent.file_safety import get_terminal_secret_access_error

    video = fake_home / "cache" / "videos" / "clip.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"x")
    assert get_terminal_secret_access_error(f"ls -la {video}") is None


def test_non_string_command_returns_none():
    from agent.file_safety import get_terminal_secret_access_error

    assert get_terminal_secret_access_error(None) is None  # type: ignore[arg-type]
    assert get_terminal_secret_access_error("") is None


def test_unbalanced_quotes_does_not_raise():
    """shlex.split can raise ValueError on unbalanced quotes — must degrade
    to "not blocked" rather than propagate."""
    from agent.file_safety import get_terminal_secret_access_error

    assert get_terminal_secret_access_error("echo 'unterminated") is None
