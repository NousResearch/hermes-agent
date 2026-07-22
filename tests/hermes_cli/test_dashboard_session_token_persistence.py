"""Tests for persistent dashboard session token loader."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli import _dashboard_session_token as session_token


@pytest.fixture
def home(tmp_path):
    """An isolated HERMES_HOME used as the persistence root for one test."""
    home = tmp_path / "hermes-home"
    home.mkdir()
    return home


def test_env_var_wins_over_persisted_file(home, monkeypatch):
    """HERMES_DASHBOARD_SESSION_TOKEN overrides anything on disk."""
    monkeypatch.setenv(session_token.ENV_VAR, "operator-injected")
    (home / "state").mkdir()
    (home / "state" / "dashboard_session_token").write_text("STALE-FILE")

    assert session_token.load_or_create(home_path=home) == "operator-injected"
    # File untouched
    assert (home / "state" / "dashboard_session_token").read_text() == "STALE-FILE"


def test_first_run_creates_token_file(home, monkeypatch):
    """No env var, no file -> fresh token returned AND persisted."""
    monkeypatch.delenv(session_token.ENV_VAR, raising=False)

    token_path = home / "state" / "dashboard_session_token"
    assert not token_path.exists()

    token = session_token.load_or_create(home_path=home)
    assert token
    assert len(token) >= 32  # secrets.token_urlsafe(32) is ~43 chars

    assert token_path.exists()
    assert token_path.read_text(encoding="utf-8").strip() == token

    # POSIX: group/other cannot read. Skip on Windows.
    if os.name == "posix":
        mode = token_path.stat().st_mode & 0o777
        assert mode & 0o077 == 0  # Not readable by group or other


def test_persisted_file_reused_across_calls(home, monkeypatch):
    """Same file -> same token across many invocations (no rotation on read)."""
    monkeypatch.delenv(session_token.ENV_VAR, raising=False)

    t1 = session_token.load_or_create(home_path=home)
    t2 = session_token.load_or_create(home_path=home)
    t3 = session_token.load_or_create(home_path=home)
    assert t1 == t2 == t3


def test_existing_file_reused(home, monkeypatch):
    """A non-empty token file is reused as-is on next startup."""
    monkeypatch.delenv(session_token.ENV_VAR, raising=False)
    (home / "state").mkdir()
    (home / "state" / "dashboard_session_token").write_text(
        "stable-token-from-earlier-run"
    )

    got = session_token.load_or_create(home_path=home)
    assert got == "stable-token-from-earlier-run"


def test_whitespace_only_file_is_replaced(home, monkeypatch):
    """Empty/whitespace content treated as missing: file replaced + new token."""
    monkeypatch.delenv(session_token.ENV_VAR, raising=False)
    (home / "state").mkdir(parents=True)
    (home / "state" / "dashboard_session_token").write_text("   \n  \n")

    new_token = session_token.load_or_create(home_path=home)
    assert new_token and new_token.strip() and len(new_token) > 8
    assert (
        home / "state" / "dashboard_session_token"
    ).read_text(encoding="utf-8").strip() == new_token


def test_read_failure_falls_through_to_generate(home, monkeypatch):
    """If the file can't be read (OSError), a fresh token is still produced."""
    monkeypatch.delenv(session_token.ENV_VAR, raising=False)

    real_read_text = Path.read_text

    def _explode_on_read(self, *a, **kw):
        if self.name == "dashboard_session_token":
            raise PermissionError("simulated EACCES")
        return real_read_text(self, *a, **kw)

    with patch.object(Path, "read_text", _explode_on_read):
        token = session_token.load_or_create(home_path=home)
    assert token and len(token) > 8


def test_write_failure_returns_in_memory_token(home, monkeypatch):
    """Best-effort: file can't be written -> in-memory token still returned."""
    monkeypatch.delenv(session_token.ENV_VAR, raising=False)

    real_mkdir = Path.mkdir

    def _fail_on_state(self, *a, **kw):
        if self.name == "state" or (self.parts and self.parts[-1] == "state"):
            raise OSError("EACCES - simulated read-only HOME", 13)
        return real_mkdir(self, *a, **kw)

    with patch.object(Path, "mkdir", _fail_on_state):
        token = session_token.load_or_create(home_path=home)
    # In-memory token still produced
    assert token and len(token) > 8


def test_get_hermes_home_fetch_failure_returns_random_token(home, monkeypatch):
    """If get_hermes_home() raises, we fall back to an in-memory token."""
    monkeypatch.delenv(session_token.ENV_VAR, raising=False)

    with patch(
        "hermes_cli.config.get_hermes_home",
        side_effect=RuntimeError("config broken"),
    ):
        token = session_token.load_or_create(home_path=None)
    assert token and len(token) > 8


def test_default_home_path_resolution(monkeypatch, tmp_path):
    """When home_path=None, the helper uses HERMES_HOME via get_hermes_home."""
    monkeypatch.delenv(session_token.ENV_VAR, raising=False)
    # Point HERMES_HOME at a tmp dir
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    token = session_token.load_or_create(home_path=None)
    persisted = (tmp_path / "state" / "dashboard_session_token").read_text(
        encoding="utf-8"
    ).strip()
    assert persisted == token