"""Tests for hermes_cli.session_index — sessions.json indexing for
standalone CLI sessions (issue #29073).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Point HERMES_HOME at a tmp dir and force re-import of the module
    so its ``get_hermes_home()`` cache picks up the new value.
    """
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    # The module captures get_hermes_home at import time inside the
    # function body, so just re-import to be safe in case of caching.
    import importlib
    import hermes_cli.session_index as si
    importlib.reload(si)
    return home


def _read_index(home: Path) -> dict:
    p = home / "sessions" / "sessions.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text("utf-8"))


def test_index_creates_sessions_json(hermes_home):
    from hermes_cli.session_index import index_cli_session

    assert index_cli_session("20260520_120000_abc123") is True

    data = _read_index(hermes_home)
    assert "cli:20260520_120000_abc123" in data
    entry = data["cli:20260520_120000_abc123"]
    assert entry["session_id"] == "20260520_120000_abc123"
    assert entry["platform"] == "local"
    assert entry["chat_type"] == "dm"
    assert entry["created_at"]
    assert entry["updated_at"]


def test_index_is_idempotent_preserves_created_at(hermes_home):
    """Re-indexing the same session refreshes updated_at but keeps the
    original created_at -- regression guard for #29073.
    """
    import time
    from hermes_cli.session_index import index_cli_session

    index_cli_session("sid1")
    first = _read_index(hermes_home)["cli:sid1"]
    time.sleep(0.01)
    index_cli_session("sid1")
    second = _read_index(hermes_home)["cli:sid1"]

    assert second["created_at"] == first["created_at"]
    assert second["updated_at"] >= first["updated_at"]


def test_index_preserves_other_entries(hermes_home):
    """Upserting a CLI entry must not clobber unrelated gateway entries
    (telegram, discord, ...) already in sessions.json.
    """
    from hermes_cli.session_index import index_cli_session

    sessions_dir = hermes_home / "sessions"
    sessions_dir.mkdir()
    path = sessions_dir / "sessions.json"
    path.write_text(json.dumps({
        "agent:main:telegram:dm": {
            "session_key": "agent:main:telegram:dm",
            "session_id": "tg_sid",
            "created_at": "2026-01-01T00:00:00",
            "updated_at": "2026-01-01T00:00:00",
            "platform": "telegram",
            "chat_type": "dm",
        }
    }))

    index_cli_session("cli_sid")
    data = _read_index(hermes_home)
    assert "agent:main:telegram:dm" in data
    assert data["agent:main:telegram:dm"]["session_id"] == "tg_sid"
    assert "cli:cli_sid" in data


def test_index_returns_false_on_empty_session_id(hermes_home):
    from hermes_cli.session_index import index_cli_session
    assert index_cli_session("") is False
    assert index_cli_session(None) is False  # type: ignore[arg-type]
    # No file created
    assert not (hermes_home / "sessions" / "sessions.json").exists()


def test_index_swallows_write_errors(hermes_home, monkeypatch):
    """A broken disk must not break the live CLI session."""
    from hermes_cli import session_index

    def _boom(*_a, **_k):
        raise OSError("simulated")

    monkeypatch.setattr(session_index, "_atomic_write", _boom)
    # Returns False but does not raise.
    assert session_index.index_cli_session("sid_err") is False
