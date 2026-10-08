"""Tests for hermes_cli/terminal_breadcrumbs.py — per-terminal ``hermes -c``.

Covers terminal id derivation (tty vs env vars vs none), breadcrumb
write/read roundtrip under a temp HERMES_HOME, stale-session fallback
(breadcrumb pointing at a deleted session), compression-tip projection,
and the session.terminal_continue config gate.
"""

import json
import os
import time
from pathlib import Path

import pytest

from hermes_cli import terminal_breadcrumbs as tb

pytestmark = pytest.mark.platforms("linux")  # os.ttyname is POSIX-only

TERMINAL_ENV_VARS = (
    "ZELLIJ_PANE_ID",
    "TMUX_PANE",
    "KITTY_WINDOW_ID",
    "WEZTERM_PANE",
    "TERM_SESSION_ID",
    "WT_SESSION",
)

@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home

@pytest.fixture
def no_terminal_env(monkeypatch):
    """Strip every terminal-identity env var so tests control identity."""
    for var in TERMINAL_ENV_VARS:
        monkeypatch.delenv(var, raising=False)

def _fake_no_tty(monkeypatch):
    monkeypatch.setattr(tb.os, "ttyname", lambda fd: (_ for _ in ()).throw(OSError()))

def _fake_tty(monkeypatch, name="/dev/pts/7"):
    monkeypatch.setattr(tb.os, "ttyname", lambda fd: name)

# ---------------------------------------------------------------- identity

def test_terminal_id_prefers_tty(monkeypatch, no_terminal_env):
    _fake_tty(monkeypatch, "/dev/pts/7")
    monkeypatch.setenv("TMUX_PANE", "%3")
    assert tb.get_terminal_id() == "tty-dev-pts-7"

def test_terminal_id_env_var_order(monkeypatch, no_terminal_env):
    _fake_no_tty(monkeypatch)
    monkeypatch.setenv("KITTY_WINDOW_ID", "12")
    monkeypatch.setenv("WT_SESSION", "abc-123")
    # KITTY_WINDOW_ID comes before WT_SESSION in the preference order
    assert tb.get_terminal_id() == "kitty_window_id-12"

def test_terminal_id_sanitizes_env_value(monkeypatch, no_terminal_env):
    _fake_no_tty(monkeypatch)
    monkeypatch.setenv("TMUX_PANE", "%41")
    tid = tb.get_terminal_id()
    assert tid is not None
    assert "/" not in tid and "%" not in tid

def test_terminal_id_none_when_no_identity(monkeypatch, no_terminal_env):
    _fake_no_tty(monkeypatch)
    assert tb.get_terminal_id() is None

# ---------------------------------------------------------- write / read

def test_breadcrumb_roundtrip(hermes_home, monkeypatch, no_terminal_env):
    _fake_tty(monkeypatch)
    tb.write_breadcrumb("20260815_120000_abc123", cwd="/tmp/project")
    crumb = tb.read_breadcrumb()
    assert crumb is not None
    assert crumb["session_id"] == "20260815_120000_abc123"
    assert crumb["cwd"] == "/tmp/project"
    assert isinstance(crumb["ts"], float)
    files = list((hermes_home / "terminal-sessions").iterdir())
    assert [f.name for f in files] == ["tty-dev-pts-7"]

def test_write_skipped_without_terminal_identity(hermes_home, monkeypatch, no_terminal_env):
    _fake_no_tty(monkeypatch)
    tb.write_breadcrumb("20260815_120000_abc123")
    assert not (hermes_home / "terminal-sessions").exists()
    assert tb.read_breadcrumb() is None

def test_two_terminals_do_not_clobber(hermes_home, monkeypatch, no_terminal_env):
    _fake_tty(monkeypatch, "/dev/pts/1")
    tb.write_breadcrumb("session-one")
    _fake_tty(monkeypatch, "/dev/pts/2")
    tb.write_breadcrumb("session-two")
    assert tb.read_breadcrumb()["session_id"] == "session-two"
    _fake_tty(monkeypatch, "/dev/pts/1")
    assert tb.read_breadcrumb()["session_id"] == "session-one"

def test_stale_breadcrumb_ignored_and_pruned(hermes_home, monkeypatch, no_terminal_env):
    _fake_tty(monkeypatch, "/dev/pts/1")
    directory = hermes_home / "terminal-sessions"
    directory.mkdir(parents=True)
    stale = directory / "tty-dev-pts-1"
    stale.write_text(
        json.dumps({"session_id": "old", "cwd": "/", "ts": time.time() - 40 * 86400})
    )
    old_mtime = time.time() - 40 * 86400
    os.utime(stale, (old_mtime, old_mtime))
    # read: stale payload rejected
    assert tb.read_breadcrumb() is None
    # write from another terminal prunes the stale file
    _fake_tty(monkeypatch, "/dev/pts/2")
    tb.write_breadcrumb("fresh")
    assert not stale.exists()

def test_corrupt_breadcrumb_returns_none(hermes_home, monkeypatch, no_terminal_env):
    _fake_tty(monkeypatch, "/dev/pts/1")
    directory = hermes_home / "terminal-sessions"
    directory.mkdir(parents=True)
    (directory / "tty-dev-pts-1").write_text("not json{")
    assert tb.read_breadcrumb() is None

# ------------------------------------------------------------- resolution

def _make_session(home: Path, session_id: str):
    from hermes_state import SessionDB

    db = SessionDB()
    db.create_session(session_id, "cli")
    db.close()

def test_resolve_picks_this_terminals_session(hermes_home, monkeypatch, no_terminal_env):
    _fake_tty(monkeypatch, "/dev/pts/5")
    _make_session(hermes_home, "20260815_100000_aaaaaa")
    _make_session(hermes_home, "20260815_110000_bbbbbb")  # newer, other terminal
    tb.write_breadcrumb("20260815_100000_aaaaaa")
    assert tb.resolve_breadcrumb_session() == "20260815_100000_aaaaaa"

def test_resolve_falls_back_when_session_deleted(hermes_home, monkeypatch, no_terminal_env):
    _fake_tty(monkeypatch, "/dev/pts/5")
    _make_session(hermes_home, "20260815_110000_bbbbbb")
    tb.write_breadcrumb("20260815_100000_deleted")  # never existed / deleted
    assert tb.resolve_breadcrumb_session() is None

def test_resolve_projects_through_compression_chain(hermes_home, monkeypatch, no_terminal_env):
    _fake_tty(monkeypatch, "/dev/pts/5")
    _make_session(hermes_home, "20260815_100000_parent")
    tb.write_breadcrumb("20260815_100000_parent")

    from hermes_state import SessionDB

    monkeypatch.setattr(
        SessionDB,
        "get_compression_tip",
        lambda self, sid: "20260815_110000_child",
    )
    assert tb.resolve_breadcrumb_session() == "20260815_110000_child"


def test_selected_pg_open_failure_logs_trace_without_sqlite_fallback(
    hermes_home, monkeypatch, no_terminal_env, capsys, caplog
):
    from state_store_runtime_readiness import trap_state_db_opens
    import cli_session_store

    (hermes_home / "config.yaml").write_text(
        "state_store:\n  backend: postgresql\n  postgresql:\n"
        "    dsn_env: HERMES_STATE_STORE_TEST_DSN\n",
        encoding="utf-8",
    )
    _fake_no_tty(monkeypatch)
    monkeypatch.setenv("TMUX_PANE", "%5")
    directory = hermes_home / "terminal-sessions"
    directory.mkdir()
    (directory / "tmux_pane--5").write_text(
        json.dumps({"session_id": "20260815_100000_pgcrumb", "ts": time.time()}),
        encoding="utf-8",
    )
    assert tb.get_terminal_id() == "tmux_pane--5"
    assert tb._breadcrumbs_dir() == directory
    crumb = tb.read_breadcrumb()
    assert crumb is not None and crumb["session_id"] == "20260815_100000_pgcrumb"

    def unavailable(config):
        assert config["state_store"]["backend"] == "postgresql"
        raise RuntimeError("selected PostgreSQL store unavailable")

    monkeypatch.setattr(cli_session_store, "open_selected_read_store", unavailable)
    with trap_state_db_opens(hermes_home) as events, caplog.at_level("ERROR"):
        assert tb.resolve_breadcrumb_session() is None

    err = capsys.readouterr().err
    assert "cannot open selected state store" in err
    assert "selected PostgreSQL store unavailable" in err
    assert "skipping breadcrumb resume" in err
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert "terminal breadcrumb" in record.message
    assert record.exc_info is not None
    assert record.exc_info[0] is RuntimeError
    assert str(record.exc_info[1]) == "selected PostgreSQL store unavailable"
    assert events == []
    assert not (hermes_home / "state.db").exists()

# ------------------------------------------------------------ config gate

def test_config_gate_off_disables_writes_and_resolution(
    hermes_home, monkeypatch, no_terminal_env
):
    _fake_tty(monkeypatch, "/dev/pts/9")
    import hermes_cli.config as config_mod

    monkeypatch.setattr(
        config_mod, "load_config", lambda: {"session": {"terminal_continue": False}}
    )
    tb.write_breadcrumb("20260815_120000_abc123")
    assert not (hermes_home / "terminal-sessions").exists()
    # Even with a pre-existing breadcrumb, resolution must decline
    monkeypatch.setattr(config_mod, "load_config", lambda: {})
    tb.write_breadcrumb("20260815_120000_abc123")
    monkeypatch.setattr(
        config_mod, "load_config", lambda: {"session": {"terminal_continue": False}}
    )
    assert tb.resolve_breadcrumb_session() is None
