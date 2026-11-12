"""`hermes sessions` exit status tracks the outcome: a subcommand that fails, refuses, or could not ask for
its confirmation exits non-zero, so cron/CI wrappers and `hermes sessions repair && ...` chains can tell."""

import sqlite3
import sys
import time

import pytest


def _run_cli(monkeypatch, argv_tail):
    import hermes_cli.main as main_mod

    monkeypatch.setattr(sys, "argv", ["hermes", "sessions", *argv_tail])
    try:
        main_mod.main()
    except SystemExit as exc:
        return exc.code or 0
    return 0


def _old_ended_session(db_path):
    from hermes_state import SessionDB

    with SessionDB(db_path=db_path) as db:
        db.create_session("s-old", "cli")
        db.append_message("s-old", "user", "hello")
        db.end_session("s-old", "user_exit")
    old = time.time() - 100 * 86400
    raw = sqlite3.connect(str(db_path))
    with raw:
        raw.execute("UPDATE sessions SET started_at = ?, ended_at = ?", (old, old))
        raw.execute("UPDATE messages SET timestamp = ?", (old,))
    raw.close()


def _garbage_store(db_path):
    db_path.write_bytes(b"not a sqlite database" * 200)


def _legacy_inline_fts_store(db_path):
    from hermes_state_common import LEGACY_FTS_SQL, LEGACY_FTS_TRIGRAM_SQL, SCHEMA_SQL

    raw = sqlite3.connect(str(db_path))
    with raw:
        raw.executescript(SCHEMA_SQL + LEGACY_FTS_SQL + LEGACY_FTS_TRIGRAM_SQL)
        raw.execute("INSERT INTO sessions (id, source, started_at) VALUES ('s-old', 'cli', ?)", (time.time(),))
        raw.execute("INSERT INTO messages (session_id, timestamp, role, content) VALUES ('s-old', ?, 'user', 'hi')",
                    (time.time(),))
    raw.close()


def _vacuum_fails(monkeypatch):
    from hermes_state import SessionDB

    def vacuum(self):
        raise sqlite3.OperationalError("database is locked")
    monkeypatch.setattr(SessionDB, "vacuum", vacuum)


def _store(monkeypatch, tmp_path, build):
    import hermes_state
    from hermes_constants import get_hermes_home

    db_path = get_hermes_home() / "state.db"
    build(db_path)
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", db_path)
    monkeypatch.chdir(tmp_path)
    return db_path


@pytest.mark.parametrize("build, failing, valid, break_it", [
    (_garbage_store, ["repair"], None, None),
    (_old_ended_session, ["optimize"], ["optimize"], _vacuum_fails),
    (_old_ended_session, ["archive"], ["archive", "--source", "cli", "--yes"], None),
    (_old_ended_session, ["prune", "--never-active", "--older-than", "not-a-duration"],
     ["prune", "--never-active", "--older-than", "2d", "--yes"], None),
    (_old_ended_session, ["export", "exports", "--format", "md"],
     ["export", "exports", "--format", "md", "--older-than", "90"], None),
], ids=["repair-failed", "optimize-vacuum-fails", "archive-no-filter", "never-active-bad-duration",
        "export-md-no-filter"])
def test_failed_or_refused_subcommand_exits_1_where_the_valid_form_exits_0(
        monkeypatch, tmp_path, build, failing, valid, break_it):
    _store(monkeypatch, tmp_path, build)
    if valid:
        assert _run_cli(monkeypatch, valid) == 0
    if break_it:
        break_it(monkeypatch)
    assert _run_cli(monkeypatch, failing) == 1


@pytest.mark.parametrize("build, argv", [
    (_legacy_inline_fts_store, ["optimize-storage"]),
    (_old_ended_session, ["delete", "s-old"]),
    (_old_ended_session, ["archive", "--source", "cli"]),
    (_old_ended_session, ["prune", "--older-than", "30"]),
], ids=["optimize-storage", "delete", "archive", "prune"])
def test_confirmation_nobody_can_answer_exits_1_and_changes_nothing(monkeypatch, tmp_path, capsys, build, argv):
    """No TTY (cron, a wrapper): EOF at the prompt is not a decline. An explicit "n" still exits 0."""
    db_path = _store(monkeypatch, tmp_path, build)

    def snapshot():
        raw = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            return (raw.execute("SELECT id, archived FROM sessions ORDER BY id").fetchall(),
                    raw.execute("SELECT name FROM sqlite_master ORDER BY name").fetchall())
        finally:
            raw.close()

    monkeypatch.setattr("builtins.input", lambda _prompt="": "n")
    assert _run_cli(monkeypatch, argv) == 0
    before = snapshot()  # after the first open: SessionDB's own schema reconcile is not the command's work

    def no_stdin(_prompt=""):
        raise EOFError
    monkeypatch.setattr("builtins.input", no_stdin)
    capsys.readouterr()
    assert _run_cli(monkeypatch, argv) == 1
    assert "Pass --yes" in capsys.readouterr().out
    assert snapshot() == before

    def ctrl_c(_prompt=""):
        raise KeyboardInterrupt
    monkeypatch.setattr("builtins.input", ctrl_c)  # an interrupt is not a decline either: `&& next` must not run
    assert _run_cli(monkeypatch, argv) == 130
    assert snapshot() == before


@pytest.mark.platforms("any")
def test_redirected_null_stdin_is_unattended_on_every_os(tmp_path):
    """A real ``stdin=DEVNULL`` child: Windows reports isatty() True for NUL, which must still read as
    nobody-could-answer (exit 1), never as a decline that lets ``&& next`` run."""
    import os
    import subprocess
    from pathlib import Path

    home = tmp_path / ".hermes"
    home.mkdir()
    _old_ended_session(home / "state.db")
    root = str(Path(__file__).resolve().parents[2])
    env = {**os.environ, "HERMES_HOME": str(home), "PYTHONPATH": root}
    proc = subprocess.run([sys.executable, "-m", "hermes_cli.main", "sessions", "delete", "s-old"], cwd=root,
                          env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=120, check=False)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "Pass --yes" in proc.stdout
