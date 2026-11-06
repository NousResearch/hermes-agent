"""The post-update guard must never destroy the only copy of rows newer than its snapshot."""

import sqlite3
from pathlib import Path

from hermes_cli import update_cmd_maint
from hermes_cli.backup import _should_exclude


def _make_db(path: Path, rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE sessions (id INTEGER PRIMARY KEY, name TEXT)")
        conn.executemany("INSERT INTO sessions (name) VALUES (?)", [(f"row{i}",) for i in range(rows)])
    conn.close()


def _damage_sessions_btree(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        page = conn.execute("SELECT rootpage FROM sqlite_master WHERE name='sessions'").fetchone()[0]
        page_size = conn.execute("PRAGMA page_size").fetchone()[0]
    conn.close()
    with path.open("r+b") as stream:
        stream.seek((page - 1) * page_size)
        stream.write(b"\x00")


def _damaged_home_with_old_snapshot(tmp_path: Path) -> tuple[Path, bytes]:
    home = tmp_path / "home"
    state = home / "state.db"
    _make_db(state, 50)
    _damage_sessions_btree(state)
    _make_db(home / "state-snapshots" / "20260101-pre-update" / "state.db", 5)
    return home, state.read_bytes()


def _row_count(path: Path) -> int:
    with sqlite3.connect(path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    conn.close()
    return count


def test_auto_restore_keeps_the_replaced_db_and_names_it(tmp_path, capsys):
    home, damaged = _damaged_home_with_old_snapshot(tmp_path)
    (home / "state.db-wal").write_bytes(b"frames only this file holds")

    update_cmd_maint._verify_and_restore_one_state_db(home, label="default home")

    assert _row_count(home / "state.db") == 5
    kept = min(home.glob("state.db.replaced-*"), key=lambda path: len(path.name))
    assert kept.read_bytes() == damaged
    assert kept.with_name(kept.name + "-wal").read_bytes() == b"frames only this file holds"
    assert kept.stat().st_mode & 0o077 == 0
    out = capsys.readouterr().out
    assert str(kept) in out
    assert "hermes sessions recover" in out


def test_refused_restore_leaves_no_copy_behind(tmp_path, monkeypatch, capsys):
    home, damaged = _damaged_home_with_old_snapshot(tmp_path)
    monkeypatch.setattr("hermes_cli.backup_restore._safe_restore_db", lambda src, dst: False)

    update_cmd_maint._verify_and_restore_one_state_db(home, label="default home")

    assert (home / "state.db").read_bytes() == damaged
    assert not list(home.glob("state.db.replaced-*"))
    assert "state.db was not restored" in capsys.readouterr().out


def test_no_restore_when_the_copy_cannot_be_written(tmp_path, monkeypatch, capsys):
    home, damaged = _damaged_home_with_old_snapshot(tmp_path)

    def no_space(*_args, **_kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(update_cmd_maint.shutil, "copy2", no_space)

    update_cmd_maint._verify_and_restore_one_state_db(home, label="default home")

    assert (home / "state.db").read_bytes() == damaged
    assert not list(home.glob("state.db.replaced-*"))
    assert "No space left on device" in capsys.readouterr().out


def test_backups_skip_the_kept_copy():
    assert _should_exclude(Path("state.db.replaced-20260101-000000-1"))
    assert _should_exclude(Path("profiles/work/state.db.replaced-20260101-000000-1-wal"))
