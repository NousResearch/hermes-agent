"""Recovery retention regressions for #106087, building on #97768."""

import sqlite3

from hermes_cli import backup


def _home(tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("model: {}\n")
    for rel in ("state.db", "cron/executions.db"):
        path = home / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as conn:
            conn.execute("CREATE TABLE records (value TEXT)")
            conn.execute("INSERT INTO records VALUES (?)", ("r" * 10000 if rel == "state.db" else "recovery",))
    return home


def test_repeated_failed_captures_are_bounded_without_losing_complete_recovery(tmp_path, monkeypatch):
    home = _home(tmp_path)
    complete = backup.create_quick_snapshot(hermes_home=home, label="pre-update", keep=1)
    assert complete
    real_copy = backup._safe_copy_db
    monkeypatch.setattr(backup, "_safe_copy_db", lambda src, dst: False if src.name == "state.db" else real_copy(src, dst))
    partials = [backup.create_quick_snapshot(hermes_home=home, label="pre-update", keep=1) for _ in range(4)]
    assert all(partials)
    root = home / "state-snapshots"
    assert {p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")} == {complete, partials[-1]}
    assert backup.verify_sqlite_integrity(root / complete / "state.db")["valid"]


def test_unusable_claimed_copy_cannot_replace_complete_generation(tmp_path, monkeypatch):
    home = _home(tmp_path)
    complete = backup.create_quick_snapshot(hermes_home=home, label="pre-update", keep=1)
    assert complete
    real_copy = backup._safe_copy_db
    monkeypatch.setattr(backup, "_safe_copy_db", lambda src, dst: False if src.name == "executions.db" else real_copy(src, dst))
    newer = backup.create_quick_snapshot(hermes_home=home, label="pre-update", keep=1)
    root = home / "state-snapshots"
    # A valid but smaller SQLite replacement cannot satisfy the manifest claim.
    claimed = (root / newer / "state.db")
    claimed.unlink()
    with sqlite3.connect(claimed) as conn:
        conn.execute("CREATE TABLE records (value TEXT)")
        conn.execute("INSERT INTO records VALUES ('replacement')")
    assert backup.verify_sqlite_integrity(claimed)["valid"]
    assert claimed.stat().st_size != (root / complete / "state.db").stat().st_size
    monkeypatch.setattr(backup, "_safe_copy_db", lambda src, dst: False if src.name == "state.db" else real_copy(src, dst))
    latest = backup.create_quick_snapshot(hermes_home=home, label="pre-update", keep=1)
    assert latest
    assert (root / complete / "state.db").exists(), "unusable payload displaced the last complete generation"
    assert backup.verify_sqlite_integrity(root / complete / "state.db")["valid"]
    assert (root / latest).exists()
    assert not (root / newer).exists(), "stale partial generations must not accumulate"
