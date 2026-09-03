"""Regression tests: `hermes backup -o <bad path>` errors cleanly (round-3 SUB-01)
and action_receipts.db is restored at mode 0600 on POSIX.

Before the bad-path fix, an unwritable/nonexistent-parent output path raised a
raw PermissionError traceback from the unguarded is_dir()/mkdir() calls. It must
print a one-line error and exit 1 instead.
"""

import os
import stat
import zipfile
from argparse import Namespace
from pathlib import Path

import pytest


def _make_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("model: {}\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def test_backup_unwritable_parent_errors_cleanly(tmp_path, monkeypatch, capsys):
    _make_home(tmp_path, monkeypatch)
    import hermes_cli.backup as backup_mod

    # A parent directory that cannot be created (a file stands where the dir
    # would go) reliably triggers an OSError on mkdir without needing root.
    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file, not a dir")
    bad_out = blocker / "sub" / "backup.zip"

    with pytest.raises(SystemExit) as exc:
        backup_mod.run_backup(Namespace(output=str(bad_out)))

    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "cannot write backup" in out.lower()
    assert "Traceback" not in out


@pytest.mark.skipif(os.name != "posix", reason="POSIX file permissions only")
def test_import_restores_action_receipts_db_with_0600(tmp_path, monkeypatch):
    """action_receipts.db must land at 0600 without a public transition.

    The extraction helper creates and fills the temporary inode at the final
    secret mode before atomic publication. This test spies on that helper's
    atomic-replace boundary, so a regression to post-publication chmod is
    observable even when the end state still happens to be 0600.
    """
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    # Stub out gateway service helpers so the post-import step does not touch
    # the host's systemd/launchd (mirrors the autouse fixture in test_backup.py).
    import hermes_cli.gateway as gateway_mod
    monkeypatch.setattr(gateway_mod, "ensure_gateway_service", lambda **kw: False)
    monkeypatch.setattr(gateway_mod, "_is_service_running", lambda: False)
    import hermes_cli.backup as backup_mod

    published_modes = []
    original_replace = backup_mod.atomic_replace
    def _capture_replace(tmp, target):
        if target.name == "action_receipts.db":
            published_modes.append(stat.S_IMODE(Path(tmp).stat().st_mode))
        return original_replace(tmp, target)

    monkeypatch.setattr(backup_mod, "atomic_replace", _capture_replace)

    # Build a minimal valid backup zip that contains action_receipts.db.
    # _validate_backup_zip accepts any zip that has at least one of the
    # canonical marker files; state.db satisfies that requirement.
    zip_path = tmp_path / "backup.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("state.db", b"SQLite format 3\x00")
        zf.writestr("action_receipts.db", b"SQLite format 3\x00")

    backup_mod.run_import(Namespace(zipfile=str(zip_path), force=True))

    restored = hermes_home / "action_receipts.db"
    assert restored.exists(), "action_receipts.db was not restored"
    mode = restored.stat().st_mode & 0o777
    assert mode == 0o600, (
        f"action_receipts.db restored with mode {oct(mode)}, expected 0o600"
    )
    assert published_modes == [0o600]


@pytest.mark.skipif(os.name != "posix", reason="POSIX file permissions only")
def test_safe_restore_hardens_action_receipt_db_and_live_sidecars(tmp_path):
    import sqlite3
    import hermes_cli.backup as backup_mod

    src = tmp_path / "snapshot.db"
    with sqlite3.connect(src) as conn:
        conn.execute("CREATE TABLE marker(value TEXT)")
        conn.execute("INSERT INTO marker VALUES ('restored')")

    dst = tmp_path / "action_receipts.db"
    live = sqlite3.connect(dst)
    try:
        live.execute("PRAGMA journal_mode=WAL")
        live.execute("CREATE TABLE old(value TEXT)")
        live.commit()
        sidecars = [
            dst.with_name(dst.name + "-wal"),
            dst.with_name(dst.name + "-shm"),
        ]
        assert all(path.exists() for path in sidecars)
        for path in (dst, *sidecars):
            path.chmod(0o644)

        assert backup_mod._safe_restore_db(src, dst) is True
        assert live.execute("SELECT value FROM marker").fetchone() == ("restored",)
        for path in (dst, *sidecars):
            assert stat.S_IMODE(path.stat().st_mode) == 0o600
    finally:
        live.close()
