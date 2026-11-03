"""Interrupted full backups must recover without consuming user files or active stages.

Salvage follow-up to #112365. Exercise real child-process death on each native
host and the public full-backup paths, rather than inferring liveness from PIDs.
"""

from __future__ import annotations

import os
import signal
import sqlite3
import subprocess
import sys
import textwrap
import threading
import zipfile
from argparse import Namespace
from pathlib import Path

import pytest

from hermes_cli import backup


def _home(path: Path) -> Path:
    path.mkdir()
    (path / "config.yaml").write_bytes(b"model: {}\n")
    with sqlite3.connect(path / "state.db") as conn:
        conn.execute("CREATE TABLE messages (text TEXT)")
        conn.execute("INSERT INTO messages VALUES ('keep me')")
    return path


def _child(home: Path, output: Path) -> tuple[subprocess.Popen[str], Path]:
    script = """
        import sys
        from pathlib import Path
        from argparse import Namespace
        from hermes_cli import backup

        def pause_copy(src, dst):
            dst.write_bytes(b'\\0' * 4096)
            print('READY ' + str(dst), flush=True)
            sys.stdin.read(1)
            return False

        backup._safe_copy_db = pause_copy
        backup._run_backup_locked(Namespace(output=sys.argv[2]), Path(sys.argv[1]))
    """
    env = {**os.environ, "HERMES_HOME": str(home)}
    proc: subprocess.Popen[str] = subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent(script), str(home), str(output)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    stdout = proc.stdout
    assert stdout is not None
    ready = threading.Event()
    snapshots = []

    def read_ready():
        for line in stdout:
            if line.startswith("READY "):
                snapshots.append(Path(line.strip()[6:]))
                ready.set()
                return

    reader = threading.Thread(target=read_ready, daemon=True)
    reader.start()
    if not ready.wait(30):
        proc.kill()
        _, stderr = proc.communicate(timeout=30)
        reader.join(timeout=30)
        pytest.fail(f"backup child did not reach staging: {stderr}")
    reader.join(timeout=30)
    return proc, snapshots[0]


def _kill(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is None:
        proc.kill()
    proc.communicate(timeout=30)


@pytest.mark.platforms("windows", "posix")
@pytest.mark.parametrize("automatic", [False, True])
def test_killed_backup_recovers_and_preserves_live_peer(
    tmp_path, monkeypatch, automatic
):
    home = _home(tmp_path / "home")
    output = home / "backup.zip"
    user_files = {"tmp-user.db": b"user-owned", ".notes.partial": b"notes"}
    # Generic legacy tmp DBs cannot safely be reclaimed, so keep this valid and
    # verify its rows survive both cleanup and the next archive.
    with sqlite3.connect(home / "tmp-user.db") as conn:
        conn.execute("CREATE TABLE data (value TEXT)")
        conn.execute("INSERT INTO data VALUES ('user-owned')")
    user_files["tmp-user.db"] = (home / "tmp-user.db").read_bytes()
    (home / ".notes.partial").write_bytes(user_files[".notes.partial"])
    child, active_snapshot = _child(home, output)
    try:
        assert active_snapshot.read_bytes() == b"\0" * 4096
        # Reclaim from a different process/home while the original is active.
        from hermes_cli.backup_staging import reclaim_backup_staging

        reclaim_backup_staging(home)
        assert active_snapshot.read_bytes() == b"\0" * 4096
    finally:
        _kill(child)
    # Retry into the default excluded backups directory instead of the old root
    # output. The walk must still reclaim the abandoned root stages.
    retry_dir = home / "backups"
    retry_dir.mkdir(exist_ok=True)
    output = retry_dir / "recovered.zip"
    monkeypatch.setattr(backup, "_collect_external_entries", lambda: ([], []))
    if automatic:
        assert backup._write_full_zip_backup(output, home) == output
    else:
        assert backup._run_backup_locked(Namespace(output=str(output)), home)
    assert not any(p.name.startswith(".hermes-backup-stage-") for p in home.iterdir())
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        assert "state.db" in archive.namelist()
        assert not any(".hermes-backup-stage-" in n for n in archive.namelist())
    for name, content in user_files.items():
        assert (home / name).read_bytes() == content


@pytest.mark.platforms("posix")
@pytest.mark.parametrize(
    "signum", [signal.SIGTERM, getattr(signal, "SIGHUP", signal.SIGTERM)]
)
def test_termination_unwinds_staging_without_publishing(tmp_path, signum):
    home = _home(tmp_path / "home")
    output = home / "backup.zip"
    output.write_bytes(b"previous backup")
    child, _ = _child(home, output)
    try:
        child.send_signal(signum)
        child.communicate(timeout=30)
        assert child.returncode == 128 + signum
        assert output.read_bytes() == b"previous backup"
        assert not any(
            p.name.startswith(".hermes-backup-stage-") for p in home.iterdir()
        )
    finally:
        _kill(child)


def test_failed_snapshot_cleans_staging_but_keeps_failure(tmp_path, monkeypatch):
    home = _home(tmp_path / "home")
    output = home / "backup.zip"

    def fail(src, dst):
        dst.write_bytes(b"incomplete")
        return False

    monkeypatch.setattr(backup, "_safe_copy_db", fail)
    monkeypatch.setattr(backup, "_collect_external_entries", lambda: ([], []))
    assert not backup._run_backup_locked(Namespace(output=str(output)), home)
    assert not any(p.name.startswith(".hermes-backup-stage-") for p in home.iterdir())
    with zipfile.ZipFile(output) as archive:
        assert "state.db" not in archive.namelist()
        assert archive.read("config.yaml") == b"model: {}\n"


def test_stage_stays_on_output_filesystem_and_snapshot_preserves_wal(
    tmp_path, monkeypatch
):
    home = _home(tmp_path / "home")
    output_dir = tmp_path / "external"
    output_dir.mkdir()
    output = output_dir / "backup.zip"
    safe_copy = backup._safe_copy_db

    def copy_on_output(src, dst):
        assert dst.parent.parent == output_dir
        return safe_copy(src, dst)

    monkeypatch.setattr(backup, "_safe_copy_db", copy_on_output)
    with sqlite3.connect(home / "state.db") as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("INSERT INTO messages VALUES ('in WAL')")
        conn.commit()
        with zipfile.ZipFile(output, "w") as archive:
            assert backup._zip_sqlite_snapshot(
                archive, home / "state.db", Path("state.db"), output
            )
        with zipfile.ZipFile(output) as archive:
            restored = tmp_path / "restored.db"
            restored.write_bytes(archive.read("state.db"))
        with sqlite3.connect(restored) as snapshot:
            assert snapshot.execute("SELECT text FROM messages").fetchall() == [
                ("keep me",),
                ("in WAL",),
            ]
            assert snapshot.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    assert list(output_dir.iterdir()) == [output]


def test_recovery_leaves_unknown_directories_and_unexpected_user_files(tmp_path):
    from hermes_cli.backup_staging import reclaim_backup_staging, staged_backup_file

    lookalike = tmp_path / (".hermes-backup-stage-" + "0" * 32)
    lookalike.mkdir()
    (lookalike / "snapshot.db").write_bytes(b"user file")
    # A user's extra file in even a genuine owned directory prevents all deletion.
    with staged_backup_file(tmp_path, "snapshot.db") as staged:
        staged.write_bytes(b"snapshot")
        notes = staged.parent / "user-notes.txt"
        notes.write_bytes(b"keep these")
    reclaim_backup_staging(tmp_path)
    assert staged.read_bytes() == b"snapshot"
    assert notes.read_bytes() == b"keep these"
    assert (lookalike / "snapshot.db").read_bytes() == b"user file"
    listed = {
        str(rel) for _, rel in backup._iter_backup_files(tmp_path, tmp_path / "out.zip")
    }
    assert str(Path(lookalike.name) / "snapshot.db") in listed
    assert str(Path(staged.parent.name) / "user-notes.txt") in listed


@pytest.mark.platforms("posix")
def test_recovery_does_not_follow_staging_or_payload_symlinks(tmp_path):
    from hermes_cli.backup_staging import reclaim_backup_staging, staged_backup_file

    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "snapshot.db"
    secret.write_bytes(b"keep me")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    linked_dir = output_dir / (".hermes-backup-stage-" + "1" * 32)
    linked_dir.symlink_to(outside, target_is_directory=True)
    with staged_backup_file(output_dir, "snapshot.db") as staged:
        staged.symlink_to(secret)
    reclaim_backup_staging(output_dir)
    assert staged.is_symlink()
    assert linked_dir.is_symlink()
    assert secret.read_bytes() == b"keep me"


def test_nested_signal_guard_unwinds_and_restores_default():
    from hermes_cli.termination_guard import unwind_on_termination

    previous = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    try:
        with pytest.raises(SystemExit, match=str(128 + signal.SIGTERM)):
            with unwind_on_termination(), unwind_on_termination():
                signal.raise_signal(signal.SIGTERM)
        assert signal.getsignal(signal.SIGTERM) == signal.SIG_DFL
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        with unwind_on_termination():
            assert signal.getsignal(signal.SIGTERM) == signal.SIG_IGN
    finally:
        signal.signal(signal.SIGTERM, previous)


def test_signal_guard_preserves_embedding_handlers_and_worker_threads():
    from hermes_cli.termination_guard import unwind_on_termination

    previous = signal.getsignal(signal.SIGTERM)

    def custom(signum, frame):
        return None

    signal.signal(signal.SIGTERM, custom)
    try:
        with unwind_on_termination():
            assert signal.getsignal(signal.SIGTERM) is custom
        errors = []

        def worker():
            try:
                with unwind_on_termination():
                    assert signal.getsignal(signal.SIGTERM) is custom
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(timeout=10)
        assert not thread.is_alive()
        assert not errors
        assert signal.getsignal(signal.SIGTERM) is custom
    finally:
        signal.signal(signal.SIGTERM, previous)
