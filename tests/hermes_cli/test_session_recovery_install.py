"""Behavioral tests for quiescent recovery installation of the active state.db."""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import subprocess
import sys
import time
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_state import SessionDB


def _seed_state_db(path: Path, count: int = 12) -> None:
    db = SessionDB(db_path=path)
    try:
        for index in range(count):
            session_id = f"recovery-{index}"
            db.create_session(session_id=session_id, source="cli")
            db.append_message(session_id, role="user", content=f"preservedneedle{index}")
    finally:
        db.close()


def _corrupt_fts_data_root(path: Path) -> None:
    """Damage a real derived-index B-tree while leaving canonical rows readable."""
    conn = sqlite3.connect(str(path))
    try:
        if conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal":
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        page_size = conn.execute("PRAGMA page_size").fetchone()[0]
        root_page = conn.execute(
            "SELECT rootpage FROM sqlite_master WHERE name = 'messages_fts_data'"
        ).fetchone()[0]
    finally:
        conn.close()
    with path.open("r+b") as handle:
        handle.seek(page_size * (root_page - 1))
        handle.write(b"\xde\xad\xbe\xef" * (page_size // 4))


@pytest.mark.requires_wal
def test_install_recovers_real_corruption_under_quiescent_writer_guard(tmp_path, monkeypatch, capsys):
    """A complete candidate is installed only through the real exclusive DB guard.

    Requires WAL: the generation fingerprint asserts on the -wal sidecar, which only
    exists while the store actually runs in WAL journal mode.
    """
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    source = home / "state.db"
    output = tmp_path / "recovered-state.db"
    report_path = output.with_name(output.name + ".recovery.json")
    _seed_state_db(source)
    _corrupt_fts_data_root(source)

    from hermes_cli.sessions_cmd import _cmd_recover

    status = _cmd_recover(Namespace(
        source=source,
        output=output,
        inspect_only=False,
        allow_partial=False,
        install=True,
        report=report_path,
        work_dir=tmp_path,
        chunk_size=1000,
    ))

    assert status == 0, capsys.readouterr().out
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["complete"] is True
    assert report["partial"] is False
    assert report["verified"] is True
    assert report["installed"] is True
    assert report["install_target"] == str(source)
    assert report["generation_fence"]["source_application_id"] != report["generation_fence"]["candidate_application_id"]

    preserved = Path(report["preserved_source_bundle"])
    assert preserved.is_dir()
    assert (preserved / "source" / "state.db").is_file()
    assert (preserved / "manifest.json").is_file()

    db = SessionDB(db_path=source)
    try:
        assert db.search_messages("preservedneedle3")
        db.create_session(session_id="recovery-canary", source="system")
        db.append_message("recovery-canary", role="user", content="recoverycanaryneedle")
        assert db.get_messages("recovery-canary")[0]["content"] == "recoverycanaryneedle"
        assert db.search_messages("recoverycanaryneedle")
        db.delete_session("recovery-canary")
    finally:
        db.close()

    assert "installed" in capsys.readouterr().out.lower()


def test_install_refuses_when_a_real_sessiondb_writer_is_open(tmp_path, monkeypatch, capsys):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    source = home / "state.db"
    output = tmp_path / "must-not-install.db"
    _seed_state_db(source)
    # Damage a canonical B-tree, then hold a real writable SessionDB connection.
    conn = sqlite3.connect(str(source))
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        page_size = conn.execute("PRAGMA page_size").fetchone()[0]
        root_page = conn.execute(
            "SELECT rootpage FROM sqlite_master WHERE name = 'sessions'"
        ).fetchone()[0]
    finally:
        conn.close()
    with source.open("r+b") as handle:
        handle.seek(page_size * (root_page - 1))
        handle.write(b"\\xde\\xad\\xbe\\xef" * (page_size // 4))
    writer = SessionDB(db_path=source)

    from hermes_cli.sessions_cmd import _cmd_recover

    try:
        status = _cmd_recover(Namespace(
            source=source,
            output=output,
            inspect_only=False,
            allow_partial=False,
            install=True,
            report=None,
            work_dir=tmp_path,
            chunk_size=1000,
        ))
        shown = capsys.readouterr().out.lower()
        assert status != 0
        assert "connection" in shown or "writer" in shown
        assert not output.exists()
    finally:
        writer.close()

    with sqlite3.connect(str(source)) as check:
        result = check.execute("PRAGMA quick_check").fetchone()[0]
    assert result != "ok"


def test_install_does_not_allow_partial_recovery(tmp_path, monkeypatch, capsys):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    source = home / "state.db"
    output = tmp_path / "must-not-install-partial.db"
    _seed_state_db(source)

    from hermes_cli.sessions_cmd import _cmd_recover

    status = _cmd_recover(Namespace(
        source=source,
        output=output,
        inspect_only=False,
        allow_partial=True,
        install=True,
        report=None,
        work_dir=tmp_path,
        chunk_size=1000,
    ))

    assert status == 2
    assert "cannot be combined" in capsys.readouterr().out.lower()
    assert not output.exists()


def test_exclusive_repair_guard_refuses_a_real_sessiondb_writer(tmp_path):
    """The existing cross-platform SQLite guard refuses while a writer connection is open."""
    import hermes_state_repair

    source = tmp_path / "state.db"
    _seed_state_db(source, count=1)
    writer = SessionDB(db_path=source)
    try:
        with hermes_state_repair._exclusive_repair_db_guard(source) as (guard, error):
            assert guard is None
            assert error is not None
    finally:
        writer.close()


def test_sessions_recover_parser_exposes_install_as_an_explicit_opt_in():
    from argparse import ArgumentParser

    from hermes_cli.subcommands.sessions import build_sessions_parser

    parser = ArgumentParser()
    subparsers = parser.add_subparsers()
    build_sessions_parser(subparsers, cmd_sessions=lambda args: args)
    args = parser.parse_args([
        "sessions", "recover", "--source", "state.db", "--output", "recovered.db", "--install",
    ])

    assert args.install is True


def _sha256_main(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _no_published_backup_runs(backup_root: Path) -> bool:
    """No completed or partial backup run was left behind."""
    if not backup_root.exists():
        return True
    for child in backup_root.iterdir():
        if child.name.endswith(".partial") or child.name.startswith("."):
            return False
        if (child / "manifest.json").exists() or (child / "source").exists():
            return False
    return True


def test_install_refuses_when_shared_volume_has_no_headroom(tmp_path, monkeypatch, capsys):
    """Same-volume low space: backup+work+output share one filesystem and must refuse first."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    source = home / "state.db"
    output = tmp_path / "must-not-install-shared-lowspace.db"
    _seed_state_db(source, count=2)
    before = _sha256_main(source)

    monkeypatch.setattr(
        shutil, "disk_usage",
        lambda _path: SimpleNamespace(total=10_000_000_000, used=9_999_999_000, free=1024),
    )

    from hermes_cli.sessions_cmd import _cmd_recover

    status = _cmd_recover(Namespace(
        source=source,
        output=output,
        inspect_only=False,
        allow_partial=False,
        install=True,
        report=None,
        work_dir=tmp_path,
        chunk_size=1000,
    ))

    assert status != 0
    assert not output.exists()
    assert _no_published_backup_runs(home / "backups" / "session-recovery")
    assert _sha256_main(source) == before
    shown = capsys.readouterr().out.lower()
    assert "free disk space" in shown or "disk space" in shown or "free space" in shown


def test_install_refuses_when_backup_volume_is_full_but_work_has_space(tmp_path, monkeypatch, capsys):
    """Separate-volume low space: a full backup filesystem refuses even when work/output fit."""
    import hermes_cli.session_recovery_install as install_mod

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    source = home / "state.db"
    output = tmp_path / "must-not-install-backup-full.db"
    _seed_state_db(source, count=2)
    before = _sha256_main(source)

    def fake_volume_key(path: Path) -> int:
        # Backup lives under the profile home; work/output live beside it in tmp_path.
        text = str(path)
        if text == str(home) or text.startswith(str(home) + "/") or "session-recovery" in text:
            return 1111
        return 2222

    def fake_disk_usage(path):
        text = str(path)
        if text == str(home) or text.startswith(str(home) + "/"):
            return SimpleNamespace(total=10_000_000_000, used=9_999_999_000, free=1024)
        return SimpleNamespace(total=10_000_000_000, used=0, free=10_000_000_000)

    monkeypatch.setattr(install_mod, "_volume_key", fake_volume_key)
    monkeypatch.setattr(shutil, "disk_usage", fake_disk_usage)

    from hermes_cli.sessions_cmd import _cmd_recover

    status = _cmd_recover(Namespace(
        source=source,
        output=output,
        inspect_only=False,
        allow_partial=False,
        install=True,
        report=None,
        work_dir=tmp_path,
        chunk_size=1000,
    ))

    assert status != 0
    assert not output.exists()
    assert _no_published_backup_runs(home / "backups" / "session-recovery")
    assert _sha256_main(source) == before
    shown = capsys.readouterr().out.lower()
    assert "backup" in shown
    assert "free disk space" in shown or "disk space" in shown or "free space" in shown


def test_install_fails_closed_when_free_space_cannot_be_determined(tmp_path, monkeypatch, capsys):
    """Undeterminable disk usage refuses before any backup copy or active-DB change."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    source = home / "state.db"
    output = tmp_path / "must-not-install-unknown-space.db"
    _seed_state_db(source, count=2)
    before = _sha256_main(source)

    def boom(_path):
        raise OSError("statvfs failed")

    monkeypatch.setattr(shutil, "disk_usage", boom)

    from hermes_cli.sessions_cmd import _cmd_recover

    status = _cmd_recover(Namespace(
        source=source,
        output=output,
        inspect_only=False,
        allow_partial=False,
        install=True,
        report=None,
        work_dir=tmp_path,
        chunk_size=1000,
    ))

    assert status != 0
    assert not output.exists()
    assert _no_published_backup_runs(home / "backups" / "session-recovery")
    assert _sha256_main(source) == before
    shown = capsys.readouterr().out.lower()
    assert "could not determine" in shown or "free space" in shown or "disk space" in shown


def test_install_refuses_against_a_separate_process_holding_the_db(tmp_path, monkeypatch, capsys):
    """Cross-process writer exclusion: a holder in another process refuses the install."""
    from hermes_state_holders import foreign_state_db_holders

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    source = home / "state.db"
    output = tmp_path / "must-not-install-cross-process.db"
    _seed_state_db(source, count=2)
    before = _sha256_main(source)

    holder = subprocess.Popen(
        [
            sys.executable, "-c",
            "import sqlite3, sys; c = sqlite3.connect(sys.argv[1]); "
            "c.execute('SELECT 1').fetchone(); print('held', flush=True); "
            "sys.stdin.readline()",
            str(source),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "held"
        deadline = time.monotonic() + 10
        while not foreign_state_db_holders(source) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert foreign_state_db_holders(source), "holder subprocess was not visible to the scan"

        from hermes_cli.sessions_cmd import _cmd_recover

        status = _cmd_recover(Namespace(
            source=source,
            output=output,
            inspect_only=False,
            allow_partial=False,
            install=True,
            report=None,
            work_dir=tmp_path,
            chunk_size=1000,
        ))

        assert status != 0
        assert not output.exists()
        assert _no_published_backup_runs(home / "backups" / "session-recovery")
        assert _sha256_main(source) == before
        shown = capsys.readouterr().out.lower()
        assert "another process" in shown or "holds" in shown or "writer" in shown or "connection" in shown
    finally:
        try:
            holder.communicate(input="\n", timeout=10)
        except Exception:
            holder.kill()
            holder.wait(timeout=10)


def test_install_refuses_when_live_holds_tables_the_candidate_lacks(tmp_path, monkeypatch, capsys):
    """A live-only table with rows blocks --install instead of being silently dropped.

    Recovery copies only known tables into the candidate; installing that candidate
    over the live file would erase anything else. A table recovery never registered
    (here standing in for agent_config_overrides / agent_tool_state) must refuse
    with its name in the refusal, and the live rows must survive.
    """
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    source = home / "state.db"
    output = tmp_path / "must-not-install.db"
    report_path = output.with_name(output.name + ".recovery.json")
    _seed_state_db(source)
    _corrupt_fts_data_root(source)
    # NOTE: plain sqlite3.connect (closed explicitly) — `with conn:` would only
    # scope the transaction, leaving the handle open and tripping the live-writer guard.
    conn = sqlite3.connect(str(source))
    try:
        conn.execute("CREATE TABLE agent_config_overrides (scope TEXT PRIMARY KEY, payload TEXT)")
        conn.execute("INSERT INTO agent_config_overrides VALUES ('profile', '{\"ttl\": 60}')")
        conn.commit()
    finally:
        conn.close()

    from hermes_cli.sessions_cmd import _cmd_recover

    status = _cmd_recover(Namespace(
        source=source,
        output=output,
        inspect_only=False,
        allow_partial=False,
        install=True,
        report=report_path,
        work_dir=tmp_path,
        chunk_size=1000,
    ))

    assert status != 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["installed"] is False
    assert "agent_config_overrides" in report["install_refusal"]
    assert "agent_config_overrides" in report["verification"]["unknown_live_tables"]
    with sqlite3.connect(str(source)) as check:
        assert check.execute("SELECT COUNT(*) FROM agent_config_overrides").fetchone()[0] == 1
