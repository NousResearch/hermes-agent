"""Subprocess coverage for Kanban maintenance leases and WAL evidence bundles."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_maintenance as km


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _start_holder(
    tmp_path: Path, db_path: Path, mode: str
) -> tuple[subprocess.Popen, Path]:
    ready = tmp_path / f"{mode}-ready"
    release = tmp_path / f"{mode}-release"
    script = tmp_path / f"hold-{mode}.py"
    context_name = (
        "maintenance_lease" if mode == "maintenance" else "writer_registration"
    )
    kwargs = (
        ", action='test-holder', timeout=0.2"
        if mode == "maintenance"
        else ", owner='test-writer'"
    )
    script.write_text(
        textwrap.dedent(
            f"""
            import pathlib
            import sys
            import time
            sys.path.insert(0, {str(_REPO_ROOT)!r})
            from hermes_cli import kanban_maintenance as km

            with km.{context_name}(pathlib.Path({str(db_path)!r}){kwargs}):
                pathlib.Path({str(ready)!r}).write_text("ready", encoding="utf-8")
                for _ in range(1000):
                    if pathlib.Path({str(release)!r}).exists():
                        break
                    time.sleep(0.01)
            """
        ),
        encoding="utf-8",
    )
    child = subprocess.Popen([sys.executable, str(script)])
    for _ in range(1000):
        if ready.exists():
            break
        if child.poll() is not None:
            break
        time.sleep(0.01)
    assert ready.exists(), f"{mode} holder exited before acquiring lease"
    return child, release


def _start_write_txn_holder(
    tmp_path: Path, db_path: Path
) -> tuple[subprocess.Popen, Path]:
    ready = tmp_path / "transaction-ready"
    release = tmp_path / "transaction-release"
    script = tmp_path / "hold-transaction.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import pathlib
            import sys
            import time
            sys.path.insert(0, {str(_REPO_ROOT)!r})
            from hermes_cli import kanban_db as kb

            db = pathlib.Path({str(db_path)!r})
            with kb.connect(db_path=db) as conn:
                with kb.write_txn(conn):
                    pathlib.Path({str(ready)!r}).write_text("ready", encoding="utf-8")
                    for _ in range(1000):
                        if pathlib.Path({str(release)!r}).exists():
                            break
                        time.sleep(0.01)
            """
        ),
        encoding="utf-8",
    )
    child = subprocess.Popen([sys.executable, str(script)])
    for _ in range(1000):
        if ready.exists() or child.poll() is not None:
            break
        time.sleep(0.01)
    assert ready.exists(), "write transaction exited before acquiring admission"
    return child, release


@pytest.mark.skipif(
    sys.platform == "win32", reason="subprocess flock coverage is POSIX-only"
)
def test_two_processes_cannot_hold_maintenance_lease(tmp_path):
    db_path = tmp_path / "kanban.db"
    child, release = _start_holder(tmp_path, db_path, "maintenance")
    try:
        with pytest.raises(km.MaintenanceLeaseBusyError) as excinfo:
            with km.maintenance_lease(db_path, action="second", timeout=0.1):
                pytest.fail("second process acquired an exclusive maintenance lease")
        assert excinfo.value.holder["action"] == "test-holder"
    finally:
        release.write_text("release", encoding="utf-8")
        child.wait(timeout=15)


@pytest.mark.skipif(
    sys.platform == "win32", reason="subprocess flock coverage is POSIX-only"
)
def test_evidence_capture_refuses_active_writer_registration(tmp_path):
    db_path = tmp_path / "kanban.db"
    kb.init_db(db_path=db_path)
    child, release = _start_holder(tmp_path, db_path, "writer")
    try:
        with pytest.raises(km.MaintenanceLeaseBusyError) as excinfo:
            kb.capture_evidence_bundle(
                db_path,
                incident_id="active-writer",
                manifest_path=tmp_path / "active-writer-manifest.json",
                timeout=0.1,
            )
        assert excinfo.value.active_writers
        assert excinfo.value.active_writers[0]["owner"] == "test-writer"
    finally:
        release.write_text("release", encoding="utf-8")
        child.wait(timeout=15)


@pytest.mark.skipif(
    sys.platform == "win32", reason="subprocess flock coverage is POSIX-only"
)
def test_evidence_capture_refuses_open_write_transaction(tmp_path):
    db_path = tmp_path / "kanban.db"
    kb.init_db(db_path=db_path)
    child, release = _start_write_txn_holder(tmp_path, db_path)
    try:
        with pytest.raises(km.MaintenanceLeaseBusyError) as excinfo:
            kb.capture_evidence_bundle(
                db_path,
                incident_id="open-transaction",
                manifest_path=tmp_path / "open-transaction-manifest.json",
                timeout=0.1,
            )
        assert excinfo.value.active_writers
        assert excinfo.value.active_writers[0]["owner"] == "write_txn"
    finally:
        release.write_text("release", encoding="utf-8")
        child.wait(timeout=15)


def test_wal_bundle_restores_integrity_and_reuses_completed_manifest(tmp_path):
    db_path = tmp_path / "kanban.db"
    manifest_path = tmp_path / "incident-manifest.json"
    conn = sqlite3.connect(db_path, isolation_level=None)
    try:
        assert conn.execute("PRAGMA journal_mode=WAL").fetchone()[0].lower() == "wal"
        conn.execute("PRAGMA wal_autocheckpoint=0")
        conn.execute(
            "CREATE TABLE evidence (id INTEGER PRIMARY KEY, payload TEXT NOT NULL)"
        )
        conn.execute("BEGIN IMMEDIATE")
        conn.executemany(
            "INSERT INTO evidence(payload) VALUES (?)",
            [(f"row-{index}-" + "x" * 1024,) for index in range(64)],
        )
        conn.execute("COMMIT")
        assert Path(str(db_path) + "-wal").exists()
        assert Path(str(db_path) + "-shm").exists()

        first = kb.capture_evidence_bundle(
            db_path,
            incident_id="wal-incident",
            manifest_path=manifest_path,
            timeout=1.0,
        )
        second = kb.capture_evidence_bundle(
            db_path,
            incident_id="wal-incident",
            manifest_path=manifest_path,
            timeout=1.0,
        )
    finally:
        conn.close()

    assert first == second
    assert first["capture_consistency"] == "quiesced"
    assert first["completed"] is True
    assert first["journal_mode"] == "wal"
    assert {entry["source_name"] for entry in first["files"]} >= {
        "kanban.db",
        "kanban.db-wal",
        "kanban.db-shm",
    }
    assert len(list(tmp_path.glob("incident-manifest.json"))) == 1

    restore_dir = tmp_path / "restored"
    restore_dir.mkdir()
    for entry in first["files"]:
        if entry["kind"] not in {"database", "wal", "shm"}:
            continue
        source = Path(entry["path"])
        target = restore_dir / entry["source_name"]
        target.write_bytes(source.read_bytes())
        assert kb._sha256_file(target) == entry["sha256"]

    restored = sqlite3.connect(restore_dir / "kanban.db")
    try:
        assert restored.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert restored.execute("SELECT COUNT(*) FROM evidence").fetchone()[0] == 64
    finally:
        restored.close()

    on_disk = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert on_disk == first
