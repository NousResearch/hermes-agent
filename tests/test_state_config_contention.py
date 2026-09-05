"""Existing databases remain usable while a config writer finishes publication."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import textwrap
from threading import Event, Thread

import pytest

from hermes_cli import config
from hermes_state_wal import apply_wal_with_fallback
from utils import atomic_yaml_write
import hermes_state_wal as wal


def test_partial_config_import_cannot_invert_config_and_module_locks(tmp_path):
    script = textwrap.dedent(
        """
        from pathlib import Path
        from threading import Event, Thread
        import os, sqlite3, sys

        import providers
        import hermes_state_wal as wal

        partial_ready = Event()
        opener_inside_resolve = Event()
        importer_load_returned = Event()
        original_resolve = wal.resolve_journal_mode

        def controlled_list_providers():
            module = sys.modules["hermes_cli.config"]
            assert module.__spec__._initializing
            assert hasattr(module, "_CONFIG_LOCK") and hasattr(module, "load_config")
            partial_ready.set()
            assert opener_inside_resolve.wait(5)
            module.load_config()
            importer_load_returned.set()
            return []

        def controlled_resolve():
            opener_inside_resolve.set()
            return original_resolve()

        providers.list_providers = controlled_list_providers
        wal.resolve_journal_mode = controlled_resolve
        root = Path(os.environ["STATE_CONFIG_TEST_ROOT"]) / "cold-import"
        root.mkdir()
        os.environ["HOME"] = str(root)
        os.environ["HERMES_HOME"] = str(root / ".hermes")
        database = root / "state.db"
        with sqlite3.connect(database) as connection:
            connection.execute("CREATE TABLE retained (value TEXT)")
        outcomes = []

        def importer():
            import hermes_cli.config
            outcomes.append("import-returned")

        def opener():
            with sqlite3.connect(database) as connection:
                outcomes.append(f"open:{wal.apply_wal_with_fallback(connection)}")

        importing = Thread(target=importer)
        opening = Thread(target=opener)
        importing.start()
        assert partial_ready.wait(5)
        opening.start()
        importing.join(5)
        opening.join(5)
        assert importer_load_returned.is_set()
        assert not importing.is_alive() and not opening.is_alive(), outcomes
        assert "import-returned" in outcomes
        assert any(value.startswith("open:") for value in outcomes)
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[1],
        env={
            **os.environ,
            "HOME": str(tmp_path),
            "HERMES_HOME": str(tmp_path / ".hermes"),
            "STATE_CONFIG_TEST_ROOT": str(tmp_path),
        },
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize(
    "mode,require_wal,vulnerable",
    [
        ("delete", False, True),
        ("delete", False, False),
        ("wal", False, False),
        ("wal", True, False),
    ],
)
def test_existing_database_does_not_wait_for_config_writer(
    tmp_path, monkeypatch, caplog, mode, require_wal, vulnerable
):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(wal, "is_sqlite_wal_reset_vulnerable", lambda: vulnerable)
    config.save_config({"database": {"journal_mode": mode}, "marker": "before"})
    config.load_config_readonly()
    path = tmp_path / "existing.db"
    with closing(sqlite3.connect(path)) as conn, conn:
        assert conn.execute(f"PRAGMA journal_mode={mode}").fetchone()[0] == mode
        conn.execute("CREATE TABLE retained (value TEXT)")
        conn.execute("INSERT INTO retained VALUES ('keep')")

    entered, release = Event(), Event()

    def writer():
        with config._CONFIG_LOCK:
            atomic_yaml_write(config.get_config_path(), {
                "database": {"journal_mode": "delete"}, "marker": "after replacement",
            })
            entered.set()
            assert release.wait(10)

    def opener():
        conn = sqlite3.connect(path)
        try:
            selected = apply_wal_with_fallback(
                conn,
                db_label=f"contention-{mode}-{require_wal}-{vulnerable}",
                require_wal=require_wal,
            )
            retained = conn.execute("SELECT value FROM retained").fetchone()[0]
            conn.execute("INSERT INTO retained VALUES ('during config write')")
            conn.commit()
            return selected, retained
        finally:
            conn.close()

    worker = Thread(target=writer)
    worker.start()
    assert entered.wait(5)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(opener)
        try:
            assert future.result(timeout=2) == (mode, "keep")
        finally:
            release.set()
            worker.join(5)
    assert not worker.is_alive()
    warnings = [
        record
        for record in caplog.records
        if "WAL-reset corruption bug" in record.message
        and "using journal_mode=DELETE" in record.message
    ]
    if vulnerable:
        assert len(warnings) == 1
    else:
        assert not warnings
    assert config.load_config_readonly()["marker"] == "after replacement"
    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM retained").fetchone()[0] == 2


@pytest.mark.parametrize("state", ["empty", "unknown", "required_wal"])
def test_unestablished_or_strict_modes_still_wait_for_real_policy(tmp_path, monkeypatch, state):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(wal, "is_sqlite_wal_reset_vulnerable", lambda: False)
    config.save_config({"database": {"journal_mode": "wal"}})
    path = tmp_path / "policy.db"
    if state != "empty":
        with closing(sqlite3.connect(path)) as conn:
            conn.execute("CREATE TABLE existing (value TEXT)")
    if state == "unknown":
        monkeypatch.setattr(wal, "_on_disk_journal_mode", lambda _conn: None)
    resolving = Event()
    original = wal.resolve_journal_mode

    def resolve():
        resolving.set()
        return original()

    monkeypatch.setattr(wal, "resolve_journal_mode", resolve)

    def opener():
        with closing(sqlite3.connect(path)) as conn:
            return wal.apply_wal_with_fallback(conn, require_wal=state == "required_wal")

    with ThreadPoolExecutor(max_workers=1) as pool:
        with config._CONFIG_LOCK:
            pending = pool.submit(opener)
            assert resolving.wait(5)
            assert not pending.done()
        assert pending.result(timeout=5) == "wal"


def test_deferred_mode_change_is_applied_after_writer_finishes(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(wal, "is_sqlite_wal_reset_vulnerable", lambda: False)
    config.save_config({"database": {"journal_mode": "wal"}})
    path = tmp_path / "deferred.db"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE existing (value TEXT)")

    def opener():
        with closing(sqlite3.connect(path)) as conn:
            return wal.apply_wal_with_fallback(conn)

    with ThreadPoolExecutor(max_workers=1) as pool:
        with config._CONFIG_LOCK:
            assert pool.submit(opener).result(timeout=2) == "delete"
        assert pool.submit(opener).result(timeout=5) == "wal"
