"""The backend's private copies of application helpers against their originals: the zeroed-file probe the state file helpers run, the profile label, canonical home and gateway lock record the reconcile messages name, and the gateway key namespace profile repair writes. The admin's replayed column history ends at the backend schema, and its salvage lane logs under the original logger name."""

from __future__ import annotations

import inspect
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from gateway import runtime_ownership
from gateway import session as gateway_session
from gateway import status as gateway_status
from hermes_cli import backup
from persistence.sqlite._schema_ddl import SCHEMA_SQL
from persistence.sqlite._state import dbfile
from persistence.sqlite.admin import _lost_and_found, _reconcile, _schema_history, profile_repair


def test_the_zeroed_file_probe_copy_is_the_original_text():
    assert inspect.getsource(dbfile.is_zeroed_sqlite_file) == inspect.getsource(backup.is_zeroed_sqlite_file)


def _declared_now(table: str) -> tuple[str, ...]:
    with closing(sqlite3.connect(":memory:")) as conn:
        conn.executescript(SCHEMA_SQL)
        return tuple(str(row[1]) for row in conn.execute(f'PRAGMA table_info("{table}")'))


@pytest.mark.parametrize("table", sorted(_schema_history.SCHEMA_HISTORY))
def test_the_replayed_column_history_ends_at_the_backend_schema(table):
    assert _schema_history.current_declared_columns(table) == _declared_now(table)


def test_the_salvage_lane_logs_under_the_original_logger():
    assert _lost_and_found.logger.name == "hermes_cli.session_lost_and_found"


def _homes(root: Path) -> list[Path]:
    return [root, root / "profiles" / "work", root / "profiles" / "Bad Name", root / "profiles" / "main",
            root / "elsewhere", root / ".." / root.name / "profiles" / "work", Path("~/relative")]


def test_the_profile_label_copy_names_every_home_as_the_original(tmp_path, monkeypatch):
    root = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(root))
    for home in _homes(root):
        assert _reconcile._profile_label_for_home(home) == gateway_status._profile_label_for_home(home), home
    assert _reconcile._profile_label_for_home(root / "profiles" / "work") == "work"


def test_the_canonical_home_copy_matches_the_original(tmp_path):
    for home in _homes(tmp_path):
        assert _reconcile.canonical_home(home) == runtime_ownership.canonical_home(home)


@pytest.mark.parametrize("content", [None, "", "  \n", '{"pid": 12, "hermes_home": "/x"}', "4321", "[1, 2]",
                                     "not json", b"\xff\xfe\x00garbage", '\ufeff{"pid": 7}'])
def test_the_lock_record_reader_copy_reads_every_file_as_the_original(tmp_path, content):
    path = tmp_path / "gateway.lock"
    if isinstance(content, bytes):
        path.write_bytes(content)
    elif content is not None:
        path.write_text(content, encoding="utf-8")
    assert _reconcile._read_json_file(path, bare_pid_ok=True) == gateway_status._read_gateway_lock_record(path)


@pytest.mark.parametrize("profile", [None, "", "default", "main", "work", "main~"])
def test_profile_repair_writes_the_gateway_key_namespace(profile):
    assert profile_repair._session_key_namespace(profile) == gateway_session._session_key_namespace(profile)
