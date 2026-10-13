"""Behavioral contracts for literal, tracked SQLite read-only connections."""
from __future__ import annotations

from contextlib import closing
from pathlib import Path, PureWindowsPath
from urllib.parse import urlsplit
import sqlite3

import pytest

from hermes_cli.sqlite_safe_read import open_sqlite_readonly, read_only_db_uri
from hermes_cli import sqlite_safe_read as safe_read


def _make_db(path: Path, value: str = "literal") -> None:
    with closing(sqlite3.connect(path.resolve())) as conn:
        conn.execute("CREATE TABLE sample(value TEXT)")
        conn.execute("INSERT INTO sample VALUES (?)", (value,))
        conn.commit()


def _assert_literal_readonly(path: Path) -> None:
    _make_db(path)
    assert not safe_read.has_live_connection(path)
    with closing(open_sqlite_readonly(path, timeout=0.0)) as conn:
        assert conn.execute("SELECT value FROM sample").fetchall() == [("literal",)]
        assert Path(conn.execute("PRAGMA database_list").fetchone()[2]).resolve() == path.resolve()
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 0
        assert safe_read.has_live_connection(path)
        assert safe_read.read_header_bytes_preopen(path, length=16) is None
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            conn.execute("INSERT INTO sample VALUES ('forbidden')")
    assert not safe_read.has_live_connection(path)
    assert safe_read.read_header_bytes_preopen(path, length=16) == b"SQLite format 3\x00"


@pytest.mark.parametrize("filename", ["ordinary.db", "space name.db", "hash#name.db", "literal%23.db", "Unicode-é.db"])
def test_literal_path_is_readonly_and_tracked(tmp_path, filename):
    _assert_literal_readonly(tmp_path / filename)


@pytest.mark.parametrize(
    "windows_path, expected_path",
    [
        (r"C:\share\hash#literal%23.db", "/C:/share/hash#literal%23.db"),
        (r"\\server\share\hash#literal%23.db", "//server/share/hash#literal%23.db"),
        (r"\\localhost\share\hash#literal%23.db", "//localhost/share/hash#literal%23.db"),
    ],
)
def test_windows_uri_encoding_keeps_authority_empty(monkeypatch, windows_path, expected_path):
    # PureWindowsPath gives real stdlib encoding without requiring an SMB server.
    resolved = PureWindowsPath(windows_path)
    with monkeypatch.context() as patcher:
        patcher.setattr(Path, "resolve", lambda self: resolved)
        uri = read_only_db_uri("placeholder")
    from urllib.parse import unquote

    parsed = urlsplit(uri)
    assert parsed.scheme == "file"
    assert parsed.netloc == ""
    assert unquote(parsed.path) == expected_path
    assert parsed.query == "mode=ro"
    assert parsed.fragment == ""


@pytest.mark.platforms("windows")
def test_native_windows_unc_uri_has_an_empty_authority(monkeypatch):
    path = Path(r"\\server\share\hash#literal%23.db")
    # Exercise the native Windows Path class; do not contact a network share.
    with monkeypatch.context() as patcher:
        patcher.setattr(Path, "resolve", lambda self: self)
        parsed = urlsplit(read_only_db_uri(path))
    assert parsed.netloc == ""
    assert parsed.path == "//server/share/hash%23literal%2523.db"
    assert parsed.query == "mode=ro"


@pytest.mark.platforms("linux")
def test_question_mark_is_literal_on_posix(tmp_path):
    _assert_literal_readonly(tmp_path / "question?mode=rw.db")


@pytest.mark.platforms("windows")
def test_native_windows_readonly_paths_and_tracking(tmp_path):
    for filename in ["ordinary.db", "space name.db", "hash#name.db", "literal%23.db", "Unicode-é.db"]:
        _assert_literal_readonly(tmp_path / filename)


@pytest.mark.platforms("macos")
def test_native_macos_readonly_paths_and_tracking(tmp_path):
    for filename in ["ordinary.db", "question?mode=rw.db", "hash#name.db", "literal%23.db", "Unicode-é.db"]:
        _assert_literal_readonly(tmp_path / filename)


@pytest.mark.parametrize("error", [sqlite3.OperationalError("cannot open database"), safe_read.UntrackableConnectionError("tracking refused"), RuntimeError("registry failed")])
def test_tracking_failure_never_reopens_untracked(tmp_path, monkeypatch, error):
    path = tmp_path / "sample.db"
    _make_db(path)
    attempts = []

    def fail_tracked(*args, **kwargs):
        attempts.append(args)
        raise error

    monkeypatch.setattr(safe_read, "connect_tracked", fail_tracked)
    with pytest.raises(type(error)) as raised:
        open_sqlite_readonly(path)
    assert raised.value is error
    assert len(attempts) == 1
    assert not safe_read.has_live_connection(path)


def test_missing_file_is_not_created(tmp_path):
    path = tmp_path / "missing#fragment.db"
    with pytest.raises(sqlite3.OperationalError):
        open_sqlite_readonly(path)
    assert not path.exists()
    assert not (tmp_path / "missing").exists()
    assert not safe_read.has_live_connection(path)


def test_percent_escape_does_not_select_decoy(tmp_path):
    literal = tmp_path / "data%23name.db"
    decoy = tmp_path / "data#name.db"
    _make_db(literal, "literal")
    _make_db(decoy, "decoy")
    with closing(open_sqlite_readonly(literal)) as conn:
        assert conn.execute("SELECT value FROM sample").fetchone()[0] == "literal"
        assert safe_read.has_live_connection(literal)
        assert not safe_read.has_live_connection(decoy)


def test_failed_factory_closes_retained_connection(tmp_path):
    opened = []

    class FailingFactory(sqlite3.Connection):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            opened.append(self)
            raise RuntimeError("factory failed after open")

    path = tmp_path / "sample.db"
    _make_db(path)
    with pytest.raises(RuntimeError, match="factory failed after open"):
        open_sqlite_readonly(path, factory=FailingFactory)
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        opened[0].execute("SELECT 1")
    assert not safe_read.has_live_connection(path)


def test_connection_factory_preserves_close_and_tracking(tmp_path):
    class CustomConnection(sqlite3.Connection):
        closed = False

        def close(self):
            super().close()
            self.closed = True

    path = tmp_path / "sample.db"
    _make_db(path)
    conn = open_sqlite_readonly(path, factory=CustomConnection)
    assert isinstance(conn, CustomConnection)
    assert safe_read.has_live_connection(path)
    conn.close()
    assert conn.closed
    assert not safe_read.has_live_connection(path)
    conn.close()
    assert not safe_read.has_live_connection(path)


@pytest.mark.parametrize("override", [{"uri": False}, {"tracking_path": "different.db"}])
def test_readonly_control_options_cannot_be_overridden(tmp_path, override):
    path = tmp_path / "sample.db"
    _make_db(path)
    with pytest.raises(TypeError):
        open_sqlite_readonly(path, **override)
    assert not safe_read.has_live_connection(path)


def test_failed_tracking_closes_the_partially_opened_connection(tmp_path):
    path = tmp_path / "sample.db"
    _make_db(path)
    opened = []

    def untrackable_opener(database, **kwargs):
        # Deliberately ignore the required factory to simulate a bad opener.
        kwargs.pop("factory")
        conn = sqlite3.connect(database, **kwargs)
        opened.append(conn)
        return conn

    with pytest.raises(safe_read.UntrackableConnectionError):
        open_sqlite_readonly(path, connect_fn=untrackable_opener)
    assert len(opened) == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        opened[0].execute("SELECT 1")
    assert not safe_read.has_live_connection(path)


def test_opening_options_are_forwarded_without_changes(tmp_path):
    path = tmp_path / "sample.db"
    _make_db(path)
    calls = []

    def observed_opener(database, **kwargs):
        calls.append((database, kwargs.copy()))
        return sqlite3.connect(database, **kwargs)

    with closing(open_sqlite_readonly(
        path, connect_fn=observed_opener, timeout=1.25,
        isolation_level=None, check_same_thread=False,
    )) as conn:
        assert conn.isolation_level is None
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 1250
    assert len(calls) == 1
    assert calls[0][1]["uri"] is True
    assert calls[0][1]["timeout"] == 1.25
    assert calls[0][1]["check_same_thread"] is False
    assert calls[0][1]["isolation_level"] is None
    assert not safe_read.has_live_connection(path)


def test_readonly_snapshot_sees_committed_wal(tmp_path):
    path = tmp_path / "live.db"
    with closing(safe_read.connect_tracked(path, isolation_level=None)) as writer:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("CREATE TABLE sample(value TEXT)")
        writer.execute("INSERT INTO sample VALUES ('committed')")
        with closing(open_sqlite_readonly(path, isolation_level=None)) as reader:
            assert reader.execute("SELECT value FROM sample").fetchall() == [("committed",)]
            writer.execute("INSERT INTO sample VALUES ('next')")
            assert reader.execute("SELECT COUNT(*) FROM sample").fetchone()[0] == 2
        assert safe_read.has_live_connection(path)
    assert not safe_read.has_live_connection(path)


@pytest.mark.platforms("linux")
@pytest.mark.parametrize("filename", ["file:literal.db", "file:literal?mode=memory", "literal%2Fname.db", "literal%00name.db", ":memory:"])
def test_uri_like_names_are_literal(tmp_path, monkeypatch, filename):
    monkeypatch.chdir(tmp_path)
    _assert_literal_readonly(Path(filename))


@pytest.mark.parametrize("alias_kind", ["symlink", "hardlink", "relative"])
@pytest.mark.platforms("linux")
def test_path_aliases_refuse_raw_access(tmp_path, monkeypatch, alias_kind):
    path = tmp_path / "sample.db"
    alias = tmp_path / "alias.db"
    _make_db(path)
    if alias_kind == "symlink":
        alias.symlink_to(path)
    elif alias_kind == "hardlink":
        alias.hardlink_to(path)
    else:
        monkeypatch.chdir(tmp_path)
        alias = Path("sample.db")
    with closing(open_sqlite_readonly(path)):
        assert safe_read.has_live_connection(alias)
        assert safe_read.read_header_bytes_preopen(alias) is None
        with pytest.raises(safe_read.LiveConnectionError):
            with safe_read.offline_file_access(alias):
                pytest.fail("raw access was allowed")
    assert not safe_read.has_live_connection(alias)


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("suffix", ["", "-wal", "-shm"])
def test_hardlink_identity_extends_main_and_sidecar_guards(tmp_path, suffix):
    path = tmp_path / "sample.db"
    alias = tmp_path / "unrelated-name.db"
    with closing(safe_read.connect_tracked(path, isolation_level=None)) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE sample(value TEXT)")
        alias.hardlink_to(Path(str(path) + suffix))
        assert safe_read.has_live_connection(alias)
        assert safe_read.read_header_bytes_preopen(alias) is None
        with pytest.raises(safe_read.LiveConnectionError):
            with safe_read.offline_file_access(alias):
                pytest.fail("hardlink alias bypassed the live-file guard")
    assert not safe_read.has_live_connection(alias)


def test_factory_metadata_does_not_disable_tracking(tmp_path):
    class Factory(sqlite3.Connection):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.row_factory = lambda cursor, row: {"custom": row}
            self.text_factory = bytes

        def cursor(self, *args, **kwargs):
            raise RuntimeError("custom cursor rejects metadata")

        def execute(self, sql, *args, **kwargs):
            if sql == "PRAGMA database_list":
                raise sqlite3.OperationalError("custom execute rejects metadata")
            return super().execute(sql, *args, **kwargs)

    path = tmp_path / "sample.db"
    _make_db(path)
    with closing(safe_read.connect_tracked(path, factory=Factory)) as conn:
        assert safe_read.has_live_connection(path)
        assert conn.execute("SELECT value FROM sample").fetchone() == {"custom": (b"literal",)}
    assert not safe_read.has_live_connection(path)


def test_factory_close_failure_retains_tracking(tmp_path):
    class Factory(sqlite3.Connection):
        fail_close = True

        def close(self):
            if self.fail_close:
                raise RuntimeError("close failed")
            super().close()

    path = tmp_path / "sample.db"
    _make_db(path)
    conn = open_sqlite_readonly(path, factory=Factory)
    try:
        with pytest.raises(RuntimeError, match="close failed"):
            conn.close()
        assert safe_read.has_live_connection(path)
        assert safe_read.read_header_bytes_preopen(path) is None
    finally:
        assert isinstance(conn, Factory)
        conn.fail_close = False
        conn.close()
    assert not safe_read.has_live_connection(path)


def test_registration_failure_closes_real_connection(tmp_path, monkeypatch):
    class BrokenRegistry(dict):
        def __setitem__(self, key, value):
            raise RuntimeError("registration failed")

    path = tmp_path / "sample.db"
    _make_db(path)
    opened = []

    def opener(database, **kwargs):
        conn = sqlite3.connect(database, **kwargs)
        opened.append(conn)
        return conn

    monkeypatch.setattr(safe_read, "_live_connections", BrokenRegistry())
    with pytest.raises(RuntimeError, match="registration failed"):
        open_sqlite_readonly(path, connect_fn=opener)
    assert len(opened) == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        opened[0].execute("SELECT 1")
    assert not safe_read.has_live_connection(path)


def test_identity_lookup_failure_is_not_an_untracked_fallback(tmp_path):
    path = tmp_path / "sample.db"
    _make_db(path)
    opened = []

    def opener(database, **kwargs):
        conn = sqlite3.connect(database, **kwargs)
        conn.set_authorizer(lambda *args: sqlite3.SQLITE_DENY)
        opened.append(conn)
        return conn

    with pytest.raises(safe_read.UntrackableConnectionError):
        safe_read.connect_tracked(path, connect_fn=opener)
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        opened[0].execute("SELECT 1")
    assert not safe_read.has_live_connection(path)
