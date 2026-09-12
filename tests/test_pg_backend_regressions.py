"""Backend authority, reconnect, and cross-host lease regressions."""

import sys
import time
from types import SimpleNamespace

import pytest

import hermes_state
import hermes_state_postgres as pg
from hermes_state_errors import SessionCompressionInProgressError, SessionTurnLeaseLostError


class _RawConnection:
    def __init__(self, *, fail_readonly=False, server_version=160000):
        self.info = SimpleNamespace(server_version=server_version)
        self.executed = []
        self.closed = False
        self.fail_readonly = fail_readonly

    def execute(self, sql):
        self.executed.append(sql)
        if self.fail_readonly:
            raise RuntimeError("read-only setup failed")

    def close(self):
        self.closed = True


def test_reconnect_restores_readonly_before_publishing_connection(monkeypatch):
    old, replacement = _RawConnection(), _RawConnection()
    conn = pg._PostgresConnection(old, "postgresql://test/db")
    conn._read_only = True
    monkeypatch.setitem(sys.modules, "psycopg", SimpleNamespace(connect=lambda *a, **kw: replacement))
    conn._reconnect()
    assert old.closed
    assert conn._conn is replacement
    assert replacement.executed == ["SET default_transaction_read_only = on"]


def test_failed_readonly_reconnect_closes_replacement(monkeypatch):
    old, replacement = _RawConnection(), _RawConnection(fail_readonly=True)
    conn = pg._PostgresConnection(old, "postgresql://test/db")
    conn._read_only = True
    monkeypatch.setitem(sys.modules, "psycopg", SimpleNamespace(connect=lambda *a, **kw: replacement))
    with pytest.raises(RuntimeError, match="reconnect failed"):
        conn._reconnect()
    assert replacement.closed
    assert conn._conn is old


@pytest.mark.parametrize("dsn", [
    "postgresql://user:private-password@example/db?sslpassword=private-password",
    "host=example user=user password=private-password dbname=db",
    "invalid-private-password",
])
def test_dsn_labels_do_not_disclose_credentials(dsn):
    assert "private-password" not in pg._redact_dsn(dsn)


def test_profile_env_handles_export_and_inline_comments(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_STATE_DATABASE_URL", "postgresql://active/other")
    (tmp_path / ".env").write_text(
        'export HERMES_STATE_DATABASE_URL="postgresql://target/db" # profile credential\n')
    assert pg._dsn_from_profile_env(tmp_path) == "postgresql://target/db"
    assert pg.os.environ["HERMES_STATE_DATABASE_URL"] == "postgresql://active/other"


def test_selected_config_cannot_fall_back_when_merged_loader_fails(tmp_path, monkeypatch):
    import hermes_cli.config as config
    import hermes_constants

    (tmp_path / "config.yaml").write_text("sessions:\n  state_backend: postgres\n")
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_STATE_BACKEND", raising=False)

    def broken_loader():
        raise RuntimeError("failed to load other settings")

    monkeypatch.setattr(config, "load_config", broken_loader)
    with pytest.raises(RuntimeError, match="PostgreSQL state is selected"):
        pg.resolve_postgres_dsn()


def test_unknown_backend_is_not_a_sqlite_selection(monkeypatch):
    monkeypatch.delenv("HERMES_STATE_BACKEND", raising=False)
    with pytest.raises(RuntimeError, match="Unknown sessions.state_backend"):
        pg.resolve_postgres_dsn({"sessions": {"state_backend": "postgress"}})


@pytest.fixture
def lease_store(tmp_path, monkeypatch):
    # Execute the common lease/guard code on real rows. Only the host policy
    # flag changes; server isolation is covered by live PostgreSQL tests.
    db = hermes_state.SessionDB(tmp_path / "state.db")
    db.create_session("session", "cli")
    db.append_message("session", "user", "preserve me")
    db._is_postgres = True
    db._COMPRESSION_BUSY_WAIT_S = 0

    def forbidden_local_probe(holder):
        raise AssertionError("a remote PostgreSQL holder was probed on this host")

    monkeypatch.setattr(hermes_state, "_compression_lock_holder_process_is_dead", forbidden_local_probe)
    try:
        yield db
    finally:
        db._is_postgres = False
        db.close()


def test_live_remote_turn_holder_cannot_be_reclaimed(lease_store):
    db = lease_store
    assert db.try_acquire_session_turn_lease("session", "pid=99999999:remote", ttl_seconds=300)
    assert not db.try_acquire_session_turn_lease("session", "pid=99999998:local", ttl_seconds=300)
    with pytest.raises(SessionTurnLeaseLostError):
        db.replace_messages("session", [], reject_active_turn_lease=True)
    assert db.get_messages("session")[0]["content"] == "preserve me"


def test_expired_remote_turn_holder_can_be_reclaimed(lease_store):
    db = lease_store
    assert db.try_acquire_session_turn_lease("session", "pid=99999999:remote", ttl_seconds=300)
    db._write_sql("UPDATE session_turn_leases SET expires_at = ?", (time.time() - 1,))
    assert db.try_acquire_session_turn_lease("session", "pid=99999998:local", ttl_seconds=300)


def test_remote_compression_holder_blocks_destructive_rewrite(lease_store):
    db = lease_store
    now = time.time()
    db._write_sql(
        "INSERT INTO compression_locks (session_id, holder, acquired_at, expires_at) VALUES (?, ?, ?, ?)",
        ("session", "pid=99999999:remote", now, now + 300))
    with pytest.raises(SessionCompressionInProgressError):
        db.replace_messages("session", [], reject_active_turn_lease=True)
    assert db.get_messages("session")[0]["content"] == "preserve me"


@pytest.mark.parametrize("content", [
    "before\x00after",
    '\x01json:{"literal": true}',
    '\x00json:["literal"]',
])
def test_text_encoding_is_lossless_and_postgres_bindable(tmp_path, content):
    pytest.importorskip("psycopg")
    from psycopg._queries import PostgresQuery
    from psycopg.adapt import Transformer

    db = hermes_state.SessionDB(tmp_path / "encoding.db")
    try:
        db.create_session("encoded", "cli")
        db.append_message("encoded", "user", content)
        assert db.get_messages("encoded")[0]["content"] == content
        stored = db._conn.execute("SELECT content FROM messages").fetchone()[0]
        query = PostgresQuery(Transformer())
        query.convert("SELECT %s", (stored,))
    finally:
        db.close()


@pytest.mark.parametrize("server_version", [140015, 150010])
def test_connect_rejects_old_server_before_schema_changes(monkeypatch, server_version):
    raw = _RawConnection(server_version=server_version)
    monkeypatch.setitem(sys.modules, "psycopg", SimpleNamespace(connect=lambda *a, **kw: raw))
    with pytest.raises(RuntimeError, match="requires PostgreSQL 16 or newer"):
        pg.connect_postgres("postgresql://test/db")
    assert raw.closed
    assert raw.executed == []


@pytest.mark.parametrize("server_version", [160000, 170003])
def test_connect_accepts_supported_server_without_ddl(monkeypatch, server_version):
    raw = _RawConnection(server_version=server_version)
    monkeypatch.setitem(sys.modules, "psycopg", SimpleNamespace(connect=lambda *a, **kw: raw))
    conn = pg.connect_postgres("postgresql://test/db")
    try:
        assert conn.raw is raw
        assert raw.executed == []
    finally:
        conn.close()


def test_reconnect_rejects_old_server_without_publishing_connection(monkeypatch):
    old = _RawConnection()
    replacement = _RawConnection(server_version=150010)
    conn = pg._PostgresConnection(old, "postgresql://test/db")
    monkeypatch.setitem(sys.modules, "psycopg", SimpleNamespace(connect=lambda *a, **kw: replacement))
    with pytest.raises(RuntimeError, match="requires PostgreSQL 16 or newer"):
        conn._reconnect()
    assert replacement.closed
    assert conn._conn is old
    assert replacement.executed == []
