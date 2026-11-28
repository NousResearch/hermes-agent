"""A browse read opens ``state.db`` once.

The Desktop's session reads each take a disposable read-only handle through ``_open_session_db_at_path``.
The declared-schema probe that decides whether the store needs its one browse catch-up runs on that
handle's own connection, so a read pays one SQLite open, as the read-only SessionDB browse did.
"""
from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from fastapi import HTTPException

from hermes_cli.web_server_sessions import _open_session_db_at_path
from persistence.conversation.messages import TranscriptRead
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_holder import omit_session_list_revision
from tests.fixtures.storage_seed import seed_messages, seed_session


def _seeded(home: Path) -> Path:
    (home / "config.yaml").write_text("{}\n", encoding="utf-8")
    with open_persistence(profile_scope(home=home)) as store:
        seed_session(store, "kept", "cli")
        seed_messages(store, "kept", *({"role": "user", "content": f"m{i}"} for i in range(5)))
    return home / "state.db"


def _count_opens(monkeypatch, db_path: Path) -> list:
    import hermes_cli.sqlite_safe_read as ssr

    original, target, opens = ssr.connect_tracked, db_path.resolve(), []

    def counting(path, *args, **kwargs):
        tracked = kwargs.get("tracking_path")
        if tracked is not None and Path(tracked).resolve() == target:
            opens.append(str(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(ssr, "connect_tracked", counting)
    return opens


def test_browse_read_opens_the_store_once(tmp_path, monkeypatch):
    path = _seeded(tmp_path)
    opens = _count_opens(monkeypatch, path)

    store = _open_session_db_at_path(path, read_only=True)
    try:
        assert store.sessions.get("kept") is not None
        assert len(store.messages.read_transcript("kept", TranscriptRead(limit=50, latest=True))) == 5
    finally:
        store.close()
    assert len(opens) == 1, opens


def _lagging(home: Path) -> Path:
    path = _seeded(home)
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("DROP INDEX IF EXISTS idx_sessions_effective_activity")
        omit_session_list_revision(conn)
        conn.execute("ALTER TABLE sessions DROP COLUMN last_read_at")
        conn.commit()
    return path


def _corrupt(home: Path) -> Path:
    path = _seeded(home)
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        page_size = conn.execute("PRAGMA page_size").fetchone()[0]
    with open(path, "r+b") as fh:  # sqlite_master overwritten past the header
        fh.seek(100)
        fh.write(b"\xde\xad\xbe\xef" * ((page_size - 100) // 4))
    return path


@pytest.mark.parametrize("make, status", [(_lagging, None), (_corrupt, 503)])
def test_browse_leaves_the_process_init_error_alone(tmp_path, make, status, request):
    """A browse reports its own failure in its 503; the process's session-history notice stays the owner's."""
    import storage_user_copy

    home = tmp_path / "profiles" / "scribe"
    home.mkdir(parents=True)
    path = make(home)
    previous = storage_user_copy.get_last_init_error()
    request.addfinalizer(lambda: storage_user_copy.set_last_init_error(previous))
    owner_error = "owner open failed: database is locked"
    storage_user_copy.set_last_init_error(owner_error)
    try:
        _open_session_db_at_path(path, read_only=True).close()
        assert status is None
    except HTTPException as exc:
        assert exc.status_code == status
    assert storage_user_copy.get_last_init_error() == owner_error


def test_store_still_behind_after_the_catch_up_is_refused_by_name(tmp_path, monkeypatch):
    from hermes_cli.web_server_sessions import _SCHEMA_UNAVAILABLE

    home = tmp_path / "profiles" / "scribe"
    home.mkdir(parents=True)
    path = _lagging(home)
    monkeypatch.setattr("persistence.sqlite.admin._reconcile.reconcile_drift_for_browse", lambda _path: True)
    with pytest.raises(HTTPException) as caught:
        _open_session_db_at_path(path, read_only=True)
    assert caught.value.status_code == 503
    assert caught.value.detail == _SCHEMA_UNAVAILABLE


@pytest.mark.parametrize("route", ["/api/sessions/kept", "/api/sessions/kept/messages?limit=50&order=latest"])
def test_desktop_session_reads_open_the_store_once(route, monkeypatch, _isolate_hermes_home):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")
    from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN, app
    from hermes_constants import get_hermes_home

    path = _seeded(get_hermes_home())
    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    assert client.get(route).status_code == 200  # first request: app startup work out of the count

    opens = _count_opens(monkeypatch, path)
    response = client.get(route)
    assert response.status_code == 200, response.text
    assert len(opens) == 1, opens
