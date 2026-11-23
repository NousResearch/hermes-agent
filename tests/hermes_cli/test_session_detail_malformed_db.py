"""A corrupt state.db must not be reported as a missing session.

Regression for the 2026-08-31 incident: ``state.db`` corruption (rowid out of
order in ``sessions``, wrong entry count in ``idx_messages_session_id``) made
``resolve_session_id`` raise ``sqlite3.DatabaseError: database disk image is
malformed``. The Desktop showed "Couldn't open this session - session
unavailable" and the main process logged ``404: {"detail":"Session not
found"}`` — a session that existed was reported as absent, which sent the
diagnosis in entirely the wrong direction for a day.

A corrupt store is an unavailable store, not an empty one.
"""

import sqlite3
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from persistence.core.errors import CorruptStoreError, PermanentStoreError

from hermes_cli import web_server
import hermes_cli.web_server_sessions as _web_server_sessions
from hermes_cli.web_routers import sessions as sessions_router


class _Store:
    """Storage interface stand-in whose id lookup is ``resolve_id``; records its close."""

    def __init__(self, resolve_id):
        self.closed = False
        self.sessions = SimpleNamespace(resolve_id=resolve_id)

    def close(self):
        self.closed = True


def _corrupt_lookup(session_id):
    """Fails the way a corrupt file really fails."""
    raise CorruptStoreError("database disk image is malformed")


@pytest.fixture
def malformed_db(monkeypatch):
    store = _Store(_corrupt_lookup)
    monkeypatch.setattr(
        _web_server_sessions, "_open_session_db_for_profile", lambda profile, *, read_only: store
    )
    return store


@pytest.mark.asyncio
async def test_corrupt_db_is_not_reported_as_missing_session(malformed_db):
    with pytest.raises(HTTPException) as excinfo:
        await sessions_router.get_session_detail("20260830_180820_744f05")

    assert excinfo.value.status_code != 404, (
        "corruption reported as 'Session not found' - this is the bug"
    )
    assert excinfo.value.status_code == 503


@pytest.mark.asyncio
async def test_corrupt_db_detail_names_the_real_cause(malformed_db):
    with pytest.raises(HTTPException) as excinfo:
        await sessions_router.get_session_detail("20260830_180820_744f05")

    detail = str(excinfo.value.detail).lower()
    assert "corrupt" in detail or "malformed" in detail


@pytest.mark.asyncio
async def test_db_is_closed_even_when_corruption_raises(malformed_db):
    with pytest.raises(HTTPException):
        await sessions_router.get_session_detail("20260830_180820_744f05")

    assert malformed_db.closed, "connection leaked on the corruption path"


# ── The same lookup runs in four more handlers ──────────────────────────────
#
# delete is the dangerous one: an unresolvable id is treated as idempotent
# success ({"ok": True, "already_absent": True}), so a corrupt store made
# DELETE report that it had removed a session that is still on disk — and the
# Desktop drops the sidebar row on that answer.


@pytest.mark.asyncio
async def test_messages_endpoint_reports_corruption(malformed_db):
    with pytest.raises(HTTPException) as excinfo:
        await sessions_router.get_session_messages(
            "20260830_180820_744f05", None, None, 0, None, False
        )
    assert excinfo.value.status_code == 503


class _CorruptAuthorityDB:
    """Authority store whose first write fails the way a corrupt file really fails."""

    db_path = "/nonexistent/state.db"

    def _validate_import_payload(self, sessions):  # pragma: no cover - import only
        return sessions, []


def _mutation_request(monkeypatch):
    """Delete/rename never open a SessionDB in the handler: they admit through the session
    authority, so corruption surfaces from the authority write, not from a handler-opened store."""
    import gateway.session_mutations as _mutations

    authority = SimpleNamespace(db=_CorruptAuthorityDB(), profile_id="p", epoch=1)
    principal = object()
    monkeypatch.setattr(_web_server_sessions, "_session_mutation_context", lambda request, profile: (authority, principal))

    async def _mutate(*_args, **_kwargs):
        raise sqlite3.DatabaseError("database disk image is malformed")

    monkeypatch.setattr(_mutations, "mutate_session", _mutate)
    return SimpleNamespace()


@pytest.mark.asyncio
async def test_delete_does_not_claim_success_on_a_corrupt_store(monkeypatch):
    request = _mutation_request(monkeypatch)
    with pytest.raises(HTTPException) as excinfo:
        await sessions_router.delete_session_endpoint(
            "20260830_180820_744f05", request, request_id="r1", expected_revision=1
        )
    assert excinfo.value.status_code == 503


@pytest.mark.asyncio
async def test_rename_endpoint_reports_corruption(monkeypatch):
    from hermes_cli.web_models import SessionRename

    request = _mutation_request(monkeypatch)
    with pytest.raises(HTTPException) as excinfo:
        await sessions_router.rename_session_endpoint(
            "20260830_180820_744f05", SessionRename(title="neu", request_id="r1", expected_revision=1), request
        )
    assert excinfo.value.status_code == 503


@pytest.mark.asyncio
async def test_export_endpoint_reports_corruption(malformed_db):
    with pytest.raises(HTTPException) as excinfo:
        await sessions_router.export_session_endpoint("20260830_180820_744f05")
    assert excinfo.value.status_code == 503


# ── A genuinely absent session must still be a 404, not a 503 ──────────────


@pytest.mark.asyncio
async def test_absent_session_is_still_404(monkeypatch):
    monkeypatch.setattr(
        _web_server_sessions,
        "_open_session_db_for_profile",
        lambda profile, *, read_only: _Store(lambda sid: None),
    )
    with pytest.raises(HTTPException) as excinfo:
        await sessions_router.get_session_detail("does_not_exist")
    assert excinfo.value.status_code == 404


# ── Unrelated DatabaseErrors must not be relabelled as corruption ──────────


def _other_error_lookup(session_id):
    raise PermanentStoreError("some unrelated database failure")


@pytest.mark.asyncio
async def test_non_corruption_database_error_is_not_swallowed(monkeypatch):
    monkeypatch.setattr(
        _web_server_sessions,
        "_open_session_db_for_profile",
        lambda profile, *, read_only: _Store(_other_error_lookup),
    )
    with pytest.raises(PermanentStoreError) as excinfo:
        await sessions_router.get_session_detail("20260830_180820_744f05")
    assert not isinstance(excinfo.value, CorruptStoreError)


# ── The same two failures as the file raises them, through the backend's own translation ──


@pytest.fixture
def real_store_whose_lookup_raises(monkeypatch, tmp_path):
    """Make the id lookup of a real storage interface handle raise ``sqlite3.DatabaseError(message)``, so the
    backend's own error translation decides what the route sees."""
    from persistence.sqlite.conversation import sessions as sqlite_sessions
    from persistence_assembly import open_persistence, profile_scope

    def install(message):
        def failing_lookup(holder, session_id_or_prefix):
            raise sqlite3.DatabaseError(message)

        monkeypatch.setattr(sqlite_sessions, "resolve_session_id", failing_lookup)
        monkeypatch.setattr(
            _web_server_sessions, "_open_session_db_for_profile",
            lambda profile, *, read_only: open_persistence(profile_scope(home=tmp_path)))

    return install


@pytest.mark.asyncio
async def test_malformed_file_error_from_the_backend_is_503(real_store_whose_lookup_raises):
    real_store_whose_lookup_raises("database disk image is malformed")
    with pytest.raises(HTTPException) as excinfo:
        await sessions_router.get_session_detail("20260830_180820_744f05")
    assert excinfo.value.status_code == 503
    assert "malformed" in str(excinfo.value.detail).lower()


@pytest.mark.asyncio
async def test_unrelated_database_error_from_the_backend_is_not_relabelled(real_store_whose_lookup_raises):
    real_store_whose_lookup_raises("some unrelated database failure")
    with pytest.raises(PermanentStoreError) as excinfo:
        await sessions_router.get_session_detail("20260830_180820_744f05")
    assert not isinstance(excinfo.value, CorruptStoreError)
    assert isinstance(excinfo.value.__cause__, sqlite3.DatabaseError)
