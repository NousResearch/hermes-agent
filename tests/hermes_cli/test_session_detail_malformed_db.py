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

import pytest
from fastapi import HTTPException

from hermes_cli import web_server
import hermes_cli.web_server_sessions as _web_server_sessions
from hermes_cli.web_routers import sessions as sessions_router


class _MalformedDB:
    """SessionDB stand-in that fails the way a corrupt file really fails."""

    def __init__(self):
        self.closed = False

    def resolve_session_id(self, session_id):
        raise sqlite3.DatabaseError("database disk image is malformed")

    def get_session(self, sid):  # pragma: no cover - never reached
        raise AssertionError("resolve_session_id should have raised first")

    def close(self):
        self.closed = True


@pytest.fixture
def malformed_db(monkeypatch):
    db = _MalformedDB()
    monkeypatch.setattr(
        _web_server_sessions, "_open_session_db_for_profile", lambda profile, *, read_only: db
    )
    return db


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


@pytest.mark.asyncio
async def test_delete_does_not_claim_success_on_a_corrupt_store(malformed_db):
    with pytest.raises(HTTPException) as excinfo:
        await sessions_router.delete_session_endpoint("20260830_180820_744f05")
    assert excinfo.value.status_code == 503


@pytest.mark.asyncio
async def test_rename_endpoint_reports_corruption(malformed_db):
    from hermes_cli.web_models import SessionRename

    with pytest.raises(HTTPException) as excinfo:
        await sessions_router.rename_session_endpoint(
            "20260830_180820_744f05", SessionRename(title="neu")
        )
    assert excinfo.value.status_code == 503


@pytest.mark.asyncio
async def test_export_endpoint_reports_corruption(malformed_db):
    with pytest.raises(HTTPException) as excinfo:
        await sessions_router.export_session_endpoint("20260830_180820_744f05")
    assert excinfo.value.status_code == 503


# ── A genuinely absent session must still be a 404, not a 503 ──────────────


class _EmptyDB:
    def __init__(self):
        self.closed = False

    def resolve_session_id(self, session_id):
        return None

    def get_session(self, sid):
        return None

    def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_absent_session_is_still_404(monkeypatch):
    monkeypatch.setattr(
        _web_server_sessions,
        "_open_session_db_for_profile",
        lambda profile, *, read_only: _EmptyDB(),
    )
    with pytest.raises(HTTPException) as excinfo:
        await sessions_router.get_session_detail("does_not_exist")
    assert excinfo.value.status_code == 404


# ── Unrelated DatabaseErrors must not be relabelled as corruption ──────────


class _OtherErrorDB(_EmptyDB):
    def resolve_session_id(self, session_id):
        raise sqlite3.DatabaseError("some unrelated database failure")


@pytest.mark.asyncio
async def test_non_corruption_database_error_is_not_swallowed(monkeypatch):
    monkeypatch.setattr(
        _web_server_sessions,
        "_open_session_db_for_profile",
        lambda profile, *, read_only: _OtherErrorDB(),
    )
    with pytest.raises(sqlite3.DatabaseError):
        await sessions_router.get_session_detail("20260830_180820_744f05")
