"""``lineage_adoption.adopt_session_lineage`` over two reference stores: the donor's compression lineage imported into the adopting store and the donor segments retired.
"""

from __future__ import annotations

import logging
import time

import pytest

from lineage_adoption import adopt_session_lineage
from persistence import SessionId
from persistence.conversation.messages import NewMessage, TranscriptRead, TranscriptView

NOW = time.time()
DAY = 86400.0


def _payload(sid, started, ended=None, end_reason=None, parent=None):
    return {"id": sid, "source": "cli", "started_at": started, "ended_at": ended, "end_reason": end_reason, "model": "model-a",
            "parent_session_id": parent, "messages": [{"role": "user", "content": "hello", "timestamp": started},
                                                      {"role": "assistant", "content": "hi", "timestamp": started + 1}]}


@pytest.fixture
def stores(backend, scope_a, scope_b, monkeypatch):
    hooks: list = []
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", lambda name, **payload: hooks.append(payload))
    local, donor = backend.open(scope_a), backend.open(scope_b)
    assert donor.maintenance.import_sessions([_payload("s-aroot", NOW - 2 * DAY, NOW - 2 * DAY + 10, "compression"),
                                              _payload("s-atip", NOW - 2 * DAY + 20, parent="s-aroot")]).ok
    yield local, donor
    # CONTRACT: an adoption never notifies a plugin: the session lives on in the adopting profile.
    assert hooks == []


def _rows(store, sid):
    return len(store.messages.read_transcript(SessionId(sid), TranscriptRead(view=TranscriptView.ALL)))


def test_adoption_imports_the_lineage_and_retires_the_donor(stores):
    local, donor = stores
    result = adopt_session_lineage(local, donor, "s-atip")
    # CONTRACT: the import dict in its key order, then ``adopted`` and ``donor_retired``.
    assert list(result) == ["ok", "imported", "skipped", "detached", "imported_ids", "skipped_ids", "errors", "adopted", "donor_retired"]
    assert result == {"ok": True, "imported": 2, "skipped": 0, "detached": 0, "imported_ids": ["s-aroot", "s-atip"], "skipped_ids": [],
                      "errors": [], "adopted": True, "donor_retired": True}
    donor_rows = {sid: donor.maintenance.export_session(SessionId(sid)).session for sid in ("s-aroot", "s-atip")}
    # CONTRACT: every donor segment is ended as adopted and archived, never deleted, without the sweep's provenance.
    assert {sid: [row.end_reason, row.archived, row.auto_archived, row.ended_at is not None] for sid, row in donor_rows.items()} == {
        "s-aroot": ["adopted_by_profile", 1, 0, True], "s-atip": ["adopted_by_profile", 1, 0, True]}
    local_rows = {sid: local.maintenance.export_session(SessionId(sid)).session for sid in ("s-aroot", "s-atip")}
    assert {sid: [row.end_reason, row.archived, row.parent_session_id] for sid, row in local_rows.items()} == {
        "s-aroot": ["compression", 0, None], "s-atip": [None, 0, "s-aroot"]}
    assert {sid: _rows(local, sid) for sid in ("s-aroot", "s-atip")} == {"s-aroot": 2, "s-atip": 2}


def test_adoption_again_skips_and_retires_again(stores):
    local, donor = stores
    adopt_session_lineage(local, donor, "s-atip")
    # CONTRACT: idempotent; the retired root no longer continues into the tip, so the lineage is the tip alone.
    assert adopt_session_lineage(local, donor, "s-atip") == {
        "ok": True, "imported": 0, "skipped": 1, "detached": 0, "imported_ids": [], "skipped_ids": ["s-atip"], "errors": [],
        "adopted": True, "donor_retired": True}


def test_a_donor_ahead_of_the_local_copy_is_not_retired(stores, caplog):
    local, donor = stores
    adopt_session_lineage(local, donor, "s-atip")
    donor.messages.append(SessionId("s-atip"), NewMessage(role="user", content="more"))
    caplog.set_level(logging.DEBUG, logger="hermes_state")
    assert adopt_session_lineage(local, donor, "s-atip") == {
        "ok": True, "imported": 0, "skipped": 1, "detached": 0, "imported_ids": [], "skipped_ids": ["s-atip"], "errors": [],
        "adopted": True, "donor_retired": False}
    assert [(r.levelname, r.getMessage()) for r in caplog.records if r.name == "hermes_state" and r.levelname != "DEBUG"] == [
        ("WARNING", "adoption divergence: donor segment s-atip has 3 messages, local copy has 2 — donor will NOT be retired")]


def test_an_unknown_donor_session_is_refused(stores):
    local, donor = stores
    assert adopt_session_lineage(local, donor, "nope") == {
        "ok": False, "adopted": False, "donor_retired": False, "error": "session 'nope' not found in donor store"}


def test_adoption_without_retirement_leaves_the_donor(stores):
    local, donor = stores
    adopt_session_lineage(local, donor, "s-atip")
    assert adopt_session_lineage(local, donor, "s-aroot", retire_donor=False) == {
        "ok": True, "imported": 0, "skipped": 1, "detached": 0, "imported_ids": [], "skipped_ids": ["s-aroot"], "errors": [],
        "adopted": True, "donor_retired": False}
