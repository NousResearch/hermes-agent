"""The hosted-room authority-history read and the work-record lineage check of the SQLite backend against main's
(gateway/hosted_room_authority_history.py, gateway/hosted_room_work_lineage.py, read from the pinned checkout), on one
room log.

Outcomes are compared by value, or by error class name and message.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from persistence.sqlite.gateway import _hosted_work as backend
from tests.persistence_sqlite._legacy_store import legacy_import, legacy_store

__all__ = ["legacy_store"]

ROOM = "room-1"
CONTROL = {"kind": "system", "id": "authority-control"}


def _transfer(kind, previous, owner, epoch):
    return kind, CONTROL, epoch, {"previous_gateway_id": previous, "authority_gateway_id": owner,
                                  "authority_epoch": epoch}


def _message(owner, epoch):
    return "message.member", {"kind": "gateway", "id": owner}, epoch, {"text": "hi"}


# A room handed from one host to another and back: the outgoing owner's fence, then the new owner's claim.
HANDOFF = [_message("gw-a", 1), _transfer("authority.lost", "gw-a", "gw-b", 2), _message("gw-b", 2),
           _transfer("authority.claimed", "gw-b", "gw-a", 3), _message("gw-a", 3)]
CLAIMS = [_message("gw-a", 1), _transfer("authority.claimed", "gw-a", "gw-b", 2), _message("gw-b", 2)]
FORK = [_message("gw-a", 1), _transfer("authority.lost", "gw-a", "gw-b", 2),
        _transfer("authority.claimed", "gw-c", "gw-a", 3), _message("gw-a", 3)]
SKIPPED = [_message("gw-a", 1), _transfer("authority.lost", "gw-a", "gw-b", 3), _message("gw-b", 3)]


def _log(events):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("""CREATE TABLE hosted_room_events (room_id TEXT, seq INTEGER, kind TEXT, actor_json TEXT,
                    authority_epoch INTEGER, payload_json TEXT)""")
    for seq, (kind, actor, epoch, payload) in enumerate(events, start=1):
        conn.execute("INSERT INTO hosted_room_events VALUES (?, ?, ?, ?, ?, ?)",
                     (ROOM, seq, kind, json.dumps(actor), epoch, json.dumps(payload)))
    return conn


def _outcome(call):
    try:
        result = call()
    except Exception as exc:  # compared by class name and message
        return "error", type(exc).__name__, str(exc)
    if isinstance(result, tuple):  # the lineage check's spans and digest; each module has its own span class
        spans, digest = result
        return "ok", [(span.gateway_id, span.epoch, span.from_seq) for span in spans], digest
    return "ok", result


@pytest.mark.parametrize("events, owner, epoch", [
    (HANDOFF, "gw-a", 3), (CLAIMS, "gw-b", 2), (FORK, "gw-a", 3), (SKIPPED, "gw-b", 3)],
    ids=["handoff", "claims", "fork", "skipped-epoch"])
def test_history_and_lineage_match_the_gateway(legacy_store, events, owner, epoch):
    gateway_history = legacy_import("gateway.hosted_room_authority_history")
    gateway_lineage = legacy_import("gateway.hosted_room_work_lineage")
    conn = _log(events)
    assert _outcome(lambda: backend.read_history_locked(conn, ROOM, gateway_id=owner, epoch=epoch)) == _outcome(
        lambda: gateway_history.read_history_locked(conn, ROOM, gateway_id=owner, epoch=epoch))
    authority = {"gateway_id": owner, "epoch": epoch}
    for seq in range(1, len(events) + 1):
        ours = _outcome(lambda: backend.source_prefix_locked(conn, ROOM, authority, seq))
        theirs = _outcome(lambda: gateway_lineage.source_prefix_locked(conn, ROOM, authority, seq))
        assert ours == theirs, seq


def test_a_handed_off_room_reads_one_unbroken_chain():
    history = backend.read_history_locked(_log(HANDOFF), ROOM, gateway_id="gw-a", epoch=3)
    assert [(span["gateway_id"], span["epoch"]) for span in history] == [("gw-a", 1), ("gw-b", 2), ("gw-a", 3)]
