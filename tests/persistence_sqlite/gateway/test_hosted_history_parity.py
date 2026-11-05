"""The hosted-room authority-history read of the SQLite backend on a room log handed from one host to another and back."""

from __future__ import annotations

import json
import sqlite3

from persistence.sqlite.gateway import _hosted_work as backend


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


def _log(events):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("""CREATE TABLE hosted_room_events (room_id TEXT, seq INTEGER, kind TEXT, actor_json TEXT,
                    authority_epoch INTEGER, payload_json TEXT)""")
    for seq, (kind, actor, epoch, payload) in enumerate(events, start=1):
        conn.execute("INSERT INTO hosted_room_events VALUES (?, ?, ?, ?, ?, ?)",
                     (ROOM, seq, kind, json.dumps(actor), epoch, json.dumps(payload)))
    return conn


def test_a_handed_off_room_reads_one_unbroken_chain():
    history = backend.read_history_locked(_log(HANDOFF), ROOM, gateway_id="gw-a", epoch=3)
    assert [(span["gateway_id"], span["epoch"]) for span in history] == [("gw-a", 1), ("gw-b", 2), ("gw-a", 3)]
