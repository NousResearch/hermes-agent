"""Round-trip export/import must restore archived message state (#122679).

The JSONL transfer projection feeds ``import_sessions``: rows in-place compaction
archived must come back archived: visible in the display history, never as live
model context: or a restored compacted session silently loses every turn the
summary replaced. ``active``/``compacted`` are the row state; absent flags (older
exports) keep meaning live rows.
"""

from __future__ import annotations

import json

from persistence.sqlite import SQLiteBackend
from persistence.sqlite.conversation._store.messages import append_message, archive_and_compact, get_messages
from persistence.sqlite.conversation._store.sessions import create_session, get_session
from persistence.sqlite.conversation._store.titles import set_session_title
from tests.contracts._scopes import profile_scope
from tests.persistence_sqlite.conftest import profile_home, store_for

STRANDED_ID = "20260823_043331_c93770"


def _seed_and_compact(holder, turns=4):
    create_session(holder, STRANDED_ID, source="tui")
    set_session_title(holder, STRANDED_ID, "Bot Chat")
    for i in range(1, turns + 1):
        append_message(holder, STRANDED_ID, "user", f"question {i}")
        append_message(holder, STRANDED_ID, "assistant", f"answer {i}")
    tail = get_messages(holder, STRANDED_ID)[-2:]
    archive_and_compact(holder,
        STRANDED_ID, [{"role": "user", "content": "[summary of turns 1-3]"}, *tail], tail_count=2)


def _shape(holder, **flags):
    return [(m["role"], m["content"], m.get("active", 1), m.get("compacted", 0))
            for m in get_messages(holder, STRANDED_ID, **flags)]


def test_export_all_round_trips_compacted_history(backend, scope_a, scope_b):
    src_store, dst_store = backend.open(scope_a), backend.open(scope_b)
    src, dst = src_store.holder(), dst_store.holder()
    _seed_and_compact(src)
    shown_before = _shape(src, include_compacted=True)
    live_before = _shape(src)
    all_before = _shape(src, include_inactive=True)
    assert len(shown_before) > len(live_before), "compacted turns must be archived, not deleted"

    # The transfer projection through real JSONL (what `sessions export` writes).
    payload = json.loads(json.dumps([{**export.session.as_dict(), "messages": [m.as_wire_row() for m in export.messages]}
                                     for export in src_store.maintenance.export_all(include_inactive=True)]))

    result = dst_store.maintenance.import_sessions(payload)

    assert result.ok and result.imported == 1, result
    assert _shape(dst, include_inactive=True) == all_before, "every row must survive with its state"
    assert _shape(dst, include_compacted=True) == shown_before, "display history must be unchanged"
    assert _shape(dst) == live_before, "archived turns must not re-enter live model context"
    # Session counters count live rows only.
    assert get_session(dst, STRANDED_ID)["message_count"] == len(live_before)

    # Hand-edited/foreign JSONL: string flags ("0" is truthy) and null for live rows must not
    # flip archived rows live or live rows archived.
    for msg in payload[0]["messages"]:
        msg["active"], msg["compacted"] = (str(msg["active"]) if not msg["active"] else None,
                                           str(msg["compacted"]))
    scope_c = profile_scope("profile-c")
    profile_home(scope_c.name)
    edited = SQLiteBackend({scope_c.hermes_home_key: store_for(scope_c)})
    try:
        edited_store = edited.open(scope_c)
        assert edited_store.maintenance.import_sessions(payload).ok
        assert _shape(edited_store.holder(), include_inactive=True) == all_before
    finally:
        edited.close()
