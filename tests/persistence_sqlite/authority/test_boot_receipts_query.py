"""The boot's receipt query reads the local receipts through the key index, not a table scan.

The boot restores the local receipts with one query over ``state_meta``. It matched the receipt
prefix with ``LIKE``, which SQLite cannot serve from an index here: ``state_meta.key`` keeps the
default BINARY collation, and SQLite only uses an index for a prefix ``LIKE`` when the column is
NOCASE (or ``case_sensitive_like`` is on). So the plan was a full scan of ``state_meta`` on every
boot - walking rows the query never reads (the ``gateway.admission.result:*`` blobs alone are
hundreds of MB on a long-lived install). ``GLOB`` is case-sensitive and index-backed, so the same
receipts arrive through a range scan of ``sqlite_autoindex_state_meta_1`` - the convention the
store's own local helpers already use for this very prefix ("GLOB uses the key index").

The receipts themselves must not change: the same rows come back, and the same ones are dropped.
"""
from __future__ import annotations

import json
from typing import Any

from persistence.sqlite.authority import _local
from persistence.sqlite.conversation._store.local import POLICY_PREFIX
from persistence.sqlite.conversation._store.retirement import VIEWER_CLOSE_PREFIX as CLOSE_PREFIX

# The query this replaced: the same rows, planned as a full scan of ``state_meta``.
_LIKE_RECEIPTS_SQL = _local._BOOT_RECEIPTS_SQL.replace("m.key GLOB ?", "m.key LIKE ?")

_START = len(POLICY_PREFIX) + 1

# A receipt whose session row is open, one whose session ended, one whose session was compressed,
# one with an unsettled admission, one with a stored close, and one whose session row is gone.
_KEPT = ["sid-admitted", "sid-closing", "sid-compressed", "sid-open", "sid-orphan"]
_DROPPED = ["sid-ended"]


def _receipt(session_id: str) -> str:
    """A local creation receipt: its value names the session it points at."""
    route = "agent:main:local:dm:" + session_id
    return json.dumps({"entry": {"session_id": session_id, "session_key": route},
                       "policy": {}, "route": route})


def _seed(holder: Any) -> None:
    """One receipt per boot case, plus the rows this query must never return."""
    def write(conn):
        for name, ended_at, end_reason in (("open", None, None), ("ended", 100.0, "user_exit"),
                                           ("compressed", 100.0, "compression"),
                                           ("admitted", 100.0, "user_exit"),
                                           ("closing", 100.0, "user_exit")):
            conn.execute("INSERT INTO sessions(id, source, started_at, ended_at, end_reason) "
                         "VALUES(?,?,?,?,?)", ("sid-" + name, "local", 1.0, ended_at, end_reason))
        conn.execute("INSERT INTO session_admissions(admission_id, request_id, principal_id, "
                     "target_session_id, lineage_json, payload_json, payload_digest, intent, status) "
                     "VALUES(?,?,?,?,?,?,?,?,?)",
                     ("adm-1", "r1", "owner", "sid-admitted", "[]", "{}", "d", "queue", "queued"))
        rows = [(POLICY_PREFIX + "sid-" + name, _receipt("sid-" + name))
                for name in ("open", "ended", "compressed", "admitted", "closing")]
        # Its session row is gone: the receipt still restores (the join finds nothing).
        rows.append((POLICY_PREFIX + "sid-orphan", _receipt("sid-missing")))
        rows.append((CLOSE_PREFIX + "sid-closing", "1"))
        rows.append(("goal:sid-gone", '{"x":1}'))
        rows.append(("gateway.admission.result:abc", '{"y":2}'))
        conn.executemany("INSERT INTO state_meta(key, value) VALUES(?,?)", rows)

    holder._execute_write(write)


def _ids(rows) -> list[str]:
    return sorted(row[0][len(POLICY_PREFIX):] for row in rows)


def test_the_boot_receipts_query_reads_the_key_index(store):
    """The plan seeks the receipts through the key index instead of scanning the table."""
    holder = store.holder()
    _seed(holder)
    with holder._read_ctx() as conn:
        steps = [row[3] for row in conn.execute("EXPLAIN QUERY PLAN " + _local._BOOT_RECEIPTS_SQL,
                                                (POLICY_PREFIX + "*", _START, CLOSE_PREFIX, _START))]
    assert not any(step.startswith("SCAN m") for step in steps), steps
    assert any("USING INDEX" in step and "state_meta" in step for step in steps), steps


def test_the_boot_receipts_query_returns_the_same_receipts_as_the_scan(store):
    """Same rows as the LIKE scan it replaced: the keep and drop cases are unchanged."""
    holder = store.holder()
    _seed(holder)
    with holder._read_ctx() as conn:
        got = _ids(conn.execute(_local._BOOT_RECEIPTS_SQL,
                                 (POLICY_PREFIX + "*", _START, CLOSE_PREFIX, _START)).fetchall())
        reference = _ids(conn.execute(_LIKE_RECEIPTS_SQL,
                                       (POLICY_PREFIX + "%", _START, CLOSE_PREFIX, _START)).fetchall())
    assert got == reference
    assert got == _KEPT
    assert not set(got) & set(_DROPPED)


def test_local_receipt_ids_returns_the_boot_receipts(store):
    """The boot helper itself reads the same receipts (the function, not just its SQL)."""
    holder = store.holder()
    _seed(holder)
    assert sorted(_local.local_receipt_ids(holder)) == _KEPT


def test_a_receipt_outside_the_prefix_is_never_returned(store):
    """The other ``state_meta`` rows (goals, admission results, closes) are not receipts."""
    holder = store.holder()
    _seed(holder)
    with holder._read_ctx() as conn:
        keys = [row[0] for row in conn.execute("SELECT key FROM state_meta")]
    assert "goal:sid-gone" in keys and "gateway.admission.result:abc" in keys
    ids = _local.local_receipt_ids(holder)
    assert not any(not sid.startswith("sid-") for sid in ids), ids


def test_the_bindings_boot_verb_returns_the_boot_receipts(store):
    """The verb the gateway's boot actually calls reads the same receipts."""
    holder = store.holder()
    _seed(holder)
    assert sorted(str(sid) for sid in store.bindings.boot_session_ids()) == _KEPT
