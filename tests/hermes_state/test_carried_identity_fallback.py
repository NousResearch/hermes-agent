"""Regression for est-qtya — a carried-forward dict whose timestamp was lost or
re-stamped by the handoff/continuation re-derivation must still resolve to its
ACTIVE durable original when the identity matches exactly ONE active row.

Live shape (clankville state.db, session 20261003_180729_01b81ee4): a
"finish this up" continuation re-inserted the post-merge tail as fresh rows
sharing message_uid/content/tool_call_id with the originals but stamped at
the flush instant (all copies share one timestamp, microseconds apart). The
old resolver keyed its fallback on (identity, timestamp), so a re-stamped or
missing stamp resolved NOTHING: the original stayed active beside the
re-inserted copy — the ghost tail (25 duplicated tool-result rows; md5s
verified identical, e.g. ids 19289==19425, 19309==19445).

The fix adds an identity-WITHOUT-timestamp fallback, gated on a UNIQUE active
row per identity, so the ghost copy's original is rewind-stamped
(active=0, compacted=0) at the commit instead of projecting twice. The
uniqueness gate is pinned here too: an ambiguous duplicate must never be
rewound on identity alone.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path: Path) -> SessionDB:
    d = SessionDB(tmp_path / "state.db")
    d.create_session("sess1", source="test")
    return d


def _rows(db: SessionDB):
    return [
        {"id": r.get("id"), "active": r.get("active"), "compacted": r.get("compacted"),
         "content": r.get("content")}
        for r in db.get_messages("sess1", include_inactive=True)
    ]


def _seed_tool_tail(db: SessionDB):
    db.append_message("sess1", role="user", content="kick off the merge")
    db.append_message(
        "sess1", role="assistant", content="",
        tool_calls=[{"id": "call-1", "type": "function",
                     "function": {"name": "terminal", "arguments": "{}"}}],
    )
    db.append_message("sess1", role="tool", content='{"output": "rc=0 mergeCommit"}',
                      tool_call_id="call-1", tool_name="terminal")
    db.append_message(
        "sess1", role="assistant", content="",
        tool_calls=[{"id": "call-2", "type": "function",
                     "function": {"name": "terminal", "arguments": "{}"}}],
    )
    db.append_message("sess1", role="tool", content="Error executing tool 'terminal': timed out after 420.0s",
                      tool_call_id="call-2", tool_name="terminal")


def _carried_without_timestamps(db: SessionDB, contents):
    carried = [m for m in db.get_messages_as_conversation("sess1")
               if m.get("content") in contents]
    assert carried
    for m in carried:
        m.pop("timestamp", None)  # the handoff re-derivation loses the stamp
    return carried


def test_carried_dict_without_timestamp_rewinds_its_original(db: SessionDB) -> None:
    """A stamp-less carried dict must resolve to its unique active original, so
    the commit rewind-stamps the original instead of leaving it live beside the
    freshly inserted copy (the ghost tail)."""
    _seed_tool_tail(db)
    carried = _carried_without_timestamps(
        db, {"kick off the merge", "Error executing tool 'terminal': timed out after 420.0s"})

    compacted = [
        {"role": "user", "content": "kick off the merge"},
        {"role": "assistant", "content": "[CONTEXT COMPACTION] summary of the merge exchange"},
        {"role": "tool", "content": "Error executing tool 'terminal': timed out after 420.0s"},
    ]
    db.archive_and_compact("sess1", compacted, carried_messages=carried)

    rows = _rows(db)
    timeout_rows = [r for r in rows
                    if r["content"] == "Error executing tool 'terminal': timed out after 420.0s"]
    assert len(timeout_rows) == 2, "one rewind-stamped original + one live copy"
    live = [r for r in timeout_rows if r["active"] == 1]
    originals = [r for r in timeout_rows if r["active"] == 0]
    assert len(live) == 1 and len(originals) == 1, (
        "the ghost tail is exactly the original staying active beside the copy"
    )
    assert originals[0]["compacted"] == 0, "resolved carried originals are superseded, not summarized"

    kick = [r for r in rows if r["content"] == "kick off the merge"]
    assert sum(1 for r in kick if r["active"] == 1) == 1


def test_ambiguous_identity_is_never_rewound_on_identity_alone(db: SessionDB) -> None:
    """The uniqueness gate: two ACTIVE rows share the identity and the carried
    dict has no usable stamp — no original may be rewind-stamped on identity
    alone (a false rewind would silently hide live history)."""
    db.append_message("sess1", role="user", content="ambiguous turn")
    db.append_message("sess1", role="user", content="ambiguous turn")
    carried = _carried_without_timestamps(db, {"ambiguous turn"})
    assert len(carried) == 2

    before = {r["id"]: (r["active"], r["compacted"]) for r in _rows(db)}
    db.archive_and_compact(
        "sess1",
        [{"role": "assistant", "content": "[CONTEXT COMPACTION] summary"}],
        carried_messages=carried[:1],
    )
    after = {r["id"]: (r["active"], r["compacted"]) for r in _rows(db) if r["id"] in before}
    for row_id, flags in after.items():
        assert flags != (0, 0), (
            f"ambiguous identity must not rewind row {row_id}"
        )
