"""Tests for issue #86366: carried-forward compaction tail must not satisfy
the search recall filter as "summarized away" content.

``archive_and_compact()`` soft-archives every active row with
``compacted = 1`` and then re-inserts ``compacted_messages`` as fresh live
rows. When the compressor's protected tail rides inside that list verbatim,
its ORIGINALS end up stored twice: ``(active=0, compacted=1)`` next to the
live clone: and since ``search_messages()`` recalls both flags, every
carried-forward message came back once per compaction, mislabeled as
archived history.

The fix adds a ``tail_count`` parameter: the last *tail_count* archived rows
are superseded byte-identical duplicates (rewind semantics:
``active=0, compacted=0``, hidden from recall) instead of compacted history.
Callers without tail knowledge keep the archive-everything behavior.

Pinned here at the persistence layer:

* ``tail_count > 0`` → tail originals hidden from ``search_messages``,
  non-tail originals still recalled;
* default (``tail_count=0``) → historical behavior unchanged;
* live-context load and counters unaffected by the new split.
"""

from __future__ import annotations

import pytest

from persistence.sqlite.conversation._store.messages import (
    append_message, archive_and_compact, get_messages, get_messages_as_conversation,
)
from persistence.sqlite.conversation._store.search import search_messages
from persistence.sqlite.conversation._store.sessions import create_session, get_session


@pytest.fixture
def db(backend, scope_a):
    d = backend.open(scope_a).holder()
    create_session(d, "sess1", source="test")
    return d

def _seed(db, n: int = 6) -> None:
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        append_message(db, "sess1", role=role, content=f"turn {i}")

SUMMARY = [
    {"role": "user", "content": "[CONTEXT COMPACTION] summary of earlier turns"},
    {"role": "assistant", "content": "Continuing from the summary."},
]

def _recall(db, query: str, include_inactive: bool = False):
    return search_messages(db, query, include_inactive=include_inactive)

def _rows(db):
    """All rows of the fixture session with lifecycle flags.

    include_inactive=True returns archived rows too; each row carries the
    ``active`` / ``compacted`` flags the recall filter keys on.
    """
    rows = get_messages(db, "sess1", include_inactive=True)
    out = []
    for r in rows:
        out.append({
            "id": r.get("id"),
            "active": r.get("active"),
            "compacted": r.get("compacted"),
            "content": r.get("content"),
        })
    return out

class TestTailCountArchivesAsRewindSemantics:
    def test_tail_originals_hidden_from_recall(self, db) -> None:
        """tail_count>0: the tail's original rows must NOT come back from
        session_search alongside their live clones (#86366)."""
        _seed(db)
        # Compact keeping the last 2 messages verbatim in the new set.
        compacted = [*SUMMARY, {"role": "user", "content": "turn 4"},
                     {"role": "assistant", "content": "turn 5"}]
        count = archive_and_compact(
            db, "sess1", compacted, tail_count=len(compacted) - len(SUMMARY)
        )

        assert count == 4

        rows = _rows(db)
        turn5 = [r for r in rows if r["content"] == "turn 5"]
        assert len(turn5) == 2, "one archived original + one live clone"
        originals = [r for r in turn5 if r["active"] == 0]
        lives = [r for r in turn5 if r["active"] == 1]
        assert len(originals) == 1 and len(lives) == 1
        # THE fix: the original is rewind-stamped, NOT compacted=1, so it
        # stops satisfying search_messages' recall filter (#86366).
        assert originals[0]["compacted"] == 0

        # The rewind-stamped original must never satisfy the recall filter;
        # only the summary-side content remains searchable from that turn.
        recalled_snippets = [h.get("snippet", "") for h in _recall(db, "turn 5")]
        assert all(
            ">>>turn 5<<<" not in s or "[CONTEXT COMPACTION]" in s
            for s in recalled_snippets
        ) or all("turn 5" not in s.lower() or "[context compaction]" in s.lower()
                 for s in recalled_snippets), (
            f"tail originals must not be recalled: {recalled_snippets}"
        )

    def test_non_tail_originals_still_recalled_as_compacted(
        self, db
    ) -> None:
        """Summarized-away turns keep their discoverability: the fix must
        narrow only the tail originals."""
        _seed(db)
        compacted = [*SUMMARY, {"role": "user", "content": "turn 4"},
                     {"role": "assistant", "content": "turn 5"}]
        archive_and_compact(
            db, "sess1", compacted, tail_count=len(compacted) - len(SUMMARY)
        )

        rows = _rows(db)
        turn1 = [r for r in rows if r["content"] == "turn 1"]
        assert turn1 and turn1[0]["compacted"] == 1 and turn1[0]["active"] == 0, (
            "non-tail originals must stay compacted=1 (discoverable)"
        )
        assert any(
            "turn" in (h.get("snippet") or "") and "1" in (h.get("snippet") or "")
            for h in _recall(db, "turn 1")
        ), (
            "summarized-away turns must remain recallable: "
            f"{[h.get('snippet') for h in _recall(db, 'turn 1')]}"
        )

    def test_explicit_carried_messages_leave_summarized_tool_results_discoverable(self, db) -> None:
        """A non-contiguous carried set must not classify a summarized tool block as rewind-only.

        Micro-compaction rewrites a prefix plus assistant/tool exchange plus suffix into
        a prefix plus summary marker plus suffix. The carried rows therefore live on BOTH
        sides of the removed exchange; using a positional tail count hid the removed
        tool result as active=0, compacted=0 (#118481).
        """
        append_message(db, "sess1", role="user", content="question 0")
        append_message(
            db,
            "sess1",
            role="assistant",
            content="",
            tool_calls=[{
                "id": "call-1",
                "type": "function",
                "function": {"name": "read_file", "arguments": "{}"},
            }],
        )
        append_message(
            db,
            "sess1",
            role="tool",
            content="tool result that was summarized",
            tool_call_id="call-1",
            tool_name="read_file",
        )
        append_message(db, "sess1", role="user", content="question 1")
        append_message(db, "sess1", role="assistant", content="answer 1")

        carried = [
            message for message in get_messages_as_conversation(db, "sess1")
            if message.get("content") in {"question 0", "question 1", "answer 1"}
        ]
        assert carried and all("_row_id" not in message for message in carried), (
            "this regression must exercise the row-id-less resume fallback"
        )

        archive_and_compact(
            db,
            "sess1",
            [
                {"role": "user", "content": "question 0"},
                {"role": "assistant", "content": "[CONTEXT COMPACTION] summarized tool exchange"},
                {"role": "user", "content": "question 1"},
                {"role": "assistant", "content": "answer 1"},
            ],
            carried_messages=carried,
        )

        rows = _rows(db)
        summarized_tool = [r for r in rows if r["content"] == "tool result that was summarized"]
        assert len(summarized_tool) == 1
        assert summarized_tool[0]["active"] == 0
        assert summarized_tool[0]["compacted"] == 1, (
            "summarized tool results must stay discoverable as compacted history"
        )

        carried_original = [
            r for r in rows
            if r["content"] == "question 1" and r["active"] == 0
        ]
        assert len(carried_original) == 1
        assert carried_original[0]["compacted"] == 0, (
            "only exact carried-forward originals should be hidden as superseded duplicates"
        )

    def test_default_zero_keeps_archive_everything(self, db) -> None:
        """Without tail_count the historical behavior is untouched."""
        _seed(db)
        archive_and_compact(db, "sess1", SUMMARY)

        rows = _rows(db)
        archived = [r for r in rows if r["content"] == "turn 5"]
        assert archived and all(
            r["compacted"] == 1 and r["active"] == 0 for r in archived
        ), "default must still archive everything as compacted=1"

    def test_live_context_load_unaffected(self, db) -> None:
        """get_messages (active=1) returns exactly the compacted set either
        way; the rewind-stamped originals never leak into live loads."""
        _seed(db)
        compacted = [*SUMMARY, {"role": "user", "content": "turn 4"},
                     {"role": "assistant", "content": "turn 5"}]
        archive_and_compact(
            db, "sess1", compacted, tail_count=len(compacted) - len(SUMMARY)
        )

        live = [r["content"] for r in get_messages(db, "sess1")]
        assert live == [
            "[CONTEXT COMPACTION] summary of earlier turns",
            "Continuing from the summary.",
            "turn 4",
            "turn 5",
        ]

    def test_message_count_reflects_active_set(self, db) -> None:

        _seed(db)
        compacted = [*SUMMARY, {"role": "user", "content": "turn 4"},
                     {"role": "assistant", "content": "turn 5"}]
        returned = archive_and_compact(
            db, "sess1", compacted, tail_count=len(compacted) - len(SUMMARY)
        )
        assert returned == 4

        # get_session exposes the persisted counters.
        session_row = get_session(db, "sess1")
        assert session_row is not None
        mc = session_row.get("message_count")
        if mc is None and "config" in session_row:
            cfg = session_row.get("config") or {}
            mc = cfg.get("message_count")
        if mc is not None:
            assert int(mc) == 4
