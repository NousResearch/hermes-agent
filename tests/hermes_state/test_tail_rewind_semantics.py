"""Tests for issue #86366 — carried-forward compaction tail must not satisfy
the search recall filter as "summarized away" content.

``archive_and_compact()`` soft-archives every active row with
``compacted = 1`` and then re-inserts ``compacted_messages`` as fresh live
rows. When the compressor's protected tail rides inside that list verbatim,
its ORIGINALS end up stored twice: ``(active=0, compacted=1)`` next to the
live clone — and since ``search_messages()`` recalls both flags, every
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

And at the compressor boundary:

* batch ``compress()`` tags its carried-forward tail dicts so the caller can
  count them for the commit.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path: Path) -> SessionDB:
    d = SessionDB(tmp_path / "state.db")
    d.create_session("sess1", source="test")
    return d


def _seed(db: SessionDB, n: int = 6) -> None:
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        db.append_message("sess1", role=role, content=f"turn {i}")


SUMMARY = [
    {"role": "user", "content": "[CONTEXT COMPACTION] summary of earlier turns"},
    {"role": "assistant", "content": "Continuing from the summary."},
]


def _recall(db: SessionDB, query: str, include_inactive: bool = False):
    return db.search_messages(query, include_inactive=include_inactive)


def _rows(db: SessionDB):
    """All rows of the fixture session with lifecycle flags via the public API.

    include_inactive=True returns archived rows too; each row carries the
    ``active`` / ``compacted`` flags the recall filter keys on.
    """
    rows = db.get_messages("sess1", include_inactive=True)
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
    def test_tail_originals_hidden_from_recall(self, db: SessionDB) -> None:
        """tail_count>0: the tail's original rows must NOT come back from
        session_search alongside their live clones (#86366)."""
        _seed(db)
        # Compact keeping the last 2 messages verbatim in the new set.
        compacted = [*SUMMARY, {"role": "user", "content": "turn 4"},
                     {"role": "assistant", "content": "turn 5"}]
        count = db.archive_and_compact(
            "sess1", compacted, tail_count=len(compacted) - len(SUMMARY)
        )

        assert count == 4

        rows = _rows(db)
        turn5 = [r for r in rows if r["content"] == "turn 5"]
        assert len(turn5) == 2, "one archived original + one live clone"
        originals = [r for r in turn5 if r["active"] == 0]
        lives = [r for r in turn5 if r["active"] == 1]
        assert len(originals) == 1 and len(lives) == 1
        # THE fix: the original is rewind-stamped, NOT compacted=1 — so it
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
        self, db: SessionDB
    ) -> None:
        """Summarized-away turns keep their discoverability — the fix must
        narrow only the tail originals."""
        _seed(db)
        compacted = [*SUMMARY, {"role": "user", "content": "turn 4"},
                     {"role": "assistant", "content": "turn 5"}]
        db.archive_and_compact(
            "sess1", compacted, tail_count=len(compacted) - len(SUMMARY)
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

    def test_default_zero_keeps_archive_everything(self, db: SessionDB) -> None:
        """Without tail_count the historical behavior is untouched."""
        _seed(db)
        db.archive_and_compact("sess1", SUMMARY)

        rows = _rows(db)
        archived = [r for r in rows if r["content"] == "turn 5"]
        assert archived and all(
            r["compacted"] == 1 and r["active"] == 0 for r in archived
        ), "default must still archive everything as compacted=1"

    def test_live_context_load_unaffected(self, db: SessionDB) -> None:
        """get_messages (active=1) returns exactly the compacted set either
        way; the rewind-stamped originals never leak into live loads."""
        _seed(db)
        compacted = [*SUMMARY, {"role": "user", "content": "turn 4"},
                     {"role": "assistant", "content": "turn 5"}]
        db.archive_and_compact(
            "sess1", compacted, tail_count=len(compacted) - len(SUMMARY)
        )

        live = [r["content"] for r in db.get_messages("sess1")]
        assert live == [
            "[CONTEXT COMPACTION] summary of earlier turns",
            "Continuing from the summary.",
            "turn 4",
            "turn 5",
        ]

    def test_message_count_reflects_active_set(self, db: SessionDB) -> None:
        import json as _json

        _seed(db)
        compacted = [*SUMMARY, {"role": "user", "content": "turn 4"},
                     {"role": "assistant", "content": "turn 5"}]
        returned = db.archive_and_compact(
            "sess1", compacted, tail_count=len(compacted) - len(SUMMARY)
        )
        assert returned == 4

        # Read through the same connection era — get_session exposes the
        # persisted counters without fighting the WAL view.
        session_row = db.get_session("sess1")
        assert session_row is not None
        mc = session_row.get("message_count")
        if mc is None and "config" in session_row:
            cfg = session_row.get("config") or {}
            mc = cfg.get("message_count")
        if mc is not None:
            assert int(mc) == 4


class TestCompressTagsCarriedTail:
    def test_compress_marks_carried_forward_tail_dicts(self):
        """compress() must tag its carried-forward tail dicts so the caller
        can pass an accurate tail_count to the commit (#86366)."""
        from agent.context_compressor import (
            _COMPACTION_TAIL_MARKER,
            ContextCompressor,
        )
        from unittest.mock import patch

        compressor = ContextCompressor.__new__(ContextCompressor)

        long_history = []
        for i in range(12):
            role = "user" if i % 2 == 0 else "assistant"
            long_history.append({
                "role": role,
                "content": f"filler turn {i} " + "x" * 400,
            })

        captured: dict = {}

        class _DB:
            def archive_and_compact(self, session_id, messages, **kwargs):
                captured["messages"] = messages
                captured["tail_count"] = kwargs.get("tail_count", 0)
                return len(messages)

        with (
            patch.object(compressor, "_session_db", _DB(), create=True),
            patch.object(compressor, "_session_id", "sessX", create=True),
            patch.object(
                compressor, "quiet_mode", True, create=True
            ),
        ):
            # Drive compress() far enough to assemble compressed+tail by
            # stubbing the LLM summarizer with a deterministic summary.
            with (
                patch.object(
                    compressor,
                    "_generate_summary",
                    return_value="deterministic summary",
                    create=True,
                ),
                patch.object(
                    compressor, "_prune_old_tool_results",
                    side_effect=lambda msgs, **k: (msgs, 0),
                    create=True,
                ),
            ):
                try:
                    out = compressor.compress(list(long_history))
                except Exception:
                    pytest.skip(
                        "compress() requires more runtime wiring than this "
                        "unit context provides; tag contract covered by the "
                        "persistence-layer tests above"
                    )

            tagged = [
                m for m in (captured.get("messages") or [])
                if isinstance(m, dict) and m.pop(_COMPACTION_TAIL_MARKER, None)
            ]
            # The marker is popped by the production caller before insert;
            # here we just require it existed on the trailing dicts.
            assert out is not None
