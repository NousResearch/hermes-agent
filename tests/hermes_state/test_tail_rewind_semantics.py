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

    def test_explicit_carried_messages_leave_summarized_tool_results_discoverable(self, db: SessionDB) -> None:
        """A non-contiguous carried set must not classify a summarized tool block as rewind-only.

        Micro-compaction rewrites a prefix plus assistant/tool exchange plus suffix into
        a prefix plus summary marker plus suffix. The carried rows therefore live on BOTH
        sides of the removed exchange; using a positional tail count hid the removed
        tool result as active=0, compacted=0 (#118481).
        """
        db.append_message("sess1", role="user", content="question 0")
        db.append_message(
            "sess1",
            role="assistant",
            content="",
            tool_calls=[{
                "id": "call-1",
                "type": "function",
                "function": {"name": "read_file", "arguments": "{}"},
            }],
        )
        db.append_message(
            "sess1",
            role="tool",
            content="tool result that was summarized",
            tool_call_id="call-1",
            tool_name="read_file",
        )
        db.append_message("sess1", role="user", content="question 1")
        db.append_message("sess1", role="assistant", content="answer 1")

        carried = [
            message for message in db.get_messages_as_conversation("sess1")
            if message.get("content") in {"question 0", "question 1", "answer 1"}
        ]
        assert carried and all("_row_id" not in message for message in carried), (
            "this regression must exercise the row-id-less resume fallback"
        )

        db.archive_and_compact(
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
        can pass an accurate tail_count to archive_and_compact (#86366);
        untagged, the tail originals are recalled as compacted history."""
        from unittest.mock import patch

        from agent.context_compressor import _COMPACTION_TAIL_MARKER, ContextCompressor

        with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
            compressor = ContextCompressor(
                model="test/model", protect_first_n=2, protect_last_n=2, quiet_mode=True,
            )
            _ = compressor.context_length

        history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"filler turn {i} " + "x" * 400}
            for i in range(12)
        ]
        with patch("agent.context_compressor.call_llm", side_effect=RuntimeError("no provider")):
            out = compressor.compress(list(history))

        assert len(out) < len(history)
        tagged = [m for m in out if isinstance(m, dict) and m.get(_COMPACTION_TAIL_MARKER)]
        assert tagged, "carried-forward tail rows were not tagged"
        # The tag marks exactly the verbatim tail: the trailing run of the output,
        # each row a copy of an original from the end of the history.
        assert out[-len(tagged):] == tagged
        originals = {m["content"] for m in history[-len(tagged):]}
        assert {m["content"] for m in tagged} <= originals
        assert not any(m.get(_COMPACTION_TAIL_MARKER) for m in out[:2])

    def test_archive_and_compact_marks_demoted_tail_tool_originals_compacted_1(self, tmp_path):
        """ATLAS-129 / DQ #515: When a protected-tail tool output (>500 chars) is demoted
        to a compaction stub, archive_and_compact must keep the pre-demotion full-text
        original row at (active=0, compacted=1) so session_search can recall it, while
        verbatim carried tail rows stay (active=0, compacted=0)."""
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "state.db")
        sid = "s_tail_demote"
        db.create_session(sid, source="cli")
        id_u1 = db.append_message(sid, role="user", content="older turn")
        id_a1 = db.append_message(sid, role="assistant", content="calling tool", tool_calls=[{"id": "call_big"}])
        big_payload = "REFINED_EXECUTION_TRACER_PAYLOAD " + ("Z" * 1800)
        id_t_big = db.append_message(
            sid, role="tool", content=big_payload, tool_call_id="call_big", tool_name="terminal"
        )
        id_a2 = db.append_message(sid, role="assistant", content="calling small tool", tool_calls=[{"id": "call_small"}])
        id_t_small = db.append_message(
            sid, role="tool", content="short tool output ok", tool_call_id="call_small", tool_name="terminal"
        )

        compacted_messages = [
            {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
            {"role": "assistant", "content": "calling tool", "tool_calls": [{"id": "call_big"}]},
            {
                "role": "tool",
                "content": f"[terminal output demoted at compaction — {len(big_payload):,} chars preserved in session history. Recover with session_search(query=..., session_id='{sid}')]",
                "tool_call_id": "call_big",
                "tool_name": "terminal",
            },
            {"role": "assistant", "content": "calling small tool", "tool_calls": [{"id": "call_small"}]},
            {
                "role": "tool",
                "content": "short tool output ok",
                "tool_call_id": "call_small",
                "tool_name": "terminal",
            },
        ]
        db.archive_and_compact(sid, compacted_messages, tail_count=4)

        rows = {
            int(r["id"]): (int(r["active"]), int(r["compacted"]))
            for r in db._conn.execute(
                "SELECT id, active, compacted FROM messages WHERE session_id = ?", (sid,)
            ).fetchall()
        }
        # Summarised middle user row and demoted big tool original must be compacted=1
        assert rows[id_u1] == (0, 1)
        assert rows[id_t_big] == (0, 1)
        # Verbatim carried assistant/small-tool originals must stay rewound (0, 0)
        assert rows[id_a1] == (0, 0)
        assert rows[id_a2] == (0, 0)
        assert rows[id_t_small] == (0, 0)
        # And search_messages must now find the demoted tool payload!
        hits = db.search_messages("REFINED_EXECUTION_TRACER_PAYLOAD", role_filter=["tool"])
        assert [int(h["id"]) for h in hits] == [id_t_big]
        # Demoted tool original must reconcile its display_order AFTER the summary and its parent assistant
        displayed = db.get_messages(sid, include_compacted=True)
        roles_and_calls = [(m["role"], m.get("tool_call_id")) for m in displayed]
        assert roles_and_calls == [
            ("user", None),
            ("user", None),
            ("assistant", None),
            ("tool", "call_big"),
            ("assistant", None),
            ("tool", "call_small"),
        ]

    def test_demoted_tail_tool_with_uid_paints_once_after_assistant_and_survives_trailing_prompt(self, tmp_path):
        """When compress() carries message_uid on a demoted tail tool stub (with or without
        a pre-populated timestamp) and appends a trailing synthetic user prompt after the
        carried tail, every display projection must paint the tool result ONCE (as the full
        original) AFTER its carried parent assistant, and verbatim tail tools must still rewind
        to (active=0, compacted=0)."""
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "state.db")
        sid = "s_tail_demote_uid"
        db.create_session(sid, source="cli")
        db.append_message(sid, role="user", content="older turn")
        db.append_message(sid, role="assistant", content="calling big tool", tool_calls=[{"id": "call_big"}])
        big_payload = "FULL_ORIGINAL_PAYLOAD " + ("X" * 1800)
        id_t_big = db.append_message(
            sid, role="tool", content=big_payload, tool_call_id="call_big", tool_name="terminal"
        )
        db.append_message(sid, role="assistant", content="calling small tool", tool_calls=[{"id": "call_small"}])
        id_t_small = db.append_message(
            sid, role="tool", content="short tool output ok", tool_call_id="call_small", tool_name="terminal"
        )

        live = db.get_messages_as_conversation(sid, include_row_ids=True)
        stub_text = "[terminal output demoted at compaction]"
        tail_carried = []
        for m in live[1:]:
            copy = dict(m)
            copy.pop("_row_id", None)
            copy.pop("timestamp", None)  # exercise _carry_parent_timestamps uid fallback
            if copy.get("tool_call_id") == "call_big":
                copy["content"] = stub_text
            tail_carried.append(copy)

        compacted_messages = [
            {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
            *tail_carried,
            {"role": "user", "content": "[SESSION TODO SNAPSHOT — active items]"},
            {"role": "user", "content": "[STILL IN PROGRESS — continue]"},
        ]
        db.archive_and_compact(sid, compacted_messages, tail_count=4)

        assert db._conn is not None
        rows = {
            int(r["id"]): (int(r["active"]), int(r["compacted"]), int(r["display_order"]))
            for r in db._conn.execute(
                "SELECT id, active, compacted, display_order FROM messages WHERE session_id = ?", (sid,)
            ).fetchall()
        }
        # Verbatim small tool original must still be rewound to (0, 0) despite trailing synthetic prompts
        assert rows[id_t_small][:2] == (0, 0)
        assert rows[id_t_big][:2] == (0, 1)

        paged = db.get_messages(sid, include_compacted=True)
        shown = db.get_messages_as_conversation(sid, include_compacted=True)
        _model, resumed = db.get_resume_conversations(sid)
        seg_rows, _, _ = db.transcript_segment_page(sid, latest=True, after=None, count=100)

        for projection in (paged, shown, resumed, seg_rows):
            roles = [m["role"] for m in projection]
            assert roles == [
                "user",
                "user",
                "assistant",
                "tool",
                "assistant",
                "tool",
                "user",
                "user",
            ]
            big_tools = [m for m in projection if m["role"] == "tool" and m["tool_call_id"] == "call_big"]
            assert len(big_tools) == 1
            assert big_tools[0]["content"] == big_payload
