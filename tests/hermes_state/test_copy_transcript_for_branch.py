"""Tests for SessionDB.copy_transcript_for_branch — the durable-verbatim branch seed.

A branch child's transcript must be the parent's transcript: tool calls/results, reasoning
sidecars, api_content bytes and compaction flags, copied row-for-row in one INSERT..SELECT
per chunk. The prose rebuild it replaces dropped every tool-bearing row, so the child's
first turn sent the provider a context the parent never had and models that validate
tool-call/tool-result alternation rejected every tool turn of the branched session.
"""

import json
import sqlite3

import pytest

from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path):
    d = SessionDB(db_path=tmp_path / "state.db")
    yield d
    d.close()


def _tool_turn(db, session_id, n):
    """A canonical tool-bearing turn: user -> assistant(tool_calls) -> tool -> assistant(text)."""
    db.append_messages_batch(session_id, [
        {"role": "user", "content": f"question {n}"},
        {"role": "assistant", "content": f"let me check {n}",
         "tool_calls": [{"name": "terminal", "arguments": json.dumps({"command": f"echo {n}"})}],
         "reasoning_content": f"thinking {n}", "finish_reason": "tool_calls"},
        {"role": "tool", "content": f"output {n}", "tool_name": "terminal", "tool_call_id": f"call_{n}"},
        {"role": "assistant", "content": f"answer {n}", "finish_reason": "stop"},
    ])


_COPYED_COLUMNS = (
    "role", "content", "tool_call_id", "tool_calls", "tool_name", "timestamp", "finish_reason",
    "reasoning_content", "api_content", "display_kind", "message_uid", "tool_call_uids")


def _raw_rows(conn, session_id):
    order = ", ".join(_COPYED_COLUMNS)
    return conn.execute(
        f"SELECT {order} FROM messages WHERE session_id = ? ORDER BY id", (session_id,)).fetchall()


class TestVerbatimCopy:
    def test_child_rows_match_parent_row_for_row(self, db, tmp_path):
        """Every copied column is byte-identical: tools, reasoning, api_content, identity."""
        db.create_session("parent", source="desktop")
        _tool_turn(db, "parent", 1)
        _tool_turn(db, "parent", 2)
        db.create_session("child", source="desktop")

        inserted = db.copy_transcript_for_branch("parent", "child")

        assert inserted == 8
        child_db = sqlite3.connect(tmp_path / "state.db")
        try:
            parent_rows = _raw_rows(child_db, "parent")
            child_rows = _raw_rows(child_db, "child")
            assert len(child_rows) == len(parent_rows) == 8
            for pr, cr in zip(parent_rows, child_rows):
                assert tuple(pr) == tuple(cr)
        finally:
            child_db.close()

    def test_child_model_projection_equals_parent(self, db):
        """The headline fix: the branched conversation's MODEL context keeps its tool calls/results."""
        db.create_session("parent", source="desktop")
        _tool_turn(db, "parent", 1)
        _tool_turn(db, "parent", 2)
        db.create_session("child", source="desktop")

        db.copy_transcript_for_branch("parent", "child")

        parent_model, _ = db.get_resume_conversations("parent")
        child_model, _ = db.get_resume_conversations("child")
        assert [(m.get("role"), m.get("content"), m.get("tool_calls"), m.get("tool_call_id"))
                for m in child_model] == \
               [(m.get("role"), m.get("content"), m.get("tool_calls"), m.get("tool_call_id"))
                for m in parent_model]
        # The tool-bearing turn is intact in the child, not narration-only. The model
        # projection ABSORBS the tool result into the assistant row (by design — that is
        # how a tool turn resumes), so the durable evidence is the absorbed row id the
        # child's tool-call assistant row carries. A prose-rebuilt child has no such row.
        absorbed = [m for m in child_model if m.get("_absorbed_row_ids")]
        assert absorbed, child_model
        assert absorbed[0].get("finish_reason") == "tool_calls"

    def test_platform_message_id_is_not_duplicated(self, db, tmp_path):
        """A platform row id must not exist on two sessions (CLI /branch parity)."""
        db.create_session("parent", source="desktop")
        db.append_message(session_id="parent", role="user", content="hi", platform_message_id="plat-1")
        db.create_session("child", source="desktop")

        db.copy_transcript_for_branch("parent", "child")

        conn = sqlite3.connect(tmp_path / "state.db")
        try:
            child_plat = conn.execute(
                "SELECT platform_message_id FROM messages WHERE session_id = 'child'").fetchall()
            assert all(p[0] is None for p in child_plat)
            parent_plat = conn.execute(
                "SELECT platform_message_id FROM messages WHERE session_id = 'parent'").fetchall()
            assert parent_plat == [("plat-1",)]
        finally:
            conn.close()

    def test_parent_is_never_mutated_and_child_is_session_scoped(self, db, tmp_path):
        db.create_session("parent", source="desktop")
        _tool_turn(db, "parent", 1)
        db.create_session("child", source="desktop")

        before = _raw_rows(sqlite3.connect(tmp_path / "state.db"), "parent")
        db.copy_transcript_for_branch("parent", "child")

        conn = sqlite3.connect(tmp_path / "state.db")
        try:
            assert _raw_rows(conn, "parent") == before  # copy is read-only on the source
            stray = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id NOT IN ('parent', 'child')").fetchone()[0]
            assert stray == 0
        finally:
            conn.close()

    def test_counters_set_from_child_rows(self, db):
        db.create_session("parent", source="desktop")
        _tool_turn(db, "parent", 1)
        _tool_turn(db, "parent", 2)
        db.create_session("child", source="desktop")

        db.copy_transcript_for_branch("parent", "child")

        child = db.get_session("child")
        assert child["message_count"] == 8
        assert child["tool_call_count"] == 2


class TestCompactionFlags:
    def test_tip_flags_kept_and_ancestor_rows_archived(self, db, tmp_path):
        """A compacted parent branches WITHOUT re-inflating its model context: the child's model
        projection is the tip's active rows (summary + tail), while its display projection keeps
        the full lineage the desktop branch paints."""
        db.create_session("root", source="desktop")
        _tool_turn(db, "root", 1)
        # Simulate in-place compaction: the root's rows become display-only archive and the session
        # ends 'compression' (what publish_compression_child stamps)...
        db.end_session("root", "compression")
        conn = sqlite3.connect(tmp_path / "state.db")
        try:
            conn.execute("UPDATE messages SET active = 0, compacted = 1 WHERE session_id = 'root'")
            conn.commit()
        finally:
            conn.close()
        # ...and a compression child carries the summary + a live tail.
        db.create_session("tip", source="desktop", parent_session_id="root")
        db.append_messages_batch("tip", [
            {"role": "assistant", "content": "summary of earlier turns", "_compressed_summary": True},
        ])
        _tool_turn(db, "tip", 2)
        db.create_session("child", source="desktop")

        db.copy_transcript_for_branch("tip", "child")

        child_model, child_display = db.get_resume_conversations("child")
        tip_model, _ = db.get_resume_conversations("tip")
        # The child's model context IS the tip's, verbatim: summary + live tail, never re-inflating
        # the compacted root. The resume preamble is a gateway-layer concern (prepended at resume
        # time, never stored in a projection), so the projection itself carries no marker row.
        assert [m.get("content") for m in child_model] == [
            "summary of earlier turns", "question 2", "let me check 2", "answer 2"]
        assert [m.get("content") for m in child_model] == [m.get("content") for m in tip_model]
        assert child_model[0].get("_compressed_summary") is True  # summary flag copied, not re-derived
        assert "question 1" in json.dumps([m.get("content") for m in child_display])
        # The child's tool turn survived the compaction mapping (absorbed result row).
        assert any(m.get("_absorbed_row_ids") for m in child_model)
        conn = sqlite3.connect(tmp_path / "state.db")
        try:
            archive = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id = 'child' AND active = 0"
                " AND compacted = 1").fetchone()[0]
            assert archive == 4  # root's rows landed display-only, never model-visible
        finally:
            conn.close()


class TestCountBoundary:
    def test_up_to_row_id_bounds_the_copy(self, db, tmp_path):
        db.create_session("parent", source="desktop")
        _tool_turn(db, "parent", 1)
        _tool_turn(db, "parent", 2)
        db.create_session("child", source="desktop")

        conn = sqlite3.connect(tmp_path / "state.db")
        try:
            # Boundary at the first turn's closing assistant row: rows 1..4 land (turn 1
            # with its tool call + result), turn 2 stays in the parent.
            turn_one_end = conn.execute(
                "SELECT id FROM messages WHERE session_id = 'parent' AND content = 'answer 1'"
            ).fetchone()[0]
        finally:
            conn.close()

        inserted = db.copy_transcript_for_branch("parent", "child", up_to_row_id=turn_one_end)

        assert inserted == 4
        child_model, _ = db.get_resume_conversations("child")
        assert [(m.get("role"), m.get("content")) for m in child_model] == [
            ("user", "question 1"), ("assistant", "let me check 1"), ("assistant", "answer 1")]
        # The bounded turn is COMPLETE: its tool call and result both landed (the result
        # rides absorbed in the tool-call row — no dangling call in the child).
        assert any(m.get("_absorbed_row_ids") for m in child_model)


class TestChunkingAndSchema:
    def test_copy_is_chunked_and_complete_on_large_transcripts(self, db):
        """1500 rows copy across chunk boundaries with every row landing (efficiency: one
        INSERT..SELECT per chunk, not a Python decode/encode round-trip per row)."""
        db.create_session("parent", source="desktop")
        rows = [{"role": "user" if i % 2 else "assistant", "content": f"row {i}"} for i in range(1500)]
        db.append_messages_batch("parent", rows, chunk_rows=500)
        db.create_session("child", source="desktop")

        inserted = db.copy_transcript_for_branch("parent", "child", chunk_rows=500)

        assert inserted == 1500
        assert db.get_session("child")["message_count"] == 1500

    def test_branch_copy_column_list_matches_schema(self, tmp_path):
        """A new messages column must never be silently dropped from branch children: the copy's
        column list is pinned to the table schema (everything except the rowid ``id``)."""
        d = SessionDB(db_path=tmp_path / "state.db")
        try:
            conn = sqlite3.connect(tmp_path / "state.db")
            try:
                schema_cols = {r[1] for r in conn.execute("PRAGMA table_info(messages)").fetchall()}
            finally:
                conn.close()
            copy_cols = set(d._BRANCH_COPY_COLUMNS)
            assert copy_cols == schema_cols - {"id"}
        finally:
            d.close()
