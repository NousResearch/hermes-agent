"""Regression tests for the cached session summary (issue #45103).

These exercise the REAL SessionDB against a REAL sqlite file — the code
under test is never mocked. Only the LLM call itself is substituted,
because a test must not spend an API call.

Pinned here:

  1. The three summary columns exist on a NORMAL database — they are
     declared in ``SCHEMA_SQL`` (``hermes_state_common.py``), so a
     freshly created DB already has them.
  2. ``summarize_session()`` persists the summary together with the model
     that produced it and the time it was produced.
  3. A failed LLM call keeps the last good summary in place instead of
     clearing it.
  4. Only user and assistant text reaches the summarizer prompt — tool
     output and system prompts never do.
"""

import time

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    d = SessionDB(db_path=tmp_path / "state.db")
    yield d
    d.close()


def _make_session(db, session_id="s1", *, ended=True, messages=8, summary=None,
                  summary_updated_at=None, summary_model=None):
    """Create a session row with the shape the summary cache expects."""
    db.create_session(session_id, "cli")
    for i in range(messages):
        db.append_message(session_id, "user", content=f"message {i}")
    now = time.time()
    db._conn.execute(
        "UPDATE sessions SET ended_at = ?, message_count = ?, archived = 0 WHERE id = ?",
        (now if ended else None, messages, session_id),
    )
    if summary is not None:
        db._conn.execute(
            "UPDATE sessions SET summary = ?, summary_updated_at = ?, summary_model = ? WHERE id = ?",
            (summary, summary_updated_at, summary_model, session_id),
        )
    db._conn.commit()
    return session_id




class TestSummarizeSession:
    def test_summarize_session_is_a_noop_on_a_fresh_summary(self, db):
        """summarize_session() must not re-spend an LLM call on a summary
        that is still fresh."""
        _make_session(
            db, "s1", ended=True,
            summary="already summarized", summary_updated_at=time.time() - 1,
            summary_model="m",
        )
        assert db.summarize_session("s1") == "already summarized"

    def test_summarize_session_returns_none_without_messages(self, db):
        db.create_session("empty", "cli")
        assert db.summarize_session("empty") is None

    def test_summarize_session_returns_none_for_unknown_session(self, db):
        assert db.summarize_session("nope") is None

    def test_summarize_session_persists_the_summary(self, db, monkeypatch):
        """A successful summary lands in sessions.summary with its model
        and timestamp."""
        _make_session(db, "s1", ended=True, summary=None)

        class _Msg:
            content = "The user asked about the parser and it was fixed."

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]
            model = "test-model"

        import agent.auxiliary_client as aux

        monkeypatch.setattr(aux, "call_llm", lambda **kwargs: _Resp())

        result = db.summarize_session("s1")
        assert result == "The user asked about the parser and it was fixed."

        row = db._conn.execute(
            "SELECT summary, summary_updated_at, summary_model FROM sessions WHERE id = 's1'"
        ).fetchone()
        assert row["summary"] == "The user asked about the parser and it was fixed."
        assert row["summary_updated_at"] > 0
        assert row["summary_model"] == "test-model"

    def test_summarize_session_keeps_the_old_summary_when_the_llm_fails(self, db, monkeypatch):
        """A failing LLM must not wipe the cached summary — the user keeps
        seeing the last good text."""
        _make_session(
            db, "s1", ended=True,
            summary="previous good summary", summary_updated_at=time.time() - 7200,
            summary_model="m",
        )
        import agent.auxiliary_client as aux

        def _boom(**kwargs):
            raise RuntimeError("provider down")

        monkeypatch.setattr(aux, "call_llm", _boom)
        assert db.summarize_session("s1") is None
        row = db._conn.execute(
            "SELECT summary FROM sessions WHERE id = 's1'"
        ).fetchone()
        assert row["summary"] == "previous good summary"

    def test_transcript_keeps_only_user_and_assistant_text(self, db):
        """Tool output and system messages must not reach the summarizer
        prompt verbatim — they bloat it and distort the summary."""
        from hermes_state import SessionDB as _SDB

        formatted = _SDB._summarize_format_conversation([
            {"role": "user", "content": "please fix it"},
            {"role": "assistant", "content": "done"},
            {"role": "tool", "content": "SECRET-TOOL-OUTPUT"},
            {"role": "system", "content": "SECRET-SYSTEM-PROMPT"},
        ])
        assert "please fix it" in formatted
        assert "done" in formatted
        assert "SECRET-TOOL-OUTPUT" not in formatted
        assert "SECRET-SYSTEM-PROMPT" not in formatted

    def test_a_multimodal_text_part_still_reaches_the_prompt(self, db):
        """A user turn whose text lives in a content part is kept.

        Multimodal turns arrive as a list of parts rather than a plain
        string. Dropping them because the content is not a string would
        silently lose the user's actual words from the summary.
        """
        from hermes_state import SessionDB as _SDB

        formatted = _SDB._summarize_format_conversation([
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what does this screenshot show"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                ],
            },
        ])
        assert "what does this screenshot show" in formatted

    def test_an_empty_turn_is_skipped_without_raising(self, db):
        """A turn with no usable text is dropped instead of raising.

        An assistant turn can be empty (a tool call with no prose); the
        formatter must not fail on it, or the whole summary fails — and
        it must not leave an empty line in the transcript either.
        """
        from hermes_state import SessionDB as _SDB

        formatted = _SDB._summarize_format_conversation([
            {"role": "user", "content": "the real question"},
            {"role": "assistant", "content": ""},
            {"role": "assistant", "content": None},
        ])
        assert "the real question" in formatted
        # The two empty turns must contribute nothing at all — no bare
        # "[assistant] " markers, and no extra blank-line separators.
        assert "[assistant]" not in formatted
        assert formatted.strip() == "[user] the real question"



class TestSchema:
    """The schema side of the summary cache.

    These live with the state layer rather than with the scheduler
    because they describe the database, not the runtime.
    """

    def test_the_sessions_table_declares_the_summary_columns(self, db):
        """A freshly created DB already carries the three columns.

        This is the failure the review of #49520 described: when the
        columns were declared only in a repair path, a normal DB did
        not have them and every SELECT against it failed.

        The scheduler's own test file asserts the same three columns —
        this one is the state layer's copy, so that a branch carrying
        only the state layer still pins them.
        """
        cols = {row[1] for row in db._conn.execute("PRAGMA table_info(sessions)")}
        assert {"summary", "summary_updated_at", "summary_model"} <= cols


class TestTipProjection:
    """The list projection must surface the LIVE conversation's summary.

    Compression seals a root and starts a tip. The root keeps the row
    the summary was written to, but the conversation the user sees in
    the sidebar is the TIP — so the summary has to be projected from
    the tip, exactly like ``preview`` and ``title`` already are.

    Without this, the hover card shows a summary of a conversation
    that has been rotated away.
    """

    def _make_chain(self, db, t0, *, root="root", tip="tip"):
        db.create_session(root, "cli")
        db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (t0, root))
        db._conn.execute(
            "UPDATE sessions SET ended_at=?, end_reason='compression' WHERE id=?",
            (t0 + 100, root),
        )
        db.create_session(tip, "cli", parent_session_id=root)
        db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (t0 + 200, tip))
        db._conn.commit()

    def test_the_projected_row_carries_the_tip_summary(self, db):
        """The summary written on the TIP is what the list projection surfaces."""
        self._make_chain(db, time.time() - 3600)
        db._conn.execute(
            "UPDATE sessions SET summary=?, summary_updated_at=?, summary_model=? WHERE id=?",
            ("root summary", 1.0, "root-model", "root"),
        )
        db._conn.execute(
            "UPDATE sessions SET summary=?, summary_updated_at=?, summary_model=? WHERE id=?",
            ("tip summary", 2.0, "tip-model", "tip"),
        )
        db._conn.commit()

        rows = db.list_sessions_rich(limit=50, order_by_last_active=True)
        assert len(rows) == 1
        row = rows[0]
        assert row["id"] == "tip"
        assert row["_lineage_root_id"] == "root"
        # The TIP's values win — the root's are the sealed conversation's.
        assert row["summary"] == "tip summary"
        assert row["summary_updated_at"] == 2.0
        assert row["summary_model"] == "tip-model"

    def test_the_projection_does_not_fall_back_to_the_root_summary(self, db):
        """Only the tip carries a summary; the root's must NOT surface.

        If the projection ever fell back to the root row, a rotated
        conversation would show a stale summary of content the user
        can no longer see.
        """
        self._make_chain(db, time.time() - 3600)
        db._conn.execute(
            "UPDATE sessions SET summary=?, summary_updated_at=?, summary_model=? WHERE id=?",
            ("root summary", 1.0, "root-model", "root"),
        )
        db._conn.commit()

        rows = db.list_sessions_rich(limit=50, order_by_last_active=True)
        assert len(rows) == 1
        assert rows[0]["summary"] is None
        assert rows[0]["summary_updated_at"] is None
        assert rows[0]["summary_model"] is None

