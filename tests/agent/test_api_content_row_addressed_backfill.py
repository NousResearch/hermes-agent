"""Row-addressed ``api_content`` backfill (NousResearch/hermes-agent#102194).

The sidecar is stamped by the turn prologue and normally reaches the DB in the
same INSERT as the clean content (the crash persist runs after the stamp). When
another writer materialized the current turn's user row FIRST — in-place
preflight compaction, or a close/early flush that raced the prologue — that
insert never happens: the crash persist marker-skips the message and the row
keeps ``api_content = NULL``, so the next turn replays clean content and the
request prefix diverges exactly at that message.

The prologue therefore backfills, but only when a row provably exists for THIS
dict. ``_row_id`` is that proof and that address: both early writers stamp it on
the live message (``_insert_message_rows`` directly, ``sync_flushed_message_markers``
after the batch commit). A positional "newest active user row" update cannot be
substituted for it — a repeated user turn ("ok", "y", "continue") makes the
previous turn's row compare equal on content, and the backfill would overwrite
that turn's sidecar with this turn's bytes.
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

import pytest

from agent.session_persistence import SessionPersistenceMixin
from agent.turn_context import _stamp_api_content_sidecar, compose_user_api_content
from hermes_state import SessionDB
from tests.agent.test_api_content_sidecar import _FakeAgent, _build


class TestSetMessageApiContent:
    """The store primitive: addressed by row id, guarded on the rest."""

    def _open(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("s1", source="cli")
        return db

    def test_older_identical_row_is_untouched(self, tmp_path):
        """Two user turns with the same text — the repeated-"ok" shape.

        Addressing the row makes the older turn's sidecar unreachable; the
        positional helper cannot tell them apart (asserted on the same DB).
        """
        db = self._open(tmp_path)
        try:
            db.append_message("s1", "user", content="ok", api_content="ok\n\nTURN-1")
            db.append_message("s1", "assistant", content="reply")
            db.append_message("s1", "user", content="ok")
            rows = db.get_messages("s1")
            turn_1_id, turn_2_id = rows[0]["id"], rows[2]["id"]

            assert db.set_message_api_content("s1", turn_2_id, "ok", "ok\n\nTURN-2") == 1
            rows = {r["id"]: r for r in db.get_messages("s1")}
            assert rows[turn_1_id]["api_content"] == "ok\n\nTURN-1"
            assert rows[turn_2_id]["api_content"] == "ok\n\nTURN-2"

        finally:
            db.close()

    def test_guards_refuse_wrong_session_or_mismatched_content_or_archived_row(self, tmp_path):
        db = self._open(tmp_path)
        try:
            db.create_session("s2", source="cli")
            db.append_message("s1", "user", content="hello")
            row_id = db.get_messages("s1")[0]["id"]

            assert db.set_message_api_content("s2", row_id, "hello", "x") == 0
            assert db.set_message_api_content("s1", row_id, "other", "x") == 0
            assert db.set_message_api_content("s1", row_id + 999, "hello", "x") == 0
            assert db.get_messages("s1")[0]["api_content"] is None

            # Archived by compaction: active = 0, so the row is off limits.
            db.archive_and_compact("s1", [{"role": "user", "content": "hello"}])
            assert db.set_message_api_content("s1", row_id, "hello", "x") == 0
        finally:
            db.close()

class TestPrologueRowAddressedBackfill:
    """The prologue gate: backfill iff a durable row exists for this dict."""

    def test_no_row_id_and_no_compaction_writes_nothing(self):
        """The normal path: the row does not exist yet and the crash persist
        writes it WITH the sidecar. A backfill here has no row to address and
        would have to guess — so it must not run at all."""
        agent = _FakeAgent()
        agent._session_db = MagicMock()
        with patch(
            "hermes_cli.plugins.invoke_hook",
            return_value=[{"context": "PLUGIN-CTX"}],
        ):
            ctx = _build(agent)

        assert (
            ctx.messages[ctx.current_turn_user_idx]["api_content"]
            == "hello\n\nPLUGIN-CTX"
        )
        agent._session_db.set_message_api_content.assert_not_called()
        agent._session_db.set_latest_user_api_content.assert_not_called()

class _RealPersistenceAgent(SessionPersistenceMixin, _FakeAgent):
    """Stand-in agent with the real SessionPersistenceMixin flush implementation."""

    def __init__(self, db=None, sid="s1"):
        _FakeAgent.__init__(self)
        self._session_db = db
        self.session_id = sid
        self._session_db_created = True
        self._flushed_db_message_ids = set()
        self._last_flushed_db_idx = 0


class TestRealEarlyFlushAndOverrideLifecycle:
    """End-to-end tests exercising real database flushes and API-only overrides."""

    def test_real_close_flush_syncs_row_id_and_prologue_backfills(self, tmp_path):
        """Proof that the real _flush_messages_to_session_db path syncs _row_id onto
        the live dict (via sync_flushed_message_markers) and the prologue backfills it."""
        path = tmp_path / "state.db"
        db = SessionDB(db_path=path)
        sid = "sess-real-flush"
        db.create_session(sid, source="cli")
        try:
            agent = _RealPersistenceAgent(db, sid)

            staged = {"role": "user", "content": "hello"}
            agent._pending_cli_user_message = staged

            # Simulate the early/close flush racing the prologue
            flushed = agent._flush_messages_to_session_db([staged], None)
            assert flushed is True
            assert staged.get("_db_persisted") is True
            assert isinstance(staged.get("_row_id"), int)
            assert staged["_row_id"] == db.get_messages(sid)[-1]["id"]
            # At this point, the row in SQLite has api_content = None
            assert db.get_messages(sid)[-1]["api_content"] is None

            # Now build_turn_context runs
            with patch(
                "hermes_cli.plugins.invoke_hook",
                return_value=[{"context": "PLUGIN-CTX"}],
            ):
                ctx = _build(agent)

            expected = compose_user_api_content("hello", "", "PLUGIN-CTX")
            assert ctx.messages[ctx.current_turn_user_idx]["api_content"] == expected
            # Backfilled to the exact row in SQLite!
            assert db.get_messages(sid)[-1]["api_content"] == expected
        finally:
            db.close()

    def test_pre_flushed_api_only_turn_without_injections_preserves_sidecar(self, tmp_path):
        """[ehz0ah bug 1]: Pre-flushed clean input where the API turn has an API-only
        variant (e.g. voice prefix) and NO memory/plugin injection is composed.
        compose_user_api_content returns None, but the differing API-only bytes must
        be preserved as api_content and backfilled onto the row."""
        path = tmp_path / "state.db"
        db = SessionDB(db_path=path)
        sid = "sess-api-only-no-inj"
        db.create_session(sid, source="cli")
        try:
            agent = _RealPersistenceAgent(db, sid)

            clean_text = "hello"
            api_text = "[voice] hello"

            staged = {"role": "user", "content": clean_text}
            agent._pending_cli_user_message = staged
            agent._flush_messages_to_session_db([staged], None)
            assert staged.get("_row_id") is not None

            # Worker resumes with API-facing message and clean persist override
            with patch("hermes_cli.plugins.invoke_hook", return_value=[]):
                ctx = _build(
                    agent,
                    user_message=api_text,
                    persist_user_message=clean_text,
                )

            # Live user message has the API text and api_content
            turn_msg = ctx.messages[ctx.current_turn_user_idx]
            assert turn_msg["content"] == api_text
            assert turn_msg["api_content"] == api_text

            # Database row has clean text as content, but api_text as api_content!
            db_rows = db.get_messages(sid)
            assert len(db_rows) == 1
            assert db_rows[0]["content"] == clean_text
            assert db_rows[0]["api_content"] == api_text

            # Replay via get_messages_as_conversation preserves clean content
            # alongside the api_content sidecar.
            conv = db.get_messages_as_conversation(sid)
            assert conv[0]["content"] == clean_text
            assert conv[0]["api_content"] == api_text
            from agent.turn_context import substitute_api_content
            substitute_api_content(conv[0])
            assert conv[0]["content"] == api_text
        finally:
            db.close()

    def test_repeated_prompt_protected_against_positional_overwrite(self, tmp_path):
        """Repeated prompts 'ok' across turns: Turn 1 has sidecar, Turn 2 is pre-flushed.
        Row-addressed backfill on Turn 2 never mutates Turn 1's stored sidecar."""
        path = tmp_path / "state.db"
        db = SessionDB(db_path=path)
        sid = "sess-repeated-ok"
        db.create_session(sid, source="cli")
        try:
            # Turn 1
            db.append_message(sid, "user", content="ok", api_content="ok\n\nTURN-1-CTX")
            db.append_message(sid, "assistant", content="acknowledged")
            t1_user_row = db.get_messages(sid)[0]

            # Turn 2: staged and pre-flushed
            staged_t2 = {"role": "user", "content": "ok"}
            agent = _RealPersistenceAgent(db, sid)
            agent._pending_cli_user_message = staged_t2

            agent._flush_messages_to_session_db([staged_t2], None)
            t2_user_row = db.get_messages(sid)[2]
            assert t2_user_row["id"] != t1_user_row["id"]
            assert t2_user_row["api_content"] is None

            # Prologue backfills Turn 2
            with patch(
                "hermes_cli.plugins.invoke_hook",
                return_value=[{"context": "TURN-2-CTX"}],
            ):
                _build(agent, user_message="ok")

            rows = {r["id"]: r for r in db.get_messages(sid)}
            assert rows[t1_user_row["id"]]["api_content"] == "ok\n\nTURN-1-CTX"
            assert rows[t2_user_row["id"]]["api_content"] == "ok\n\nTURN-2-CTX"
        finally:
            db.close()
