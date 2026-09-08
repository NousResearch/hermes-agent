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
from unittest.mock import patch

import pytest

from agent.turn_context import compose_user_api_content
from hermes_state import SessionDB
from tests.agent.test_api_content_sidecar import _FakeAgent, _build


def _open_db(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id="s1", source="cli")
    return db


def _open_db_with_sessions(tmp_path, *session_ids):
    db = SessionDB(db_path=tmp_path / "state.db")
    for sid in session_ids:
        db.create_session(session_id=sid, source="cli")
    return db


class TestSetMessageApiContent:
    """The store primitive: addressed by row id, guarded on the rest."""

    def test_updates_the_addressed_row(self, tmp_path):
        db = _open_db(tmp_path)
        try:
            db.append_message("s1", "user", content="ok")
            row_id = db.get_messages("s1")[0]["id"]
            assert db.set_message_api_content("s1", row_id, "ok", "ok\n\nCTX") == 1
            assert db.get_messages("s1")[0]["api_content"] == "ok\n\nCTX"
        finally:
            db.close()

    def test_older_identical_row_is_untouched(self, tmp_path):
        """Two user turns with the same text — the repeated-"ok" shape.

        Addressing the row makes the older turn's sidecar unreachable; the
        positional helper cannot tell them apart (asserted on the same DB).
        """
        db = _open_db(tmp_path)
        try:
            db.append_message("s1", "user", content="ok")
            db.append_message("s1", "assistant", content="hi")
            db.append_message("s1", "user", content="ok")
            rows = db.get_messages("s1")
            older_id, newer_id = rows[0]["id"], rows[2]["id"]
            assert older_id != newer_id

            db.set_message_api_content("s1", newer_id, "ok", "ok\n\nCTX-NEW")
            assert db.get_messages("s1")[2]["api_content"] == "ok\n\nCTX-NEW"
            # The older identical row was not touched.
            assert db.get_messages("s1")[0]["api_content"] is None
        finally:
            db.close()

    def test_content_mismatch_writes_nothing(self, tmp_path):
        """A row a racing rewrite changed is left untouched."""
        db = _open_db(tmp_path)
        try:
            db.append_message("s1", "user", content="ok")
            row_id = db.get_messages("s1")[0]["id"]
            assert db.set_message_api_content("s1", row_id, "different", "ok\n\nCTX") == 0
            assert db.get_messages("s1")[0]["api_content"] is None
        finally:
            db.close()

    def test_rejects_invalid_row_id_and_session(self, tmp_path):
        db = _open_db(tmp_path)
        try:
            db.append_message("s1", "user", content="ok")
            for bad_id in (0, -1, None, "5", True):
                assert db.set_message_api_content("s1", bad_id, "ok", "ok\n\nCTX") == 0
            assert db.set_message_api_content("", 1, "ok", "ok\n\nCTX") == 0
            assert db.get_messages("s1")[0]["api_content"] is None
        finally:
            db.close()

    def test_archived_row_is_untouched(self, tmp_path):
        """A compaction-archived row (active=0) must not be revived."""
        db = _open_db(tmp_path)
        try:
            db.append_message("s1", "user", content="ok")
            row_id = db.get_messages("s1")[0]["id"]
            db.archive_and_compact("s1", [{"role": "user", "content": "summary"}])
            assert db.set_message_api_content("s1", row_id, "ok", "ok\n\nCTX") == 0
        finally:
            db.close()


class TestPrologueRowAddressedBackfill:
    """The prologue backfills when — and only when — the row provably exists."""

    def _agent_with_db(self, tmp_path, **overrides):
        db = _open_db_with_sessions(tmp_path, "s1", "sess-1")
        agent = _FakeAgent()
        agent._session_db = db
        for k, v in overrides.items():
            setattr(agent, k, v)
        return agent, db

    def test_row_id_backfill_replaces_positional_call(self, tmp_path):
        """A pre-persisted row (close-flush raced the prologue) carries
        ``_row_id``; the backfill must address it, not the newest row."""
        agent, db = self._agent_with_db(tmp_path)
        # Pre-existing older turn with identical clean text.
        db.append_message(agent.session_id, "user", content="ok")
        # The current turn's message was already flushed: stamped with a row id
        # and the persisted marker, so the crash persist will skip it.
        current_row_id = db.append_message(
            agent.session_id, "user", content="ok"
        )
        try:
            with patch(
                "hermes_cli.plugins.invoke_hook",
                return_value=[{"context": "PLUGIN-CTX"}],
            ), patch(
                "agent.session_persistence._is_ephemeral_scaffolding",
                lambda _m: False,
            ):
                ctx = _build(
                    agent,
                    user_message="ok",
                    # The live dict the prologue will see, carrying the flush stamps.
                    conversation_history=[],
                )
            # Stamp the row-id path precondition directly on the live dict, as
            # sync_flushed_message_markers would have.
            turn_msg = ctx.messages[ctx.current_turn_user_idx]
            turn_msg["_row_id"] = current_row_id

            expected = compose_user_api_content(
                "ok", ctx.ext_prefetch_cache, ctx.plugin_user_context
            )
            # Re-run the stamp with the row id present: the exact-sent-bytes
            # backfill must land on the addressed row.
            from agent.turn_context import _stamp_api_content_sidecar

            _stamp_api_content_sidecar(
                agent, ctx.messages, ctx.current_turn_user_idx,
                ctx.ext_prefetch_cache, ctx.plugin_user_context,
                preflight_compressed=False,
            )
            rows = db.get_messages(agent.session_id)
            assert rows[1]["api_content"] == expected
            # The older identical row keeps its (absent) sidecar.
            assert rows[0]["api_content"] is None
        finally:
            db.close()

    def test_in_place_compaction_falls_back_to_positional(self, tmp_path):
        """Compacted copies carry no ``_row_id``; the pre-existing positional
        backfill stays for them (archive_and_compact just made this row the
        newest active user row, so targeting by position is safe there)."""
        agent, db = self._agent_with_db(tmp_path, _last_compaction_in_place=True)
        positional_calls = []
        orig_positional = db.set_latest_user_api_content

        def _spy(*args, **kwargs):
            positional_calls.append(args)
            return orig_positional(*args, **kwargs)

        db.set_latest_user_api_content = _spy
        try:
            with patch(
                "hermes_cli.plugins.invoke_hook",
                return_value=[{"context": "PLUGIN-CTX"}],
            ):
                ctx = _build(agent, user_message="hello")
            turn_msg = ctx.messages[ctx.current_turn_user_idx]
            assert "_row_id" not in turn_msg

            from agent.turn_context import _stamp_api_content_sidecar

            _stamp_api_content_sidecar(
                agent, ctx.messages, ctx.current_turn_user_idx,
                ctx.ext_prefetch_cache, ctx.plugin_user_context,
                preflight_compressed=True,
            )
            # The positional fallback fired (row-addressed path not taken:
            # no _row_id on the compacted copy).
            assert len(positional_calls) == 1
            assert positional_calls[0][0] == agent.session_id
            expected = compose_user_api_content(
                "hello", ctx.ext_prefetch_cache, ctx.plugin_user_context
            )
            assert positional_calls[0][2] == expected
        finally:
            db.close()

    def test_no_backfill_without_row_id_or_compaction(self, tmp_path):
        """Normal path: the row does not exist yet, and nothing is written
        by the stamp (the crash persist below writes the row once, with the
        sidecar in the same INSERT)."""
        agent, db = self._agent_with_db(tmp_path)
        calls = []
        orig = db.set_latest_user_api_content

        def _spy(*args, **kwargs):
            calls.append(args)
            return orig(*args, **kwargs)

        db.set_latest_user_api_content = _spy
        try:
            with patch(
                "hermes_cli.plugins.invoke_hook",
                return_value=[{"context": "PLUGIN-CTX"}],
            ):
                ctx = _build(agent, user_message="hello")
            assert ctx.messages[ctx.current_turn_user_idx]["api_content"]
            assert calls == []  # no positional backfill on the normal path
        finally:
            db.close()


def test_positional_backfill_collision_documented(tmp_path):
    """The exact failure mode that rules out an unconditional positional
    backfill: repeated identical user turns. The positional helper cannot
    distinguish the rows; the row-addressed primitive can."""
    db = _open_db(tmp_path)
    try:
        db.append_message("s1", "user", content="continue")
        db.append_message("s1", "assistant", content="doing it")
        older = db.get_messages("s1")[0]["id"]
        db.set_latest_user_api_content("s1", "continue", "continue\n\nCTX-1")
        assert db.get_messages("s1")[0]["api_content"] == "continue\n\nCTX-1"

        # New turn with the same text, row NOT yet materialized: a positional
        # backfill now would hit the older row — which the row-addressed
        # primitive refuses to do (no valid id => no write).
        assert db.set_message_api_content("s1", 0, "continue", "continue\n\nCTX-2") == 0
        assert db.get_messages("s1")[0]["api_content"] == "continue\n\nCTX-1"
    finally:
        db.close()
