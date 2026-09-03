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

from agent.turn_context import compose_user_api_content
from hermes_state import SessionDB
from tests.agent.test_api_content_sidecar import _FakeAgent, _build


class TestSetMessageApiContent:
    """The store primitive: addressed by row id, guarded on the rest."""

    def _open(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("s1", source="cli")
        return db

    def test_updates_the_addressed_row(self, tmp_path):
        db = self._open(tmp_path)
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

            # The positional helper is only safe when the caller already knows
            # the newest active user row is its own message.
            db.set_latest_user_api_content("s1", "ok", "ok\n\nTURN-3")
            rows = {r["id"]: r for r in db.get_messages("s1")}
            assert rows[turn_2_id]["api_content"] == "ok\n\nTURN-3"
        finally:
            db.close()

    def test_guards_reject_wrong_session_content_or_archived_row(self, tmp_path):
        db = self._open(tmp_path)
        db.create_session("s2", source="cli")
        try:
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

    def test_survives_lone_surrogate(self, tmp_path):
        db = self._open(tmp_path)
        try:
            db.append_message("s1", "user", content="turn text")
            row_id = db.get_messages("s1")[0]["id"]
            dirty = "text \ud83d\ude00 \ud83d more"
            assert db.set_message_api_content("s1", row_id, "turn text", dirty) == 1
            stored = db.get_messages("s1")[0]["api_content"]
            assert "\ud83d" not in stored or "\ud83d\ude00" in stored
        finally:
            db.close()

    def test_rejects_boolean_and_invalid_row_ids_and_empty_session(self, tmp_path):
        db = self._open(tmp_path)
        try:
            db.append_message("s1", "user", content="turn text")
            row_id = db.get_messages("s1")[0]["id"]
            assert db.set_message_api_content("s1", True, "turn text", "sidecar") == 0
            assert db.set_message_api_content("s1", False, "turn text", "sidecar") == 0
            assert db.set_message_api_content("s1", 0, "turn text", "sidecar") == 0
            assert db.set_message_api_content("s1", -5, "turn text", "sidecar") == 0
            assert db.set_message_api_content("", row_id, "turn text", "sidecar") == 0
            assert db.set_message_api_content(None, row_id, "turn text", "sidecar") == 0
            assert db.get_messages("s1")[0]["api_content"] is None
        finally:
            db.close()




class TestPrologueRowAddressedBackfill:
    """The prologue gate: backfill iff a durable row exists for this dict."""

    def test_preexisting_row_receives_the_sidecar(self, tmp_path):
        """A close/early flush wrote the staged CLI input before the stamp and
        synced ``_row_id`` back onto it. The crash persist then skips the
        message, so the prologue must push the sidecar into that exact row."""
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("s1", source="cli")
        try:
            db.append_message("s1", "user", content="hello")
            row_id = db.get_messages("s1")[0]["id"]

            agent = _FakeAgent()
            agent.session_id = "s1"
            agent._session_db = db
            agent._pending_cli_user_message = {
                "role": "user",
                "content": "hello",
                "_db_persisted": True,
                "_row_id": row_id,
            }
            with patch(
                "hermes_cli.plugins.invoke_hook",
                return_value=[{"context": "PLUGIN-CTX"}],
            ):
                ctx = _build(agent)

            expected = compose_user_api_content("hello", "", "PLUGIN-CTX")
            assert ctx.messages[ctx.current_turn_user_idx]["api_content"] == expected
            assert db.get_messages("s1")[0]["api_content"] == expected
        finally:
            db.close()

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

    def test_db_persisted_alone_does_not_arm_the_backfill(self):
        """``_db_persisted`` is stamped on resumed history dicts whose row id
        is unknown, so it cannot stand in for ``_row_id``: arming the
        positional backfill from it re-opens the wrong-row write."""
        agent = _FakeAgent()
        agent._session_db = MagicMock()
        agent._pending_cli_user_message = {
            "role": "user",
            "content": "hello",
            "_db_persisted": True,
        }
        with patch(
            "hermes_cli.plugins.invoke_hook",
            return_value=[{"context": "PLUGIN-CTX"}],
        ):
            _build(agent)

        agent._session_db.set_message_api_content.assert_not_called()
        agent._session_db.set_latest_user_api_content.assert_not_called()

    def test_boolean_row_id_does_not_arm_the_backfill(self):
        """In Python isinstance(True, int) is True; a boolean _row_id must not
        be mistaken for a valid SQLite primary key."""
        agent = _FakeAgent()
        agent._session_db = MagicMock()
        agent._pending_cli_user_message = {
            "role": "user",
            "content": "hello",
            "_db_persisted": True,
            "_row_id": True,
        }
        with patch(
            "hermes_cli.plugins.invoke_hook",
            return_value=[{"context": "PLUGIN-CTX"}],
        ):
            _build(agent)

        agent._session_db.set_message_api_content.assert_not_called()
        agent._session_db.set_latest_user_api_content.assert_not_called()


    def test_row_id_wins_over_the_compaction_fallback(self):
        """A compacted copy that kept its fresh row id is addressed by id; the
        positional fallback stays for a copy that carries none."""
        agent = _make_in_place_compaction_agent(row_id=41)
        with patch(
            "hermes_cli.plugins.invoke_hook",
            return_value=[{"context": "PLUGIN-CTX"}],
        ):
            _build(agent)
        agent._session_db.set_message_api_content.assert_called_once_with(
            "sess-1", 41, "hello", "hello\n\nPLUGIN-CTX"
        )
        agent._session_db.set_latest_user_api_content.assert_not_called()

    def test_compaction_without_row_id_keeps_positional_fallback(self):
        agent = _make_in_place_compaction_agent(row_id=None)
        with patch(
            "hermes_cli.plugins.invoke_hook",
            return_value=[{"context": "PLUGIN-CTX"}],
        ):
            _build(agent)
        agent._session_db.set_latest_user_api_content.assert_called_once_with(
            "sess-1", "hello", "hello\n\nPLUGIN-CTX"
        )
        agent._session_db.set_message_api_content.assert_not_called()


def _make_in_place_compaction_agent(*, row_id):
    """Agent whose preflight compression compacts in place, mirroring
    ``archive_and_compact``: the current-turn user dict is replaced by a fresh
    copy whose row already exists (and carries ``_row_id`` when the insert
    stamped one)."""
    agent = _FakeAgent()
    agent.compression_enabled = True
    agent._session_db = MagicMock()

    calls = {"n": 0}

    def _should_compress(_tokens):
        calls["n"] += 1
        return calls["n"] == 1

    agent.context_compressor = types.SimpleNamespace(
        protect_first_n=0,
        protect_last_n=0,
        threshold_tokens=1,
        context_length=1000,
        last_prompt_tokens=-1,
        should_compress=_should_compress,
        should_defer_preflight_to_real_usage=lambda _t: False,
        get_active_compression_failure_cooldown=lambda: None,
    )

    def _compress(messages, _system, approx_tokens=None, task_id=None):
        agent._last_compaction_in_place = True
        survivor = dict(messages[-1])
        if row_id is not None:
            survivor["_row_id"] = row_id
        return (
            [{"role": "assistant", "content": "compaction summary"}, survivor],
            "SYSTEM",
        )

    agent._compress_context = _compress
    return agent
