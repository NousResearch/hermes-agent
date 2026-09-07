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
            assert rows[turn_1_id]["api_content"] == "ok\n\nTURN-1"
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

    def test_duck_typed_store_does_not_fall_back_to_positional_when_row_id_present(self):
        """When a valid _row_id exists, a store lacking set_message_api_content
        must NOT fall back to set_latest_user_api_content (fails closed to
        prevent wrong-row corruption on repeated inputs)."""
        agent = _FakeAgent()
        # Mock defining ONLY set_latest_user_api_content (like older/external stores)
        mock_db = MagicMock(spec=["set_latest_user_api_content"])
        agent._session_db = mock_db
        agent._pending_cli_user_message = {
            "role": "user",
            "content": "hello",
            "_db_persisted": True,
            "_row_id": 42,
        }
        with patch(
            "hermes_cli.plugins.invoke_hook",
            return_value=[{"context": "PLUGIN-CTX"}],
        ):
            _build(agent)

        mock_db.set_latest_user_api_content.assert_not_called()

    def test_wrapper_lacking_set_message_api_content_fails_closed_without_corrupting_newer_row(
        self, tmp_path
    ):
        """[ehz0ah blocking feedback]: A wrapper exposing only set_latest_user_api_content
        and delegating to SessionDB must NOT be called when _row_id is present.
        With repeated 'ok' rows and _row_id=1, falling back would update the newer row at id 3;
        failing closed ensures row 3 is untouched and row 1 remains unchanged."""
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("s1", source="cli")
        try:
            db.append_message("s1", "user", content="ok")  # id=1
            db.append_message("s1", "assistant", content="reply")  # id=2
            db.append_message("s1", "user", content="ok")  # id=3

            rows = db.get_messages("s1")
            row_1_id, row_3_id = rows[0]["id"], rows[2]["id"]

            class _LegacyStoreWrapper:
                def __init__(self, real_db):
                    self._real = real_db

                def set_latest_user_api_content(self, session_id, content, api_content):
                    return self._real.set_latest_user_api_content(
                        session_id, content, api_content
                    )

            wrapper = _LegacyStoreWrapper(db)
            agent = _FakeAgent()
            agent.session_id = "s1"
            agent._session_db = wrapper
            agent._pending_cli_user_message = {
                "role": "user",
                "content": "ok",
                "_db_persisted": True,
                "_row_id": row_1_id,
            }

            with patch(
                "hermes_cli.plugins.invoke_hook",
                return_value=[{"context": "SIDE-1"}],
            ):
                _build(agent, user_message="ok")

            # Must fail closed: neither row 1 nor row 3 was updated
            rows = {r["id"]: r for r in db.get_messages("s1")}
            assert rows[row_1_id]["api_content"] is None
            assert rows[row_3_id]["api_content"] is None
        finally:
            db.close()

    def test_duck_typed_store_safely_skips_when_neither_method_present(self):
        """A store mock/wrapper defining neither method skips cleanly."""
        agent = _FakeAgent()
        mock_db = MagicMock(spec=[])
        agent._session_db = mock_db
        agent._pending_cli_user_message = {
            "role": "user",
            "content": "hello",
            "_db_persisted": True,
            "_row_id": 42,
        }
        with patch(
            "hermes_cli.plugins.invoke_hook",
            return_value=[{"context": "PLUGIN-CTX"}],
        ):
            # Must not raise AttributeError
            _build(agent)


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

    def test_pre_flushed_api_only_turn_with_injections_guards_on_durable_content(self, tmp_path):
        """[ehz0ah bug 2]: Pre-flushed clean input with API-only variant AND plugin context.
        The row update must use the durable clean content ('hello') as the SQL guard,
        not the restored API-facing content ('[voice] hello'), so the row is updated."""
        path = tmp_path / "state.db"
        db = SessionDB(db_path=path)
        sid = "sess-api-only-with-inj"
        db.create_session(sid, source="cli")
        try:
            agent = _RealPersistenceAgent(db, sid)

            clean_text = "hello"
            api_text = "[voice] hello"

            staged = {"role": "user", "content": clean_text}
            agent._pending_cli_user_message = staged
            agent._flush_messages_to_session_db([staged], None)
            assert staged.get("_row_id") is not None

            with patch(
                "hermes_cli.plugins.invoke_hook",
                return_value=[{"context": "PLUGIN-CTX"}],
            ):
                ctx = _build(
                    agent,
                    user_message=api_text,
                    persist_user_message=clean_text,
                )

            expected_sidecar = compose_user_api_content(api_text, "", "PLUGIN-CTX")
            turn_msg = ctx.messages[ctx.current_turn_user_idx]
            assert turn_msg["api_content"] == expected_sidecar

            # Database row was updated successfully by row_id with durable content guard!
            db_rows = db.get_messages(sid)
            assert db_rows[0]["content"] == clean_text
            assert db_rows[0]["api_content"] == expected_sidecar

            # Replay restores the clean content and the composed sidecar:
            conv = db.get_messages_as_conversation(sid)
            assert conv[0]["content"] == clean_text
            assert conv[0]["api_content"] == expected_sidecar
            from agent.turn_context import substitute_api_content
            substitute_api_content(conv[0])
            assert conv[0]["content"] == expected_sidecar
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
