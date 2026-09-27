"""Regressions for #124731: the persist user-message override must not drop an
earlier unanswered user message that alternation repair merged into the current
turn's user row.

A crash or restart between turn start and the reply leaves a user row R
durable in state.db with no assistant answer. When the next user message U
arrives, ``_merge_consecutive_users`` folds U into R's dict and re-anchoring
falls back to that merged row. Both override writers then rewrite the merged
row to U alone — R's bytes leave the live list at finalize (ACP hosts keep it
across turns), and on the session-row-heal replay path (the only row writer
that reaches a merged row; by then R's own row is confirmed gone, so the
merged bytes must be kept rather than duplicated) they leave the durable
transcript.

``_merge_consecutive_users`` stamps the pre-merge bytes onto its survivor
(``USER_MERGE_PREFIX_KEY``); both writers share ``_post_override_content``,
which re-attaches the prefix for a str override whose content still starts
with it. A plain row (no stamp) takes the override alone.
"""

from __future__ import annotations

from run_agent import AIAgent
from agent.agent_runtime_helpers import repair_message_sequence
from agent.context_compressor import USER_MERGE_PREFIX_KEY
from agent.session_persistence import (
    SessionPersistenceMixin,
    _db_flush_row,
    _post_override_content,
)
from agent.turn_context import reanchor_current_turn_user_idx
from hermes_state import SessionDB

R = "please deploy build 42 to staging"
U_WIRE = "[03:00] can you also run the smoke tests"
U_CLEAN = "can you also run the smoke tests"
MERGED = R + "\n\n" + U_CLEAN


class TestReplayFlushRow:
    """The session-row-heal writer: the replay re-writes the history prefix as
    rows (``_db_flush_collect(replay_history=True)``); the merged row is the
    only user row in that snapshot, so the prefix must ride the row content
    and the exact pre-override wire bytes the api_content sidecar."""

    def _merged_agent(self):
        messages = [
            {"role": "user", "content": R},
            {"role": "user", "content": U_WIRE},
        ]
        assert repair_message_sequence(None, messages) == 1
        agent = object.__new__(SessionPersistenceMixin)
        agent._persist_user_message_override = U_CLEAN
        agent._persist_user_message_timestamp = None
        agent._persist_user_message_platform_id = None
        return agent, messages

    def test_replay_flush_row_keeps_prefix_and_exact_api_sidecar(self):
        agent, messages = self._merged_agent()
        row = _db_flush_row(agent, messages[0], True)
        assert row["content"] == MERGED
        assert row["api_content"] == messages[0]["content"]  # pre-override wire bytes

    def test_no_op_merge_no_stamp_takes_plain_override(self):
        """A merge that reproduces the persisted bytes (empty incoming turn)
        stamps no prefix; the override stays plain and nothing is lost."""
        messages = [
            {"role": "user", "content": R},
            {"role": "user", "content": ""},
        ]
        assert repair_message_sequence(None, messages) == 1
        assert messages[0]["content"] == R
        assert USER_MERGE_PREFIX_KEY not in messages[0]
        agent = object.__new__(SessionPersistenceMixin)
        agent._persist_user_message_override = R
        agent._persist_user_message_timestamp = None
        agent._persist_user_message_platform_id = None
        row = _db_flush_row(agent, messages[0], True)
        assert row["content"] == R
        assert row["api_content"] is None

    def test_stale_prefix_takes_the_plain_override(self):
        """An intervening rewrite that moved the content past the stamped prefix
        makes the prefix stale; only the plain override is safe then."""
        agent = object.__new__(SessionPersistenceMixin)
        stale = {"role": "user", "content": "something else entirely", USER_MERGE_PREFIX_KEY: R}
        assert _post_override_content(stale, stale["content"], U_CLEAN) == U_CLEAN
        # The live writer agrees on the stale row.
        stale_list = [stale]
        agent._persist_user_message_idx = 0
        agent._persist_user_message_override = U_CLEAN
        agent._persist_user_message_timestamp = None
        agent._persist_user_message_platform_id = None
        agent._apply_persist_user_message_override(stale_list)
        assert stale["content"] == U_CLEAN


class TestCrashRecoveryRoundTrip:
    def test_crash_recovery_round_trip_live_and_reload_agree(self, tmp_path):
        """End to end: crash-orphan row R, next turn U. The turn-start flush
        writes U's row; repair merges U into R's live dict; the finalize
        override must keep R's bytes in the live list; the reload-time repair
        of the two store rows must reproduce that live slot, and the store
        keeps its idempotent two-user-row shape (never a merged third row)."""
        db = SessionDB(db_path=tmp_path / "state.db")
        sid = "sess-crash-recover"
        db.create_session(session_id=sid, source="cli")
        try:
            # Pre-crash: R was flushed at its own turn start and never answered.
            db.append_message(sid, "user", content=R)

            agent = AIAgent(
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                session_db=db,
                session_id=sid,
            )
            agent._session_db_created = True

            # Restart: the host reloads the history for the next turn.
            history = db.get_messages_as_conversation(
                sid, repair_alternation=False, include_row_ids=True
            )
            assert [m["content"] for m in history] == [R]

            # Turn prologue: append U (API-only live bytes) and anchor the override.
            messages = list(history) + [{"role": "user", "content": U_WIRE}]
            agent._persist_user_message_idx = len(messages) - 1
            agent._persist_user_message_override = U_CLEAN
            agent._persist_user_message_timestamp = None

            # The turn-start flush writes U's row from U's own dict (pre-merge,
            # no stamp): plain override content, wire bytes on the sidecar.
            agent._flush_messages_to_session_db(messages, history)
            rows = db.get_messages_as_conversation(sid, repair_alternation=False)
            assert [m["content"] for m in rows if m.get("role") == "user"] == [R, U_CLEAN]
            assert rows[1].get("api_content") == U_WIRE

            # Repair merges U into R's dict; re-anchoring falls back to it.
            assert repair_message_sequence(agent, messages) == 1
            agent._persist_user_message_idx = reanchor_current_turn_user_idx(
                messages, U_WIRE
            )
            assert agent._persist_user_message_idx == 0

            messages.append({"role": "assistant", "content": "service deployed"})

            # finalize_turn's override step, then the persist flush (history
            # passed so the loaded rows stay skipped, exactly like the host's
            # finalize call).
            agent._apply_persist_user_message_override(messages)
            agent._flush_messages_to_session_db(messages, history)

            live = messages[0]["content"]
            assert live == MERGED, (
                "R's bytes left the live list; ACP hosts keep this list across turns"
            )

            # Store shape: R's row and U's row stay separate active rows — the
            # merged row was history-skipped, so no merged third row can
            # duplicate R on the next load-time repair.
            active = db.get_messages_as_conversation(sid, repair_alternation=False)
            assert [m["content"] for m in active if m.get("role") == "user"] == [R, U_CLEAN]

            # Reload (twice): the load-time repair reproduces the live slot
            # byte-for-byte and stays idempotent.
            for _round in (1, 2):
                reloaded = db.get_messages_as_conversation(
                    sid, repair_alternation=True, include_row_ids=True
                )
                user_rows = [m for m in reloaded if m.get("role") == "user"]
                assert user_rows[0]["content"] == live
        finally:
            db.close()
