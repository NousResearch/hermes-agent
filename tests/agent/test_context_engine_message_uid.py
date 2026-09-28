"""A context engine sees ``message_uid`` on every message the host hands it.

The engine-facing surfaces are ``compress()`` (the live list), ``post_llm_call``'s
``conversation_history`` (the same live dicts after the turn flush) and ``on_turn_complete``
(structural clones). The id must reach all three after a cold restore that never asked for
``_row_id``, survive the persist override and an in-place compaction commit, and never reach the
provider. A consecutive-user merge keeps the first constituent's uid and records the absorbed ones.
"""

from __future__ import annotations

import logging
import os
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.context_engine import ContextEngine
from hermes_state import SessionDB

UID_LEN = 32


class _CapturingEngine(ContextEngine):
    """Records what compress() and on_turn_complete() receive; compress() returns marker-swept copies."""

    last_prompt_tokens = 0

    def __init__(self) -> None:
        self.compress_input = None
        self.turn_complete_messages = None
        self._last_compress_aborted = False
        self._last_summary_error = None
        self.compression_count = 1

    @property
    def name(self) -> str:
        return "capturing"

    def update_from_response(self, usage):
        pass

    def should_compress(self, prompt_tokens=None):
        return False

    def compress(self, messages, current_tokens=None, focus_topic=None, force=False):
        self.compress_input = [dict(m) for m in messages]
        summary = {"role": "user", "content": "[CONTEXT COMPACTION] summary of prior turns"}
        return [summary] + [dict(m) for m in messages[-2:]]

    def on_turn_complete(self, messages, usage=None, **kwargs):
        self.turn_complete_messages = messages

    def _record_compression_failure_cooldown(self, *a, **k):
        pass


def _make_agent(session_db, session_id):
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key", base_url="https://openrouter.ai/api/v1", model="test/model", quiet_mode=True,
            session_db=session_db, session_id=session_id, skip_context_files=True, skip_memory=True,
        )
    agent.compression_in_place = True
    agent.context_compressor = _CapturingEngine()
    return agent


def _seed(db, sid, n=6):
    db.create_session(sid, "cli", model="test/model")
    for i in range(n):
        db.append_message(session_id=sid, role="user" if i % 2 == 0 else "assistant", content=f"seed msg {i}")
    return [dict(r) for r in db._conn.execute(
        "SELECT id, content, message_uid FROM messages WHERE session_id = ? AND active = 1 ORDER BY id",
        (sid,)).fetchall()]


@pytest.fixture()
def db(tmp_path):
    with closing(SessionDB(db_path=Path(tmp_path) / "state.db")) as handle:
        yield handle


def test_compress_input_after_a_cold_restore_carries_every_rows_uid(db):
    """The ACP/gateway restore shape (no ``include_row_ids``) → compress() input → committed generation."""
    from agent.conversation_compression import compress_context

    sid = "20260928_120000_uid"
    stored = _seed(db, sid)
    restored = db.get_messages_as_conversation(sid, repair_alternation=True)
    assert all("_row_id" not in m for m in restored)
    agent = _make_agent(db, sid)
    agent._session_db_created = True
    agent._last_flushed_db_idx = len(restored)

    compressed, _sp = compress_context(agent, restored, approx_tokens=100_000, system_message="sys")

    engine = agent.context_compressor
    assert [m.get("message_uid") for m in engine.compress_input] == [r["message_uid"] for r in stored]
    # The committed generation: the copied tail keeps its uids, the summary got a fresh one.
    active = [dict(r) for r in db._conn.execute(
        "SELECT content, message_uid FROM messages WHERE session_id = ? AND active = 1 ORDER BY id", (sid,))]
    assert [r["content"] for r in active] == [
        "[CONTEXT COMPACTION] summary of prior turns", "seed msg 4", "seed msg 5"]
    assert [r["message_uid"] for r in active[1:]] == [stored[4]["message_uid"], stored[5]["message_uid"]]
    assert len(active[0]["message_uid"]) == UID_LEN
    # And the live list the caller keeps carries the same uids (what the next compress()/post_llm_call sees).
    assert [m.get("message_uid") for m in compressed] == [r["message_uid"] for r in active]


def test_turn_flush_stamps_uids_on_the_live_dicts_post_llm_call_hands_over(db):
    """``post_llm_call(conversation_history=list(messages))`` passes the live dicts; after the turn flush
    every one of them carries the durable uid, including the current-turn user row."""
    sid = "20260928_120100_flush"
    db.create_session(sid, "cli", model="test/model")
    agent = _make_agent(db, sid)
    agent._session_db_created = True
    messages = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
    agent._persist_user_message_idx = 0

    agent._persist_session(messages, conversation_history=None)

    assert all(len(m.get("message_uid", "")) == UID_LEN for m in messages)
    stored = [r[0] for r in db._conn.execute(
        "SELECT message_uid FROM messages WHERE session_id = ? AND active = 1 ORDER BY id", (sid,))]
    assert [m["message_uid"] for m in messages] == stored


def test_persist_override_keeps_the_uid(db):
    """The ACP/gateway persist override rewrites the current-turn user CONTENT in place; the id is the row's."""
    sid = "20260928_120200_override"
    db.create_session(sid, "cli", model="test/model")
    agent = _make_agent(db, sid)
    agent._session_db_created = True
    messages = [{"role": "user", "content": "api-only variant with injected context"}]
    agent._persist_user_message_idx = 0
    agent._persist_user_message_override = "what the user typed"
    agent._persist_session(messages, conversation_history=None)
    uid = messages[0]["message_uid"]

    agent._apply_persist_user_message_override(messages)

    assert messages[0]["content"] == "what the user typed"
    assert messages[0]["message_uid"] == uid
    row = db._conn.execute(
        "SELECT content, message_uid FROM messages WHERE session_id = ? AND active = 1", (sid,)).fetchone()
    assert (row[0], row[1]) == ("what the user typed", uid)


def test_on_turn_complete_clones_carry_the_uid():
    from agent.conversation_loop import _notify_context_engine_turn_complete

    engine = _CapturingEngine()

    class _Agent:
        context_compressor = engine
        session_id = "s"

    messages = [{"role": "user", "content": "q", "message_uid": "a" * UID_LEN},
                {"role": "assistant", "content": "r", "message_uid": "b" * UID_LEN}]
    _notify_context_engine_turn_complete(_Agent(), messages, usage=None, logger=logging.getLogger("t"))

    seen = engine.turn_complete_messages
    assert [m["message_uid"] for m in seen] == ["a" * UID_LEN, "b" * UID_LEN]
    assert all(clone is not original for clone, original in zip(seen, messages))


def test_consecutive_user_merge_keeps_the_first_uid_and_records_the_absorbed_ones():
    from agent.agent_runtime_helpers import _merge_consecutive_users

    a = {"role": "user", "content": "first", "message_uid": "a" * UID_LEN}
    b = {"role": "user", "content": "second", "message_uid": "b" * UID_LEN}
    c = {"role": "user", "content": "third", "message_uid": "c" * UID_LEN}

    merged, repairs = _merge_consecutive_users([a, b, c])

    assert repairs == 2 and merged == [a]
    assert a["content"] == "first\n\nsecond\n\nthird"
    assert a["message_uid"] == "a" * UID_LEN
    assert a["_absorbed_message_uids"] == ["b" * UID_LEN, "c" * UID_LEN]
    # A restored (no ``_row_id``) absorbed dict still leaves its uid on the survivor.
    assert "_absorbed_row_ids" not in a


def test_restart_after_a_merge_still_sees_the_composites_constituents(db):
    """A dangling user row, a restart, the next prompt merged into it, a turn flush, another restart: the
    restored survivor names the absorbed row by uid instead of leaving the engine to parse ``\\n\\n``."""
    sid = "20260928_120300_witness"
    db.create_session(sid, "cli", model="test/model")
    db.append_message(session_id=sid, role="user", content="unanswered before the crash")
    dangling_uid = db.get_messages_as_conversation(sid)[0]["message_uid"]
    # Restart: the ACP/gateway restore shape, then the next prompt lands and the pre-request repair merges.
    history = db.get_messages_as_conversation(sid, repair_alternation=True, include_row_ids=True)
    agent = _make_agent(db, sid)
    agent._session_db_created = True
    prompt = {"role": "user", "content": "next prompt"}
    messages = history + [prompt]
    agent._persist_user_message_idx = 1
    agent._persist_session(messages, conversation_history=history)  # the turn-start flush of the prompt
    from agent.agent_runtime_helpers import repair_message_sequence
    repair_message_sequence(agent, messages)
    assert len(messages) == 1 and messages[0]["message_uid"] == dangling_uid
    assert messages[0]["_absorbed_message_uids"] == [prompt["message_uid"]]
    messages.append({"role": "assistant", "content": "reply"})
    agent._persist_session(messages, conversation_history=None)
    # Second restart.
    again = db.get_messages_as_conversation(sid, repair_alternation=True)
    survivor = next(m for m in again if m["message_uid"] == dangling_uid)
    assert survivor["content"].startswith("unanswered before the crash")
    assert prompt["message_uid"] in survivor["_absorbed_message_uids"]


def test_the_uid_never_reaches_the_provider_copy():
    from agent.message_metadata import PERSISTENCE_ONLY_MESSAGE_FIELDS

    assert {"message_uid", "_absorbed_message_uids"} <= PERSISTENCE_ONLY_MESSAGE_FIELDS
