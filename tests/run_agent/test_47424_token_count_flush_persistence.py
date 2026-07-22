"""Regression tests for PR #47424 — production boundary proof that
``HermesAgent._flush_messages_to_session_db`` propagates ``token_count``
from the running transcript into the real SessionDB messages table.

The change adds a per-message token-burn column (messages.token_count)
that is populated when the conversation loop assembles a successful
assistant API response.  Per teknium1's review, "the new tests cover
only the already-existing ``SessionDB.append_message(token_count=...)``
contract ... [they] do not verify either changed production boundary:
assigning response usage to an assistant message **or** flushing that
message into SQLite."

These tests close that gap against the *real* flush path with *real*
SQLite, using ``object.__new__(AIAgent)`` to skip the 60-parameter
``__init__`` (same pattern as ``tests/run_agent/
test_file_mutation_verifier.py``).

Note on test isolation: each test uses its own agent+SESSION_DB fixture.
CPython reuses ids for short string/int objects, so sharing one bare agent
across multiple flushes would cause the second flush's transcribed
messages to collide with already-tracked ids in ``_flushed_db_message_ids``
and skip the new rows.  One agent per test sidesteps that entirely.
"""

from __future__ import annotations

import pytest

from run_agent import AIAgent
from hermes_state import SessionDB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bare_agent_with_session_db(db: SessionDB) -> AIAgent:
    """Skip AIAgent.__init__ and attach only the state needed by
    ``_flush_messages_to_session_db``.

    ``__init__`` takes ~60 parameters and touches network/auth/filesystem.
    Two instance attributes are required for the flush path:

    - ``self._session_db`` (an instance of SessionDB)
    - ``self.session_id`` (str; used as the row key)

    Plus cursor-tracking attributes that ``_flush_messages_to_session_db``
    inspects/resets:
    - ``_session_db_created`` (True so the ensure-DB skip path runs)
    - ``_last_flushed_db_idx`` (0 = first flush on a new session)
    - ``_flushed_db_message_ids`` (an empty set)
    - ``_flushed_db_message_session_id`` (None by default)

    ``_apply_persist_user_message_override`` is stubbed to a no-op so
    the per-turn override hook doesn't try to mutate messages mid-test.
    """
    agent = object.__new__(AIAgent)
    agent._session_db = db
    agent._session_db_created = True
    # agent.session_id is set in __init__; we skip __init__, so type-checkers
    # can't see it. Mirror how tests/run_agent/test_message_sequence_repair.py:289
    # sets it: bypass the attribute-access type error via setattr.
    setattr(agent, "session_id", "s1")
    agent._last_flushed_db_idx = 0  # type: ignore[attr-defined]
    agent._flushed_db_message_ids = set()  # type: ignore[attr-defined]
    agent._flushed_db_message_session_id = None  # type: ignore[attr-defined]
    agent._apply_persist_user_message_override = lambda messages: None
    return agent


@pytest.fixture()
def db(tmp_path):
    """Real SessionDB with a temp database file (mirrors tests/test_hermes_state.py)."""
    db_path = tmp_path / "test_state.db"
    session_db = SessionDB(db_path=db_path)
    session_db.create_session(session_id="s1", source="cli")
    yield session_db
    session_db.close()


def _find_message_by_content(db: SessionDB, content: str):
    rows = [m for m in db.get_messages("s1") if m["content"] == content]
    assert len(rows) == 1, (
        f"expected exactly 1 message with content={content!r}, got {len(rows)}"
    )
    return rows[0]


# ---------------------------------------------------------------------------
# Production-boundary tests
# ---------------------------------------------------------------------------


class TestTokenCountFlowsThroughFlush:
    """token_count that the conversation loop writes to ``msg["token_count"]``
    must survive the production ``_flush_messages_to_session_db`` call and
    land as a non-NULL row in messages.token_count (#47424 #47201).

    Each test uses its OWN agent to avoid CPython object-id reuse across
    two flush sequences — see module docstring.
    """

    def test_usage_bearing_assistant_turn_persists_token_count(self, db):
        """A successful assistant turn carrying usage output_tokens through
        the run_agent flush must round-trip in SQLite."""
        agent = _bare_agent_with_session_db(db)
        messages = [
            {"role": "user", "content": "what is 2+2?"},
            {
                "role": "assistant",
                "content": "4",
                # Set by agent/conversation_loop.py on success.
                "token_count": 150,
                "finish_reason": "stop",
            },
        ]

        AIAgent._flush_messages_to_session_db(agent, messages, conversation_history=[])

        row = _find_message_by_content(db, "4")
        assert row["role"] == "assistant"
        # 150 travelled msg -> append_message -> messages.token_count
        # (not the legacy default None).
        assert row["token_count"] == 150

    def test_assistant_turn_without_usage_persists_null(self, db):
        """An assistant turn that the model delivered without a usage block
        (some providers omit it) leaves messages.token_count NULL — never 0,
        never a previous turn's count."""
        agent = _bare_agent_with_session_db(db)
        messages = [
            {"role": "user", "content": "what time is it?"},
            {
                "role": "assistant",
                "content": "11:42",
                "finish_reason": "stop",
                # No `token_count` key — provider omitted usage.
                # run_conversation() leaves msg.token_count absent on this
                # path so flush should also leave it None.
            },
        ]

        AIAgent._flush_messages_to_session_db(agent, messages, conversation_history=[])

        row = _find_message_by_content(db, "11:42")
        # NULL is the contract: distinguishes "no usage data" from "0 tokens"
        # (0 is a legitimate value when the model emits completion_tokens=0,
        # which serves a different analytics purpose — see #47424 comment).
        assert row["token_count"] is None, (
            f"expected NULL for no-usage turn, got {row['token_count']!r}"
        )

    def test_user_turn_token_count_is_null(self, db):
        """User messages never carry output_tokens; flush must not invent one."""
        agent = _bare_agent_with_session_db(db)
        messages = [{"role": "user", "content": "hi there"}]  # no token_count

        AIAgent._flush_messages_to_session_db(agent, messages, conversation_history=[])

        row = _find_message_by_content(db, "hi there")
        assert row["role"] == "user"
        assert row["token_count"] is None
