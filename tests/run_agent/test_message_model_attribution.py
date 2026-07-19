"""Per-message model/provider attribution write-path tests.

Verifies ``AIAgent._flush_messages_to_session_db`` stamps each written
message row with the model/provider that was ACTUALLY active at write
time (read fresh via getattr, never cached) — so a mid-conversation
``/model`` switch (an in-place ``agent.switch_model`` on the same
session/thread) leaves earlier rows attributed to the old model while
later rows pick up the new one.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

SESSION_ID = "test-model-attribution"


def _make_agent(session_db, session_id=SESSION_ID, model="claude-sonnet-5", provider="anthropic"):
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model=model,
            provider=provider,
            quiet_mode=True,
            session_db=session_db,
            session_id=session_id,
            skip_context_files=True,
            skip_memory=True,
        )
    agent._ensure_db_session()
    return agent


class TestMessageModelAttributionWritePath:
    def test_flush_stamps_current_model_and_provider(self):
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "t.db")
            try:
                agent = _make_agent(db, model="claude-sonnet-5", provider="anthropic")
                turn = [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi"},
                ]
                agent._flush_messages_to_session_db(turn, [])

                rows = db.get_messages(SESSION_ID)
                assert [r["model"] for r in rows] == ["claude-sonnet-5", "claude-sonnet-5"]
                assert [r["provider"] for r in rows] == ["anthropic", "anthropic"]
            finally:
                db.close()

    def test_mid_conversation_model_switch_keeps_old_rows_on_old_model(self):
        """Mirrors an in-place ``agent.switch_model(...)`` call: the same
        agent/session continues, only ``agent.model``/``agent.provider``
        change. Rows flushed before the switch must keep the old
        attribution; rows flushed after must carry the new one.
        """
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db = SessionDB(db_path=Path(tmpdir) / "t.db")
            try:
                agent = _make_agent(db, model="claude-sonnet-5", provider="anthropic")

                first_turn = [
                    {"role": "user", "content": "first prompt"},
                    {"role": "assistant", "content": "first reply"},
                ]
                agent._flush_messages_to_session_db(first_turn, [])

                # Simulate the runtime effect of a /model switch — switch_model()
                # ultimately mutates these two attributes in place on the same
                # live agent/session; no new session is created.
                agent.model = "grok-4"
                agent.provider = "xai"

                history = [dict(m) for m in first_turn]
                second_turn = history + [
                    {"role": "user", "content": "second prompt"},
                    {"role": "assistant", "content": "second reply"},
                ]
                agent._flush_messages_to_session_db(second_turn, history)

                rows = db.get_messages(SESSION_ID)
                assert [r["content"] for r in rows] == [
                    "first prompt", "first reply", "second prompt", "second reply",
                ]
                assert [r["model"] for r in rows] == [
                    "claude-sonnet-5", "claude-sonnet-5", "grok-4", "grok-4",
                ]
                assert [r["provider"] for r in rows] == [
                    "anthropic", "anthropic", "xai", "xai",
                ]
                # Same session/thread throughout — the switch never forked a
                # new session id.
                assert {r["session_id"] for r in rows} == {SESSION_ID}
            finally:
                db.close()
