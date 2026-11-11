"""An explicit ``/resume <delegate-child>`` sticks: the next human message stays on it.

The #138210 route guard treats a row carrying ``_delegate_from`` as a stale capture and moves the
chat off it. A session the user deliberately resumed is the conversation, so ``/resume`` adopts it
(drops the marker) so the route self-heal (``_is_session_ended_in_db``) does not evict it. Real ``SessionDB``.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource, build_session_key
from gateway.session_lifecycle import is_delegate_child_row
from hermes_state import AsyncSessionDB, SessionDB


@pytest.mark.asyncio
async def test_resumed_delegate_child_keeps_the_next_message(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    source = SessionSource(platform=Platform.TELEGRAM, user_id="12345", chat_id="67890", user_name="u")
    key = build_session_key(source)
    db.create_session("parent_1", "telegram", session_key=key, user_id="12345", chat_id="67890")
    db.create_session("child_1", "telegram", parent_session_id="parent_1", session_key=key,
                      user_id="12345", chat_id="67890", model_config={"_delegate_from": "parent_1"})
    db.append_message("child_1", "assistant", "subagent result")
    db.end_session("parent_1", "session_switch")

    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)
    runner.adapters, runner.config, runner._voice_mode = {}, SimpleNamespace(platforms={}), {}
    runner._session_db, runner._running_agents = AsyncSessionDB(db), {}
    runner._is_user_authorized = lambda _s: True
    current = MagicMock(session_id="parent_1", session_key=key)
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = current
    runner.session_store.load_transcript.return_value = []
    runner.session_store.switch_session.return_value = MagicMock(session_id="child_1", session_key=key)

    await runner._handle_resume_command(MessageEvent(text="/resume child_1 --all", source=source))

    assert runner.session_store.switch_session.call_args.args[1] == "child_1"
    assert not is_delegate_child_row(db.get_session("child_1")), "route self-heal would evict it"
    db.close()
