"""Turns that are not user-originated do not drive memory.

Refs #141912. A notice is injected as a user-role turn, but it is typed at turn start
(``persist_user_display_kind``). Memory used to ignore the
display kind: it was told of the notice, recalled against it, counted it toward the
memory review and synced it as something the user said. The turn now asks the
canonical rule, ``is_user_originated_turn``, once, and the per-turn memory calls
follow the answer. A user-originated turn still drives all of them.
"""

from __future__ import annotations

import shutil
import tempfile
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.turn_context import build_turn_context
from hermes_state import SessionDB
from run_agent import AIAgent

NOTICE = "The nightly backup finished: 412 files, 0 errors."
TYPED = "What did the nightly backup cover?"


@pytest.fixture()
def agent():
    tmp = tempfile.mkdtemp(prefix="memory_turn_origin_")
    db = SessionDB(Path(tmp) / "state.db")
    sid = "sess-origin"
    db.create_session(session_id=sid, source="cli", model="test-model")
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
    agent._cached_system_prompt = "SYSTEM"
    agent._skip_mcp_refresh = True
    # A live memory slot, and a review that fires on every counted turn.
    agent._memory_manager = MagicMock()
    agent._memory_manager.prefetch_all.return_value = ""
    agent._memory_store = object()
    agent._memory_nudge_interval = 1
    agent.valid_tool_names = {"memory"}
    try:
        yield agent
    finally:
        db.close()
        shutil.rmtree(tmp, ignore_errors=True)


def _start_turn(agent, user_message, **overrides):
    kwargs = dict(
        agent=agent,
        user_message=user_message,
        system_message=None,
        conversation_history=None,
        task_id=None,
        stream_callback=None,
        persist_user_message=None,
        restore_or_build_system_prompt=lambda *a, **k: None,
        install_safe_stdio=lambda: None,
        sanitize_surrogates=lambda s: s,
        summarize_user_message_for_log=lambda s: s,
        set_session_context=lambda _sid: None,
        set_current_write_origin=lambda _o: None,
        ra=lambda: types.SimpleNamespace(_set_interrupt=lambda *a, **k: None),
    )
    kwargs.update(overrides)
    with patch("agent.auxiliary_client.set_runtime_main", lambda *a, **k: None):
        return build_turn_context(**kwargs)


def test_a_notice_turn_does_not_drive_memory_at_turn_start(agent):
    notice = _start_turn(agent, NOTICE, persist_user_display_kind="internal_notification")
    agent._memory_manager.on_turn_start.assert_not_called()
    agent._memory_manager.prefetch_all.assert_not_called()
    assert notice.should_review_memory is False

    typed = _start_turn(agent, TYPED)
    agent._memory_manager.on_turn_start.assert_called_once()
    agent._memory_manager.prefetch_all.assert_called_once()
    assert typed.should_review_memory is True


def test_a_notice_turn_is_not_synced_at_turn_end(agent):
    _start_turn(agent, NOTICE, persist_user_display_kind="internal_notification")
    agent._sync_external_memory_for_turn(original_user_message=NOTICE, final_response="Noted.", interrupted=False)
    agent._memory_manager.sync_all.assert_not_called()
    agent._memory_manager.queue_prefetch_all.assert_not_called()

    _start_turn(agent, TYPED)
    agent._sync_external_memory_for_turn(original_user_message=TYPED, final_response="Everything.", interrupted=False)
    agent._memory_manager.sync_all.assert_called_once()
    agent._memory_manager.queue_prefetch_all.assert_called_once()
