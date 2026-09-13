"""Isolated E2E coverage for the normal-turn State Answer Shadow seam."""

from __future__ import annotations

import queue
import sqlite3
from typing import Any
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


class Collector:
    def __init__(self):
        self.events: queue.Queue[tuple[str, Any]] = queue.Queue()

    def enqueue_initial(self, event):
        self.events.put(("initial", event))

    def enqueue_terminal(self, event_id, update):
        self.events.put(("terminal", (event_id, update)))


@pytest.fixture()
def loop_agent(tmp_path):
    from hermes_state import SessionDB
    from run_agent import AIAgent

    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            enabled_toolsets=[],
        )
        agent.client = MagicMock()
        agent._cached_system_prompt = "You are helpful."
        agent._use_prompt_caching = False
        agent.tool_delay = 0
        agent.compression_enabled = False
        agent.save_trajectories = False
        agent._session_db = SessionDB(tmp_path / "state.db")
        agent._owns_session_db = True
        yield agent
        agent._session_db.close()


def _response(content="Hello"):
    message = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def _take_events(collector):
    return [collector.events.get(timeout=2) for _ in range(2)]


def test_real_normal_turn_emits_ordered_shadow_lifecycle(loop_agent):
    collector = Collector()
    loop_agent._state_answer_shadow_collector = collector
    loop_agent.client.chat.completions.create.return_value = _response("isolated answer")

    result = loop_agent.run_conversation("hello")
    events = _take_events(collector)

    assert loop_agent.client.chat.completions.create.call_count == 1
    assert result["final_response"] == "isolated answer"
    assert result["api_calls"] == 1
    assert result["completed"] is True
    assert result["persistence_confirmed"] is True
    assert [kind for kind, _ in events] == ["initial", "terminal"]

    initial = events[0][1]
    terminal_event_id, terminal = events[1][1]
    assert terminal_event_id == initial["shadow_event_id"]
    assert initial["input_status"] == "input_unavailable"
    assert initial["shadow_decision"] == "not_evaluated"
    assert terminal["terminal_status"] == "completed"
    assert terminal["model_call_status"] == "observed"
    assert terminal["persistence_receipt_status"] == "true"

    with sqlite3.connect(loop_agent._session_db.db_path) as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id",
            (loop_agent.session_id,),
        ).fetchall()
    assert [(role, content) for role, content in rows if role == "assistant"][-1] == (
        "assistant",
        "isolated answer",
    )


def test_real_normal_turn_without_injection_preserves_legacy_result(loop_agent):
    loop_agent.client.chat.completions.create.return_value = _response("legacy answer")

    result = loop_agent.run_conversation("hello")

    assert result["final_response"] == "legacy answer"
    assert result["api_calls"] == 1
    assert not hasattr(loop_agent, "_state_answer_shadow_collector")
