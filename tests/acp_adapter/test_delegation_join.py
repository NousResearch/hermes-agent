"""ACP must return delegated results before ending a turn without a completion consumer."""

import json
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import acp
import pytest
from acp.schema import TextContentBlock

from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager


@pytest.mark.asyncio
async def test_prompt_joins_parallel_delegates_before_end_turn(tmp_path, monkeypatch):
    import tools.delegate_tool as dt

    from gateway.session_context import async_delivery_supported
    from tools.process_registry import process_registry

    parent = MagicMock()
    parent._delegate_depth = 0
    parent._interrupt_requested = False
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._session_db = None
    parent.model = "test-model"
    parent.provider = "openrouter"
    manager = SessionManager(agent_factory=lambda: parent)
    server = HermesACPAgent(session_manager=manager)
    server._conn = MagicMock(spec=acp.Client)
    server._conn.session_update = AsyncMock()
    session = await server.new_session(cwd=str(tmp_path))
    parent.session_id = session.session_id
    rendezvous = threading.Barrier(2, timeout=5)
    creds = {
        "model": "test-model", "provider": None, "base_url": None, "api_key": None,
        "api_mode": None, "command": None, "args": None,
    }
    monkeypatch.setattr(dt, "_resolve_delegation_credentials", lambda *a, **k: creds)
    monkeypatch.setattr(
        dt, "_build_child_agent",
        lambda **kw: SimpleNamespace(_delegate_role="leaf", _subagent_id=kw["goal"]),
    )

    def run_child(task_index, goal, **kwargs):
        rendezvous.wait()  # Both children must run concurrently, even when joined inline.
        return {
            "task_index": task_index, "status": "completed", "summary": f"reviewed {goal}",
            "api_calls": 1, "duration_seconds": 0.1, "model": "test-model",
            "exit_reason": "completed",
        }

    monkeypatch.setattr(dt, "_run_single_child", run_child)
    observed = {}

    def run_conversation(**kwargs):
        observed["async_delivery"] = async_delivery_supported()
        observed["result"] = json.loads(dt.delegate_task(
            tasks=[{"goal": "Review the code"}, {"goal": "Review the docs"}],
            background=True, parent_agent=parent,
        ))
        return {"final_response": "reviews complete", "messages": []}

    parent.run_conversation = run_conversation
    response = await server.prompt(
        prompt=[TextContentBlock(type="text", text="review code and docs")],
        session_id=session.session_id,
    )

    assert response.stop_reason == "end_turn"
    assert "results" in observed["result"], observed["result"]
    assert [(r["status"], r["summary"]) for r in observed["result"]["results"]] == [
        ("completed", "reviewed Review the code"), ("completed", "reviewed Review the docs"),
    ]
    assert observed["async_delivery"] is False
    assert "SYNCHRONOUSLY" in observed["result"]["note"]
    assert process_registry.completion_queue.empty()
    assert parent._active_children == []
