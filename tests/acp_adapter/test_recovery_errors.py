"""Recovery faults must preserve the original session and remain retryable."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from acp.agent.router import build_agent_router
from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager
from hermes_state import SessionDB


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["session/load", "session/resume"])
@pytest.mark.parametrize("failure", ["query", "history", "agent", "database_init"])
async def test_recovery_fault_preserves_identity_and_history(tmp_path, method, failure):
    db = SessionDB(tmp_path / "state.db")
    try:
        factory = lambda: SimpleNamespace(model="recovery-test")
        seed = SessionManager(agent_factory=factory, db=db)
        state = seed.create_session(cwd=str(tmp_path))
        state.history = [{"role": "user", "content": "saved question"},
                         {"role": "assistant", "content": "saved answer"}]
        seed.save_session(state.session_id)
        before = db.get_messages_as_conversation(state.session_id)
        manager = SessionManager(agent_factory=factory, db=db)
        fault = RuntimeError("recovery dependency unavailable")
        if failure == "database_init":
            manager = SessionManager(agent_factory=factory)
            seam = patch("hermes_state_registry.acquire", side_effect=fault)
        elif failure == "agent":
            seam = patch.object(manager, "_make_agent", side_effect=fault)
        else:
            name = "get_session" if failure == "query" else "get_messages_as_conversation"
            seam = patch.object(db, name, side_effect=fault)
        router = build_agent_router(HermesACPAgent(session_manager=manager), use_unstable_protocol=True)
        request = {"cwd": str(tmp_path), "sessionId": state.session_id, "mcpServers": []}
        with seam, pytest.raises(RuntimeError, match="recovery dependency unavailable"):
            await router(method, request, False)
        assert manager._sessions == {}
        assert db.get_session(state.session_id) is not None
        assert db.get_messages_as_conversation(state.session_id) == before
        assert [row["id"] for row in db.search_sessions(source="acp", limit=10)] == [state.session_id]
        with patch("hermes_state_registry.acquire", return_value=db) as acquire:
            await router(method, request, False)
            if failure == "database_init":
                acquire.assert_called_once()
            else:
                acquire.assert_not_called()
        restored = manager.get_session(state.session_id)
        assert restored.session_id == state.session_id
        assert [(r["role"], r["content"]) for r in restored.history] == [
            (r["role"], r["content"]) for r in before
        ]
    finally:
        db.close()
