"""ACP gating is presentation/pending state, never a synthetic conversation turn."""
import asyncio
from types import SimpleNamespace

import pytest
from acp.schema import TextContentBlock

from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager
from hermes_state import SessionDB


class Agent:
    model = "welcome"
    provider = "nous"
    base_url = "https://inference-api.nousresearch.com/v1"
    tools = []

    def __init__(self, **kwargs):
        self.session_id = kwargs.get("session_id")
        self.calls = []
        self.blocked = True

    def run_conversation(self, user_message, conversation_history, **kwargs):
        self.calls.append(user_message)
        if self.blocked:
            return {"messages": list(conversation_history), "final_response": "Use /login or /model to continue.",
                    "failed": True, "completed": False, "partial": False, "api_calls": 0,
                    "failure_reason": "free_tier_limit", "code": "free_tier_limit",
                    "retryable": False, "retryable_after_recovery": True}
        self.stream_delta_callback("Finished.")
        return {"messages": conversation_history + [{"role": "user", "content": user_message},
                                                    {"role": "assistant", "content": "Finished."}],
                "final_response": "Finished.\n\nConfigure a provider to continue.",
                "completed": True, "free_tier_notice": "Configure a provider to continue.",
                "response_transformed": True, "free_tier": {"capped": True}}


class Conn:
    def __init__(self):
        self.updates = []

    async def session_update(self, *args, **kwargs):
        self.updates.append(kwargs.get("update") if kwargs else args[1])

    async def request_permission(self, **kwargs):
        return SimpleNamespace(outcome="allow")


def text_updates(conn):
    return [u.content.text for u in conn.updates if getattr(u, "session_update", None) == "agent_message_chunk"]


@pytest.mark.asyncio
async def test_refusal_retains_prompt_and_queue_until_explicit_recovery(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    manager = SessionManager(agent_factory=Agent, db=db)
    server = HermesACPAgent(manager)
    state = manager.create_session(str(tmp_path))
    state.history = [{"role": "user", "content": "old"}, {"role": "assistant", "content": "answer"}]
    manager.save_session(state.session_id)
    before = db.get_messages_as_conversation(state.session_id)
    state.queued_prompts = ["later"]
    conn = Conn()
    server.on_connect(conn)
    prompt = [TextContentBlock(type="text", text="retained request")]
    response = await server.prompt(prompt, state.session_id)
    assert response.stop_reason == "refusal"
    assert state.pending_prompt == prompt
    assert state.queued_prompts == ["later"]
    assert state.agent.calls == ["retained request"]
    assert db.get_messages_as_conversation(state.session_id) == before
    assert response.field_meta["retryable"] is False
    login = await server.prompt([TextContentBlock(type="text", text="/login")], state.session_id)
    assert login.stop_reason == "end_turn"
    assert "hermes setup" in text_updates(conn)[-1]
    assert state.agent.calls == ["retained request"]
    state.agent.blocked = False
    conn.updates.clear()
    response = await server.prompt(prompt, state.session_id)
    await asyncio.sleep(0)
    assert response.stop_reason == "end_turn"
    assert state.pending_prompt is None
    assert state.queued_prompts == ["later"]  # crossing pauses the next queued turn, too
    assert "".join(text_updates(conn)) == "Finished.\n\nConfigure a provider to continue."
    assert state.history[-1]["content"] == "Finished."
    # A cap reached while draining a queued turn must also pause its caller's drain.
    state.queued_prompts = ["crossing queued turn", "must stay queued"]
    conn.updates.clear()
    await server._finish_turn(state, state.session_id, conn,
        {"completed": True, "final_response": "previous turn"}, state.agent.session_id, False)
    assert state.queued_prompts == ["must stay queued"]
    assert state.agent.calls[-1] == "crossing queued turn"
    db.close()


@pytest.mark.asyncio
async def test_authentication_refreshes_welcome_runtime_without_draining_pending(tmp_path, monkeypatch):
    db = SessionDB(tmp_path / "auth.db")
    manager = SessionManager(agent_factory=Agent, db=db)
    server = HermesACPAgent(manager)
    state = manager.create_session(str(tmp_path))
    from hermes_cli.auth_constants import DEFAULT_NOUS_WELCOME_URL
    state.agent.base_url = DEFAULT_NOUS_WELCOME_URL
    old_agent = state.agent
    state.pending_prompt = [TextContentBlock(type="text", text="retry me")]
    state.queued_prompts = ["later"]
    monkeypatch.setattr("acp_adapter.server.detect_provider", lambda: "nous")
    from acp_adapter.auth import TERMINAL_SETUP_AUTH_METHOD_ID
    assert await server.authenticate(TERMINAL_SETUP_AUTH_METHOD_ID) is not None
    assert state.agent is not old_agent
    assert state.pending_prompt[0].text == "retry me"
    assert state.queued_prompts == ["later"]
    assert not state.agent.calls
    # Recovery can also be a same-provider model selection. Never carry the welcome host
    # onto a paid model; the factory must resolve that provider's configured endpoint.
    state.agent.base_url = DEFAULT_NOUS_WELCOME_URL
    created = []
    original_factory = manager._make_agent
    def capture_factory(**kwargs):
        created.append(kwargs)
        return original_factory(**kwargs)
    monkeypatch.setattr(manager, "_make_agent", capture_factory)
    await server.set_session_model("nous:paid-model", state.session_id)
    assert not created[-1].get("base_url")
    assert state.pending_prompt[0].text == "retry me"
    db.close()
