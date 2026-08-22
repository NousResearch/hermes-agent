import pytest
from acp.schema import TextContentBlock

from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager


class NoopDb:
    def get_session(self, *_args, **_kwargs):
        return None

    def create_session(self, *_args, **_kwargs):
        return None

    def update_session(self, *_args, **_kwargs):
        return None


class DummyAgent:
    def __init__(self, session_id: str, cwd: str):
        self.session_id = session_id
        self.cwd = cwd
        self.model = "dummy-model"
        self.provider = "dummy-provider"
        self.conversation_history = []
        self.runs = []

    def run_conversation(self, *, user_message, conversation_history, **_kwargs):
        self.runs.append(user_message)
        messages = list(conversation_history or [])
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": "done"})
        return {"final_response": "done", "messages": messages}


def make_manager() -> SessionManager:
    return SessionManager(
        agent_factory=lambda: DummyAgent("unused", "."),
        db=NoopDb(),
    )


def test_create_session_can_use_client_supplied_session_id():
    manager = make_manager()

    state = manager.create_session(cwd="/tmp/project", session_id="client-session-id")

    assert state.session_id == "client-session-id"
    assert manager.get_session("client-session-id") is state


@pytest.mark.asyncio
async def test_resume_missing_session_preserves_requested_session_id():
    manager = make_manager()
    agent = HermesACPAgent(session_manager=manager)

    await agent.resume_session(cwd="/tmp/project", session_id="client-session-id")

    state = manager.get_session("client-session-id")
    assert state is not None
    assert state.session_id == "client-session-id"
    assert state.cwd == "/tmp/project"

    response = await agent.prompt(
        session_id="client-session-id",
        prompt=[TextContentBlock(type="text", text="continue the task")],
    )

    assert response.stop_reason == "end_turn"
    assert state.agent.runs == ["continue the task"]
