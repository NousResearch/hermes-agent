"""ACP must report unloadable sessions as JSON-RPC errors, not silent success.

Regression coverage for #74678: session/load, session/resume and session/prompt
used to report success for an unknown/unrestorable session id — load returned
None (empty result), resume silently created a replacement session under a new
id a spec-conformant client cannot discover, and prompt answered
stopReason="refusal" for a turn that never ran. ACP defines the -32002
"Resource not found" error for exactly this condition; clients use it to fall
back to session/new deliberately.
"""

import pytest
from acp.exceptions import RequestError
from acp.schema import TextContentBlock

from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager

UNKNOWN_SESSION_ID = "00000000-0000-0000-0000-0000000000ff"


class DummyAgent:
    """Minimal AIAgent stand-in: accepts callback attrs, records runs."""

    def __init__(self):
        self.model = "dummy-model"
        self.provider = "dummy-provider"
        self.conversation_history = []
        self.runs = []
        self._supports_active_turn_redirect = True

    def run_conversation(self, *, user_message, conversation_history, **_kwargs):
        self.runs.append(user_message)
        messages = list(conversation_history or [])
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": "done"})
        return {"final_response": "done", "messages": messages}


class NoopDb:
    def get_session(self, *_args, **_kwargs):
        return None

    def create_session(self, *_args, **_kwargs):
        return None

    def update_session(self, *_args, **_kwargs):
        return None


class UnrestorableDb:
    """DB has the session row, but rebuilding its agent fails.

    Mirrors the named-custom-provider loss scenario (#63681/#74628): the
    session exists in state.db but can no longer be recreated, so
    ``_restore`` returns None and the caller must surface -32002.
    """

    def get_session(self, session_id, **_kwargs):
        return {
            "id": session_id,
            "source": "acp",
            "model": "lost-model",
            "billing_provider": None,
            "billing_base_url": None,
            "model_config": "{}",
        }

    def get_messages_as_conversation(self, *_args, **_kwargs):
        return []

    def create_session(self, *_args, **_kwargs):
        return None

    def update_session(self, *_args, **_kwargs):
        return None


class ExplodingAgentFactory:
    def __call__(self):
        raise RuntimeError("No LLM provider configured")


def make_agent() -> tuple[HermesACPAgent, SessionManager]:
    manager = SessionManager(agent_factory=DummyAgent, db=NoopDb())
    return HermesACPAgent(session_manager=manager), manager


@pytest.mark.asyncio
async def test_load_session_unknown_id_raises_resource_not_found():
    agent, _ = make_agent()

    with pytest.raises(RequestError) as exc_info:
        await agent.load_session(cwd=".", session_id=UNKNOWN_SESSION_ID)

    assert exc_info.value.code == -32002
    assert UNKNOWN_SESSION_ID in str(exc_info.value)
    assert exc_info.value.data == {"sessionId": UNKNOWN_SESSION_ID}


@pytest.mark.asyncio
async def test_resume_session_unknown_id_raises_and_creates_no_replacement():
    agent, manager = make_agent()

    with pytest.raises(RequestError) as exc_info:
        await agent.resume_session(cwd=".", session_id=UNKNOWN_SESSION_ID)

    assert exc_info.value.code == -32002
    # No silent replacement session may be created under any id.
    assert manager.get_session(UNKNOWN_SESSION_ID) is None
    assert len(manager._sessions) == 0


@pytest.mark.asyncio
async def test_prompt_unknown_id_raises_resource_not_found():
    agent, _ = make_agent()

    with pytest.raises(RequestError) as exc_info:
        await agent.prompt(
            session_id=UNKNOWN_SESSION_ID,
            prompt=[TextContentBlock(type="text", text="hello")],
        )

    assert exc_info.value.code == -32002


@pytest.mark.asyncio
async def test_load_session_unrestorable_db_row_raises_resource_not_found():
    # A session that exists in state.db but cannot be rebuilt (provider lost,
    # #63681/#74628) must produce the same -32002 as a never-created id —
    # not an empty success result.
    manager = SessionManager(agent_factory=ExplodingAgentFactory(), db=UnrestorableDb())
    agent = HermesACPAgent(session_manager=manager)

    with pytest.raises(RequestError) as exc_info:
        await agent.load_session(cwd=".", session_id=UNKNOWN_SESSION_ID)

    assert exc_info.value.code == -32002


@pytest.mark.asyncio
async def test_load_session_known_id_still_succeeds():
    agent, manager = make_agent()
    state = manager.create_session(cwd=".")

    response = await agent.load_session(cwd=".", session_id=state.session_id)

    assert response is not None
    assert response.modes is not None
    assert response.models is not None


@pytest.mark.asyncio
async def test_resume_session_known_id_still_succeeds():
    agent, manager = make_agent()
    state = manager.create_session(cwd=".")

    response = await agent.resume_session(cwd=".", session_id=state.session_id)

    assert response is not None
    assert manager.get_session(state.session_id) is state


@pytest.mark.asyncio
async def test_prompt_known_id_still_runs():
    agent, manager = make_agent()
    state = manager.create_session(cwd=".")

    response = await agent.prompt(
        session_id=state.session_id,
        prompt=[TextContentBlock(type="text", text="continue the task")],
    )

    assert response.stop_reason == "end_turn"
    assert state.agent.runs == ["continue the task"]
