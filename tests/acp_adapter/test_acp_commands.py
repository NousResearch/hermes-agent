from types import SimpleNamespace

import pytest
from acp.schema import TextContentBlock

from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager


class FakeAgent:
    def __init__(self):
        self.model = "fake-model"
        self.provider = "fake-provider"
        self.enabled_toolsets = ["hermes-acp"]
        self.disabled_toolsets = []
        self.tools = []
        self.valid_tool_names = set()
        self.steers = []
        self.runs = []

    def steer(self, text):
        self.steers.append(text)
        return True

    def run_conversation(self, *, user_message, conversation_history, task_id, **kwargs):
        self.runs.append(user_message)
        messages = list(conversation_history or [])
        messages.append({"role": "user", "content": user_message})
        final = f"ran: {user_message}"
        messages.append({"role": "assistant", "content": final})
        return {"final_response": final, "messages": messages}


class CaptureConn:
    def __init__(self):
        self.updates = []

    async def session_update(self, *args, **kwargs):
        if kwargs:
            self.updates.append((kwargs.get("session_id"), kwargs.get("update")))
        else:
            self.updates.append((args[0], args[1]))

    async def request_permission(self, *args, **kwargs):
        return SimpleNamespace(outcome="allow")


class NoopDb:
    def get_session(self, *_args, **_kwargs):
        return None

    def create_session(self, *_args, **_kwargs):
        return None

    def update_session(self, *_args, **_kwargs):
        return None


def make_agent_and_state():
    fake = FakeAgent()
    manager = SessionManager(agent_factory=lambda **kwargs: fake, db=NoopDb())
    acp_agent = HermesACPAgent(session_manager=manager)
    state = manager.create_session(cwd=".")
    conn = CaptureConn()
    acp_agent.on_connect(conn)
    return acp_agent, state, fake, conn


@pytest.mark.asyncio
async def test_acp_steer_slash_command_injects_into_running_agent():
    acp_agent, state, fake, _conn = make_agent_and_state()
    state.is_running = True

    response = await acp_agent.prompt(
        session_id=state.session_id,
        prompt=[TextContentBlock(type="text", text="/steer prefer the simpler fix")],
    )

    assert response.stop_reason == "end_turn"
    assert fake.steers == ["prefer the simpler fix"]
    assert fake.runs == []


@pytest.mark.asyncio
async def test_acp_steer_after_zed_interrupt_replays_interrupted_prompt_with_guidance():
    acp_agent, state, fake, _conn = make_agent_and_state()
    state.interrupted_prompt_text = "write hi to a text file"

    response = await acp_agent.prompt(
        session_id=state.session_id,
        prompt=[TextContentBlock(type="text", text="/steer write HELLO instead")],
    )

    assert response.stop_reason == "end_turn"
    assert fake.steers == []
    assert fake.runs == [
        "write hi to a text file\n\nUser correction/guidance after interrupt: write HELLO instead"
    ]
    assert state.interrupted_prompt_text == ""


@pytest.mark.asyncio
async def test_acp_queue_slash_command_adds_next_turn_without_running_now():
    acp_agent, state, fake, _conn = make_agent_and_state()

    response = await acp_agent.prompt(
        session_id=state.session_id,
        prompt=[TextContentBlock(type="text", text="/queue run the tests after this")],
    )

    assert response.stop_reason == "end_turn"
    assert state.queued_prompts == ["run the tests after this"]
    assert fake.runs == []


@pytest.mark.asyncio
async def test_acp_prompt_drains_queued_turns_after_current_run():
    acp_agent, state, fake, conn = make_agent_and_state()
    state.queued_prompts.append("then run tests")

    response = await acp_agent.prompt(
        session_id=state.session_id,
        prompt=[TextContentBlock(type="text", text="make the change")],
    )

    assert response.stop_reason == "end_turn"
    assert fake.runs == ["make the change", "then run tests"]
    assert state.queued_prompts == []
    agent_messages = [u for _sid, u in conn.updates if getattr(u, "session_update", None) == "agent_message_chunk"]
    assert len(agent_messages) >= 2
