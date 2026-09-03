import asyncio
import sys
import threading
from types import ModuleType, SimpleNamespace

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
        self._supports_active_turn_redirect = True
        self.steers = []
        self.redirects = []
        self.runs = []

    def steer(self, text):
        self.steers.append(text)
        return True

    def redirect(self, text):
        self.redirects.append(text)
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


def test_acp_real_agent_gets_session_db_for_recall(monkeypatch):
    """ACP sessions persist to SessionDB; recall must receive the same DB handle."""
    captured = {}
    sentinel_db = NoopDb()

    class CapturingAgent(FakeAgent):
        def __init__(self, **kwargs):
            super().__init__()
            captured.update(kwargs)

    def mod(name, **attrs):
        module = ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        return module

    monkeypatch.setitem(sys.modules, "run_agent", mod("run_agent", AIAgent=CapturingAgent))
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.config",
        mod("hermes_cli.config", load_config=lambda: {"model": {"default": "m", "provider": "p"}}),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.runtime_provider",
        mod(
            "hermes_cli.runtime_provider",
            resolve_runtime_provider=lambda **_kwargs: {
                "provider": "p",
                "api_mode": "chat_completions",
                "base_url": "u",
                "api_key": "k",
                "command": None,
                "args": [],
            },
        ),
    )

    manager = SessionManager(db=sentinel_db)
    agent = manager._make_agent(session_id="acp-session", cwd=".")

    assert isinstance(agent, CapturingAgent)
    assert captured["session_db"] is sentinel_db
    assert captured["platform"] == "acp"
    assert captured["session_id"] == "acp-session"


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
async def test_acp_cancel_publishes_hard_stop_while_holding_runtime_lock():
    acp_agent, state, fake, _conn = make_agent_and_state()
    state.is_running = True
    state.current_prompt_text = "original request"
    observed = {}

    def interrupt():
        acquired = state.runtime_lock.acquire(blocking=False)
        observed["lock_held"] = not acquired
        if acquired:
            state.runtime_lock.release()

    fake.interrupt = interrupt

    await acp_agent.cancel(state.session_id)

    assert observed["lock_held"] is True
    assert state.cancel_event.is_set()
    assert state.interrupted_prompt_text == "original request"


@pytest.mark.asyncio
async def test_cancelled_prompt_releases_runtime_and_drains_follow_up_queue():
    class BlockingFirstRunAgent(FakeAgent):
        def __init__(self):
            super().__init__()
            self._supports_active_turn_redirect = False
            self.started = threading.Event()
            self.release = threading.Event()
            self.interrupt_calls = 0

        def interrupt(self):
            self.interrupt_calls += 1

        def run_conversation(
            self, *, user_message, conversation_history, task_id, **kwargs
        ):
            self.runs.append(user_message)
            if len(self.runs) == 1:
                self.started.set()
                assert self.release.wait(timeout=2), "test did not release first turn"
                return {
                    "final_response": "interrupted",
                    "messages": list(conversation_history or []),
                    "interrupted": True,
                }
            return {
                "final_response": f"ran: {user_message}",
                "messages": list(conversation_history or []),
            }

    fake = BlockingFirstRunAgent()
    manager = SessionManager(agent_factory=lambda **kwargs: fake, db=NoopDb())
    acp_agent = HermesACPAgent(session_manager=manager)
    state = manager.create_session(cwd=".")
    conn = CaptureConn()
    acp_agent.on_connect(conn)

    active_prompt = asyncio.create_task(
        acp_agent.prompt(
            session_id=state.session_id,
            prompt=[TextContentBlock(type="text", text="review everything")],
        )
    )
    assert await asyncio.to_thread(fake.started.wait, 1)

    active_prompt.cancel()
    await asyncio.sleep(0)
    assert state.is_running is True
    assert fake.interrupt_calls == 1

    queued_response = await acp_agent.prompt(
        session_id=state.session_id,
        prompt=[TextContentBlock(type="text", text="continue with the review")],
    )
    assert queued_response.stop_reason == "end_turn"
    assert state.queued_prompts == ["continue with the review"]

    fake.release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(active_prompt, timeout=2)

    assert state.is_running is False
    assert state.queued_prompts == []
    assert fake.runs == [
        "review everything",
        (
            "review everything\n\n"
            "User correction/guidance after interrupt: continue with the review"
        ),
    ]


@pytest.mark.asyncio
async def test_session_cancel_with_no_final_text_releases_runtime():
    class InterruptedAgent(FakeAgent):
        def __init__(self):
            super().__init__()
            self.started = threading.Event()
            self.release = threading.Event()

        def interrupt(self):
            self.release.set()

        def run_conversation(
            self, *, user_message, conversation_history, task_id, **kwargs
        ):
            self.runs.append(user_message)
            self.started.set()
            assert self.release.wait(timeout=2), "test did not cancel first turn"
            return {
                "final_response": None,
                "messages": list(conversation_history or []),
                "interrupted": True,
            }

    fake = InterruptedAgent()
    manager = SessionManager(agent_factory=lambda **kwargs: fake, db=NoopDb())
    acp_agent = HermesACPAgent(session_manager=manager)
    state = manager.create_session(cwd=".")
    acp_agent.on_connect(CaptureConn())

    active_prompt = asyncio.create_task(
        acp_agent.prompt(
            session_id=state.session_id,
            prompt=[TextContentBlock(type="text", text="wait for cancellation")],
        )
    )
    assert await asyncio.to_thread(fake.started.wait, 1)

    await acp_agent.cancel(state.session_id)
    response = await asyncio.wait_for(active_prompt, timeout=2)

    assert response.stop_reason == "cancelled"
    assert state.is_running is False
    assert state.current_prompt_text == ""

    follow_up = await acp_agent.prompt(
        session_id=state.session_id,
        prompt=[TextContentBlock(type="text", text="follow up")],
    )
    assert follow_up.stop_reason == "end_turn"
    assert fake.runs == [
        "wait for cancellation",
        (
            "wait for cancellation\n\n"
            "User correction/guidance after interrupt: follow up"
        ),
    ]


