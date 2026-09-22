import asyncio
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

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
async def test_acp_reset_rejected_while_turn_running():
    """prompt() dispatches slash commands before the is_running claim; /reset
    must be refused there rather than clearing state.history mid-turn."""
    acp_agent, state, fake, _conn = make_agent_and_state()
    state.is_running = True
    state.history = [{"role": "user", "content": "earlier"}]

    response = await acp_agent.prompt(
        session_id=state.session_id,
        prompt=[TextContentBlock(type="text", text="/reset")],
    )

    assert response.stop_reason == "end_turn"
    assert state.history == [{"role": "user", "content": "earlier"}]
    assert fake.runs == []
    assert state.queued_prompts == []


@pytest.mark.asyncio
async def test_acp_compress_rejected_while_turn_running():
    """Mid-turn /compress would compress a torn history and rebind
    state.history while the live turn still appends to the old list."""
    acp_agent, state, fake, _conn = make_agent_and_state()
    state.is_running = True
    state.history = [{"role": "user", "content": "earlier"}]
    sentinel_db = object()
    fake._session_db = sentinel_db
    fake._cached_system_prompt = "sys"
    fake._compress_context = lambda *a, **k: ([{"role": "user", "content": "summary"}], "new-sys")

    response = await acp_agent.prompt(
        session_id=state.session_id,
        prompt=[TextContentBlock(type="text", text="/compress")],
    )

    assert response.stop_reason == "end_turn"
    assert state.history == [{"role": "user", "content": "earlier"}]
    assert fake._session_db is sentinel_db
    assert fake.runs == []


@pytest.mark.asyncio
async def test_acp_prompt_during_mutating_command_queues_then_runs():
    """The command_op flag closes the check-then-act window: a prompt arriving while
    /reset is mid-flight must queue behind it, then run on the cleared history."""
    acp_agent, state, fake, _conn = make_agent_and_state()
    loop = asyncio.get_running_loop()
    state.history = [{"role": "user", "content": "earlier"}]

    # Inject inside _cmd_reset: at that point _handle_slash_command already holds
    # command_op, so the concurrent prompt must queue rather than claim the turn.
    orig_reset = acp_agent._cmd_reset

    def patched_reset(args, st):
        fut = asyncio.run_coroutine_threadsafe(
            acp_agent.prompt(
                session_id=st.session_id,
                prompt=[TextContentBlock(type="text", text="follow-up")],
            ), loop)
        fut.result(timeout=10)
        return orig_reset(args, st)

    with patch.object(acp_agent, "_cmd_reset", patched_reset):
        response = await acp_agent.prompt(
            session_id=state.session_id,
            prompt=[TextContentBlock(type="text", text="/reset")],
        )

    assert response.stop_reason == "end_turn"
    # The follow-up queued behind the op, then ran on the cleared history.
    assert fake.runs == ["follow-up"]
    assert state.history == [
        {"role": "user", "content": "follow-up"},
        {"role": "assistant", "content": "ran: follow-up"},
    ]
    assert state.command_op is False








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






