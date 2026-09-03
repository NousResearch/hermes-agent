import asyncio
import sys
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
async def test_acp_cancelled_prompt_task_does_not_wedge_session():
    """A prompt task cancelled mid-turn (client disconnect after cancel) must
    leave the session idle, not permanently stuck with is_running=True.

    Regression (#79196): cancel() + client disconnect can cancel the prompt()
    task while the agent runs in the executor. asyncio.CancelledError is a
    BaseException in Python 3.11+, so the surrounding ``except Exception``
    does not catch it, the ``state.is_running = False`` reset is skipped, and
    every later prompt is appended to queued_prompts forever.
    """
    import threading

    # Pre-warm the lazy heavy imports _run_agent performs on its first turn
    # (gateway.session_context, tools.terminal_tool, edit_approval); otherwise
    # the executor thread spends seconds importing them and run_started is not
    # set within a short wait.
    import gateway.session_context  # noqa: F401
    import acp_adapter.edit_approval  # noqa: F401
    from tools import terminal_tool  # noqa: F401

    acp_agent, state, fake, _conn = make_agent_and_state()
    run_started = threading.Event()
    release_run = threading.Event()

    def run_conversation(**kwargs):
        run_started.set()
        release_run.wait(timeout=60)
        fake.runs.append(kwargs["user_message"])
        messages = list(kwargs.get("conversation_history") or [])
        messages.append({"role": "user", "content": kwargs["user_message"]})
        messages.append({"role": "assistant", "content": "ran"})
        return {"final_response": "ran", "messages": messages}

    fake.run_conversation = run_conversation

    task = asyncio.create_task(
        acp_agent.prompt(
            session_id=state.session_id,
            prompt=[TextContentBlock(type="text", text="long running prompt")],
        )
    )
    # _run_agent performs heavy lazy imports on its first turn (session
    # context, terminal approval wiring), so the executor thread may take
    # several seconds to reach run_conversation. Poll with sleeps instead of a
    # blocking threading.Event.wait(): this coroutine IS the event loop, and
    # blocking it would starve the very prompt task we are waiting on.
    import time

    deadline = time.monotonic() + 30
    while not run_started.is_set() and time.monotonic() < deadline:
        await asyncio.sleep(0.1)
    assert run_started.is_set(), "prompt task never reached run_conversation"
    assert state.is_running is True

    # Client hits stop then disconnects: the in-flight prompt task is cancelled.
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    release_run.set()

    # The session must be idle again — not wedged.
    assert state.is_running is False

    # A subsequent prompt must run, not queue forever. (The executor thread
    # finishes the cancelled first run in the background, so runs may hold
    # both prompts in either order — what matters is that the follow-up was
    # executed at all instead of being appended to queued_prompts.)
    response = await acp_agent.prompt(
        session_id=state.session_id,
        prompt=[TextContentBlock(type="text", text="follow up")],
    )
    assert response.stop_reason == "end_turn"
    assert state.is_running is False
    assert state.queued_prompts == []
    assert "follow up" in fake.runs
