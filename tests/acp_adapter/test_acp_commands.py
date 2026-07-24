import sys
from types import ModuleType, SimpleNamespace

import pytest
from acp.schema import TextContentBlock

import hermes_cli.goals as goals
from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager
from hermes_cli.goals import CONTINUATION_PROMPT_TEMPLATE, GoalManager


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

    def replace_messages(self, *_args, **_kwargs):
        return None

    def get_session_title(self, *_args, **_kwargs):
        return "Test ACP session"


class InMemoryGoalDb:
    def __init__(self):
        self.meta = {}

    def get_meta(self, key):
        return self.meta.get(key)

    def set_meta(self, key, value):
        self.meta[key] = value


@pytest.fixture()
def goal_db(monkeypatch):
    db = InMemoryGoalDb()
    monkeypatch.setattr(goals, "_get_session_db", lambda: db)
    return db


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
async def test_acp_plain_correction_redirects_running_turn():
    acp_agent, state, fake, _conn = make_agent_and_state()
    state.is_running = True

    response = await acp_agent.prompt(
        session_id=state.session_id,
        prompt=[TextContentBlock(type="text", text="No, use Postgres instead")],
    )

    assert response.stop_reason == "end_turn"
    assert fake.redirects == ["No, use Postgres instead"]
    assert state.queued_prompts == []
    assert fake.runs == []


@pytest.mark.asyncio
async def test_acp_plain_correction_after_cancel_replays_original_prompt():
    acp_agent, state, fake, _conn = make_agent_and_state()
    state.interrupted_prompt_text = "implement it with SQLite"

    response = await acp_agent.prompt(
        session_id=state.session_id,
        prompt=[TextContentBlock(type="text", text="No, use Postgres instead")],
    )

    assert response.stop_reason == "end_turn"
    assert fake.runs == [
        "implement it with SQLite\n\n"
        "User correction/guidance after interrupt: No, use Postgres instead"
    ]
    assert state.interrupted_prompt_text == ""


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
async def test_acp_steer_on_idle_session_runs_as_regular_prompt():
    # /steer on an idle session (no running turn, nothing to salvage) should
    # run the steer payload as a normal user prompt — NOT silently append it
    # to state.queued_prompts. Without this, users on Zed / other ACP clients
    # see their /steer turn into "queued for the next turn" when they never
    # typed /queue. Matches gateway/run.py ~L4898 idle-/steer behavior.
    acp_agent, state, fake, _conn = make_agent_and_state()

    response = await acp_agent.prompt(
        session_id=state.session_id,
        prompt=[TextContentBlock(type="text", text="/steer summarize the README")],
    )

    assert response.stop_reason == "end_turn"
    assert fake.steers == []
    assert fake.runs == ["summarize the README"]
    assert state.queued_prompts == []


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


@pytest.mark.asyncio
async def test_acp_goal_status_without_active_goal_reports_empty_state(goal_db):
    acp_agent, state, fake, conn = make_agent_and_state()

    response = await acp_agent.prompt(
        session_id=state.session_id,
        prompt=[TextContentBlock(type="text", text="/goal status")],
    )

    assert response.stop_reason == "end_turn"
    assert fake.runs == []
    text_updates = [
        u for _sid, u in conn.updates
        if getattr(u, "session_update", None) == "agent_message_chunk"
    ]
    assert any(
        "No active goal" in getattr(getattr(u, "content", None), "text", "")
        for u in text_updates
    )


@pytest.mark.asyncio
async def test_acp_goal_set_stores_goal_and_queues_kickoff(goal_db):
    acp_agent, state, fake, _conn = make_agent_and_state()

    response = await acp_agent.prompt(
        session_id=state.session_id,
        prompt=[TextContentBlock(type="text", text="/goal ship the ACP goal command")],
    )

    assert response.stop_reason == "end_turn"
    assert fake.runs == []
    mgr = GoalManager(session_id=state.session_id)
    assert mgr.is_active()
    assert mgr.state.goal == "ship the ACP goal command"
    assert state.queued_prompts == ["ship the ACP goal command"]


@pytest.mark.asyncio
async def test_acp_goal_pause_resume_clear_updates_state_and_preserves_user_queue(goal_db):
    acp_agent, state, fake, _conn = make_agent_and_state()
    GoalManager(session_id=state.session_id).set("finish ACP parity")
    synthetic = CONTINUATION_PROMPT_TEMPLATE.format(goal="finish ACP parity")
    state.queued_prompts.extend(["manual follow-up", synthetic])

    pause_response = await acp_agent.prompt(
        session_id=state.session_id,
        prompt=[TextContentBlock(type="text", text="/goal pause")],
    )

    assert pause_response.stop_reason == "end_turn"
    assert GoalManager(session_id=state.session_id).state.status == "paused"
    assert state.queued_prompts == ["manual follow-up"]

    resume_response = await acp_agent.prompt(
        session_id=state.session_id,
        prompt=[TextContentBlock(type="text", text="/goal resume")],
    )

    assert resume_response.stop_reason == "end_turn"
    assert GoalManager(session_id=state.session_id).is_active()
    assert state.queued_prompts == ["manual follow-up"]

    clear_response = await acp_agent.prompt(
        session_id=state.session_id,
        prompt=[TextContentBlock(type="text", text="/goal clear")],
    )

    assert clear_response.stop_reason == "end_turn"
    assert not GoalManager(session_id=state.session_id).has_goal()
    assert state.queued_prompts == ["manual follow-up"]
    assert fake.runs == []


@pytest.mark.asyncio
async def test_acp_subgoal_without_goal_reports_empty_state(goal_db):
    acp_agent, state, fake, conn = make_agent_and_state()

    response = await acp_agent.prompt(
        session_id=state.session_id,
        prompt=[TextContentBlock(type="text", text="/subgoal")],
    )

    assert response.stop_reason == "end_turn"
    assert fake.runs == []
    text_updates = [
        u for _sid, u in conn.updates
        if getattr(u, "session_update", None) == "agent_message_chunk"
    ]
    assert any(
        "No active goal. Set one with /goal <text>." in getattr(getattr(u, "content", None), "text", "")
        for u in text_updates
    )


@pytest.mark.asyncio
async def test_acp_subgoal_add_list_remove_clear_flow(goal_db):
    acp_agent, state, fake, conn = make_agent_and_state()
    GoalManager(session_id=state.session_id).set("finish ACP parity")

    add_response = await acp_agent.prompt(
        session_id=state.session_id,
        prompt=[TextContentBlock(type="text", text="/subgoal keep the patch small")],
    )

    assert add_response.stop_reason == "end_turn"
    assert GoalManager(session_id=state.session_id).state.subgoals == ["keep the patch small"]

    list_response = await acp_agent.prompt(
        session_id=state.session_id,
        prompt=[TextContentBlock(type="text", text="/subgoal")],
    )

    assert list_response.stop_reason == "end_turn"
    text_updates = [
        u for _sid, u in conn.updates
        if getattr(u, "session_update", None) == "agent_message_chunk"
    ]
    assert any(
        "- 1. keep the patch small" in getattr(getattr(u, "content", None), "text", "")
        for u in text_updates
    )

    remove_response = await acp_agent.prompt(
        session_id=state.session_id,
        prompt=[TextContentBlock(type="text", text="/subgoal remove 1")],
    )

    assert remove_response.stop_reason == "end_turn"
    assert GoalManager(session_id=state.session_id).state.subgoals == []

    GoalManager(session_id=state.session_id).add_subgoal("verify focused tests")
    clear_response = await acp_agent.prompt(
        session_id=state.session_id,
        prompt=[TextContentBlock(type="text", text="/subgoal clear")],
    )

    assert clear_response.stop_reason == "end_turn"
    assert GoalManager(session_id=state.session_id).state.subgoals == []
    assert fake.runs == []


@pytest.mark.asyncio
async def test_acp_subgoal_invalid_inputs(goal_db):
    acp_agent, state, fake, conn = make_agent_and_state()
    GoalManager(session_id=state.session_id).set("finish ACP parity")

    for command in (
        "/subgoal remove",
        "/subgoal remove nope",
        "/subgoal clear",
    ):
        response = await acp_agent.prompt(
            session_id=state.session_id,
            prompt=[TextContentBlock(type="text", text=command)],
        )
        assert response.stop_reason == "end_turn"

    assert fake.runs == []
    text = "\n".join(
        getattr(getattr(u, "content", None), "text", "")
        for _sid, u in conn.updates
        if getattr(u, "session_update", None) == "agent_message_chunk"
    )
    assert "Usage: /subgoal remove <n>" in text
    assert "/subgoal remove: <n> must be an integer" in text
    assert "No subgoals to clear." in text


@pytest.mark.asyncio
async def test_acp_goal_continuation_is_queued_and_drained_after_turn(goal_db, monkeypatch):
    acp_agent, state, fake, _conn = make_agent_and_state()
    GoalManager(session_id=state.session_id).set("finish ACP parity")
    verdicts = iter([
        ("continue", "more work remains", False),
        ("done", "goal completed", False),
    ])
    monkeypatch.setattr(goals, "judge_goal", lambda *_args, **_kwargs: next(verdicts))

    response = await acp_agent.prompt(
        session_id=state.session_id,
        prompt=[TextContentBlock(type="text", text="make the change")],
    )

    assert response.stop_reason == "end_turn"
    assert fake.runs[0] == "make the change"
    assert fake.runs[1].startswith("[Continuing toward your standing goal]")
    assert state.queued_prompts == []
    assert GoalManager(session_id=state.session_id).state.status == "done"


@pytest.mark.asyncio
async def test_acp_goal_done_does_not_queue_continuation(goal_db, monkeypatch):
    acp_agent, state, fake, _conn = make_agent_and_state()
    GoalManager(session_id=state.session_id).set("finish ACP parity")
    monkeypatch.setattr(
        goals,
        "judge_goal",
        lambda *_args, **_kwargs: ("done", "already complete", False),
    )

    response = await acp_agent.prompt(
        session_id=state.session_id,
        prompt=[TextContentBlock(type="text", text="make the change")],
    )

    assert response.stop_reason == "end_turn"
    assert fake.runs == ["make the change"]
    assert state.queued_prompts == []
    assert GoalManager(session_id=state.session_id).state.status == "done"
