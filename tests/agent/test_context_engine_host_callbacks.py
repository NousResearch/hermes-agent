"""Selection preparation reports real progress, cancels, and never sends unsafe fallback."""

from __future__ import annotations

import threading
import time

import pytest

from agent.context_engine import ContextEngine, ContextEngineError


@pytest.fixture(autouse=True)
def isolated_activity(monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)


class PluginEngine(ContextEngine):
    """A concrete third-party-style engine, without compressor implementation details."""

    context_length = 160_000

    @property
    def name(self):
        return "selection-plugin"

    def update_from_response(self, usage):
        self.last_prompt_tokens = usage.get("prompt_tokens", 0)

    def should_compress(self, prompt_tokens=None):
        return False

    def compress(self, messages, current_tokens=None, focus_topic=None):
        return messages


class SelectionCancelled(KeyboardInterrupt):
    pass


@pytest.fixture
def turn_agent(tmp_path):
    """Exercise the real agent and HTTP client against a loopback provider."""
    from run_agent import AIAgent
    from hermes_state import SessionDB
    from tests.fakes.fake_llm_provider import FakeLLMServer, Text

    database = SessionDB(db_path=tmp_path / "state.db")
    database.create_session("context-preparation", source="cli", model="fake-model")
    requests = []
    reply = [Text("answer from selected context")]

    def answer(request):
        requests.append(request["body"])
        return reply[0]

    with FakeLLMServer(script=answer, aux=answer) as provider:
        agent = AIAgent(
            model="fake-model", provider="openai",
            api_key="fixture-key", base_url=provider.base_url,
            enabled_toolsets=[], max_iterations=3,
            quiet_mode=True, skip_context_files=True, skip_memory=True,
            save_trajectories=False,
            session_id="context-preparation", session_db=database,
        )
        agent.compression_enabled = False
        agent.skip_background_review = True
        try:
            yield agent, requests, reply
        finally:
            agent.close()
            database.close()


@pytest.mark.parametrize("signature", ["request_only", "progress_only", "kwargs", "opaque"])
def test_selected_context_reaches_provider_without_rewriting_transcript(
    turn_agent, monkeypatch, signature,
):
    agent, requests, _ = turn_agent
    selected = [{"role": "user", "content": "bounded selected context"}]
    activity = {}

    def report(progress):
        before = agent.get_activity_summary()
        # The clock is an external boundary; observe the public watchdog snapshot,
        # not private timestamp fields or the activity tracker's implementation.
        with monkeypatch.context() as clock:
            clock.setattr(time, "time", lambda: before["last_activity_at"] + 10)
            progress("fixture archive commit")
            activity.update(agent.get_activity_summary())
        assert activity["seconds_since_activity"] == 0
        assert activity["last_activity_at"] == before["last_activity_at"] + 10

    def request_only(self, request_messages):
        return selected

    def progress_only(self, request_messages, *, progress=None):
        report(progress)
        return selected

    def kwargs(self, request_messages, **callbacks):
        report(callbacks["progress"])
        return selected

    class OpaqueSelection:
        @property
        def __signature__(self):
            raise ValueError("Extension callable has no Python signature")

        def __call__(
            self, request_messages, *, conversation_messages=None,
            incoming_message=None, budget_tokens=0,
        ):
            return selected

    class Selecting(PluginEngine):
        select_context = {
            "request_only": request_only, "progress_only": progress_only, "kwargs": kwargs,
            "opaque": OpaqueSelection(),
        }[signature]

    agent.context_compressor = Selecting()
    result = agent.run_conversation("current task", conversation_history=[
        {"role": "user", "content": "original history"},
        {"role": "assistant", "content": "original answer"},
    ])

    assert result["completed"] is True
    assert result["final_response"] == "answer from selected context"
    assert len(requests) == 1
    assert [m["content"] for m in requests[0]["messages"]] == ["bounded selected context"]
    transcript = [m.get("content") for m in result["messages"]]
    assert "original history" in transcript
    assert "current task" in transcript
    assert "bounded selected context" not in transcript


def test_ordinary_selection_error_preserves_provider_history(turn_agent):
    agent, requests, _ = turn_agent

    class Selecting(PluginEngine):
        def select_context(self, request_messages, **kwargs):
            raise RuntimeError("fixture selection unavailable")

    agent.context_compressor = Selecting()
    result = agent.run_conversation("current task")

    assert result["completed"] is True
    assert len(requests) == 1
    assert any(message.get("content") == "current task" for message in requests[0]["messages"])


def test_preparation_failure_returns_terminal_error_without_provider_call(turn_agent):
    agent, requests, _ = turn_agent
    reason = "Context preparation timed out; retry the turn."

    class Selecting(PluginEngine):
        def select_context(self, request_messages, **kwargs):
            raise ContextEngineError(reason)

    agent.context_compressor = Selecting()
    result = agent.run_conversation("current task")

    assert requests == []
    assert result["failed"] is True
    assert result["completed"] is False
    assert result["interrupted"] is False
    assert result["error"] == result["final_response"] == reason
    assert result["turn_exit_reason"] == "context_engine_error"
    assert result["failure_reason"] == "host_error"
    assert result["failure_retryable"] is True
    assert all(message.get("content") != reason for message in result["messages"])
    assert result["messages"][-1]["display_kind"] == "failed_turn"


def test_requested_selection_cancellation_finalizes_without_provider_call(turn_agent):
    agent, requests, _ = turn_agent
    outcome = {}
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()

    class Selecting(PluginEngine):
        def select_context(self, request_messages, *, cancelled=None):
            entered.set()
            while cancelled is None or not cancelled():
                if release.wait(0.01):
                    return
            raise SelectionCancelled("selection interrupted")

    def run():
        try:
            outcome["result"] = agent.run_conversation("current task")
        except BaseException as exc:
            outcome["error"] = exc
        finally:
            finished.set()

    agent.context_compressor = Selecting()
    caller = threading.Thread(target=run)
    caller.start()
    try:
        assert entered.wait(2), "Selection did not start"
        agent.interrupt()
        assert finished.wait(2), "Plugin never observed the live cancellation callback"
        assert "error" not in outcome
        result = outcome["result"]
        assert requests == []
        assert result["interrupted"] is True
        assert result["completed"] is False
    finally:
        release.set()
        caller.join(10)


def test_unrequested_selection_keyboard_interrupt_is_not_swallowed(turn_agent):
    agent, requests, _ = turn_agent

    class Selecting(PluginEngine):
        def select_context(self, request_messages, **kwargs):
            raise KeyboardInterrupt("stop selecting")

    agent.context_compressor = Selecting()
    with pytest.raises(KeyboardInterrupt):
        agent.run_conversation("current task")
    assert requests == []


def test_context_engine_tool_cancellation_closes_every_call_and_finalizes(turn_agent):
    from tests.fakes.fake_llm_provider import ToolCall

    agent, requests, reply = turn_agent
    name = "context_lookup"
    reply[0] = ToolCall(name, {"offset": 0}, parallel=[(name, {"offset": 1})])

    class Selecting(PluginEngine):
        def handle_tool_call(self, function_name, args, **kwargs):
            agent.interrupt()
            raise SelectionCancelled("lookup cancelled")

    agent.context_compressor = Selecting()
    agent._context_engine_tool_names = {name}
    agent.valid_tool_names.add(name)
    agent.tools = [{"type": "function", "function": {
        "name": name, "description": "Retrieve archived context",
        "parameters": {"type": "object", "properties": {"offset": {"type": "integer"}}},
    }}]

    try:
        result = agent.run_conversation("retrieve archived context")
    except KeyboardInterrupt:
        pytest.fail("Context-engine tool cancellation escaped turn finalization")

    assert result["interrupted"] is True
    assert result["completed"] is False
    assert len(requests) == 1
    issued_calls = [
        call for message in result["messages"] for call in message.get("tool_calls", [])
    ]
    tool_results = [message for message in result["messages"] if message.get("role") == "tool"]
    assert len(issued_calls) == 2
    assert sorted(message["tool_call_id"] for message in tool_results) == sorted(
        call["id"] for call in issued_calls
    )
