"""Selection preparation reports real progress, cancels, and never sends unsafe fallback."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.activity_tracking import ActivityTrackingMixin
from agent.context_engine import ContextEngine, ContextEngineError
from agent.conversation_loop import _apply_context_engine_selection


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


class HostAgent(ActivityTrackingMixin):
    def __init__(self, engine):
        self.context_compressor = engine
        self.session_id = "selection-callback-test"
        self._interrupt_requested = False
        self._last_activity_ts = 0.0
        self._last_activity_desc = "before selection"
        self._turn_liveness_activity_generation = 0


def select(agent):
    history = [{"role": "user", "content": "current task"}]
    request = [{"role": "system", "content": "system"}, *history]
    result = _apply_context_engine_selection(
        agent, request, history, history[-1], logger=logging.getLogger(__name__)
    )
    return result, request


def test_base_selection_accepts_optional_callbacks_without_reporting_progress():
    reported = []
    assert PluginEngine().select_context(
        [{"role": "user", "content": "task"}],
        progress=reported.append,
        cancelled=lambda: False,
    ) is None
    assert reported == []


def test_declared_callbacks_use_actual_activity_clock_and_live_interrupt_flag():
    observed_cancellation = []
    selected = [{"role": "user", "content": "bounded selected context"}]

    class Selecting(PluginEngine):
        def select_context(
            self, request_messages, *, conversation_messages=None, incoming_message=None,
            budget_tokens=0, progress=None, cancelled=None,
        ):
            assert budget_tokens == 160_000
            assert conversation_messages[-1]["content"] == "current task"
            assert incoming_message["content"] == "current task"
            assert agent._turn_liveness_activity_generation == 0
            observed_cancellation.append(cancelled())
            progress("selection committed 12 messages")
            agent._interrupt_requested = True
            observed_cancellation.append(cancelled())
            return selected

    agent = HostAgent(Selecting())
    result, _ = select(agent)

    assert result is selected
    assert observed_cancellation == [False, True]
    assert agent._last_activity_ts > 0
    assert agent._last_activity_desc == "selection committed 12 messages"
    assert agent._turn_liveness_activity_generation == 1


def test_kwargs_engine_receives_callbacks_without_synthetic_activity():
    selected = [{"role": "user", "content": "selected"}]
    received = {}

    class Selecting(PluginEngine):
        def select_context(self, request_messages, **kwargs):
            received.update(kwargs)
            assert kwargs["cancelled"]() is False
            # An idle wait is not progress: merely receiving callbacks must not touch activity.
            return selected

    agent = HostAgent(Selecting())
    result, _ = select(agent)

    assert result is selected
    assert callable(received["progress"])
    assert callable(received["cancelled"])
    assert agent._last_activity_ts == 0.0
    assert agent._turn_liveness_activity_generation == 0


def test_only_declared_callback_is_passed():
    selected = [{"role": "user", "content": "selected"}]

    class Selecting(PluginEngine):
        def select_context(self, request_messages, *, progress=None):
            progress("selection committed")
            return selected

    agent = HostAgent(Selecting())
    result, _ = select(agent)

    assert result is selected
    assert agent._last_activity_desc == "selection committed"


def test_engine_remains_usable_when_host_has_no_activity_callback():
    selected = [{"role": "user", "content": "selected"}]

    class Selecting(PluginEngine):
        def select_context(self, request_messages, **kwargs):
            assert kwargs["progress"] is None
            assert kwargs["cancelled"]() is True
            return selected

    agent = SimpleNamespace(context_compressor=Selecting(), _interrupt_requested=True)
    result, _ = select(agent)
    assert result is selected


def test_legacy_documented_selection_signature_remains_usable():
    selected = [{"role": "user", "content": "legacy selected context"}]

    class Selecting(PluginEngine):
        def select_context(
            self, request_messages, *, conversation_messages=None,
            incoming_message=None, budget_tokens=0,
        ):
            assert budget_tokens == 160_000
            return selected

    result, _ = select(HostAgent(Selecting()))
    assert result is selected


def test_request_only_legacy_signature_remains_usable():
    selected = [{"role": "user", "content": "old selected context"}]

    class Selecting(PluginEngine):
        def select_context(self, request_messages):
            return selected

    result, _ = select(HostAgent(Selecting()))
    assert result is selected


def test_ordinary_selection_error_still_falls_open():
    class Selecting(PluginEngine):
        def select_context(self, request_messages, **kwargs):
            raise RuntimeError("selection service unavailable")

    result, request = select(HostAgent(Selecting()))
    assert result is request


class SelectionCancelled(KeyboardInterrupt):
    pass


def test_cooperative_cancellation_cannot_fall_open_to_raw_history():
    class Selecting(PluginEngine):
        def select_context(self, request_messages, *, cancelled=None):
            if cancelled():
                raise SelectionCancelled("selection interrupted")
            raise AssertionError("host did not expose its interrupt flag")

    agent = HostAgent(Selecting())
    agent._interrupt_requested = True

    with pytest.raises(SelectionCancelled, match="selection interrupted"):
        select(agent)


def test_keyboard_interrupt_propagates_even_for_legacy_engine():
    class Selecting(PluginEngine):
        def select_context(
            self, request_messages, *, conversation_messages=None,
            incoming_message=None, budget_tokens=0,
        ):
            raise KeyboardInterrupt("stop selecting")

    with pytest.raises(KeyboardInterrupt, match="stop selecting"):
        select(HostAgent(Selecting()))


def test_typed_preparation_failure_never_falls_back_to_raw_history():
    class Selecting(PluginEngine):
        def select_context(self, request_messages, **kwargs):
            raise ContextEngineError("Memory preparation deadline: 12 pending nodes; retry the turn.")

    with pytest.raises(ContextEngineError, match="12 pending nodes"):
        select(HostAgent(Selecting()))


@pytest.fixture
def turn_agent(monkeypatch):
    """Run the real selection/dispatch/finalization path with an offline provider."""
    from run_agent import AIAgent

    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            model="test-model", provider="openai",
            api_key="test-key-1234567890", base_url="https://example.invalid/v1",
            quiet_mode=True, skip_context_files=True, skip_memory=True,
            save_trajectories=False,
        )
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent._disable_streaming = True
    agent.compression_enabled = False
    agent.skip_background_review = True
    requests = []

    def send(api_kwargs):
        requests.append(api_kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(
                    content="answer from selected context", tool_calls=None,
                    reasoning=None, reasoning_content=None, reasoning_details=None,
                ),
                finish_reason="stop",
            )],
            usage=None, model="test-model",
        )

    monkeypatch.setattr(agent, "_interruptible_api_call", send)
    yield agent, requests
    agent.close()


def test_selected_context_reaches_provider_without_rewriting_transcript(turn_agent):
    agent, requests = turn_agent

    class Selecting(PluginEngine):
        def select_context(self, request_messages, **kwargs):
            return [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "bounded selected context"},
            ]

    agent.context_compressor = Selecting()
    result = agent.run_conversation("current task", conversation_history=[
        {"role": "user", "content": "original history"},
        {"role": "assistant", "content": "original answer"},
    ])

    assert result["completed"] is True
    assert result["final_response"] == "answer from selected context"
    assert len(requests) == 1
    assert [m["content"] for m in requests[0]["messages"]] == [
        "You are helpful.", "bounded selected context",
    ]
    transcript = [m.get("content") for m in result["messages"]]
    assert "original history" in transcript
    assert "current task" in transcript
    assert "bounded selected context" not in transcript


def test_preparation_failure_returns_terminal_error_without_provider_call(turn_agent):
    agent, requests = turn_agent
    observed = {}
    reason = "Context preparation timed out; retry the turn."

    class Selecting(PluginEngine):
        def select_context(self, request_messages, **kwargs):
            raise ContextEngineError(reason)

        def on_turn_complete(self, messages, usage=None, **kwargs):
            observed.update(usage=usage, **kwargs)

    agent.context_compressor = Selecting()
    result = agent.run_conversation("current task")

    assert requests == []
    assert result["failed"] is True
    assert result["completed"] is False
    assert result["interrupted"] is False
    assert result["error"] == result["final_response"] == reason
    assert result["turn_exit_reason"] == "context_engine_error"
    assert result["messages"][-1]["content"] == reason
    assert observed["failed"] is True
    assert observed["usage"] is None


def test_requested_selection_cancellation_finalizes_without_provider_call(turn_agent):
    agent, requests = turn_agent
    observed = {}

    class Selecting(PluginEngine):
        def select_context(self, request_messages, *, cancelled=None):
            assert cancelled() is False
            agent.interrupt()
            if cancelled():
                raise SelectionCancelled("selection interrupted")
            raise AssertionError("host did not expose the live interrupt")

        def on_turn_complete(self, messages, usage=None, **kwargs):
            observed.update(usage=usage, **kwargs)

    agent.context_compressor = Selecting()
    result = agent.run_conversation("current task")

    assert requests == []
    assert result["interrupted"] is True
    assert result["completed"] is False
    assert observed["interrupted"] is True
    assert observed["usage"] is None
    assert agent._interrupt_requested is False


def test_unrequested_selection_keyboard_interrupt_is_not_swallowed(turn_agent):
    agent, requests = turn_agent

    class Selecting(PluginEngine):
        def select_context(self, request_messages, **kwargs):
            raise KeyboardInterrupt("stop selecting")

    agent.context_compressor = Selecting()
    with pytest.raises(KeyboardInterrupt, match="stop selecting"):
        agent.run_conversation("current task")
    assert requests == []
