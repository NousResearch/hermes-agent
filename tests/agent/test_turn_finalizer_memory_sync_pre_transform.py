"""Verify the display/memory split for `transform_llm_output` (#57282).

A `transform_llm_output` plugin can append display-only content (a citation,
a sponsored-message footer) to the response. Before this change, that
appended content also flowed into external memory sync, so a memory
provider (e.g. Honcho) would store it as if the model had said it. Plugins
can now optionally return `{"display": str, "memory": str}` instead of a
plain string to keep the two separate. A plain string return, or a dict
without a `"memory"` key, behaves exactly as before: memory receives the
same value as display.
"""

from types import SimpleNamespace
from typing import Any

from agent.turn_finalizer import finalize_turn
from agent.memory_provider import MemoryProvider
from agent.memory_manager import MemoryManager


class FakeAgent:
    """Matches the FakeAgent convention in test_turn_finalizer_final_response_persistence.py."""

    def __init__(self):
        self.max_iterations = 90
        self.iteration_budget = SimpleNamespace(remaining=10, used=1, max_total=90)
        self.quiet_mode = True
        self.model = "test-model"
        self.provider = "test-provider"
        self.base_url = ""
        self.session_id = "sess-test"
        self.context_compressor = SimpleNamespace(last_prompt_tokens=0)
        self.session_input_tokens = 0
        self.session_output_tokens = 0
        self.session_cache_read_tokens = 0
        self.session_cache_write_tokens = 0
        self.session_reasoning_tokens = 0
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_total_tokens = 0
        self.session_estimated_cost_usd = 0
        self.session_cost_status = "unknown"
        self.session_cost_source = "test"
        self._tool_guardrail_halt_decision = None
        self._interrupt_message = None
        self._response_was_previewed = True
        self._skill_nudge_interval = 0
        self._iters_since_skill = 0
        self.valid_tool_names = []
        self.persisted_messages: list[dict[str, Any]] | None = None
        self._persist_user_message_idx: int | None = None
        self._persist_user_message_override: Any = None
        self._persist_user_message_timestamp: float | None = None
        self.sync_calls: list[dict] = []

    def _handle_max_iterations(self, messages, api_call_count):
        raise AssertionError("not expected")

    def _emit_status(self, *_args, **_kwargs):
        pass

    def _safe_print(self, *_args, **_kwargs):
        pass

    def _save_trajectory(self, *_args, **_kwargs):
        pass

    def _cleanup_task_resources(self, *_args, **_kwargs):
        pass

    def _drop_trailing_empty_response_scaffolding(self, messages):
        pass

    def _persist_session(self, messages, conversation_history):
        self.persisted_messages = [dict(message) for message in messages]

    def _apply_persist_user_message_override(self, messages):
        idx = self._persist_user_message_idx
        override = self._persist_user_message_override
        if idx is not None and override is not None:
            messages[idx]["content"] = override

    def _file_mutation_verifier_enabled(self):
        return False

    def _turn_completion_explainer_enabled(self):
        return False

    def _drain_pending_steer(self):
        return None

    def clear_interrupt(self):
        pass

    def _sync_external_memory_for_turn(self, **kwargs):
        self.sync_calls.append(kwargs)


_BASE_KWARGS = dict(
    effective_task_id="task-test",
    turn_id="turn-test",
    conversation_history=[],
    _should_review_memory=False,
    _turn_exit_reason="text_response(final)",
)


def _messages_for(user_message: str) -> list[dict]:
    return [{"role": "user", "content": user_message}]


def test_memory_sync_receives_unchanged_response_when_no_transform(monkeypatch):
    """No transform fires -> memory sync gets the original response."""
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = FakeAgent()

    finalize_turn(
        agent,
        final_response="Hi there!",
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=_messages_for("Hello"),
        user_message="Hello",
        original_user_message="Hello",
        **_BASE_KWARGS,
    )

    assert len(agent.sync_calls) == 1
    assert agent.sync_calls[0]["final_response"] == "Hi there!"


def test_memory_sync_receives_transformed_response_with_string_return(monkeypatch):
    """A plain string return is used for both display and memory (prior behavior, unchanged)."""
    raw = "Here is the answer."
    appended = "\n\n[Source: example.com]"

    def _fake_invoke_hook(hook_name, **_kwargs):
        if hook_name == "transform_llm_output":
            return [raw + appended]
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _fake_invoke_hook)
    agent = FakeAgent()

    result = finalize_turn(
        agent,
        final_response=raw,
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=_messages_for("What is the answer?"),
        user_message="What is the answer?",
        original_user_message="What is the answer?",
        **_BASE_KWARGS,
    )

    assert result["final_response"] == raw + appended
    assert result["response_transformed"] is True
    assert len(agent.sync_calls) == 1
    assert agent.sync_calls[0]["final_response"] == raw + appended


def test_memory_sync_receives_dict_memory_override(monkeypatch):
    """A dict return with an explicit "memory" key sends that value to memory sync,
    while the user still sees the full "display" value."""
    raw = "Here is the answer."
    displayed = raw + "\n\n---\nSponsored: Acme Widgets"

    def _fake_invoke_hook(hook_name, **_kwargs):
        if hook_name == "transform_llm_output":
            return [{"display": displayed, "memory": raw}]
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _fake_invoke_hook)
    agent = FakeAgent()

    result = finalize_turn(
        agent,
        final_response=raw,
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=_messages_for("What is the answer?"),
        user_message="What is the answer?",
        original_user_message="What is the answer?",
        **_BASE_KWARGS,
    )

    assert result["final_response"] == displayed
    assert result["response_transformed"] is True
    assert len(agent.sync_calls) == 1
    assert agent.sync_calls[0]["final_response"] == raw
    assert "Sponsored" not in agent.sync_calls[0]["final_response"]


def test_memory_sync_dict_without_memory_key_falls_back_to_display(monkeypatch):
    """A dict return with only "display" set behaves like a plain string return:
    memory receives the same (post-transform) value as display."""
    displayed = "Rewritten response."

    def _fake_invoke_hook(hook_name, **_kwargs):
        if hook_name == "transform_llm_output":
            return [{"display": displayed}]
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _fake_invoke_hook)
    agent = FakeAgent()

    result = finalize_turn(
        agent,
        final_response="Original.",
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=_messages_for("Hello"),
        user_message="Hello",
        original_user_message="Hello",
        **_BASE_KWARGS,
    )

    assert result["final_response"] == displayed
    assert len(agent.sync_calls) == 1
    assert agent.sync_calls[0]["final_response"] == displayed


def test_memory_sync_dict_with_non_string_memory_falls_back_to_display(monkeypatch):
    """A "memory" value that isn't a string is ignored, same fallback as a missing key."""
    displayed = "Rewritten response."

    def _fake_invoke_hook(hook_name, **_kwargs):
        if hook_name == "transform_llm_output":
            return [{"display": displayed, "memory": None}]
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _fake_invoke_hook)
    agent = FakeAgent()

    finalize_turn(
        agent,
        final_response="Original.",
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=_messages_for("Hello"),
        user_message="Hello",
        original_user_message="Hello",
        **_BASE_KWARGS,
    )

    assert agent.sync_calls[0]["final_response"] == displayed


def test_memory_sync_passes_interrupted_flag(monkeypatch):
    """Interrupted turns still pass interrupted=True through (sync itself skips internally)."""
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = FakeAgent()

    finalize_turn(
        agent,
        final_response="Partial...",
        api_call_count=1,
        interrupted=True,
        failed=False,
        messages=_messages_for("Hello"),
        user_message="Hello",
        original_user_message="Hello",
        **_BASE_KWARGS,
    )

    assert len(agent.sync_calls) == 1
    assert agent.sync_calls[0]["interrupted"] is True
    assert agent.sync_calls[0]["final_response"] == "Partial..."


class _RecordingProvider(MemoryProvider):
    """Minimal provider that just records what it was asked to sync."""

    def __init__(self):
        self.synced: list[tuple[str, str]] = []

    @property
    def name(self) -> str:
        return "recording"

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id: str = "", **kwargs) -> None:
        pass

    def sync_turn(self, user_content, assistant_content, *, session_id: str = "", messages=None) -> None:
        self.synced.append((user_content, assistant_content))

    def get_tool_schemas(self):
        return []


class _RealMemorySyncAgent(FakeAgent):
    """Same as FakeAgent, but `_sync_external_memory_for_turn` mirrors the real
    implementation in `agent/run_agent.py` closely enough to exercise a real
    `MemoryManager` + provider end-to-end, instead of just recording the kwargs
    it was called with."""

    def __init__(self, memory_manager: MemoryManager):
        super().__init__()
        self._memory_manager = memory_manager

    def _sync_external_memory_for_turn(self, *, original_user_message, final_response,
                                        interrupted, messages=None):
        if interrupted:
            return
        if not (self._memory_manager and final_response and original_user_message):
            return
        self._memory_manager.sync_all(
            original_user_message, final_response, session_id=self.session_id or "",
        )


def test_memory_sync_recording_provider_receives_memory_value_end_to_end(monkeypatch):
    """End-to-end: a real MemoryManager + provider must receive the "memory"
    value, not the "display" value, when a plugin splits the two."""
    raw = "Here is the answer."
    displayed = raw + "\n\n---\nSponsored: Acme Widgets"

    def _fake_invoke_hook(hook_name, **_kwargs):
        if hook_name == "transform_llm_output":
            return [{"display": displayed, "memory": raw}]
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _fake_invoke_hook)

    mgr = MemoryManager()
    provider = _RecordingProvider()
    mgr.add_provider(provider)
    agent = _RealMemorySyncAgent(mgr)

    finalize_turn(
        agent,
        final_response=raw,
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=_messages_for("What is the answer?"),
        user_message="What is the answer?",
        original_user_message="What is the answer?",
        **_BASE_KWARGS,
    )

    assert mgr.flush_pending(timeout=10) is True
    assert len(provider.synced) == 1
    _user_content, assistant_content = provider.synced[0]
    assert assistant_content == raw
    assert "Sponsored" not in assistant_content
