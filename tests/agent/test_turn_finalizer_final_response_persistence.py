from types import SimpleNamespace
from typing import Any

from agent.turn_finalizer import finalize_turn


class FakeAgent:
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
        self.compression_calls: list[dict[str, Any]] = []
        self._persist_user_message_idx: int | None = None
        self._persist_user_message_override: Any = None
        self._persist_user_message_timestamp: float | None = None

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
        # Capture the durable write before finalization restores API-local
        # guidance to the returned/live transcript.
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

    def _sync_external_memory_for_turn(self, **_kwargs):
        pass


def test_finalizer_restores_clean_api_local_text_before_return(monkeypatch):
    """One-shot CLI notes do not replay through same-process history."""
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = FakeAgent()
    messages = [
        {"role": "user", "content": "[MODEL SWITCH NOTE]\n\nclean prompt"},
        {"role": "assistant", "content": "Done."},
    ]
    agent._persist_user_message_idx = 0
    agent._persist_user_message_override = "clean prompt"
    agent._persist_user_message_timestamp = None

    result = finalize_turn(
        agent,
        final_response="Done.",
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="[MODEL SWITCH NOTE]\n\nclean prompt",
        original_user_message="clean prompt",
        _should_review_memory=False,
        _turn_exit_reason="text_response(finish_reason=stop)",
    )

    assert agent.persisted_messages is not None
    assert agent.persisted_messages[0]["content"] == "clean prompt"
    assert result["messages"][0]["content"] == "clean prompt"


def test_finalizer_restores_clean_api_local_multimodal_before_return(monkeypatch):
    """A queued note does not remain in the next-turn native image payload."""
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = FakeAgent()
    clean_content = [
        {"type": "text", "text": "Describe the image"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]
    api_content = [
        {"type": "text", "text": "[MODEL SWITCH NOTE]\n\nDescribe the image"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]
    messages = [
        {"role": "user", "content": api_content},
        {"role": "assistant", "content": "Done."},
    ]
    agent._persist_user_message_idx = 0
    agent._persist_user_message_override = clean_content
    agent._persist_user_message_timestamp = None

    result = finalize_turn(
        agent,
        final_response="Done.",
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message=api_content,
        original_user_message=clean_content,
        _should_review_memory=False,
        _turn_exit_reason="text_response(finish_reason=stop)",
    )

    assert agent.persisted_messages is not None
    assert agent.persisted_messages[0]["content"] == clean_content
    assert result["messages"][0]["content"] == clean_content


def test_final_response_closes_tool_tail_before_persistence(monkeypatch):
    """A recovered/previewed final response must be durable in session history.

    Regression for turns where the caller receives a non-empty final_response,
    but the message transcript still ends at a tool result. If persisted that
    way, the next turn reloads a stale/malformed history and can appear to loop
    because the assistant's visible final answer is missing from durable state.
    """
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = FakeAgent()
    messages = [
        {"role": "user", "content": "do it"},
        {
            "role": "assistant",
            "content": "I'll check.",
            "tool_calls": [
                {"id": "call-1", "function": {"name": "terminal", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "name": "terminal", "content": "ok"},
    ]

    result = finalize_turn(
        agent,
        final_response="Done.",
        api_call_count=2,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=[],
        effective_task_id="task",
        turn_id="turn",
        user_message="do it",
        original_user_message="do it",
        _should_review_memory=False,
        _turn_exit_reason="fallback_prior_turn_content",
    )

    assert result["messages"][-1] == {"role": "assistant", "content": "Done."}
    assert agent.persisted_messages is not None
    assert agent.persisted_messages[-1] == {"role": "assistant", "content": "Done."}


def test_final_response_fills_pure_tool_call_tail(monkeypatch):
    """A tail assistant row that is a *pure tool-call turn* carries no answer.

    The role check alone ("tail is assistant ⇒ nothing to do") leaves the
    #43849/#44100 invariant unmet when the tail is ``assistant(tool_calls)``
    with no text of its own: the caller and the gateway already delivered
    ``final_response``, but it never reaches the transcript. The next turn then
    replays the user backlog and the model re-answers it — the exact symptom
    that block exists to prevent.
    """
    agent = FakeAgent()
    messages = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "t1", "type": "function",
                 "function": {"name": "f", "arguments": "{}"}}
            ],
        },
    ]

    result = finalize_turn(
        agent,
        final_response="Here is your answer.",
        api_call_count=3,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=[],
        effective_task_id="t",
        turn_id="tid",
        user_message="q",
        original_user_message="q",
        _should_review_memory=False,
        _turn_exit_reason="text_response(final)",
    )

    persisted = agent.persisted_messages
    assert any(
        m.get("role") == "assistant" and m.get("content") == result["final_response"]
        for m in persisted
    ), "delivered final_response never reached the durable transcript"
    # Filled in place — no assistant→assistant pair, tool_calls preserved.
    assert persisted[-1]["content"] == "Here is your answer."
    assert persisted[-1]["tool_calls"]
    assert sum(1 for m in persisted if m.get("role") == "assistant") == 1


def test_final_response_does_not_clobber_tool_call_tail_with_text(monkeypatch):
    """A tail tool-call turn that already carries model text must be left alone."""
    agent = FakeAgent()
    messages = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": "partial text",
            "tool_calls": [
                {"id": "t1", "type": "function",
                 "function": {"name": "f", "arguments": "{}"}}
            ],
        },
    ]

    finalize_turn(
        agent,
        final_response="Here is your answer.",
        api_call_count=3,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=[],
        effective_task_id="t",
        turn_id="tid",
        user_message="q",
        original_user_message="q",
        _should_review_memory=False,
        _turn_exit_reason="text_response(final)",
    )

    assert agent.persisted_messages[-1]["content"] == "partial text"


def test_fill_pops_db_persisted_marker_for_durable_rewrite(monkeypatch):
    """The incremental tool-call persist stamps ``_db_persisted`` on the row.

    If finalize_turn fills the tail's content but leaves the marker, the next
    ``_flush_messages_to_session_db`` skips the row and the durable SQLite
    store keeps ``content=""`` — so ``/resume`` reloads the empty content and
    the bug resurfaces cross-session. The fix pops the marker so the filled
    content is re-written.
    """
    agent = FakeAgent()
    messages = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "t1", "type": "function",
                 "function": {"name": "f", "arguments": "{}"}}
            ],
            "_db_persisted": True,  # stamped by conversation_loop.py:4990
        },
    ]

    finalize_turn(
        agent,
        final_response="Here is your answer.",
        api_call_count=3,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=[],
        effective_task_id="t",
        turn_id="tid",
        user_message="q",
        original_user_message="q",
        _should_review_memory=False,
        _turn_exit_reason="text_response(final)",
    )

    persisted = agent.persisted_messages
    assert persisted is not None
    assert persisted[-1]["content"] == "Here is your answer."
    assert persisted[-1]["tool_calls"]
    assert "_db_persisted" not in persisted[-1], (
        "marker must be popped so the next flush re-writes the filled content"
    )


class _BoundaryCompressor:
    threshold_tokens = 80
    context_length = 100
    max_tokens = 0

    def __init__(self, prompt_tokens=81):
        self.last_prompt_tokens = prompt_tokens

    def should_compress(self, tokens):
        return tokens >= self.threshold_tokens


def _enable_turn_end_compression(agent, *, fail=False):
    agent.compression_enabled = True
    agent.compression_defer_until_turn_end = True
    agent.compression_emergency_threshold = 0.92
    agent.context_compressor = _BoundaryCompressor()
    agent.tools = []
    agent._last_compaction_in_place = True
    agent.compression_calls = []

    def _compress(messages, system_message, **kwargs):
        # The boundary hook must run only after the completed answer has closed
        # the transcript. Otherwise compaction can lose the visible final turn.
        assert messages[-1].get("role") == "assistant"
        assert messages[-1].get("content") == "Done."
        agent.compression_calls.append(
            {
                "system_message": system_message,
                "approx_tokens": kwargs.get("approx_tokens"),
                "task_id": kwargs.get("task_id"),
            }
        )
        if fail:
            raise RuntimeError("summary backend unavailable")
        agent.context_compressor.last_prompt_tokens = -1
        compacted = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "[CONTEXT SUMMARY] earlier turns"},
            {"role": "assistant", "content": "Done."},
        ]
        return compacted, "system"

    agent._compress_context = _compress


def _finalize_boundary_turn(agent):
    return finalize_turn(
        agent,
        final_response="Done.",
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=[
            {"role": "user", "content": "finish the task"},
            {"role": "assistant", "content": "Done."},
        ],
        conversation_history=[],
        effective_task_id="task-boundary",
        turn_id="turn-boundary",
        user_message="finish the task",
        original_user_message="finish the task",
        system_message="base system prompt",
        _should_review_memory=False,
        _turn_exit_reason="text_response(finish_reason=stop)",
    )


def test_completed_turn_compacts_at_soft_threshold_before_persistence(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = FakeAgent()
    _enable_turn_end_compression(agent)

    result = _finalize_boundary_turn(agent)

    assert agent.compression_calls == [
        {
            "system_message": "base system prompt",
            "approx_tokens": 81,
            "task_id": "task-boundary",
        }
    ]
    assert agent.persisted_messages == result["messages"]
    assert result["messages"][1]["content"].startswith("[CONTEXT SUMMARY]")
    # compress_context parks real usage at -1 until the next provider call;
    # callers must receive a non-negative context reading.
    assert result["last_prompt_tokens"] == 0
    assert result["completed"] is True


def test_failed_turn_end_compression_preserves_completed_answer(monkeypatch):
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    agent = FakeAgent()
    _enable_turn_end_compression(agent, fail=True)

    result = _finalize_boundary_turn(agent)

    assert result["final_response"] == "Done."
    assert agent.persisted_messages is not None
    assert agent.persisted_messages[-1] == {"role": "assistant", "content": "Done."}
    assert result["cleanup_errors"] == [
        "turn_end_compression: summary backend unavailable"
    ]
