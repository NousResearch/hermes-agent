"""Regression tests for #84207.

A turn interrupted while a long-running tool is executing ends with
``reason=interrupted_by_user`` and ``response_len=0``: the killed tool's
result (exit 130, ``[Command interrupted]``) is only an implementation
detail, no assistant text was produced, and gateways deliver an empty
bubble — the user gets zero feedback and the session looks dead.

Two coupled behaviors are covered here:

1. ``finalize_turn`` synthesizes a short visible closing message when an
   interrupted turn produced no content at all (``final_response`` empty),
   so every delivery surface shows *something* instead of nothing.  The
   message distinguishes a deliberate stop from a dropped connection via
   the agent's structured ``_interrupt_stop_kind``.
2. ``AIAgent.interrupt(...)`` records that structured provenance and
   propagates it to child agents (with a legacy-ABI fallback for
   third-party agents written against ``interrupt(message=None)``).
"""

from agent.turn_finalizer import finalize_turn


class _StubBudget:
    used = 3
    max_total = 90
    remaining = 87


class _StubCompressor:
    last_prompt_tokens = 0


class _StubAgent:
    """Minimal agent surface that ``finalize_turn`` reads from."""

    def __init__(self, *, stop_kind=None):
        self.max_iterations = 90
        self.iteration_budget = _StubBudget()
        self.context_compressor = _StubCompressor()
        self.model = "stub/model"
        self.provider = "stub"
        self.base_url = "http://stub"
        self.session_id = "sess-84207"
        self.quiet_mode = True
        self.platform = "webui"
        self._interrupt_requested = True
        self._interrupt_message = None
        self._interrupt_stop_kind = stop_kind
        self._tool_guardrail_halt_decision = None
        self._response_was_previewed = False
        self._skill_nudge_interval = 0
        self._iters_since_skill = 0
        for attr in (
            "session_input_tokens",
            "session_output_tokens",
            "session_cache_read_tokens",
            "session_cache_write_tokens",
            "session_reasoning_tokens",
            "session_prompt_tokens",
            "session_completion_tokens",
            "session_total_tokens",
            "session_estimated_cost_usd",
        ):
            setattr(self, attr, 0)
        self.session_cost_status = "ok"
        self.session_cost_source = "stub"

    def _save_trajectory(self, *a, **k):
        pass

    def _cleanup_task_resources(self, *a, **k):
        pass

    def _drop_trailing_empty_response_scaffolding(self, messages):
        pass

    def _persist_session(self, messages, conversation_history):
        pass

    def _emit_status(self, *a, **k):
        pass

    def _safe_print(self, *a, **k):
        pass

    def _file_mutation_verifier_enabled(self):
        return False

    def _turn_completion_explainer_enabled(self):
        return False

    def _drain_pending_steer(self):
        return None

    def clear_interrupt(self):
        pass

    def _sync_external_memory_for_turn(self, **k):
        pass


def _killed_tool_transcript():
    """The exact #84207 shape: a turn that was pure tool calls — the tool
    was killed mid-execution and no assistant text ever streamed."""
    return [
        {"role": "user", "content": "run the long export"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "function": {"name": "terminal", "arguments": "{}"}}
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "c1",
            "content": "\n[Command interrupted]\nExit Code: 130",
        },
    ]


def _finalize(agent, messages, *, final_response=None):
    return finalize_turn(
        agent,
        final_response=final_response,
        api_call_count=2,
        interrupted=True,
        failed=False,
        messages=messages,
        conversation_history=None,
        effective_task_id="task-1",
        turn_id="turn-1",
        user_message="run the long export",
        original_user_message="run the long export",
        _should_review_memory=False,
        _turn_exit_reason="interrupted_by_user",
    )


def test_interrupted_empty_turn_yields_visible_closing_message():
    """Core #84207: response_len=0 on an interrupted turn must not reach
    the gateway — a closing message is synthesized instead."""
    agent = _StubAgent()
    result = _finalize(agent, _killed_tool_transcript(), final_response=None)

    assert result["interrupted"] is True
    assert isinstance(result["final_response"], str)
    assert result["final_response"].strip(), (
        "interrupted turn reached the gateway with an empty final_response "
        "(response_len=0 — the silent-death symptom of #84207)"
    )
    assert "interrupt" in result["final_response"].lower()


def test_interrupted_user_stop_wording():
    agent = _StubAgent(stop_kind="user_stop")
    result = _finalize(agent, _killed_tool_transcript(), final_response=None)
    assert "stopped" in result["final_response"].lower()
    assert "disconnect" not in result["final_response"].lower()


def test_interrupted_client_disconnect_wording():
    agent = _StubAgent(stop_kind="client_disconnect")
    result = _finalize(agent, _killed_tool_transcript(), final_response=None)
    assert "disconnect" in result["final_response"].lower()


def test_interrupted_recovery_content_is_never_clobbered():
    """The loop's partial-stream recovery wins over the fallback — the
    synthesized message only fires when there is genuinely nothing else."""
    agent = _StubAgent(stop_kind="user_stop")
    result = _finalize(
        agent, _killed_tool_transcript(), final_response="partial streamed answer"
    )
    assert result["final_response"] == "partial streamed answer"


# ---------------------------------------------------------------------------
# interrupt() stop_kind plumbing
# ---------------------------------------------------------------------------


def _bare_agent():
    """An AIAgent-shaped object with only the attributes interrupt() touches;
    ``object.__new__`` skips __init__ like the repo's own test doubles do."""
    from run_agent import AIAgent

    agent = object.__new__(AIAgent)
    import threading

    agent._interrupt_requested = False
    agent._interrupt_message = None
    agent._interrupt_stop_kind = None
    agent._hard_interrupt_requested = threading.Event()
    agent._pending_redirect_lock = None
    agent._pending_redirect = None
    agent._execution_thread_id = None
    agent._interrupt_thread_signal_pending = False
    agent._tool_worker_threads = set()
    agent._tool_worker_threads_lock = threading.Lock()
    agent._active_children = []
    agent._active_children_lock = threading.Lock()
    agent.verbose_logging = False
    agent.quiet_mode = True
    agent.api_mode = "chat_completions"
    agent._active_request_abort = None
    agent._codex_session = None
    return agent


def test_interrupt_records_stop_kind():
    agent = _bare_agent()
    agent.interrupt(stop_kind="user_stop")
    assert agent._interrupt_requested is True
    assert agent._interrupt_stop_kind == "user_stop"


def test_interrupt_stop_kind_defaults_to_none():
    agent = _bare_agent()
    agent.interrupt("next message")
    assert agent._interrupt_stop_kind is None


def test_interrupt_propagates_stop_kind_to_children():
    parent = _bare_agent()
    child = _bare_agent()
    parent._active_children.append(child)
    parent.interrupt("halt", stop_kind="client_disconnect")
    assert child._interrupt_stop_kind == "client_disconnect"


def test_interrupt_propagates_to_legacy_abi_child():
    """A third-party child written against interrupt(message=None) must not
    break when the parent forwards stop_kind."""
    parent = _bare_agent()
    calls = []

    class _LegacyChild:
        def interrupt(self, message=None):
            calls.append(message)

    parent._active_children.append(_LegacyChild())
    parent.interrupt("halt", stop_kind="user_stop")
    assert calls == ["halt"]


def test_request_hard_interrupt_marks_stop_kind():
    from agent.interrupt_compat import request_hard_interrupt

    agent = _bare_agent()
    accepted = request_hard_interrupt(
        agent, "SSE client disconnected", stop_kind="client_disconnect"
    )
    assert accepted is True
    assert agent._interrupt_stop_kind == "client_disconnect"


def test_request_hard_interrupt_stop_kind_legacy_hard_interrupt():
    """An agent whose hard_interrupt predates the stop_kind parameter must
    still receive the provenance stamp via the direct-attribute fallback."""
    from agent.interrupt_compat import request_hard_interrupt

    calls = []

    class _LegacyHardInterruptAgent:
        _interrupt_stop_kind = None

        def hard_interrupt(self, message=None):
            calls.append(message)

    agent = _LegacyHardInterruptAgent()
    accepted = request_hard_interrupt(
        agent, "SSE client disconnected", stop_kind="client_disconnect"
    )
    assert accepted is True
    assert calls == ["SSE client disconnected"]
    assert agent._interrupt_stop_kind == "client_disconnect"