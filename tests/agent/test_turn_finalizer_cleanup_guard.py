"""Regression test for #8049.

When the post-loop cleanup chain in ``finalize_turn`` raises — trajectory
save (file I/O), resource teardown (remote VM/browser), or session
persistence (SQLite) — the partial ``final_response`` the caller is waiting
for must still be returned.  Previously any of those raised straight out of
``run_conversation``, so a subprocess wrapper saw an empty stdout with no
traceback and lost the whole turn.
"""

import pytest

from agent.turn_finalizer import finalize_turn


class _StubBudget:
    used = 5
    max_total = 3
    remaining = 0


class _StubCompressor:
    last_prompt_tokens = 0


class _StubAgent:
    """Minimal agent surface that ``finalize_turn`` reads from."""

    def __init__(self, *, raise_in):
        self._raise_in = set(raise_in)
        self.max_iterations = 3
        self.iteration_budget = _StubBudget()
        self.context_compressor = _StubCompressor()
        self.model = "stub/model"
        self.provider = "stub"
        self.base_url = "http://stub"
        self.session_id = "sess-1"
        self.quiet_mode = True
        self.platform = "cli"
        self._interrupt_requested = False
        self._interrupt_message = None
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

    # --- fallible cleanup surfaces -------------------------------------
    def _save_trajectory(self, *a, **k):
        if "save_trajectory" in self._raise_in:
            raise RuntimeError("trajectory disk full")

    def _cleanup_task_resources(self, *a, **k):
        if "cleanup_task_resources" in self._raise_in:
            raise RuntimeError("docker teardown EOF")

    def _drop_trailing_empty_response_scaffolding(self, *a, **k):
        pass

    def _persist_session(self, *a, **k):
        if "persist_session" in self._raise_in:
            raise RuntimeError("sqlite database is locked")

    # --- harmless no-ops ------------------------------------------------
    def _emit_status(self, *a, **k):
        pass

    def _safe_print(self, *a, **k):
        pass

    def _handle_max_iterations(self, messages, n):
        return "PARTIAL SUMMARY FROM MODEL"

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


def _run(
    agent,
    *,
    final_response=None,
    api_call_count=3,
    turn_exit_reason="unknown",
):
    messages = [
        {"role": "user", "content": "do a thing"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "function": {"name": "read_file", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "file contents"},
    ]
    return finalize_turn(
        agent,
        final_response=final_response,
        api_call_count=api_call_count,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=None,
        effective_task_id="task-1",
        turn_id="turn-1",
        user_message="do a thing",
        original_user_message="do a thing",
        _should_review_memory=False,
        _turn_exit_reason=turn_exit_reason,
    )




@pytest.mark.parametrize(
    "step", ["save_trajectory", "cleanup_task_resources", "persist_session"]
)
def test_single_cleanup_step_raises_does_not_skip_others(step):
    agent = _StubAgent(raise_in=(step,))
    result = _run(agent)
    # Response survives.
    assert result["final_response"] == "PARTIAL SUMMARY FROM MODEL"
    # Exactly the failing step is recorded; the others ran without error.
    assert result["cleanup_errors"] == [
        next(
            e
            for e in result["cleanup_errors"]
            if e.startswith(step)
        )
    ]
    assert len(result["cleanup_errors"]) == 1


def test_clean_turn_has_no_cleanup_errors_key():
    agent = _StubAgent(raise_in=())
    result = _run(agent)
    assert result["final_response"] == "PARTIAL SUMMARY FROM MODEL"
    assert result["completed"] is False
    assert "cleanup_errors" not in result


def test_required_final_output_hook_failure_does_not_leak_original(monkeypatch):
    import hermes_cli.plugins as plugins_mod
    from hermes_cli.plugins import PluginManager

    manager = PluginManager()

    def _boom(**kwargs):
        raise RuntimeError("final safety failed")

    manager._hooks["finalize_llm_output"] = [_boom]
    monkeypatch.setattr(plugins_mod, "get_plugin_manager", lambda: manager)
    agent = _StubAgent(raise_in=())
    agent.platform = "telegram"

    result = _run(agent)

    assert result["final_response"] != "PARTIAL SUMMARY FROM MODEL"
    assert "無法安全顯示" in result["final_response"]
    assert result["response_transformed"] is True


def test_required_finalizer_runs_after_legacy_transform(monkeypatch):
    import hermes_cli.plugins as plugins_mod
    from hermes_cli.plugins import PluginManager

    manager = PluginManager()
    manager._hooks["transform_llm_output"] = [
        lambda **kwargs: kwargs["response_text"] + " legacy-added-detail"
    ]
    manager._hooks["finalize_llm_output"] = [lambda **kwargs: "SAFE FINAL"]
    monkeypatch.setattr(plugins_mod, "get_plugin_manager", lambda: manager)
    agent = _StubAgent(raise_in=())
    agent.platform = "telegram"

    result = _run(agent)

    assert result["final_response"] == "SAFE FINAL"


def test_terminal_validator_blocks_post_sanitizer_callback_mutation(monkeypatch):
    import hermes_cli.plugins as plugins_mod
    from hermes_cli.plugins import PluginManager

    manager = PluginManager()
    manager._hooks["finalize_llm_output"] = [
        lambda **kwargs: "SAFE FINAL",
        lambda **kwargs: kwargs["response_text"] + "\n" + "sk-" + "proj-unsafe-appended-value",
    ]

    def _terminal_validator(**kwargs):
        if "sk-proj-" in kwargs["response_text"]:
            raise RuntimeError("terminal validation rejected output")
        return True

    manager._hooks["validate_llm_output"] = [_terminal_validator]
    monkeypatch.setattr(plugins_mod, "get_plugin_manager", lambda: manager)
    agent = _StubAgent(raise_in=())
    agent.platform = "telegram"

    result = _run(agent)

    assert "sk-proj-" not in result["final_response"]
    assert "無法安全顯示" in result["final_response"]
    assert result["response_transformed"] is True
