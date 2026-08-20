"""Regression test for #75772: verifier footer defeats silence suppression.

When a cron job agent responds with ``[SILENT]`` but a write_file call was
denied (e.g. outside HERMES_WRITE_SAFE_ROOT), the file-mutation verifier
footer must NOT be appended — doing so pushes [SILENT] off the last line,
causing is_autonomous_silence_response to return False and delivering the
message despite the agent's explicit silence intent.
"""

from types import SimpleNamespace
from typing import Any

from agent.turn_finalizer import finalize_turn


class _StubAgent:
    """Minimal agent stub exposing the attributes finalize_turn reads."""

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
        self.valid_tool_names: list[str] = []
        self._turn_failed_file_mutations: dict[str, Any] = {}

    def _handle_max_iterations(self, _messages, _api_call_count):
        raise AssertionError("not expected")

    def _emit_status(self, *_a, **_kw):
        pass

    def _safe_print(self, *_a, **_kw):
        pass

    def _save_trajectory(self, *_a, **_kw):
        pass

    def _cleanup_task_resources(self, *_a, **_kw):
        pass

    def _drop_trailing_empty_response_scaffolding(self, _messages):
        pass

    def _persist_session(self, _messages, _conversation_history):
        pass

    def _file_mutation_verifier_enabled(self):
        return True

    def _turn_completion_explainer_enabled(self):
        return False

    def _format_file_mutation_failure_footer(self, failed):
        lines = ["⚠️ File-mutation verifier: file(s) NOT modified:"]
        for path, reason in failed.items():
            lines.append(f"  • {path} — {reason}")
        return "\n".join(lines)

    def _drain_pending_steer(self):
        return None

    def clear_interrupt(self):
        pass

    def _sync_external_memory_for_turn(self, **_kwargs):
        pass


def _run(agent, final_response, *, interrupted=False):
    """Call finalize_turn with the minimal kwargs it needs."""
    messages = [
        {"role": "user", "content": "check"},
        {"role": "assistant", "content": final_response},
    ]
    return finalize_turn(
        agent=agent,
        final_response=final_response,
        api_call_count=1,
        interrupted=interrupted,
        failed=False,
        messages=messages,
        conversation_history=list(messages),
        effective_task_id=None,
        turn_id="turn-test",
        user_message="check",
        original_user_message="check",
        _should_review_memory=False,
        _turn_exit_reason=None,
    )


def test_verifier_footer_not_appended_after_silent_marker(monkeypatch):
    """#75772: [SILENT] response with denied writes must stay silent."""
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])

    agent = _StubAgent()
    agent._turn_failed_file_mutations = {
        "/tmp/check_release.py": "[write_file] Write denied: outside HERMES_WRITE_SAFE_ROOT",
    }

    # Model responded with prose + [SILENT] on its own line.
    response = (
        "The current state is v1.2.3 and the latest stable release is still v1.2.3.\n\n"
        "[SILENT]"
    )

    result = _run(agent, response)
    final = result["final_response"]

    # The footer must NOT appear — silence must be preserved.
    assert "File-mutation verifier" not in final, (
        f"Footer was appended to a [SILENT] response, defeating delivery suppression: {final!r}"
    )
    # The [SILENT] marker must still be the last meaningful content.
    assert "[SILENT]" in final


def test_verifier_footer_still_appended_to_normal_response(monkeypatch):
    """Non-silent responses with denied writes still get the footer."""
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])

    agent = _StubAgent()
    agent._turn_failed_file_mutations = {
        "/tmp/check.py": "[write_file] Write denied",
    }

    result = _run(agent, "I've updated the config file for you.")
    final = result["final_response"]

    # Footer SHOULD appear for normal (non-silent) responses.
    assert "File-mutation verifier" in final


def test_silent_with_own_line_explanation_no_footer(monkeypatch):
    """[SILENT] with trailing explanation must not get footer."""
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])

    agent = _StubAgent()
    agent._turn_failed_file_mutations = {
        "/opt/data/helper.py": "[write_file] Write denied",
    }

    response = "2 deals filtered\n\n[SILENT]"
    result = _run(agent, response)
    final = result["final_response"]

    assert "File-mutation verifier" not in final
