"""Tests for Peter's optional understanding-gate final-response footer.

The gate is deliberately config-driven and disabled by default. When enabled,
it prevents a model's final answer from reaching the user without a parseable
status/proof surface. It is a small runtime check, not a claim of subjective
understanding.
"""

from __future__ import annotations

from types import SimpleNamespace

from run_agent import AIAgent


def _bare_agent() -> AIAgent:
    return object.__new__(AIAgent)


def _force_understanding_gate(monkeypatch) -> None:
    from hermes_cli import config as hermes_config

    monkeypatch.setattr(
        hermes_config,
        "load_config",
        lambda: {"agent": {"understanding_gate": {"mode": "enforced"}}},
    )


class TestUnderstandingGateConfig:
    def test_disabled_by_default(self, monkeypatch):
        from hermes_cli import config as hermes_config

        monkeypatch.setattr(hermes_config, "load_config", lambda: {})
        agent = _bare_agent()

        assert agent._understanding_gate_mode() == "off"
        assert agent._understanding_gate_enabled() is False

    def test_enforced_from_agent_config_dict(self, monkeypatch):
        _force_understanding_gate(monkeypatch)
        agent = _bare_agent()

        assert agent._understanding_gate_mode() == "enforced"
        assert agent._understanding_gate_enabled() is True

    def test_boolean_config_maps_to_enforced(self, monkeypatch):
        from hermes_cli import config as hermes_config

        monkeypatch.setattr(
            hermes_config,
            "load_config",
            lambda: {"agent": {"understanding_gate": True}},
        )
        agent = _bare_agent()

        assert agent._understanding_gate_mode() == "enforced"


class TestUnderstandingGateFooter:
    def test_footer_not_added_when_status_proof_exists(self):
        response = "Done.\n\nstatus=verified proof=tests_run+artifact_reread"

        assert AIAgent._format_understanding_gate_footer(response) == ""

    def test_footer_marks_missing_status_proof_partial(self):
        footer = AIAgent._format_understanding_gate_footer("Done.")

        assert "Understanding gate" in footer
        assert "status=partial proof=understanding_gate_missing_status_proof" in footer

    def test_footer_handles_status_without_proof_as_missing(self):
        footer = AIAgent._format_understanding_gate_footer("Done.\nstatus=verified")

        assert "status=partial proof=understanding_gate_missing_status_proof" in footer

    def test_apply_footer_updates_latest_assistant_message(self, monkeypatch):
        _force_understanding_gate(monkeypatch)
        agent = _bare_agent()
        messages = [
            {"role": "user", "content": "say done"},
            {"role": "assistant", "content": "Done."},
        ]

        final = agent._apply_understanding_gate_footer(
            "Done.",
            messages,
            turn_exit_reason="text_response(finish_reason=stop)",
        )

        assert final.endswith("status=partial proof=understanding_gate_missing_status_proof")
        assert messages[-1]["content"] == final

    def test_apply_footer_is_not_duplicated(self, monkeypatch):
        _force_understanding_gate(monkeypatch)
        agent = _bare_agent()
        response = "Done.\n\nstatus=verified proof=tests_run+artifact_reread"
        messages = [{"role": "assistant", "content": response}]

        final = agent._apply_understanding_gate_footer(
            response,
            messages,
            turn_exit_reason="text_response(finish_reason=stop)",
        )

        assert final == response
        assert final.count("status=verified proof=") == 1

    def test_finalizer_persists_gated_latest_assistant_message(self, monkeypatch):
        from agent.turn_finalizer import finalize_turn

        _force_understanding_gate(monkeypatch)
        monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *a, **k: [])
        agent = _bare_agent()
        dynamic_attrs = {
            "max_iterations": 10,
            "iteration_budget": SimpleNamespace(remaining=9, used=1, max_total=10),
            "model": "test-model",
            "provider": "test-provider",
            "base_url": "",
            "session_id": "session-understanding-gate",
            "platform": None,
            "_tool_guardrail_halt_decision": None,
            "_skill_nudge_interval": 0,
            "_iters_since_skill": 0,
            "valid_tool_names": set(),
            "context_compressor": SimpleNamespace(last_prompt_tokens=0),
            "_response_was_previewed": False,
            "session_cost_status": "unknown",
            "session_cost_source": "unknown",
            "_turn_failed_file_mutations": {},
            "_save_trajectory": lambda *a, **k: None,
            "_cleanup_task_resources": lambda *a, **k: None,
            "_drop_trailing_empty_response_scaffolding": lambda *a, **k: None,
            "_file_mutation_verifier_enabled": lambda: False,
            "_turn_completion_explainer_enabled": lambda: False,
            "_drain_pending_steer": lambda: None,
            "clear_interrupt": lambda: None,
            "_sync_external_memory_for_turn": lambda *a, **k: None,
        }
        for attr, value in dynamic_attrs.items():
            setattr(agent, attr, value)
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
            setattr(agent, attr, 0)
        persisted = {}

        def _persist_session(messages, conversation_history=None):
            persisted["messages"] = [dict(m) for m in messages]
            persisted["conversation_history"] = conversation_history

        setattr(agent, "_persist_session", _persist_session)
        messages = [
            {"role": "user", "content": "say done"},
            {"role": "assistant", "content": "Done."},
        ]

        result = finalize_turn(
            agent,
            final_response="Done.",
            api_call_count=1,
            interrupted=False,
            failed=False,
            messages=messages,
            conversation_history=None,
            effective_task_id="task-1",
            turn_id="turn-1",
            user_message="say done",
            original_user_message="say done",
            _should_review_memory=False,
            _turn_exit_reason="text_response(finish_reason=stop)",
        )

        assert result["final_response"].endswith(
            "status=partial proof=understanding_gate_missing_status_proof"
        )
        assert persisted["messages"][-1]["content"] == result["final_response"]
        assert result["messages"][-1]["content"] == result["final_response"]
