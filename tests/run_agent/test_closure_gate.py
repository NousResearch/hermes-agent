"""Tests for the configurable pre-final closure gate.

The closure gate is intentionally runtime-side, not prompt-only: when enabled
for a profile such as Peter, the final response must carry an explicit
``status=... proof=...`` closure line or get an automatic advisory before it is
returned to the user.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


def _bare_agent() -> AIAgent:
    return object.__new__(AIAgent)


def _mock_response(content: str = "Done."):
    msg = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    return SimpleNamespace(
        choices=[choice],
        model="test/model",
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=3, total_tokens=8),
    )


def _make_agent_with_response(content: str) -> AIAgent:
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            session_db=MagicMock(),
            session_id="closure-gate-test",
            platform="cli",
        )
    agent.client = MagicMock()
    agent.client.chat.completions.create.return_value = _mock_response(content)
    return agent


class TestClosureGateEnabled:
    def test_default_is_disabled(self, monkeypatch):
        monkeypatch.delenv("HERMES_CLOSURE_GATE", raising=False)
        import hermes_cli.config as _cfg_mod

        monkeypatch.setattr(_cfg_mod, "load_config", lambda: {})
        assert _bare_agent()._closure_gate_enabled() is False

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
    def test_env_enables(self, monkeypatch, value):
        monkeypatch.setenv("HERMES_CLOSURE_GATE", value)
        assert _bare_agent()._closure_gate_enabled() is True

    @pytest.mark.parametrize("value", ["0", "false", "FALSE", "no", "off"])
    def test_env_disables_over_config(self, monkeypatch, value):
        monkeypatch.setenv("HERMES_CLOSURE_GATE", value)
        import hermes_cli.config as _cfg_mod

        monkeypatch.setattr(
            _cfg_mod,
            "load_config",
            lambda: {"display": {"closure_gate": {"enabled": True}}},
        )
        assert _bare_agent()._closure_gate_enabled() is False

    def test_config_object_enables_when_no_env(self, monkeypatch):
        monkeypatch.delenv("HERMES_CLOSURE_GATE", raising=False)
        import hermes_cli.config as _cfg_mod

        monkeypatch.setattr(
            _cfg_mod,
            "load_config",
            lambda: {"display": {"closure_gate": {"enabled": True}}},
        )
        assert _bare_agent()._closure_gate_enabled() is True

    def test_config_bool_enables_when_no_env(self, monkeypatch):
        monkeypatch.delenv("HERMES_CLOSURE_GATE", raising=False)
        import hermes_cli.config as _cfg_mod

        monkeypatch.setattr(
            _cfg_mod,
            "load_config",
            lambda: {"display": {"closure_gate": True}},
        )
        assert _bare_agent()._closure_gate_enabled() is True


class TestClosureGateFooter:
    def test_footer_added_when_missing_explicit_closure_line(self):
        out = _bare_agent()._apply_closure_gate_footer("Done.")

        assert out.startswith("Done.")
        assert "Closure gate:" in out
        assert "status=unverified" in out
        assert "proof=missing explicit status/proof closure" in out

    @pytest.mark.parametrize(
        "text",
        [
            "Done.\n\nstatus=verified proof=pytest tests/run_agent/test_x.py",
            "Done.\nstatus=done proof=git status reread after patch",
            "Blocked.\nSTATUS=blocked PROOF=missing credentials",
        ],
    )
    def test_footer_not_added_when_closure_line_present(self, text):
        assert _bare_agent()._apply_closure_gate_footer(text) == text

    def test_empty_response_not_augmented(self):
        assert _bare_agent()._apply_closure_gate_footer("") == ""


class TestClosureGateRuntimeHook:
    def test_run_conversation_appends_footer_before_return_when_enabled(self, monkeypatch):
        monkeypatch.setenv("HERMES_CLOSURE_GATE", "1")
        agent = _make_agent_with_response("Done.")

        result = agent.run_conversation("finish the task")

        assert result["final_response"].startswith("Done.")
        assert "Closure gate:" in result["final_response"]
        assert "status=unverified" in result["final_response"]

    def test_run_conversation_syncs_footer_into_persisted_messages(self, monkeypatch):
        monkeypatch.setenv("HERMES_CLOSURE_GATE", "1")
        agent = _make_agent_with_response("Done.")

        result = agent.run_conversation("finish the task")

        assert result["messages"][-1]["role"] == "assistant"
        assert result["messages"][-1]["content"] == result["final_response"]
        append_calls = getattr(agent._session_db.append_message, "call_args_list")
        assistant_db_rows = [
            call.kwargs
            for call in append_calls
            if call.kwargs.get("role") == "assistant"
        ]
        assert assistant_db_rows[-1]["content"] == result["final_response"]

    def test_run_conversation_does_not_duplicate_explicit_closure(self, monkeypatch):
        monkeypatch.setenv("HERMES_CLOSURE_GATE", "1")
        text = "Done.\n\nstatus=verified proof=targeted pytest passed"
        agent = _make_agent_with_response(text)

        result = agent.run_conversation("finish the task")

        assert result["final_response"] == text


class TestClosureGateDeliveryHelpers:
    def test_sync_final_response_to_messages_replaces_latest_assistant_text(self):
        messages = [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "Done."},
        ]
        final = "Done.\n\nClosure gate: status=unverified proof=missing"

        assert _bare_agent()._sync_final_response_to_messages(
            messages,
            final,
            previous_response="Done.",
        ) is True

        assert messages[-1]["content"] == final

    def test_stream_suffix_helper_emits_only_footer_after_streamed_answer(self):
        agent = _bare_agent()
        deltas = []
        agent.stream_delta_callback = deltas.append
        agent._stream_callback = None
        agent._current_streamed_assistant_text = "Done."
        agent._stream_think_scrubber = None
        agent._stream_context_scrubber = None
        agent._stream_needs_break = False
        final = "Done.\n\nClosure gate: status=unverified proof=missing"

        assert agent._emit_postprocessed_stream_suffix("Done.", final) is True

        assert deltas == ["\n\nClosure gate: status=unverified proof=missing"]
        assert agent._current_streamed_assistant_text == final
