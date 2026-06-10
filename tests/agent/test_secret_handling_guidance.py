"""Tests for SECRET_HANDLING_GUIDANCE injection (#43083).

When global secret redaction is enabled (the default), tool-call arguments
are redacted to ``***`` in conversation history. The model must therefore
reference secrets indirectly (env expansion, source .env) instead of
inlining literal values. SECRET_HANDLING_GUIDANCE teaches that contract;
it goes into the STABLE tier of the system prompt so it never changes
mid-session (cache-safe), and is gated on:

  * the agent having tools (the failure mode is credentials round-tripping
    through tool-call history), and
  * redaction being enabled (process-lifetime flag).
"""

from types import SimpleNamespace
from unittest.mock import patch

from agent.prompt_builder import SECRET_HANDLING_GUIDANCE
from agent.system_prompt import build_system_prompt_parts


def _make_agent(**overrides):
    base = dict(
        load_soul_identity=False,
        skip_context_files=True,
        valid_tool_names=["terminal"],
        _task_completion_guidance=False,
        _tool_use_enforcement=False,
        _environment_probe=False,
        _kanban_worker_guidance="",
        _memory_store=None,
        _memory_manager=None,
        model="",
        provider="",
        platform="",
        pass_session_id=False,
        session_id="",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _build_stable(agent):
    with (
        patch("run_agent.load_soul_md", return_value=""),
        patch("run_agent.build_nous_subscription_prompt", return_value=""),
        patch("run_agent.build_environment_hints", return_value=""),
        patch("run_agent.build_context_files_prompt", return_value=""),
        patch("run_agent.get_toolset_for_tool", return_value=None),
    ):
        return build_system_prompt_parts(agent)["stable"]


class TestSecretHandlingGuidance:
    def test_injected_when_redaction_on_and_tools_present(self):
        with patch("agent.redact._REDACT_ENABLED", True):
            stable = _build_stable(_make_agent())
        assert SECRET_HANDLING_GUIDANCE in stable

    def test_skipped_when_redaction_disabled(self):
        with patch("agent.redact._REDACT_ENABLED", False):
            stable = _build_stable(_make_agent())
        assert SECRET_HANDLING_GUIDANCE not in stable

    def test_skipped_when_no_tools(self):
        # Without tools there is no tool-call history for secrets to
        # round-trip through — keep the prompt lean.
        with patch("agent.redact._REDACT_ENABLED", True):
            stable = _build_stable(_make_agent(valid_tool_names=[]))
        assert SECRET_HANDLING_GUIDANCE not in stable

    def test_guidance_teaches_indirection_not_placeholder_copying(self):
        # Behavior contract: the guidance must tell the model (a) not to
        # copy *** placeholders and (b) to use indirection. Asserts the
        # relationship, not exact wording.
        text = SECRET_HANDLING_GUIDANCE
        assert "***" in text
        assert "$PGPASSWORD" in text or "source .env" in text


class TestRedactionEnabledFlag:
    def test_is_redaction_enabled_reflects_flag(self):
        from agent import redact

        with patch("agent.redact._REDACT_ENABLED", True):
            assert redact.is_redaction_enabled() is True
        with patch("agent.redact._REDACT_ENABLED", False):
            assert redact.is_redaction_enabled() is False
