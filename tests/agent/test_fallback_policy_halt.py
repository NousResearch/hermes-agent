"""``fallback_policy.halt``: refuse fallback activation when the operator demands it.

A subscription-backed primary that exhausts its usage window can otherwise start spending paid
fallback API budget invisibly (weeks, on long-running unattended work). With ``halt: true`` the
chain is never walked: the refusal is emitted as a visible diagnostic and the primary failure
surfaces through the normal terminal path. Contracts pinned here:

  1. halt → ``try_activate_fallback`` returns False, emits the refusal, touches no chain state.
  2. default (no policy) → activation proceeds exactly as before.
  3. ``_fallback_halt_enabled`` reads the effective config and fails OPEN on read errors.
"""

from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def _make_agent(fallback_model=None):
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fallback_model,
        )
        agent.client = MagicMock()
        return agent


def _mock_client():
    mock = MagicMock()
    mock.base_url = "https://example.invalid/v1"
    mock.api_key = "fb-key"
    return mock


class TestHaltPolicy:
    def test_halt_blocks_activation_and_emits_refusal(self):
        from agent import chat_completion_helpers as cch
        agent = _make_agent(fallback_model=[{"provider": "openai", "model": "gpt-4o"}])
        with patch.object(cch, "_fallback_halt_enabled", return_value=True), \
             patch.object(agent, "_emit_diagnostic_status") as emit:
            activated = agent._try_activate_fallback()
        assert activated is False
        assert agent._fallback_index == 0          # chain untouched
        assert agent._fallback_activated is False
        assert "fallback_policy.halt" in emit.call_args[0][0]

    def test_default_policy_activates_chain(self):
        from agent import chat_completion_helpers as cch
        agent = _make_agent(fallback_model=[{"provider": "openai", "model": "gpt-4o"}])
        with patch.object(cch, "_fallback_halt_enabled", return_value=False), \
             patch("agent.auxiliary_client.resolve_provider_client",
                   return_value=(_mock_client(), "gpt-4o")):
            activated = agent._try_activate_fallback()
        assert activated is True
        assert agent._fallback_index == 1


class TestHaltReader:
    def test_reads_effective_config(self, tmp_path, monkeypatch):
        import hermes_constants
        (tmp_path / "config.yaml").write_text("fallback_policy:\n  halt: true\n")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: str(tmp_path))
        from agent.chat_completion_helpers import _fallback_halt_enabled
        assert _fallback_halt_enabled(None) is True

    def test_fails_open_when_unreadable(self, tmp_path, monkeypatch):
        import hermes_constants
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "does-not-exist"))
        monkeypatch.setattr(hermes_constants, "get_hermes_home",
                            lambda: str(tmp_path / "does-not-exist"))
        from agent.chat_completion_helpers import _fallback_halt_enabled
        assert _fallback_halt_enabled(None) is False
