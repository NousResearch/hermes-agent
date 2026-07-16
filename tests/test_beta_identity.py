from unittest.mock import patch

from agent.beta_identity import (
    BETA_AGENT_IDENTITY,
    BETA_MODE,
    HERMES_MODE,
    ResolvedIdentity,
    identity_for_mode,
    normalize_agent_mode,
    resolve_agent_identity,
    resolve_agent_mode,
)
from agent.prompt_builder import DEFAULT_AGENT_IDENTITY


def test_hermes_is_default_mode():
    assert resolve_agent_mode({}) == HERMES_MODE
    assert identity_for_mode("Hermes identity", {}) == "Hermes identity"


def test_beta_mode_selects_beta_identity():
    config = {"agent": {"mode": "beta"}}
    assert resolve_agent_mode(config) == BETA_MODE
    assert identity_for_mode("Hermes identity", config) == BETA_AGENT_IDENTITY


def test_mode_is_case_and_whitespace_insensitive():
    assert resolve_agent_mode({"agent": {"mode": "  BETA  "}}) == BETA_MODE


def test_unknown_mode_falls_back_to_hermes():
    assert normalize_agent_mode("executor") == HERMES_MODE
    assert resolve_agent_mode({"agent": {"mode": "executor"}}) == HERMES_MODE


def test_unknown_mode_warns_and_falls_back(caplog):
    assert resolve_agent_mode({"agent": {"mode": "executor"}}) == HERMES_MODE
    assert "Unsupported agent.mode" in caplog.text


def test_malformed_agent_section_falls_back_to_hermes():
    assert resolve_agent_mode({"agent": "beta"}) == HERMES_MODE


def test_soul_compatibility_policy():
    soul = "Custom SOUL instructions"
    assert ResolvedIdentity(HERMES_MODE, "Hermes").compose(soul) == soul
    assert resolve_agent_identity("Hermes", {"agent": {"mode": "beta"}}).compose(soul) == (
        f"{BETA_AGENT_IDENTITY}\n\n{soul}"
    )


def test_runtime_initialization_resolves_beta_from_config(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("agent:\n  mode: beta\n", encoding="utf-8")

    with patch("run_agent.OpenAI"):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    assert agent._resolved_identity.mode == BETA_MODE
    assert agent._resolved_identity.prompt == BETA_AGENT_IDENTITY


def test_runtime_initialization_keeps_hermes_default():
    with patch("run_agent.OpenAI"), patch("hermes_cli.config.load_config", return_value={}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    assert agent._resolved_identity == ResolvedIdentity(HERMES_MODE, DEFAULT_AGENT_IDENTITY)
