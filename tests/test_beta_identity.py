from agent.beta_identity import (
    BETA_AGENT_IDENTITY,
    BETA_MODE,
    HERMES_MODE,
    identity_for_mode,
    normalize_agent_mode,
    resolve_agent_mode,
)


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


def test_malformed_agent_section_falls_back_to_hermes():
    assert resolve_agent_mode({"agent": "beta"}) == HERMES_MODE
