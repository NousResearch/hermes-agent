"""Tests that model.context_length in config.yaml is scoped to model.default.

Regression coverage for the leak where a global context_length override
written for the config's default model applied to *every* model a session
could run — including a per-session model_override to a completely
different provider/model (e.g. mobile/desktop picking a model other than
the local default). See _scope_config_context_length_to_default_model in
agent/agent_init.py, and the analogous /model switch guard in
agent_runtime_helpers.py (agent._config_context_length = None on swap).
"""

from unittest.mock import patch

from agent.agent_init import _scope_config_context_length_to_default_model


def _fake_get_model_context_length(*args, config_context_length=None, **kwargs):
    """Stand-in for agent.model_metadata.get_model_context_length: honors an
    explicit config override (step-0 resolution in the real function), else
    falls back to a fixed "auto-detected" window distinct from the override
    under test (131072) so the two are easy to tell apart in assertions."""
    if config_context_length is not None:
        return config_context_length
    return 372_000


def _build_agent(model_cfg, model, base_url="http://localhost:8090/v1"):
    """Build a real AIAgent with the given model config, mirroring the
    pattern in tests/run_agent/test_invalid_context_length_warning.py."""
    cfg = {"model": model_cfg}

    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch(
            "agent.model_metadata.get_model_context_length",
            side_effect=_fake_get_model_context_length,
        ),
        # context_compressor.py imports get_model_context_length at module
        # load time, so it needs its own patch target.
        patch(
            "agent.context_compressor.get_model_context_length",
            side_effect=_fake_get_model_context_length,
        ),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        from run_agent import AIAgent

        agent = AIAgent(
            model=model,
            api_key="test-key-1234567890",
            base_url=base_url,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    return agent


# ── init_agent integration tests ─────────────────────────────────────────


def test_override_applies_when_session_model_matches_config_default():
    """agent.model == config default → the explicit override still applies."""
    model_cfg = {
        "default": "Qwen3.6-27B-NVFP4-MTP-GGUF.gguf",
        "provider": "custom:trainlocal-qwen36-27b",
        "base_url": "http://localhost:8090/v1",
        "context_length": 131072,
    }
    agent = _build_agent(model_cfg, model="Qwen3.6-27B-NVFP4-MTP-GGUF.gguf")
    assert agent._config_context_length == 131072
    assert agent.context_compressor.context_length == 131072


def test_override_does_not_leak_onto_session_model_override():
    """agent.model != config default (e.g. per-session model_override to a
    different provider/model) → the override must not apply; auto-detection
    resolves the real window for the actual model instead."""
    model_cfg = {
        "default": "Qwen3.6-27B-NVFP4-MTP-GGUF.gguf",
        "provider": "custom:trainlocal-qwen36-27b",
        "base_url": "http://localhost:8090/v1",
        "context_length": 131072,
    }
    agent = _build_agent(
        model_cfg,
        model="some-other-model",
        base_url="http://localhost:8080/v1",
    )
    assert agent._config_context_length is None
    # Auto-detected window (mocked get_model_context_length) is used, not
    # the stale 131072 written for the local default model.
    assert agent.context_compressor.context_length == 372_000
    assert agent.context_compressor.context_length != 131072


def test_plain_string_model_config_unchanged():
    """A plain-string model config (no dict) never had an override to
    begin with — behavior is unchanged by the scoping."""
    agent = _build_agent("Qwen3.6-27B-NVFP4-MTP-GGUF.gguf", model="Qwen3.6-27B-NVFP4-MTP-GGUF.gguf")
    assert agent._config_context_length is None


# ── unit tests for the pure scoping helper ───────────────────────────────


def test_scope_helper_returns_none_unchanged():
    assert _scope_config_context_length_to_default_model(
        {"default": "model-a"}, "model-a", None
    ) is None


def test_scope_helper_non_dict_config_returns_value_unchanged():
    assert _scope_config_context_length_to_default_model(
        "model-a", "model-b", 131072
    ) == 131072


def test_scope_helper_matching_model_keeps_override():
    assert _scope_config_context_length_to_default_model(
        {"default": "model-a"}, "model-a", 131072
    ) == 131072


def test_scope_helper_mismatched_model_drops_override():
    assert _scope_config_context_length_to_default_model(
        {"default": "model-a"}, "model-b", 131072
    ) is None


def test_scope_helper_falls_back_to_name_key():
    assert _scope_config_context_length_to_default_model(
        {"name": "model-a"}, "model-a", 131072
    ) == 131072
    assert _scope_config_context_length_to_default_model(
        {"name": "model-a"}, "model-b", 131072
    ) is None


def test_scope_helper_strips_whitespace():
    assert _scope_config_context_length_to_default_model(
        {"default": " model-a "}, "model-a", 131072
    ) == 131072


def test_scope_helper_no_default_name_keeps_override():
    """If the config block names no default/name at all, there is nothing
    to compare against — leave existing behavior alone."""
    assert _scope_config_context_length_to_default_model(
        {"provider": "custom"}, "model-a", 131072
    ) == 131072


def test_scope_helper_no_agent_model_keeps_override():
    assert _scope_config_context_length_to_default_model(
        {"default": "model-a"}, None, 131072
    ) == 131072
