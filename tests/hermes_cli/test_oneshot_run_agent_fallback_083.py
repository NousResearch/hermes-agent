"""HSK-083: exercise the real ``_run_agent()`` caller wiring for startup fallback.

HSK-081's tests pinned ``_resolve_oneshot_runtime`` in isolation but never proved that
``_run_agent()`` actually consumes its result the way the docstring claims:

    runtime, choice.model = _resolve_oneshot_runtime(choice)

This module drives ``_run_agent()`` itself (not the helper) with a real ``_ModelChoice``,
mocking only the heavy/external dependencies (session store, MCP discovery, and ``AIAgent``
construction/execution). It proves that when the primary provider raises ``AuthError``, the
configured fallback runtime/model reach ``AIAgent`` construction.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from hermes_cli.auth import AuthError
from hermes_cli.oneshot import _ModelChoice, _run_agent


def _fake_load_config():
    return {
        "model": {"provider": "openai-codex", "default": "gpt-5.6-sol"},
        "fallback_providers": [{"provider": "ollama", "model": "llama-4-maverick"}],
    }


def test_run_agent_wires_configured_fallback_runtime_into_aiagent(monkeypatch):
    """Primary AuthError -> configured fallback runtime/model reach AIAgent construction."""

    def fake_resolve_model_and_provider(cfg, model, provider):
        return _ModelChoice(model="gpt-5.6-sol", provider="openai-codex")

    def fake_resolve_runtime_provider(*, requested, target_model=None,
                                       explicit_base_url=None, explicit_api_key=None):
        if requested == "openai-codex":
            raise AuthError("missing OPENAI_CODEX_API_KEY")
        assert requested == "ollama"
        return {
            "provider": "ollama", "api_key": None, "base_url": "http://localhost:11434/v1",
            "requested_provider": "ollama", "api_mode": "openai_chat",
            "credential_pool": None,
        }

    captured_kwargs: dict = {}

    class _FakeAIAgent:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)
            self.suppress_status_output = None
            self.stream_delta_callback = None
            self.tool_gen_callback = None

        def run_conversation(self, prompt, conversation_history=None):
            return {"final_response": "ok", "provider": captured_kwargs.get("provider")}

    with (
        patch("hermes_cli.oneshot._resolve_model_and_provider", side_effect=fake_resolve_model_and_provider),
        patch("hermes_cli.oneshot._create_session_db_for_oneshot", return_value=None),
        patch("hermes_cli.oneshot._load_resume_target", return_value=(None, [], None)),
        patch("hermes_cli.tools_config._get_platform_tools", return_value=set()),
        patch("hermes_cli.mcp_startup.ensure_mcp_discovery_before_agent_build"),
        patch("hermes_cli.oneshot._build_preloaded_skills_prompt", return_value=None),
        patch("hermes_cli.oneshot._close_agent"),
        patch("hermes_cli.config.load_config", side_effect=_fake_load_config),
        patch("hermes_cli.runtime_provider.resolve_runtime_provider", side_effect=fake_resolve_runtime_provider),
        patch("run_agent.AIAgent", _FakeAIAgent),
    ):
        final_response, result = _run_agent("hello", use_config_toolsets=False)

    assert final_response == "ok"
    # The fallback runtime (not the primary openai-codex runtime) reached AIAgent construction.
    assert captured_kwargs.get("provider") == "ollama"
    assert captured_kwargs.get("base_url") == "http://localhost:11434/v1"
    # The fallback model reached the effective-model path passed to AIAgent.
    assert captured_kwargs.get("model") == "llama-4-maverick"
