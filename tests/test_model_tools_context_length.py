"""Regression tests for model_tools active context resolution."""

from __future__ import annotations

from unittest.mock import patch

import model_tools


def test_tool_search_context_gate_uses_provider_aware_model_config(monkeypatch):
    """Tool Search must not treat Codex gpt-5.5 like direct OpenAI gpt-5.5.

    The generic catalog may advertise a 1.05M window for the slug, but the
    active ``openai-codex`` route uses the Codex OAuth cap/configured value.
    """
    import hermes_cli.config as cfg_mod

    monkeypatch.setattr(cfg_mod, "load_config", lambda: {
        "model": {
            "default": "gpt-5.5",
            "provider": "openai-codex",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "context_length": 272_000,
        },
        "custom_providers": [{"name": "custom"}],
    })

    with patch("agent.model_metadata.get_model_context_length", return_value=272_000) as ctx:
        assert model_tools._resolve_active_context_length() == 272_000

    ctx.assert_called_once_with(
        "gpt-5.5",
        base_url="https://chatgpt.com/backend-api/codex",
        provider="openai-codex",
        config_context_length=272_000,
        custom_providers=[{"name": "custom"}],
    )
