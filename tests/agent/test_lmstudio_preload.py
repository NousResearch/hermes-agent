"""Tests for _ensure_lmstudio_runtime_loaded JIT bypass fix (#25989).

When no explicit context_length override is configured, the function should
skip the manual /api/v1/models/load POST and let LM Studio's JIT loader
handle the model loading.  The manual lane bypasses JIT, creating a no-TTL
entry that blocks VRAM and prevents model switches from evicting it.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent(**overrides):
    """Return a minimal mock AIAgent with lmstudio defaults."""
    agent = MagicMock()
    agent.provider = "lmstudio"
    agent.model = "qwen/qwen3.6-27b"
    agent.base_url = "http://localhost:1234/v1"
    agent.api_key = ""
    agent.api_mode = "chat_completions"
    agent._config_context_length = None
    for k, v in overrides.items():
        setattr(agent, k, v)
    return agent


def _patch_models_module(mock_load):
    """Return a patch for hermes_cli.models.ensure_lmstudio_model_loaded."""
    return patch(
        "hermes_cli.models.ensure_lmstudio_model_loaded",
        mock_load,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEnsureLmstudioRuntimeLoaded:
    """Tests for _ensure_lmstudio_runtime_loaded."""

    def test_skips_preload_when_no_context_override(self):
        """No context_length configured → function returns early, no POST."""
        from run_agent import AIAgent

        agent = _make_agent()
        agent._config_context_length = None

        mock_load = MagicMock()
        with _patch_models_module(mock_load):
            AIAgent._ensure_lmstudio_runtime_loaded(agent, config_context_length=None)

        mock_load.assert_not_called()

    def test_skips_preload_when_stored_context_is_none(self):
        """config_context_length arg is None and self._config_context_length
        is also None → no preload."""
        from run_agent import AIAgent

        agent = _make_agent()
        agent._config_context_length = None

        mock_load = MagicMock()
        with _patch_models_module(mock_load):
            AIAgent._ensure_lmstudio_runtime_loaded(agent)

        mock_load.assert_not_called()

    def test_passes_explicit_context_to_load(self):
        """Explicit context_length → passes value through to load function."""
        from run_agent import AIAgent

        agent = _make_agent()
        agent._config_context_length = 100_000

        mock_load = MagicMock(return_value=100_000)
        with _patch_models_module(mock_load):
            AIAgent._ensure_lmstudio_runtime_loaded(agent, config_context_length=100_000)

        mock_load.assert_called_once_with(
            "qwen/qwen3.6-27b", "http://localhost:1234/v1", "", 100_000,
        )

    def test_uses_stored_context_when_arg_is_none(self):
        """config_context_length arg is None but self._config_context_length
        is set → uses stored value."""
        from run_agent import AIAgent

        agent = _make_agent()
        agent._config_context_length = 80_000

        mock_load = MagicMock(return_value=80_000)
        with _patch_models_module(mock_load):
            AIAgent._ensure_lmstudio_runtime_loaded(agent)

        mock_load.assert_called_once_with(
            "qwen/qwen3.6-27b", "http://localhost:1234/v1", "", 80_000,
        )

    def test_skips_for_non_lmstudio_provider(self):
        """Non-lmstudio provider → early return, no load attempted."""
        from run_agent import AIAgent

        agent = _make_agent(provider="openai")

        mock_load = MagicMock()
        with _patch_models_module(mock_load):
            AIAgent._ensure_lmstudio_runtime_loaded(agent, config_context_length=100_000)

        mock_load.assert_not_called()

    def test_updates_compressor_on_successful_load(self):
        """After successful load, context_compressor.update_model is called."""
        from run_agent import AIAgent

        agent = _make_agent()
        agent._config_context_length = 100_000
        cc = MagicMock()
        agent.context_compressor = cc

        mock_load = MagicMock(return_value=100_000)
        with _patch_models_module(mock_load):
            AIAgent._ensure_lmstudio_runtime_loaded(agent, config_context_length=100_000)

        cc.update_model.assert_called_once()
        call_kwargs = cc.update_model.call_args
        assert call_kwargs.kwargs["context_length"] == 100_000
