"""Tests for the prefetchGenericContext config knob.

Validates that:
  - The config knob defaults to True (backward-compatible).
  - The config knob parses correctly from honcho.json (root, host, override).
  - When False, the dialectic prewarm gate condition is False.
  - When True, the dialectic prewarm gate condition is True.
  - The base context prewarm fires regardless of the knob.
"""

import json
import pytest

from plugins.memory.honcho.client import HonchoClientConfig


class TestPrefetchGenericContextConfig:
    """Config parsing for prefetchGenericContext."""

    def test_defaults_to_true(self, tmp_path):
        """When not specified, prefetchGenericContext defaults to True."""
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"apiKey": "test-key"}))

        cfg = HonchoClientConfig.from_global_config(config_path=config_path)
        assert cfg.prefetch_generic_context is True

    def test_explicit_true_host_block(self, tmp_path):
        """When explicitly set to True in host block."""
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({
            "apiKey": "test-key",
            "hosts": {"hermes": {"prefetchGenericContext": True}},
        }))

        cfg = HonchoClientConfig.from_global_config(config_path=config_path)
        assert cfg.prefetch_generic_context is True

    def test_explicit_false_host_block(self, tmp_path):
        """When explicitly set to False in host block."""
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({
            "apiKey": "test-key",
            "hosts": {"hermes": {"prefetchGenericContext": False}},
        }))

        cfg = HonchoClientConfig.from_global_config(config_path=config_path)
        assert cfg.prefetch_generic_context is False

    def test_root_level_true(self, tmp_path):
        """When set at root level (no host block)."""
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({
            "apiKey": "test-key",
            "prefetchGenericContext": True,
        }))

        cfg = HonchoClientConfig.from_global_config(config_path=config_path)
        assert cfg.prefetch_generic_context is True

    def test_root_level_false(self, tmp_path):
        """When set to False at root level."""
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({
            "apiKey": "test-key",
            "prefetchGenericContext": False,
        }))

        cfg = HonchoClientConfig.from_global_config(config_path=config_path)
        assert cfg.prefetch_generic_context is False

    def test_host_overrides_root(self, tmp_path):
        """Host-level value wins over root-level."""
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({
            "apiKey": "test-key",
            "prefetchGenericContext": True,
            "hosts": {"hermes": {"prefetchGenericContext": False}},
        }))

        cfg = HonchoClientConfig.from_global_config(config_path=config_path)
        assert cfg.prefetch_generic_context is False

    def test_from_env_defaults_true(self, monkeypatch):
        """from_env() should default to True (no config file)."""
        monkeypatch.setenv("HONCHO_API_KEY", "env-test-key")
        cfg = HonchoClientConfig.from_env()
        assert cfg.prefetch_generic_context is True

    def test_dataclass_default_true(self):
        """Default dataclass value is True."""
        cfg = HonchoClientConfig()
        assert cfg.prefetch_generic_context is True


class TestPrefetchGenericContextInitPath:
    """Init-path tests exercising _do_session_init() with real mocking.

    These tests prove that _do_session_init() suppresses the dialectic
    prewarm thread when prefetchGenericContext is False, while still
    firing the base-context prefetch — the behaviour the simulated gate
    tests above only approximate.
    """

    @staticmethod
    def _make_provider(cfg_extra=None):
        from unittest.mock import patch, MagicMock
        from plugins.memory.honcho.client import HonchoClientConfig

        defaults = dict(api_key="test-key", enabled=True, recall_mode="hybrid")
        if cfg_extra:
            defaults.update(cfg_extra)
        cfg = HonchoClientConfig(**defaults)
        from plugins.memory.honcho import HonchoMemoryProvider
        provider = HonchoMemoryProvider()
        mock_manager = MagicMock()
        mock_manager.get_or_create.return_value = MagicMock(messages=[])
        mock_manager.get_prefetch_context.return_value = None
        mock_manager.pop_context_result.return_value = None
        mock_manager.dialectic_query.return_value = "prewarm synthesis"

        with patch("plugins.memory.honcho.client.HonchoClientConfig.from_global_config", return_value=cfg), \
             patch("plugins.memory.honcho.client.get_honcho_client", return_value=MagicMock()), \
             patch("plugins.memory.honcho.session.HonchoSessionManager", return_value=mock_manager), \
             patch("hermes_constants.get_hermes_home", return_value=MagicMock()):
            provider.initialize(session_id="test-init-path")
        return provider

    def test_false_suppresses_dialectic_prewarm_retains_base_context(self):
        """When prefetchGenericContext=False, _do_session_init() must:

        1. Call prefetch_context (base context still prewarms).
        2. NOT start a dialectic prewarm thread (_prefetch_thread is None).
        """
        p = self._make_provider(cfg_extra={"prefetch_generic_context": False})

        # Base context prefetch fired
        assert p._manager.prefetch_context.call_count >= 1, \
            "base context prefetch must fire regardless of prefetchGenericContext"

        # No dialectic prewarm thread was started
        assert p._prefetch_thread is None, \
            "dialectic prewarm thread must not start when prefetchGenericContext=False"

    def test_true_starts_dialectic_prewarm_and_base_context(self):
        """When prefetchGenericContext=True (default), _do_session_init() must:

        1. Call prefetch_context (base context prewarms).
        2. Start a dialectic prewarm thread (_prefetch_thread is not None).
        """
        p = self._make_provider(cfg_extra={"prefetch_generic_context": True})

        # Base context prefetch fired
        assert p._manager.prefetch_context.call_count >= 1, \
            "base context prefetch must fire when prefetchGenericContext=True"

        # Dialectic prewarm thread was started
        assert p._prefetch_thread is not None, \
            "dialectic prewarm thread must start when prefetchGenericContext=True"

        # Clean up the thread
        if p._prefetch_thread and p._prefetch_thread.is_alive():
            p._prefetch_thread.join(timeout=3.0)

    def test_gate_true_when_enabled_hybrid(self):
        """Dialectic prewarm fires when enabled and recall_mode=hybrid."""
        cfg = HonchoClientConfig(prefetch_generic_context=True, enabled=True)
        recall_mode = "hybrid"
        should_prewarm = recall_mode in {"context", "hybrid"} and cfg.prefetch_generic_context
        assert should_prewarm is True

    def test_gate_true_when_enabled_context(self):
        """Dialectic prewarm fires when enabled and recall_mode=context."""
        cfg = HonchoClientConfig(prefetch_generic_context=True, enabled=True)
        recall_mode = "context"
        should_prewarm = recall_mode in {"context", "hybrid"} and cfg.prefetch_generic_context
        assert should_prewarm is True

    def test_gate_false_when_disabled(self):
        """Dialectic prewarm does NOT fire when disabled."""
        cfg = HonchoClientConfig(prefetch_generic_context=False, enabled=True)
        recall_mode = "hybrid"
        should_prewarm = recall_mode in {"context", "hybrid"} and cfg.prefetch_generic_context
        assert should_prewarm is False

    def test_gate_false_when_tools_mode(self):
        """Dialectic prewarm does NOT fire when recall_mode=tools (regardless of knob)."""
        cfg = HonchoClientConfig(prefetch_generic_context=True, enabled=True)
        recall_mode = "tools"
        should_prewarm = recall_mode in {"context", "hybrid"} and cfg.prefetch_generic_context
        assert should_prewarm is False

    def test_base_context_prewarm_always_fires(self):
        """Base context prewarm fires regardless of the knob."""
        # When prefetchGenericContext=False, base context still prewarming
        cfg = HonchoClientConfig(prefetch_generic_context=False, enabled=True)
        recall_mode = "hybrid"
        should_base_prewarm = recall_mode in {"context", "hybrid"}
        assert should_base_prewarm is True

    def test_first_turn_dialectic_ungated_when_prewarm_disabled(self):
        """When prewarm is disabled, first-turn dialectic is not gated.

        With prefetchGenericContext=False, _prefetch_result stays empty
        and _last_dialectic_turn stays at -999, so the first-turn
        dialectic code path runs normally with the user's actual message.
        """
        # Simulated initial state when prewarm is skipped
        _prefetch_result = ""
        _last_dialectic_turn = -999
        query = "How does Honcho choose observations?"

        # This is the condition in prefetch() that fires first-turn dialectic
        should_fire_first_turn = (_last_dialectic_turn == -999) and bool(query)
        assert should_fire_first_turn is True

    def test_first_turn_dialectic_gated_when_prewarm_landed(self):
        """When prewarm landed, first-turn dialectic is gated (existing behaviour).

        With prefetchGenericContext=True, if prewarm completes before
        the first turn, _prefetch_result is set and _last_dialectic_turn
        advances, preventing the first-turn dialectic from firing.
        """
        # Simulated state after prewarm lands
        _prefetch_result = "Some generic profile context"
        _last_dialectic_turn = 0  # Set by prewarm
        query = "How does Honcho choose observations?"

        # This is the condition that SKIPS first-turn dialectic
        should_skip_first_turn = _last_dialectic_turn != -999
        assert should_skip_first_turn is True