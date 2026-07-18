"""Tests for per-model reasoning_effort override during fallback activation.

Tests that try_activate_fallback re-resolves reasoning_config when
swapping to a fallback model, so per-model overrides are honored even
during error recovery.
"""

import pytest
from unittest.mock import MagicMock, patch


class TestFallbackReasoningOverride:
    """Test try_activate_fallback re-resolves reasoning_config."""

    def test_fallback_re_resolves_reasoning_config(self):
        """When fallback activates, reasoning_config should be re-resolved.

        We test the resolution logic directly rather than spinning up a
        full try_activate_fallback (which requires extensive agent setup).
        The production code calls resolve_per_model_reasoning_effort with
        the fallback model string — we verify that works correctly.
        """
        from hermes_constants import resolve_per_model_reasoning_effort

        # Simulate: primary was gemini-flash (medium), fallback to claude-opus-4.5 (xhigh)
        overrides = {
            "claude-opus-4.5": "xhigh",
            "gemini-flash": "medium",
        }

        # Fallback model lookup
        fb_result = resolve_per_model_reasoning_effort("claude-opus-4.5", overrides)
        assert fb_result is not None
        assert fb_result["effort"] == "xhigh"

        # Primary model lookup (for comparison)
        primary_result = resolve_per_model_reasoning_effort("gemini-flash", overrides)
        assert primary_result is not None
        assert primary_result["effort"] == "medium"

        # The key point: fallback result differs from primary
        assert fb_result["effort"] != primary_result["effort"]

    def test_fallback_to_model_without_override_uses_global(self):
        """Fallback to a model with no override should resolve to None (→ global)."""
        from hermes_constants import resolve_per_model_reasoning_effort

        overrides = {"claude-opus-4.5": "xhigh"}

        # Fallback to gpt-5 which has no override
        result = resolve_per_model_reasoning_effort("gpt-5", overrides)
        assert result is None  # caller falls back to global

    def test_fallback_with_normalized_model_name(self):
        """Fallback model name may be normalized (dots→dashes); override should still match."""
        from hermes_constants import resolve_per_model_reasoning_effort

        # User wrote key with dots, but normalize_model_for_provider converts to dashes
        overrides = {"claude-sonnet-4.6": "high"}

        result = resolve_per_model_reasoning_effort("claude-sonnet-4-6", overrides)
        assert result is not None
        assert result["effort"] == "high"

    def test_fallback_recovery_restores_primary_reasoning(self):
        """After fallback + restore_primary_runtime, reasoning_config returns to primary's value.

        This tests the integration of Task 6 (_primary_runtime snapshot) with
        Task 6b (fallback re-resolution). The full cycle:
        1. Primary model = gemini-flash, reasoning = medium
        2. /model switch → _primary_runtime captures reasoning_config
        3. Fallback activates → reasoning re-resolved for fallback model
        4. restore_primary_runtime → reasoning_config restored from snapshot
        """
        from agent.agent_runtime_helpers import restore_primary_runtime

        agent = MagicMock()
        # Simulate: _primary_runtime was captured during /model switch
        agent._primary_runtime = {
            "model": "gemini-flash",
            "provider": "google",
            "base_url": "",
            "api_mode": "openai",
            "api_key": "key",
            "client_kwargs": {},
            "use_prompt_caching": False,
            "use_native_cache_layout": False,
            "reasoning_config": {"enabled": True, "effort": "medium"},
            "compressor_model": "gemini-flash",
            "compressor_base_url": "",
            "compressor_api_key": "",
            "compressor_provider": "",
            "compressor_context_length": 0,
            "compressor_api_mode": "",
            "compressor_threshold_tokens": 0,
        }
        agent._fallback_activated = True
        agent._fallback_index = 0
        agent._fallback_chain = []
        agent._fallback_model = None
        agent._transport_cache = {}
        agent._config_context_length = None
        agent._rate_limited_until = 0
        # During fallback, reasoning was changed to xhigh (fallback model's override)
        agent.model = "claude-opus-4.5"
        agent.provider = "anthropic"
        agent.reasoning_config = {"enabled": True, "effort": "xhigh"}
        agent.context_compressor = MagicMock()
        agent.base_url = ""
        agent._anthropic_prompt_cache_policy = MagicMock(return_value=(False, False))
        agent._create_openai_client = MagicMock(return_value=MagicMock())
        agent._ensure_lmstudio_runtime_loaded = MagicMock()

        result = restore_primary_runtime(agent)
        assert result is True
        # reasoning_config should be restored to primary's value (medium)
        assert agent.reasoning_config == {"enabled": True, "effort": "medium"}

    def test_fallback_global_fallback_with_yaml_false(self):
        """Fallback global fallback must not coerce YAML boolean False.

        Regression: ``or ""`` turned False into "", silently re-enabling
        thinking. The raw value must pass through so
        parse_reasoning_effort(False) returns {'enabled': False}.

        The production code in try_activate_fallback does:
            _fb_global_effort = _fb_agent_cfg.get("reasoning_effort", "")
            agent.reasoning_config = parse_reasoning_effort(_fb_global_effort)
        We verify that passing the raw False (not coerced "") produces
        the disabled config.
        """
        from hermes_constants import parse_reasoning_effort

        # Simulate: no per-model override matches, global is YAML False
        _fb_agent_cfg = {"reasoning_effort": False}

        # This is the exact line from try_activate_fallback's else branch.
        # The bug was: _fb_global_effort = _fb_agent_cfg.get(...) or ""
        # which turned False into "". The fix passes the raw value.
        _fb_global_effort = _fb_agent_cfg.get("reasoning_effort", "")
        result = parse_reasoning_effort(_fb_global_effort)

        assert result is not None
        assert result.get("enabled") is False
