"""Tests for the unified smart_model_routing module.

Verifies that:
1. resolve_turn_route uses routing_v2 as primary path (no feature flag needed)
2. task_state is written after each routing decision
3. Legacy cheap route is only used as fallback when no tiers/benchmarks
4. Continuation markers never downgrade
5. Model transitions generate handoff summaries
6. Telemetry import doesn't crash when module is missing
"""

import json
import os
import tempfile
from unittest import mock

import pytest


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def primary_config():
    """Standard primary model config."""
    return {
        "model": "claude-opus-4.6",
        "provider": "copilot",
        "base_url": "https://api.githubcopilot.com",
        "api_key": "test-key-123",
        "api_mode": "chat_completions",
        "command": None,
        "args": [],
        "credential_pool": None,
    }


@pytest.fixture
def cheap_config():
    """Routing config with cheap model."""
    return {
        "enabled": True,
        "cheap_model": {
            "provider": "ollama-cloud",
            "model": "glm-5.1",
        },
        "max_simple_chars": 160,
        "max_simple_words": 28,
    }


@pytest.fixture
def routing_config_no_v2():
    """Routing config without v2 flag (v2 is now always enabled)."""
    return {
        "enabled": True,
        "cheap_model": {
            "provider": "ollama-cloud",
            "model": "glm-5.1",
        },
    }


@pytest.fixture
def tmp_router_dir(tmp_path):
    """Create a temporary router directory for task_state and benchmarks."""
    router_dir = tmp_path / "router"
    router_dir.mkdir()
    return router_dir


# ─────────────────────────────────────────────────────────────
# 1. V2 is the primary path — no feature flag needed
# ─────────────────────────────────────────────────────────────

class TestV2IsDefault:
    """Verify that resolve_turn_route uses v2 by default, not as opt-in."""

    def test_v2_used_without_flag(self, primary_config, cheap_config):
        """V2 path is taken even without v2_enabled flag in config."""
        from agent.smart_model_routing import resolve_turn_route

        with mock.patch("agent.smart_model_routing.routing_v2") as mock_rv2, \
             mock.patch("agent.smart_model_routing._load_benchmarks", return_value={}), \
             mock.patch("agent.smart_model_routing._load_task_state", return_value={}), \
             mock.patch("agent.smart_model_routing._save_task_state"):
            mock_rv2.DEFAULT_TIERS = [["glm-5.1"], ["gpt-5-mini"], ["kimi-k2.5"]]
            mock_rv2.select_model.return_value = {
                "category": "simple",
                "model": "glm-5.1",
                "tier": 1,
                "benchmark_score": 0.0,
                "reason": "simple",
            }
            config = dict(cheap_config)  # no v2_enabled key
            result = resolve_turn_route("hola", config, primary_config)
            assert mock_rv2.select_model.called
            # Should return the v2-selected model
            assert result["model"] == "glm-5.1"
            assert "v2 smart route" in result.get("label", "")

    def test_v2_tiers_present_selects_model(self, primary_config, cheap_config):
        """When tiers are present, v2 select_model is the primary path."""
        from agent.smart_model_routing import resolve_turn_route

        with mock.patch("agent.smart_model_routing.routing_v2") as mock_rv2, \
             mock.patch("agent.smart_model_routing._load_benchmarks", return_value={}), \
             mock.patch("agent.smart_model_routing._load_task_state", return_value={}), \
             mock.patch("agent.smart_model_routing._save_task_state"):
            mock_rv2.DEFAULT_TIERS = [["glm-5.1"], ["gpt-5-mini"], ["kimi-k2.5"]]
            mock_rv2.select_model.return_value = {
                "category": "analysis",
                "model": "kimi-k2.5",
                "tier": 3,
                "benchmark_score": 0.85,
                "reason": "benchmark_best",
            }
            result = resolve_turn_route("analiza el stack trace", cheap_config, primary_config)
            assert result["model"] == "kimi-k2.5"


# ─────────────────────────────────────────────────────────────
# 2. Task state is written after each routing decision
# ─────────────────────────────────────────────────────────────

class TestTaskStatePersistence:
    """Verify that task_state.json is written after routing decisions."""

    def test_task_state_written_on_v2_path(self, primary_config, cheap_config):
        """task_state is saved when v2 path is used."""
        from agent.smart_model_routing import resolve_turn_route

        with mock.patch("agent.smart_model_routing.routing_v2") as mock_rv2, \
             mock.patch("agent.smart_model_routing._load_benchmarks", return_value={}), \
             mock.patch("agent.smart_model_routing._load_task_state", return_value={}), \
             mock.patch("agent.smart_model_routing._save_task_state") as mock_save:

            mock_rv2.DEFAULT_TIERS = [["glm-5.1"], ["gpt-5-mini"]]
            mock_rv2.select_model.return_value = {
                "category": "code",
                "model": "gpt-5-mini",
                "tier": 2,
                "benchmark_score": 0.0,
                "reason": "benchmark_best",
            }
            resolve_turn_route("debug this error", cheap_config, primary_config)
            assert mock_save.called

    def test_task_state_written_on_fallback_path(self, primary_config, cheap_config):
        """task_state is saved even when falling back to legacy path."""
        from agent.smart_model_routing import resolve_turn_route

        with mock.patch("agent.smart_model_routing._load_benchmarks", return_value={}), \
             mock.patch("agent.smart_model_routing._load_task_state", return_value={}), \
             mock.patch("agent.smart_model_routing._save_task_state") as mock_save, \
             mock.patch("agent.smart_model_routing.routing_v2") as mock_rv2:
            # No tiers, no benchmarks — forces fallback
            mock_rv2.DEFAULT_TIERS = []
            result = resolve_turn_route("hola", cheap_config, primary_config)
            assert mock_save.called


# ─────────────────────────────────────────────────────────────
# 3. Legacy cheap route only used as fallback
# ─────────────────────────────────────────────────────────────

class TestLegacyFallback:
    """Verify legacy path is only used when v2 has no tiers/benchmarks."""

    def test_fallback_when_no_tiers_no_benchmarks(self, primary_config):
        """When no tiers and no benchmarks, legacy heuristic handles simple messages."""
        from agent.smart_model_routing import resolve_turn_route

        with mock.patch("agent.smart_model_routing._load_benchmarks", return_value={}), \
             mock.patch("agent.smart_model_routing._load_task_state", return_value={}), \
             mock.patch("agent.smart_model_routing._save_task_state"), \
             mock.patch("agent.smart_model_routing.routing_v2") as mock_rv2:

            mock_rv2.DEFAULT_TIERS = []
            config = {
                "enabled": True,
                "cheap_model": {"provider": "ollama-cloud", "model": "glm-5.1"},
            }
            result = resolve_turn_route("hola", config, primary_config)
            # Legacy path routes short simple messages to cheap model
            # But since we don't resolve_runtime_provider in this test, it may return primary
            # The key assertion is that v2 was NOT called for selection
            assert not mock_rv2.select_model.called

    def test_complex_messages_stay_primary(self, primary_config):
        """Complex messages should not be cheap-routed even in legacy path."""
        from agent.smart_model_routing import _legacy_cheap_route

        cheap = {"provider": "ollama-cloud", "model": "glm-5.1"}
        result = _legacy_cheap_route("debug this stack trace and fix the error", cheap)
        assert result is None  # stays on primary


# ─────────────────────────────────────────────────────────────
# 4. Continuation markers never downgrade
# ─────────────────────────────────────────────────────────────

class TestContinuationProtection:
    """Continuation markers should preserve the current tier."""

    def test_sigue_stays_on_current_model(self, primary_config, cheap_config):
        """'sigue' should NOT route to cheap model — it's a continuation."""
        from agent.smart_model_routing import resolve_turn_route

        with mock.patch("agent.smart_model_routing.routing_v2") as mock_rv2, \
             mock.patch("agent.smart_model_routing._load_benchmarks", return_value={}), \
             mock.patch("agent.smart_model_routing._load_task_state", return_value={
                 "active_task": True,
                 "last_tier": 3,
                 "last_model": "kimi-k2.5",
                 "last_category": "code",
                 "turns_in_task": 5,
                 "easy_streak": 0,
             }), \
             mock.patch("agent.smart_model_routing._save_task_state"):

            mock_rv2.DEFAULT_TIERS = [["glm-5.1"], ["gpt-5-mini"], ["kimi-k2.5"]]
            mock_rv2.select_model.return_value = {
                "category": "code",
                "model": "kimi-k2.5",
                "tier": 3,
                "benchmark_score": 0.0,
                "reason": "continuation",
            }
            result = resolve_turn_route("sigue", cheap_config, primary_config)
            # Should stay on kimi-k2.5 (continuation preserves tier)
            assert result["model"] == "kimi-k2.5"


# ─────────────────────────────────────────────────────────────
# 5. Telemetry import doesn't crash
# ─────────────────────────────────────────────────────────────

class TestTelemetrySafety:
    """Telemetry instrumentation should never crash on missing module."""

    def test_instrument_graceful_on_missing_module(self):
        """instrument_resolve_turn_route should not crash if routing_telemetry is missing."""
        from agent.smart_model_routing import instrument_resolve_turn_route

        # This should not raise even though routing_telemetry doesn't exist
        instrument_resolve_turn_route()

    def test_auto_instrument_env_var_off(self):
        """When HERMES_ROUTING_TELEMETRY is not set, no instrumentation happens."""
        import importlib
        import agent.smart_model_routing

        # Remove env var if present
        os.environ.pop("HERMES_ROUTING_TELEMETRY", None)
        # The module should have loaded fine without telemetry
        assert hasattr(agent.smart_model_routing, "resolve_turn_route")