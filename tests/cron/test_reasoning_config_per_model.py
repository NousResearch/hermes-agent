"""Tests for per-model reasoning_effort override in cron scheduler."""

import pytest


class TestCronPerModelReasoningConfig:
    """Test cron scheduler respects per-model reasoning overrides.

    Rather than spinning up a full CronScheduler (heavy), we verify the
    resolution logic by testing the helper directly against a config dict
    shaped the same way the scheduler reads it.
    """

    def test_per_model_override_resolves_for_cron_model(self):
        """The spelling-tolerant helper resolves the cron config's model."""
        from hermes_constants import resolve_per_model_reasoning_effort

        # Simulate cron scheduler config shape
        _cfg = {
            "model": {"default": "anthropic/claude-opus-4.5"},
            "agent": {
                "reasoning_effort": "medium",
                "reasoning_overrides": {
                    "anthropic/claude-opus-4.5": "xhigh",
                },
            },
        }
        _model_cfg = _cfg.get("model", {})
        _model = str(_model_cfg.get("default", "") or "").strip()
        _overrides = (_cfg.get("agent", {}) or {}).get("reasoning_overrides", {}) or {}

        result = resolve_per_model_reasoning_effort(_model, _overrides)
        assert result is not None
        assert result["effort"] == "xhigh"

    def test_cron_falls_back_to_global_when_no_override(self):
        """When no per-model override matches, global effort is used."""
        from hermes_constants import parse_reasoning_effort, resolve_per_model_reasoning_effort

        _cfg = {
            "model": {"default": "gpt-5"},
            "agent": {
                "reasoning_effort": "low",
                "reasoning_overrides": {
                    "anthropic/claude-opus-4.5": "xhigh",
                },
            },
        }
        _model = _cfg["model"]["default"]
        _overrides = _cfg["agent"]["reasoning_overrides"]

        per_model = resolve_per_model_reasoning_effort(_model, _overrides)
        assert per_model is None  # no match

        # Scheduler falls back to global
        effort = _cfg["agent"]["reasoning_effort"]
        result = parse_reasoning_effort(effort)
        assert result is not None
        assert result["effort"] == "low"

    def test_cron_handles_missing_model_key(self):
        """Works when config has no model.default."""
        from hermes_constants import resolve_per_model_reasoning_effort

        _cfg = {
            "agent": {
                "reasoning_effort": "medium",
                "reasoning_overrides": {"claude-opus-4.5": "high"},
            },
        }
        _model_cfg = _cfg.get("model", {}) if isinstance(_cfg.get("model", {}), dict) else {}
        _model = str(_model_cfg.get("default", "") or _model_cfg.get("model", "") or "").strip()
        _overrides = (_cfg.get("agent", {}) or {}).get("reasoning_overrides", {}) or {}

        # Empty model → resolve returns None → scheduler uses global
        result = resolve_per_model_reasoning_effort(_model, _overrides)
        assert result is None

    def test_global_fallback_with_yaml_false(self):
        """YAML boolean False must reach parse_reasoning_effort uncoerced.

        Regression: str(... or "").strip() turned False into "", silently
        re-enabling thinking. The raw value must pass through so
        parse_reasoning_effort(False) returns {'enabled': False}.
        """
        from hermes_constants import parse_reasoning_effort, resolve_per_model_reasoning_effort

        _cfg = {
            "model": {"default": "gpt-5"},
            "agent": {
                "reasoning_effort": False,  # YAML boolean, not string
                "reasoning_overrides": {"claude-opus-4.5": "xhigh"},
            },
        }
        _model = _cfg["model"]["default"]
        _overrides = _cfg["agent"]["reasoning_overrides"]

        per_model = resolve_per_model_reasoning_effort(_model, _overrides)
        assert per_model is None  # no match

        # Scheduler global fallback — raw value, no coercion
        result = parse_reasoning_effort(
            _cfg.get("agent", {}).get("reasoning_effort", "")
        )
        assert result is not None
        assert result.get("enabled") is False
