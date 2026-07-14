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


class TestPerJobReasoningEffort:
    """Per-job reasoning_effort field: validation, storage, and resolution
    priority (job > per-model override > global)."""

    def test_normalize_accepts_valid_levels(self):
        from cron.jobs import _normalize_job_reasoning_effort
        for level in ("none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"):
            assert _normalize_job_reasoning_effort(level) == level
        # Case/whitespace normalized
        assert _normalize_job_reasoning_effort("  HIGH ") == "high"
        # YAML boolean False = disable alias
        assert _normalize_job_reasoning_effort(False) == "none"

    def test_normalize_empty_clears(self):
        from cron.jobs import _normalize_job_reasoning_effort
        assert _normalize_job_reasoning_effort(None) is None
        assert _normalize_job_reasoning_effort("") is None
        assert _normalize_job_reasoning_effort("   ") is None

    def test_normalize_rejects_invalid(self):
        from cron.jobs import _normalize_job_reasoning_effort
        with pytest.raises(ValueError, match="Invalid reasoning_effort"):
            _normalize_job_reasoning_effort("turbo-max")

    def test_create_job_stores_and_validates(self, tmp_path):
        from cron.jobs import use_cron_store, create_job
        with use_cron_store(tmp_path):
            job = create_job(
                prompt="test", schedule="every 1h", reasoning_effort="XHigh"
            )
            assert job["reasoning_effort"] == "xhigh"
            with pytest.raises(ValueError, match="Invalid reasoning_effort"):
                create_job(prompt="t2", schedule="every 1h", reasoning_effort="bogus")

    def test_update_job_sets_and_clears(self, tmp_path):
        from cron.jobs import use_cron_store, create_job, update_job
        with use_cron_store(tmp_path):
            job = create_job(prompt="test", schedule="every 1h")
            assert job["reasoning_effort"] is None
            updated = update_job(job["id"], {"reasoning_effort": "low"})
            assert updated is not None
            assert updated["reasoning_effort"] == "low"
            cleared = update_job(job["id"], {"reasoning_effort": ""})
            assert cleared is not None
            assert cleared["reasoning_effort"] is None
            with pytest.raises(ValueError, match="Invalid reasoning_effort"):
                update_job(job["id"], {"reasoning_effort": "hyper"})

    def test_resolution_priority_job_beats_per_model_and_global(self):
        """Mirror of the scheduler's resolution block: job field wins."""
        from hermes_constants import parse_reasoning_effort, resolve_reasoning_config

        _cfg = {
            "model": {"default": "gpt-5"},
            "agent": {
                "reasoning_effort": "medium",
                "reasoning_overrides": {"gpt-5": "low"},
            },
        }
        job = {"reasoning_effort": "xhigh"}

        # This is the scheduler's exact resolution sequence.
        _job_effort = str(job.get("reasoning_effort") or "").strip()
        reasoning_config = parse_reasoning_effort(_job_effort) if _job_effort else None
        if reasoning_config is None:
            reasoning_config = resolve_reasoning_config(_cfg, "gpt-5")
        assert reasoning_config == {"enabled": True, "effort": "xhigh"}

        # Without the job field, per-model override applies.
        job = {}
        _job_effort = str(job.get("reasoning_effort") or "").strip()
        reasoning_config = parse_reasoning_effort(_job_effort) if _job_effort else None
        if reasoning_config is None:
            reasoning_config = resolve_reasoning_config(_cfg, "gpt-5")
        assert reasoning_config == {"enabled": True, "effort": "low"}

    def test_job_none_disables_despite_per_model_override(self):
        from hermes_constants import parse_reasoning_effort, resolve_reasoning_config

        _cfg = {
            "model": {"default": "gpt-5"},
            "agent": {
                "reasoning_effort": "medium",
                "reasoning_overrides": {"gpt-5": "xhigh"},
            },
        }
        job = {"reasoning_effort": "none"}
        _job_effort = str(job.get("reasoning_effort") or "").strip()
        reasoning_config = parse_reasoning_effort(_job_effort) if _job_effort else None
        if reasoning_config is None:
            reasoning_config = resolve_reasoning_config(_cfg, "gpt-5")
        assert reasoning_config == {"enabled": False}
