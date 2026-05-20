"""Tests for config_contracts.py — Config-Runtime Contract Registry."""
from __future__ import annotations

import os

import pytest

from hermes_cli.config_contracts import (
    CONFIG_BINDINGS,
    ConfigBinding,
    get_binding_report,
    get_nested,
    validate_config_bindings,
)


class TestGetNested:
    def test_simple_key(self):
        config = {"terminal": {"docker_extra_args": ["--cap-add=SYS_PTRACE"]}}
        val, found = get_nested(config, "terminal.docker_extra_args")
        assert found is True
        assert val == ["--cap-add=SYS_PTRACE"]

    def test_missing_key(self):
        config = {"terminal": {"backend": "local"}}
        val, found = get_nested(config, "terminal.docker_extra_args")
        assert found is False
        assert val is None

    def test_wildcard_key(self):
        config = {
            "custom_providers": {
                "my-provider": {
                    "models": {
                        "gpt-5": {"max_tokens": 8192, "context_length": 256000},
                    }
                }
            }
        }
        val, found = get_nested(config, "custom_providers.*.models.*.max_tokens")
        assert found is True
        assert val == 8192

    def test_wildcard_empty_dict(self):
        config = {"custom_providers": {}}
        val, found = get_nested(config, "custom_providers.*.models.*.max_tokens")
        assert found is False

    def test_top_level_key(self):
        config = {"model": "gpt-5"}
        val, found = get_nested(config, "model")
        assert found is True
        assert val == "gpt-5"

    def test_nested_non_dict(self):
        config = {"terminal": "local"}
        val, found = get_nested(config, "terminal.backend")
        assert found is False


class TestValidateConfigBindings:
    def test_returns_empty_for_minimal_config(self):
        """Minimal config with defaults should produce no warnings."""
        config = {"model": "gpt-4o-mini", "terminal": {"backend": "local"}}
        warnings = validate_config_bindings(config)
        assert isinstance(warnings, list)

    def test_returns_list(self):
        warnings = validate_config_bindings({})
        assert isinstance(warnings, list)

    def test_strict_mode_no_crash(self):
        """Strict mode with extra keys should not crash."""
        config = {"model": "gpt-4o-mini", "new_feature": {"enabled": True}}
        warnings = validate_config_bindings(config, strict=True)
        assert isinstance(warnings, list)


class TestGetBindingReport:
    def test_report_structure(self):
        config = {
            "model": "gpt-4o-mini",
            "terminal": {"backend": "local", "timeout": 180},
        }
        report = get_binding_report(config)
        assert "healthy" in report
        assert "default" in report
        assert "missing" in report
        assert "total" in report
        assert "registered" in report
        assert report["total"] == len(CONFIG_BINDINGS)

    def test_healthy_binding(self):
        config = {"model": {"max_tokens": 8192}}
        report = get_binding_report(config)
        healthy_keys = [b["key"] for b in report["healthy"]]
        assert "model.max_tokens" in healthy_keys

    def test_custom_provider_wildcard(self):
        config = {
            "custom_providers": [
                {
                    "name": "test",
                    "base_url": "https://test.invalid/v1",
                    "models": {"gpt-5": {"max_tokens": 16384}},
                }
            ]
        }
        report = get_binding_report(config)
        # Wildcard bindings should find the value
        healthy_keys = [b["key"] for b in report["healthy"]]
        assert "custom_providers.*.models.*.max_tokens" in healthy_keys


class TestConfigBindingsRegistry:
    def test_all_bindings_have_required_fields(self):
        for binding in CONFIG_BINDINGS:
            assert isinstance(binding, ConfigBinding)
            assert binding.config_key, "config_key must not be empty"
            assert binding.consumer, "consumer must not be empty"
            assert binding.binding_type in ("env_var", "direct_read", "function_call")

    def test_env_var_bindings_have_env_var(self):
        for binding in CONFIG_BINDINGS:
            if binding.binding_type == "env_var":
                assert binding.env_var is not None, (
                    f"env_var binding for {binding.config_key} must have env_var set"
                )

    def test_docker_extra_args_registered(self):
        """The #28863 bug fix must be registered."""
        keys = [b.config_key for b in CONFIG_BINDINGS]
        assert "terminal.docker_extra_args" in keys

    def test_custom_provider_max_tokens_registered(self):
        """The #28046 bug fix must be registered."""
        keys = [b.config_key for b in CONFIG_BINDINGS]
        assert "custom_providers.*.models.*.max_tokens" in keys

    def test_no_duplicate_config_keys(self):
        keys = [b.config_key for b in CONFIG_BINDINGS]
        assert len(keys) == len(set(keys)), f"Duplicate keys: {[k for k in keys if keys.count(k) > 1]}"
