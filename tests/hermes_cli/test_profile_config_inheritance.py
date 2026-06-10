"""Regression tests for profile config inheritance (issue #43713).

When a profile config is loaded, the default root user config
(~/.hermes/config.yaml) should be merged as a base layer so that
providers, custom_providers, and other dict-typed keys are inherited.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


def _make_config_files(tmp_path, *, default_config=None, profile_config=None):
    """Create default root and profile config files in tmp_path."""
    hermes_root = tmp_path / ".hermes"
    hermes_root.mkdir(parents=True, exist_ok=True)

    # Default root config
    if default_config is not None:
        (hermes_root / "config.yaml").write_text(default_config, encoding="utf-8")

    # Profile config
    profile_dir = hermes_root / "profiles" / "testprof"
    profile_dir.mkdir(parents=True, exist_ok=True)
    if profile_config is not None:
        (profile_dir / "config.yaml").write_text(profile_config, encoding="utf-8")

    return hermes_root, profile_dir


class TestProfileConfigInheritance:
    """Profile configs should inherit from the default root user config."""

    def test_profile_inherits_providers_from_default(self, tmp_path, monkeypatch):
        """Profile with empty providers inherits default root providers."""
        from hermes_cli.config import _load_config_impl, _LOAD_CONFIG_CACHE

        _LOAD_CONFIG_CACHE.clear()
        hermes_root, profile_dir = _make_config_files(
            tmp_path,
            default_config="providers:\n  openrouter:\n    api_key_env: OPENROUTER_API_KEY\n",
            profile_config="model:\n  default: openrouter/claude-sonnet-4\n",
        )

        # Point HERMES_HOME to the profile directory
        monkeypatch.setenv("HERMES_HOME", str(profile_dir))

        # Patch get_default_hermes_root to return our test root
        with patch("hermes_cli.config.get_default_hermes_root", return_value=hermes_root):
            config = _load_config_impl(want_deepcopy=True)

        providers = config.get("providers", {})
        assert "openrouter" in providers, f"Expected openrouter in providers, got {list(providers.keys())}"
        assert providers["openrouter"]["api_key_env"] == "OPENROUTER_API_KEY"

    def test_profile_overrides_default_provider(self, tmp_path, monkeypatch):
        """Profile-specific provider overrides the default's provider."""
        from hermes_cli.config import _load_config_impl, _LOAD_CONFIG_CACHE

        _LOAD_CONFIG_CACHE.clear()
        hermes_root, profile_dir = _make_config_files(
            tmp_path,
            default_config="providers:\n  openrouter:\n    api_key_env: OPENROUTER_KEY\n",
            profile_config="providers:\n  openrouter:\n    api_key_env: MY_CUSTOM_KEY\n",
        )

        monkeypatch.setenv("HERMES_HOME", str(profile_dir))

        with patch("hermes_cli.config.get_default_hermes_root", return_value=hermes_root):
            config = _load_config_impl(want_deepcopy=True)

        providers = config.get("providers", {})
        assert providers["openrouter"]["api_key_env"] == "MY_CUSTOM_KEY"

    def test_profile_adds_provider_to_default(self, tmp_path, monkeypatch):
        """Profile can add a provider not present in the default config."""
        from hermes_cli.config import _load_config_impl, _LOAD_CONFIG_CACHE

        _LOAD_CONFIG_CACHE.clear()
        hermes_root, profile_dir = _make_config_files(
            tmp_path,
            default_config="providers:\n  openrouter:\n    api_key_env: OR_KEY\n",
            profile_config="providers:\n  anthropic:\n    api_key_env: ANTHROPIC_KEY\n",
        )

        monkeypatch.setenv("HERMES_HOME", str(profile_dir))

        with patch("hermes_cli.config.get_default_hermes_root", return_value=hermes_root):
            config = _load_config_impl(want_deepcopy=True)

        providers = config.get("providers", {})
        assert "openrouter" in providers, "Default provider should be inherited"
        assert "anthropic" in providers, "Profile provider should be added"

    def test_profile_without_config_file_inherits_default(self, tmp_path, monkeypatch):
        """Profile with no config.yaml still inherits default root config."""
        from hermes_cli.config import _load_config_impl, _LOAD_CONFIG_CACHE

        _LOAD_CONFIG_CACHE.clear()
        hermes_root, profile_dir = _make_config_files(
            tmp_path,
            default_config="providers:\n  openrouter:\n    api_key_env: OR_KEY\n",
            profile_config=None,  # No profile config file
        )

        monkeypatch.setenv("HERMES_HOME", str(profile_dir))

        with patch("hermes_cli.config.get_default_hermes_root", return_value=hermes_root):
            config = _load_config_impl(want_deepcopy=True)

        providers = config.get("providers", {})
        assert "openrouter" in providers, "Default provider should be inherited even without profile config"

    def test_default_config_not_affected_by_inheritance(self, tmp_path, monkeypatch):
        """Loading the default config should NOT trigger inheritance logic."""
        from hermes_cli.config import _load_config_impl, _LOAD_CONFIG_CACHE

        _LOAD_CONFIG_CACHE.clear()
        hermes_root, _ = _make_config_files(
            tmp_path,
            default_config="providers:\n  openrouter:\n    api_key_env: OR_KEY\n",
        )

        # Point HERMES_HOME to the root (not a profile)
        monkeypatch.setenv("HERMES_HOME", str(hermes_root))

        with patch("hermes_cli.config.get_default_hermes_root", return_value=hermes_root):
            config = _load_config_impl(want_deepcopy=True)

        providers = config.get("providers", {})
        assert "openrouter" in providers
        # Should only have the default config's providers, no extra merge
        assert providers["openrouter"]["api_key_env"] == "OR_KEY"

    def test_profile_inherits_custom_providers(self, tmp_path, monkeypatch):
        """custom_providers list is inherited from the default config."""
        from hermes_cli.config import _load_config_impl, _LOAD_CONFIG_CACHE

        _LOAD_CONFIG_CACHE.clear()
        hermes_root, profile_dir = _make_config_files(
            tmp_path,
            default_config=(
                "custom_providers:\n"
                "  - name: my-provider\n"
                "    base_url: https://my.api.com/v1\n"
            ),
            profile_config="model:\n  default: my-provider/model-x\n",
        )

        monkeypatch.setenv("HERMES_HOME", str(profile_dir))

        with patch("hermes_cli.config.get_default_hermes_root", return_value=hermes_root):
            config = _load_config_impl(want_deepcopy=True)

        custom = config.get("custom_providers", [])
        assert len(custom) == 1
        assert custom[0]["name"] == "my-provider"
