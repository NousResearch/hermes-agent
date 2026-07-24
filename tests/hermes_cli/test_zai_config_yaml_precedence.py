"""Regression tests for Z.AI base_url precedence: GLM_BASE_URL > model.base_url > probe.

Mirrors the design in PR #58088 ("honor configured base url"), adapted to
the runtime-pool layer added in PR #61575. Verifies that:
  - GLM_BASE_URL env var remains the highest-priority override
  - model.base_url from config.yaml wins when model.provider is a Z.AI alias
  - model.base_url is ignored when model.provider is NOT Z.AI
  - cached detected_endpoint and live probe still work as fallbacks
"""
from __future__ import annotations

import os
from unittest import mock


class TestConfiguredBaseUrlPrecedence:
    """_resolve_zai_base_url must consult config.yaml with the right precedence."""

    def _resolve(self, *, glm_env="", config_override="", api_key="sk-zai-fake"):
        from hermes_cli.auth import _resolve_zai_base_url
        # Always mock detect_zai_endpoint so we don't make real network calls
        with mock.patch(
            "hermes_cli.auth.detect_zai_endpoint",
            return_value=None,
        ):
            # Mock _configured_zai_base_url to return the desired config value
            with mock.patch(
                "hermes_cli.auth._configured_zai_base_url",
                return_value=config_override,
            ):
                return _resolve_zai_base_url(
                    api_key,
                    "https://api.z.ai/api/paas/v4",
                    glm_env.strip().rstrip("/"),
                )

    def test_glm_base_url_wins_over_config(self):
        """GLM_BASE_URL has higher priority than model.base_url."""
        result = self._resolve(
            glm_env="https://from-env.example/v4",
            config_override="https://from-config.example/v4",
            api_key="sk-zai-fake",
        )
        assert result == "https://from-env.example/v4"

    def test_config_base_url_wins_when_no_env(self):
        """model.base_url is consulted when GLM_BASE_URL is unset."""
        result = self._resolve(
            glm_env="",
            config_override="https://api.z.ai/api/coding/paas/v4",
            api_key="sk-zai-fake",
        )
        assert result == "https://api.z.ai/api/coding/paas/v4"

    def test_config_base_url_ignored_for_other_provider(self):
        """When config_override is empty (model.provider != zai), fall back to default."""
        # _configured_zai_base_url returns "" when model.provider != zai
        result = self._resolve(
            glm_env="",
            config_override="",
            api_key="sk-zai-fake",
        )
        # Falls through to probe, which returns None, so falls back to default
        assert result == "https://api.z.ai/api/paas/v4"

    def test_config_base_url_wins_even_without_key(self):
        """model.base_url wins even without an API key — explicit user intent > probe."""
        result = self._resolve(
            glm_env="",
            config_override="https://from-config.example/v4",
            api_key="",
        )
        assert result == "https://from-config.example/v4"

    def test_configured_helper_reads_zai_provider(self):
        """_configured_zai_base_url returns model.base_url only for zai provider."""
        from hermes_cli.auth import _configured_zai_base_url

        # Mock load_config to return a Z.AI config
        with mock.patch(
            "hermes_cli.config.load_config",
            return_value={
                "model": {
                    "provider": "zai",
                    "base_url": "https://api.z.ai/api/coding/paas/v4",
                }
            },
        ):
            assert _configured_zai_base_url() == "https://api.z.ai/api/coding/paas/v4"

        # Mock for glm alias
        with mock.patch(
            "hermes_cli.config.load_config",
            return_value={
                "model": {
                    "provider": "glm",
                    "base_url": "https://open.bigmodel.cn/api/coding/paas/v4",
                }
            },
        ):
            assert (
                _configured_zai_base_url()
                == "https://open.bigmodel.cn/api/coding/paas/v4"
            )

    def test_configured_helper_returns_empty_for_other_providers(self):
        """_configured_zai_base_url returns '' when model.provider is not zai."""
        from hermes_cli.auth import _configured_zai_base_url

        with mock.patch(
            "hermes_cli.config.load_config",
            return_value={
                "model": {
                    "provider": "openrouter",
                    "base_url": "https://openrouter.ai/api/v1",
                }
            },
        ):
            assert _configured_zai_base_url() == ""

        with mock.patch(
            "hermes_cli.config.load_config",
            return_value={
                "model": {
                    "provider": "anthropic",
                    "base_url": "https://api.anthropic.com",
                }
            },
        ):
            assert _configured_zai_base_url() == ""

    def test_configured_helper_handles_missing_config(self):
        """_configured_zai_base_url returns '' when config is missing or malformed."""
        from hermes_cli.auth import _configured_zai_base_url

        # Empty config
        with mock.patch("hermes_cli.config.load_config", return_value={}):
            assert _configured_zai_base_url() == ""

        # No model section
        with mock.patch(
            "hermes_cli.config.load_config",
            return_value={"other_section": {}},
        ):
            assert _configured_zai_base_url() == ""

        # Provider is not a string
        with mock.patch(
            "hermes_cli.config.load_config",
            return_value={"model": {"provider": None, "base_url": "https://x"}},
        ):
            assert _configured_zai_base_url() == ""

        # No base_url in config
        with mock.patch(
            "hermes_cli.config.load_config",
            return_value={"model": {"provider": "zai"}},
        ):
            assert _configured_zai_base_url() == ""

    def test_configured_helper_handles_load_failure(self):
        """_configured_zai_base_url returns '' when config loading raises."""
        from hermes_cli.auth import _configured_zai_base_url

        with mock.patch(
            "hermes_cli.config.load_config",
            side_effect=Exception("disk full"),
        ):
            assert _configured_zai_base_url() == ""