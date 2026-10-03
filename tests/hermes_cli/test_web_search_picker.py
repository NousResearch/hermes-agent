"""Tests for the Web Search & Extract category rows in `hermes tools` (tools_config).

Covers the two firecrawl setup-flow rows that share ``web_backend: firecrawl``: a row
declaring required env vars must not report active until those vars are configured, so a
cloud-key install (only ``FIRECRAWL_API_KEY`` set) no longer marks the
"Firecrawl Self-Hosted" row active — nor defaults the picker to it.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hermes_cli.tools_config import TOOL_CATEGORIES, _is_provider_active  # noqa: E402
from hermes_cli.tools_config_providers import _detect_active_provider_index  # noqa: E402


def _web_provider_named(name):
    return next(p for p in TOOL_CATEGORIES["web"]["providers"] if p["name"] == name)


@pytest.fixture()
def env_values(monkeypatch):
    """Control the row env lookup without touching real ~/.hermes or os.environ."""
    values = {}

    def fake_get_env_value(key):
        return values.get(key)

    monkeypatch.setattr(
        "hermes_cli.tools_config.get_env_value", fake_get_env_value, raising=True
    )
    return values


def _cloud_firecrawl_row():
    """Mirrors the plugin-registered "Firecrawl" row shape (cloud API key, shared backend)."""
    return {
        "name": "Firecrawl",
        "web_backend": "firecrawl",
        "env_vars": [{"key": "FIRECRAWL_API_KEY"}],
    }


class TestSharedWebBackendRows:
    def test_self_hosted_row_inactive_without_url(self, env_values):
        env_values["FIRECRAWL_API_KEY"] = "cloud-key"
        config = {"web": {"backend": "firecrawl"}}
        assert _is_provider_active(_web_provider_named("Firecrawl Self-Hosted"), config) is False

    def test_self_hosted_row_active_with_url(self, env_values):
        env_values["FIRECRAWL_API_URL"] = "http://localhost:3002"
        config = {"web": {"backend": "firecrawl"}}
        assert _is_provider_active(_web_provider_named("Firecrawl Self-Hosted"), config) is True

    def test_cloud_row_active_with_key(self, env_values):
        env_values["FIRECRAWL_API_KEY"] = "cloud-key"
        config = {"web": {"backend": "firecrawl"}}
        assert _is_provider_active(_cloud_firecrawl_row(), config) is True

    def test_detection_prefers_configured_row(self, env_values):
        """The picker default must land on the cloud row, not the unconfigured Self-Hosted row."""
        env_values["FIRECRAWL_API_KEY"] = "cloud-key"
        config = {"web": {"backend": "firecrawl"}}
        rows = [_web_provider_named("Firecrawl Self-Hosted"), _cloud_firecrawl_row()]
        assert _detect_active_provider_index(rows, config) == 1


class TestNoKeyRowsUnaffected:
    def test_row_without_env_vars_still_active(self, env_values):
        row = {"name": "DuckDuckGo (ddgs)", "web_backend": "ddgs", "env_vars": []}
        assert _is_provider_active(row, {"web": {"backend": "ddgs"}}) is True
