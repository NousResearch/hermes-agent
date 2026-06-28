"""Tests for config helpers in hermes_cli.browser_connect."""

from __future__ import annotations


def _seed_config(tmp_path, browser_overrides: dict):
    """Write a config.yaml with the given browser overrides merged."""
    import yaml
    config = {
        "_config_version": 9,
        "browser": {
            "inactivity_timeout": 120,
            "cdp_url": "",
            **browser_overrides,
        },
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")


class TestMcpServerNameConfig:
    def test_default_name(self, tmp_path, monkeypatch):
        """mcp_server_name should default to chrome-devtools."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _seed_config(tmp_path, {})
        from hermes_cli.browser_connect import get_mcp_server_name
        assert get_mcp_server_name() == "chrome-devtools"

    def test_custom_name(self, tmp_path, monkeypatch):
        """Should read custom mcp_server_name from config."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _seed_config(tmp_path, {"mcp_server_name": "chromium-cdp"})
        from hermes_cli.browser_connect import get_mcp_server_name
        assert get_mcp_server_name() == "chromium-cdp"


class TestMcpBrowserUrlArgConfig:
    def test_default_arg(self, tmp_path, monkeypatch):
        """mcp_browser_url_arg should default to --browser-url."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _seed_config(tmp_path, {})
        from hermes_cli.browser_connect import get_mcp_browser_url_arg
        assert get_mcp_browser_url_arg() == "--browser-url"

    def test_custom_arg(self, tmp_path, monkeypatch):
        """Should read custom mcp_browser_url_arg from config."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _seed_config(tmp_path, {"mcp_browser_url_arg": "--cdp-url"})
        from hermes_cli.browser_connect import get_mcp_browser_url_arg
        assert get_mcp_browser_url_arg() == "--cdp-url"
