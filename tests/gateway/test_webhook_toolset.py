"""Tests for hermes-webhook toolset and webhook platform registration."""
from hermes_cli.tools_config import PLATFORMS
from toolsets import get_toolset, resolve_toolset, validate_toolset


class TestHermesWebhookToolset:
    def test_toolset_exists(self):
        assert get_toolset("hermes-webhook") is not None

    def test_toolset_validates(self):
        assert validate_toolset("hermes-webhook")

    def test_toolset_includes_core_tools(self):
        tools = resolve_toolset("hermes-webhook")
        for tool in ["web_search", "web_extract", "terminal", "process",
                     "read_file", "write_file", "patch", "search_files",
                     "execute_code", "delegate_task", "todo", "memory",
                     "session_search"]:
            assert tool in tools, f"Missing expected tool: {tool}"

    def test_toolset_includes_browser_tools(self):
        tools = resolve_toolset("hermes-webhook")
        for tool in ["browser_navigate", "browser_snapshot", "browser_click",
                     "browser_type", "browser_scroll", "browser_back",
                     "browser_press", "browser_close"]:
            assert tool in tools, f"Missing browser tool: {tool}"


class TestWebhookPlatformConfig:
    def test_platforms_dict_includes_webhook(self):
        assert "webhook" in PLATFORMS

    def test_webhook_default_toolset(self):
        assert PLATFORMS["webhook"]["default_toolset"] == "hermes-webhook"

    def test_webhook_has_label(self):
        assert "label" in PLATFORMS["webhook"]
        assert PLATFORMS["webhook"]["label"]
