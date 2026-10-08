"""Workspace routing is exclusive to Bedrock Mantle Anthropic endpoints."""
from unittest.mock import patch

import pytest

from agent.anthropic_adapter import build_anthropic_client

MANTLE_BASE = "https://bedrock-mantle.us-east-1.api.aws/anthropic"


@pytest.fixture(autouse=True)
def isolated_config(monkeypatch):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "bedrock-test-key")
    monkeypatch.delenv("BEDROCK_MANTLE_WORKSPACE_ID", raising=False)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
    monkeypatch.setattr("hermes_cli.config.get_custom_provider_extra_headers", lambda *a: {})


@pytest.mark.parametrize("source", ["env", "config", "env-over-config"])
def test_mantle_workspace_header(monkeypatch, source):
    if source != "config":
        monkeypatch.setenv("BEDROCK_MANTLE_WORKSPACE_ID", " proj_test123 ")
    if source != "env":
        monkeypatch.setattr("hermes_cli.config.load_config", lambda: {
            "bedrock": {"mantle_workspace_id": "proj_config" if source == "env-over-config" else "proj_test123"}})
    with patch("agent.anthropic_adapter._anthropic_sdk") as sdk:
        build_anthropic_client("bedrock-test-key", MANTLE_BASE)
        headers = sdk.Anthropic.call_args.kwargs["default_headers"]
        assert headers["anthropic-workspace-id"] == "proj_test123"
        assert "anthropic-workspace" not in headers
        assert "anthropic-beta" in headers


@pytest.mark.parametrize("base", ["https://api.anthropic.com", "https://proxy.example/anthropic",
                                     "https://bedrock-mantle.us-east-1.api.aws/v1",
                                     "https://bedrock-mantle.us-east-1.api.aws.evil.test/anthropic"])
def test_workspace_header_not_added_outside_mantle(monkeypatch, base):
    monkeypatch.setenv("BEDROCK_MANTLE_WORKSPACE_ID", "proj_test123")
    with patch("agent.anthropic_adapter._anthropic_sdk") as sdk:
        build_anthropic_client("sk-ant-api03-test", base)
        assert "anthropic-workspace-id" not in sdk.Anthropic.call_args.kwargs.get("default_headers", {})


@pytest.mark.parametrize("workspace", ["workspace-test", "proj_", "proj_bad\r\nheader"])
def test_invalid_workspace_fails_closed(monkeypatch, workspace):
    monkeypatch.setenv("BEDROCK_MANTLE_WORKSPACE_ID", workspace)
    with patch("agent.anthropic_adapter._anthropic_sdk") as sdk:
        with pytest.raises(ValueError, match="Invalid Bedrock Mantle workspace ID"):
            build_anthropic_client("bedrock-test-key", MANTLE_BASE)
        sdk.Anthropic.assert_not_called()


def test_unconfigured_workspace_is_omitted():
    with patch("agent.anthropic_adapter._anthropic_sdk") as sdk:
        build_anthropic_client("bedrock-test-key", MANTLE_BASE)
        assert "anthropic-workspace-id" not in sdk.Anthropic.call_args.kwargs.get("default_headers", {})
