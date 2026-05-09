"""Tests for hermes_cli.auth_commands.describe_azure_foundry_status."""
from __future__ import annotations

from unittest.mock import patch

from hermes_cli.auth_commands import describe_azure_foundry_status


def test_unconfigured_when_env_empty():
    with patch("hermes_cli.config.get_env_value", return_value=""):
        row = describe_azure_foundry_status({})
    assert row["configured"] is False
    assert row["resource"] == "—"
    assert row["key_source"] == "—"
    assert row["content_safety"] == "—"


def test_env_key_and_base_url():
    env = {
        "AZURE_FOUNDRY_BASE_URL": "https://my-resource.openai.azure.com/openai/v1",
        "AZURE_FOUNDRY_API_KEY": "sk-test",
    }
    row = describe_azure_foundry_status(env)
    assert row["configured"] is True
    assert row["resource"] == "my-resource.openai.azure.com"
    assert row["key_source"] == "env"
    assert row["content_safety"] == "—"


def test_dotenv_key_source():
    env = {"AZURE_FOUNDRY_BASE_URL": "https://r.openai.azure.com"}
    with patch("hermes_cli.config.get_env_value", return_value="dot-key"):
        row = describe_azure_foundry_status(env)
    assert row["configured"] is True
    assert row["key_source"] == "dotenv"
    assert row["resource"] == "r.openai.azure.com"


def test_content_safety_resource_extracted():
    env = {
        "AZURE_FOUNDRY_BASE_URL": "https://r.openai.azure.com",
        "AZURE_FOUNDRY_API_KEY": "sk",
        "AZURE_CONTENT_SAFETY_ENDPOINT": "https://my-cs.cognitiveservices.azure.com",
    }
    row = describe_azure_foundry_status(env)
    assert row["content_safety"] == "my-cs.cognitiveservices.azure.com"


def test_no_base_url_resource_is_dash():
    env = {"AZURE_FOUNDRY_API_KEY": "sk"}
    with patch("hermes_cli.config.get_env_value", return_value=""):
        row = describe_azure_foundry_status(env)
    assert row["configured"] is True
    assert row["resource"] == "—"
    assert row["key_source"] == "env"


def test_dotenv_lookup_failure_handled():
    """If hermes_cli.config import or call fails, we just stay in 'missing' source."""
    env = {"AZURE_FOUNDRY_BASE_URL": "https://r.openai.azure.com"}
    with patch("hermes_cli.config.get_env_value", side_effect=RuntimeError("boom")):
        row = describe_azure_foundry_status(env)
    assert row["configured"] is True
    assert row["key_source"] == "—"
