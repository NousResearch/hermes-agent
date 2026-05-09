"""Tests for hermes_cli.doctor.check_azure_foundry."""
from __future__ import annotations

from unittest.mock import patch, MagicMock
import urllib.error

import pytest

from hermes_cli.doctor import check_azure_foundry


def test_unconfigured_when_env_empty():
    """No env → configured=False, statuses unconfigured."""
    result = check_azure_foundry({})
    assert result["configured"] is False
    assert result["foundry_status"] == "unconfigured"
    assert result["cs_status"] == "unconfigured"
    assert result["key_source"] == "missing"
    assert result["issues"] == []


def test_base_url_set_but_key_missing_warns():
    env = {"AZURE_FOUNDRY_BASE_URL": "https://r.openai.azure.com"}
    # Block dotenv lookup so key remains missing
    with patch("hermes_cli.config.get_env_value", return_value=""):
        result = check_azure_foundry(env)
    assert result["configured"] is True
    assert result["foundry_status"] == "warn"
    assert "API_KEY" in result["foundry_detail"]


def test_key_set_no_base_warns():
    env = {"AZURE_FOUNDRY_API_KEY": "sk-test"}
    result = check_azure_foundry(env)
    assert result["configured"] is True
    assert result["foundry_status"] == "warn"
    assert "BASE_URL" in result["foundry_detail"]
    assert result["key_source"] == "env"


def test_env_and_reachable_returns_ok():
    env = {
        "AZURE_FOUNDRY_BASE_URL": "https://r.openai.azure.com",
        "AZURE_FOUNDRY_API_KEY": "sk-test",
    }
    fake_resp = MagicMock()
    fake_resp.status = 200
    fake_resp.__enter__ = lambda self: fake_resp
    fake_resp.__exit__ = lambda *a: None
    with patch("urllib.request.urlopen", return_value=fake_resp):
        result = check_azure_foundry(env)
    assert result["foundry_status"] == "ok"
    assert result["key_source"] == "env"
    assert result["foundry_detail"].startswith("https://r.openai.azure.com")


def test_http_401_treated_as_ok_endpoint_reachable():
    env = {
        "AZURE_FOUNDRY_BASE_URL": "https://r.openai.azure.com",
        "AZURE_FOUNDRY_API_KEY": "sk-test",
    }
    err = urllib.error.HTTPError("u", 401, "Unauthorized", {}, None)
    with patch("urllib.request.urlopen", side_effect=err):
        result = check_azure_foundry(env)
    assert result["foundry_status"] == "ok"
    assert "401" in result["foundry_detail"]


def test_http_500_records_issue():
    env = {
        "AZURE_FOUNDRY_BASE_URL": "https://r.openai.azure.com",
        "AZURE_FOUNDRY_API_KEY": "sk-test",
    }
    err = urllib.error.HTTPError("u", 500, "boom", {}, None)
    with patch("urllib.request.urlopen", side_effect=err):
        result = check_azure_foundry(env)
    assert result["foundry_status"] == "warn"
    assert any("500" in i for i in result["issues"])


def test_network_error_records_issue():
    env = {
        "AZURE_FOUNDRY_BASE_URL": "https://r.openai.azure.com",
        "AZURE_FOUNDRY_API_KEY": "sk-test",
    }
    with patch("urllib.request.urlopen", side_effect=ConnectionError("dns")):
        result = check_azure_foundry(env)
    assert result["foundry_status"] == "warn"
    assert any("ConnectionError" in i for i in result["issues"])


def test_dotenv_fallback_for_key():
    env = {"AZURE_FOUNDRY_BASE_URL": "https://r.openai.azure.com"}
    fake_resp = MagicMock()
    fake_resp.status = 200
    fake_resp.__enter__ = lambda self: fake_resp
    fake_resp.__exit__ = lambda *a: None
    with patch("hermes_cli.config.get_env_value", return_value="dotenv-key"), \
         patch("urllib.request.urlopen", return_value=fake_resp):
        result = check_azure_foundry(env)
    assert result["key_source"] == "dotenv"
    assert result["foundry_status"] == "ok"


def test_cs_endpoint_set_but_key_missing():
    env = {
        "AZURE_FOUNDRY_BASE_URL": "https://r.openai.azure.com",
        "AZURE_FOUNDRY_API_KEY": "sk",
        "AZURE_CONTENT_SAFETY_ENDPOINT": "https://cs.cognitiveservices.azure.com",
    }
    fake_resp = MagicMock()
    fake_resp.status = 200
    fake_resp.__enter__ = lambda self: fake_resp
    fake_resp.__exit__ = lambda *a: None
    with patch("urllib.request.urlopen", return_value=fake_resp):
        result = check_azure_foundry(env)
    assert result["cs_status"] == "key_missing"
    assert "AZURE_CONTENT_SAFETY_KEY" in result["cs_detail"]


def test_cs_endpoint_reachable_ok():
    env = {
        "AZURE_FOUNDRY_BASE_URL": "https://r.openai.azure.com",
        "AZURE_FOUNDRY_API_KEY": "sk",
        "AZURE_CONTENT_SAFETY_ENDPOINT": "https://cs.cognitiveservices.azure.com",
        "AZURE_CONTENT_SAFETY_KEY": "cs-key",
    }
    fake_resp = MagicMock()
    fake_resp.status = 200
    fake_resp.__enter__ = lambda self: fake_resp
    fake_resp.__exit__ = lambda *a: None
    with patch("urllib.request.urlopen", return_value=fake_resp):
        result = check_azure_foundry(env)
    assert result["cs_status"] == "ok"
    assert result["cs_detail"].startswith("https://cs.cognitiveservices.azure.com")


def test_cs_endpoint_unreachable_warns():
    env = {
        "AZURE_FOUNDRY_BASE_URL": "https://r.openai.azure.com",
        "AZURE_FOUNDRY_API_KEY": "sk",
        "AZURE_CONTENT_SAFETY_ENDPOINT": "https://cs.cognitiveservices.azure.com",
        "AZURE_CONTENT_SAFETY_KEY": "cs-key",
    }
    # Foundry probe ok via 401, CS probe network error.
    foundry_err = urllib.error.HTTPError("u", 401, "x", {}, None)

    def _side(req, *a, **kw):
        if "contentsafety" in req.full_url:
            raise ConnectionError("dns")
        raise foundry_err

    with patch("urllib.request.urlopen", side_effect=_side):
        result = check_azure_foundry(env)
    assert result["foundry_status"] == "ok"
    assert result["cs_status"] == "warn"
