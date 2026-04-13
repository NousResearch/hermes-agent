"""Tests for the OpenViking memory provider plugin."""

import os
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Ensure OpenViking env vars don't leak between tests."""
    monkeypatch.delenv("OPENVIKING_ACCOUNT", raising=False)
    monkeypatch.delenv("OPENVIKING_ENDPOINT", raising=False)
    monkeypatch.delenv("OPENVIKING_API_KEY", raising=False)


def _mock_httpx(json_return=None):
    """Return a mock httpx module with a configurable response."""
    mock = MagicMock()
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = json_return or {}
    mock.get.return_value = resp
    mock.post.return_value = resp
    return mock


class TestVikingClientDefaults:
    def test_default_account_matches_server_extraction(self):
        """The default account must be 'default' to match OpenViking's internal extraction."""
        with patch("plugins.memory.openviking._get_httpx", return_value=_mock_httpx()):
            from plugins.memory.openviking import _VikingClient
            client = _VikingClient("http://localhost:1933")
            assert client._account == "default"

    def test_account_header_matches_default(self):
        with patch("plugins.memory.openviking._get_httpx", return_value=_mock_httpx()):
            from plugins.memory.openviking import _VikingClient
            client = _VikingClient("http://localhost:1933")
            assert client._headers()["X-OpenViking-Account"] == "default"

    def test_env_override_takes_precedence(self, monkeypatch):
        monkeypatch.setenv("OPENVIKING_ACCOUNT", "custom-tenant")
        with patch("plugins.memory.openviking._get_httpx", return_value=_mock_httpx()):
            from plugins.memory.openviking import _VikingClient
            client = _VikingClient("http://localhost:1933")
            assert client._account == "custom-tenant"

    def test_explicit_param_takes_precedence(self):
        with patch("plugins.memory.openviking._get_httpx", return_value=_mock_httpx()):
            from plugins.memory.openviking import _VikingClient
            client = _VikingClient("http://localhost:1933", account="explicit")
            assert client._account == "explicit"
