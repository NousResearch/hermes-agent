"""Tests for OpenViking session creation in initialize()."""

import os
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("OPENVIKING_ACCOUNT", raising=False)
    monkeypatch.delenv("OPENVIKING_ENDPOINT", raising=False)
    monkeypatch.delenv("OPENVIKING_API_KEY", raising=False)


def _mock_httpx(json_return=None):
    mock = MagicMock()
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = json_return or {}
    mock.get.return_value = resp
    mock.post.return_value = resp
    return mock


class TestOpenVikingSessionCreation:
    def test_initialize_creates_server_side_session(self, monkeypatch):
        """Session ID must come from the OpenViking server, not from Hermes."""
        monkeypatch.setenv("OPENVIKING_ENDPOINT", "http://localhost:1933")
        mock = _mock_httpx({"result": {"session_id": "ov-abc123"}})

        with patch("plugins.memory.openviking._get_httpx", return_value=mock):
            from plugins.memory.openviking import OpenVikingMemoryProvider
            p = OpenVikingMemoryProvider()
            p.initialize("hermes-uuid-12345")

            assert p._session_id == "ov-abc123"
            assert p._session_id != "hermes-uuid-12345"
            assert p._client is not None

    def test_initialize_disables_on_missing_session_id(self, monkeypatch):
        """Plugin must disable when server returns no session_id."""
        monkeypatch.setenv("OPENVIKING_ENDPOINT", "http://localhost:1933")
        mock = _mock_httpx({"result": {}})

        with patch("plugins.memory.openviking._get_httpx", return_value=mock):
            from plugins.memory.openviking import OpenVikingMemoryProvider
            p = OpenVikingMemoryProvider()
            p.initialize("hermes-uuid-99999")

            assert p._client is None
            assert p._session_id == ""

    def test_initialize_disables_on_post_exception(self, monkeypatch):
        """Plugin must disable when session creation POST raises."""
        monkeypatch.setenv("OPENVIKING_ENDPOINT", "http://localhost:1933")
        mock = _mock_httpx()
        # Health check passes, but POST /sessions fails
        mock.post.side_effect = Exception("Connection refused")
        health_resp = MagicMock()
        health_resp.status_code = 200
        mock.get.return_value = health_resp

        with patch("plugins.memory.openviking._get_httpx", return_value=mock):
            from plugins.memory.openviking import OpenVikingMemoryProvider
            p = OpenVikingMemoryProvider()
            p.initialize("hermes-uuid-error")

            assert p._client is None

    def test_initialize_handles_non_dict_result(self, monkeypatch):
        """Plugin must handle when 'result' is not a dict (e.g. a list or string)."""
        monkeypatch.setenv("OPENVIKING_ENDPOINT", "http://localhost:1933")
        mock = _mock_httpx({"result": ["unexpected", "list"]})

        with patch("plugins.memory.openviking._get_httpx", return_value=mock):
            from plugins.memory.openviking import OpenVikingMemoryProvider
            p = OpenVikingMemoryProvider()
            p.initialize("hermes-uuid-weird")

            # Should gracefully disable, not crash with AttributeError
            assert p._client is None
            assert p._session_id == ""
