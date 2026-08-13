"""Tests for hermes_cli.inventory.discover_provider_models.

Pure-unit: the HTTP call is mocked via unittest.mock, so no live network is
touched. Covers the OpenAI-compatible shape, the Anthropic shape (same
data[].id contract), trailing-slash normalization, and every error path.
"""

from unittest.mock import MagicMock, patch

import pytest

from hermes_cli.inventory import ModelDiscoveryError, discover_provider_models


def _fake_response(status_code: int = 200, json_data: object = None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    return resp


def _fake_client(response: MagicMock) -> MagicMock:
    client = MagicMock()
    client.get.return_value = response
    client.__enter__.return_value = client
    client.__exit__.return_value = False
    return client


def test_openai_shape():
    resp = _fake_response(200, {"data": [{"id": "gpt-4o"}, {"id": "gpt-4o-mini", "name": "Mini"}]})
    with patch("httpx.Client", return_value=_fake_client(resp)):
        models = discover_provider_models("https://host/v1", api_key="k")
    assert models == [
        {"id": "gpt-4o", "name": "gpt-4o"},
        {"id": "gpt-4o-mini", "name": "Mini"},
    ]


def test_trailing_slash_normalized():
    resp = _fake_response(200, {"data": [{"id": "a"}]})
    captured: dict[str, str] = {}
    client = _fake_client(resp)

    def _capture(url: str) -> MagicMock:
        captured["url"] = url
        return resp

    client.get.side_effect = _capture
    with patch("httpx.Client", return_value=client):
        discover_provider_models("https://host/v1/", api_key="k")
    assert captured["url"] == "https://host/v1/models"


def test_anthropic_shape():
    resp = _fake_response(200, {"data": [{"id": "claude-opus-4-1", "type": "model"}]})
    with patch("httpx.Client", return_value=_fake_client(resp)):
        models = discover_provider_models("https://api.anthropic.com/v1", api_mode="anthropic_messages")
    assert models == [{"id": "claude-opus-4-1", "name": "claude-opus-4-1"}]


def test_401_raises():
    resp = _fake_response(401, {"error": "unauthorized"})
    with patch("httpx.Client", return_value=_fake_client(resp)):
        with pytest.raises(ModelDiscoveryError):
            discover_provider_models("https://host/v1")


def test_network_error_raises():
    client = MagicMock()
    client.get.side_effect = Exception("connection refused")
    client.__enter__.return_value = client
    client.__exit__.return_value = False
    with patch("httpx.Client", return_value=client):
        with pytest.raises(ModelDiscoveryError):
            discover_provider_models("https://host/v1")


def test_missing_data_array_raises():
    resp = _fake_response(200, {"object": "list"})  # no "data"
    with patch("httpx.Client", return_value=_fake_client(resp)):
        with pytest.raises(ModelDiscoveryError):
            discover_provider_models("https://host/v1")


def test_empty_base_url_raises():
    with pytest.raises(ModelDiscoveryError):
        discover_provider_models("")


def test_dedupes_ids():
    resp = _fake_response(200, {"data": [{"id": "a"}, {"id": "a"}, {"id": "b"}]})
    with patch("httpx.Client", return_value=_fake_client(resp)):
        models = discover_provider_models("https://host/v1")
    assert [m["id"] for m in models] == ["a", "b"]
