"""Response lifetime regression tests."""

import io
from unittest.mock import Mock

import pytest
import requests


@pytest.mark.parametrize("status,body,error", [
    (200, b'{"models": [{"id": "test-image-model", "input_modalities": ["text"]}]}', None),
    (503, b"{}", requests.HTTPError),
    (200, b"invalid json", requests.exceptions.JSONDecodeError),
])
def test_live_catalog_closes_response(monkeypatch, status, body, error):
    from plugins.image_gen.xai import _fetch_live_models

    response = requests.Response()
    response.status_code = status
    response._content = body
    response.raw = io.BytesIO(body)
    response.close = Mock(wraps=response.close)
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: response)
    credentials = {"api_key": "test-key", "base_url": "https://xai.example/v1"}

    if error:
        with pytest.raises(error):
            _fetch_live_models(credentials)
    else:
        assert _fetch_live_models(credentials) == {
            "test-image-model": {"input_modalities": ["text"], "aliases": []}}
    response.close.assert_called_once_with()
