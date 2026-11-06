"""Response lifetime regression tests."""

import io
from unittest.mock import Mock

import pytest
import requests


@pytest.mark.parametrize("status,body,expected", [
    (200, b'{"data": [{"id": "local-model"}]}', "local-model"),
    (503, b"{}", ""),
    (200, b"invalid json", ""),
])
def test_auto_detection_closes_response(monkeypatch, status, body, expected):
    from hermes_cli.runtime_provider import _auto_detect_local_model

    response = requests.Response()
    response.status_code = status
    response._content = body
    response.raw = io.BytesIO(body)
    response.close = Mock(wraps=response.close)
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: response)

    assert _auto_detect_local_model("http://localhost:8000/v1") == expected
    response.close.assert_called_once_with()
