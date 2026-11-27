"""Response lifetime regression tests."""

import io
from unittest.mock import Mock

import pytest
import requests


@pytest.mark.parametrize("status,body,error", [
    (200, b'{"errcode": 0, "nonce": "test-nonce"}', False),
    (503, b"{}", True),
    (200, b"invalid json", True),
])
def test_api_post_closes_response(monkeypatch, status, body, error):
    from hermes_cli.dingtalk_auth import RegistrationError, _api_post

    response = requests.Response()
    response.status_code = status
    response._content = body
    response.raw = io.BytesIO(body)
    response.close = Mock(wraps=response.close)
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: response)

    if error:
        with pytest.raises(RegistrationError):
            _api_post("/test", {})
    else:
        assert _api_post("/test", {}) == {"errcode": 0, "nonce": "test-nonce"}
    response.close.assert_called_once_with()
