"""SSRF invariants for OpenAI image_gen _load_image_bytes (salvage of #54553/#56035)."""

from unittest.mock import MagicMock, patch

import pytest


def _redirect(location: str) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 302
    resp.headers = {"Location": location}
    return resp


def _ok(body: bytes = b"\x89PNG\r\n") -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.headers = {}
    resp.content = body
    resp.raise_for_status = MagicMock()
    return resp


def _client(response: MagicMock):
    client = MagicMock()
    client.get.return_value = response
    context = MagicMock()
    context.__enter__.return_value = client
    return context, client


def test_load_image_bytes_blocks_private_url():
    from plugins.image_gen.openai import _load_image_bytes

    with pytest.raises(ValueError, match="SSRF"):
        _load_image_bytes("http://127.0.0.1/secret.png")


def test_load_image_bytes_blocks_redirect_to_metadata():
    from plugins.image_gen.openai import _load_image_bytes

    public = "https://cdn.example.com/a.png"
    evil = "http://169.254.169.254/latest/meta-data/"
    context, client = _client(_redirect(evil))

    with patch("tools.url_safety.is_safe_url", side_effect=lambda u: u == public), patch(
        "tools.url_safety.create_ssrf_safe_client", return_value=context
    ) as safe_client:
        with pytest.raises(ValueError, match="SSRF"):
            _load_image_bytes(public)
    safe_client.assert_called_once_with(timeout=60, follow_redirects=False)
    client.get.assert_called_once_with(public)


def test_load_image_bytes_allows_public_url():
    from plugins.image_gen.openai import _load_image_bytes

    public = "https://cdn.example.com/a.png"
    context, client = _client(_ok(b"png-bytes"))
    with patch("tools.url_safety.is_safe_url", return_value=True), patch(
        "tools.url_safety.create_ssrf_safe_client", return_value=context
    ) as safe_client:
        data, name = _load_image_bytes(public)
    safe_client.assert_called_once_with(timeout=60, follow_redirects=False)
    client.get.assert_called_once_with(public)
    assert data == b"png-bytes"
    assert name.endswith(".png")
