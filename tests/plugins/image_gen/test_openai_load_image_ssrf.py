"""SSRF invariants for OpenAI image_gen _load_image_bytes (salvage of #54553/#56035)."""

from unittest.mock import MagicMock, patch

import pytest


def _redirect(location: str) -> MagicMock:
    resp = MagicMock()
    resp.is_redirect = True
    resp.headers = {"Location": location}
    return resp


def _ok(body: bytes = b"\x89PNG\r\n") -> MagicMock:
    resp = MagicMock()
    resp.is_redirect = False
    resp.headers = {}
    resp.content = body
    resp.raise_for_status = MagicMock()
    return resp


def test_load_image_bytes_blocks_private_url():
    from plugins.image_gen.openai import _load_image_bytes

    with pytest.raises(ValueError, match="SSRF"):
        _load_image_bytes("http://127.0.0.1/secret.png")


def test_load_image_bytes_blocks_redirect_to_metadata():
    from plugins.image_gen.openai import _load_image_bytes

    public = "https://cdn.example.com/a.png"
    evil = "http://169.254.169.254/latest/meta-data/"

    with patch("tools.url_safety.is_safe_url", side_effect=lambda u: u == public), patch(
        "requests.get", return_value=_redirect(evil)
    ):
        with pytest.raises(ValueError, match="SSRF"):
            _load_image_bytes(public)


def test_load_image_bytes_allows_public_url():
    from plugins.image_gen.openai import _load_image_bytes

    public = "https://cdn.example.com/a.png"
    with patch("tools.url_safety.is_safe_url", return_value=True), patch(
        "requests.get", return_value=_ok(b"png-bytes")
    ):
        data, name = _load_image_bytes(public)
    assert data == b"png-bytes"
    assert name.endswith(".png")
