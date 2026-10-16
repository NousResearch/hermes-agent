"""SVG attachments reach the agent as cached documents instead of failing image sniffing.

`image/svg+xml` is XML, not a raster format: routed to the image cache it dies at the
magic-byte sniff ("Refusing to cache non-image data"), and the failure used to be reported
to the agent as a scope/auth problem. See issue #131738.
"""

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


if "slack_bolt" not in sys.modules:
    for name in (
        "slack_bolt", "slack_bolt.adapter", "slack_bolt.adapter.socket_mode",
        "slack_bolt.adapter.socket_mode.async_handler", "slack_bolt.async_app",
        "slack_sdk", "slack_sdk.web", "slack_sdk.web.async_client", "slack_sdk.errors",
    ):
        sys.modules.setdefault(name, MagicMock())
if "aiohttp" not in sys.modules:
    sys.modules.setdefault("aiohttp", MagicMock())

from gateway.config import PlatformConfig  # noqa: E402
from plugins.platforms.slack.adapter import SlackAdapter  # noqa: E402


SVG_BYTES = b'<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"></svg>'


def _adapter() -> SlackAdapter:
    adapter = SlackAdapter.__new__(SlackAdapter)
    adapter.config = PlatformConfig(token="primary-test-token")
    return adapter


@pytest.mark.parametrize("mimetype", ["image/svg+xml", "image/svg+xml; charset=utf-8"])
def test_svg_mimetype_classified_as_document(mimetype):
    assert SlackAdapter._slack_file_kind({"name": "pic.svg"}, mimetype) == "document"


@pytest.mark.parametrize(
    "mimetype, expected",
    [("image/png", "image"), ("image/jpeg", "image"), ("audio/mpeg", "audio")],
)
def test_raster_and_audio_kinds_unchanged(mimetype, expected):
    assert SlackAdapter._slack_file_kind({"name": "f"}, mimetype) == expected


@pytest.mark.asyncio
async def test_svg_attachment_cached_as_document(monkeypatch):
    adapter = _adapter()
    adapter._download_slack_file_bytes = AsyncMock(return_value=SVG_BYTES)
    cached_doc = AsyncMock(return_value="/cache/docs/doc_abc123.svg")
    monkeypatch.setattr("plugins.platforms.slack.adapter.cache_document_from_bytes_async", cached_doc)

    event = {"files": [{
        "id": "F1", "name": "CAIXINHA.svg", "mimetype": "image/svg+xml",
        "size": len(SVG_BYTES), "url_private_download": "https://files.slack.com/f1.svg",
    }]}
    media_urls, media_types, inlined, text = await adapter._collect_inbound_media(
        event, "C1", "T1", "here you go", [], [])

    assert media_urls == ["/cache/docs/doc_abc123.svg"]
    assert media_types == ["image/svg+xml"]
    assert inlined == [False]
    assert "attachment notice" not in text
    # Cached under the original filename so the .svg extension survives.
    cached_doc.assert_awaited_once_with(SVG_BYTES, "CAIXINHA.svg")


def test_non_image_payload_diagnosis_is_not_an_auth_problem():
    detail = _adapter()._describe_slack_download_failure(
        ValueError("Refusing to cache non-image data as .jpg (starts with: b'<svg')"),
        file_obj={"name": "CAIXINHA.svg"},
    )
    assert detail is not None
    assert "raster image" in detail
    assert "scope" not in detail


def test_html_login_response_keeps_auth_diagnosis():
    detail = _adapter()._describe_slack_download_failure(
        ValueError("Slack returned HTML instead of media for https://files.slack.com/f"),
        file_obj={"name": "pic.png"},
    )
    assert detail is not None
    assert "scope, auth, or file-permission" in detail
