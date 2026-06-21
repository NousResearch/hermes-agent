"""Tests for the bundled local web extraction provider."""
from __future__ import annotations

import pytest

from plugins.web.local.provider import LocalWebExtractProvider, _extract_title, _html_to_text


def test_html_helpers_extract_title_and_visible_text() -> None:
    raw = """
    <html><head><title> Example &amp; Test </title><style>.x{}</style></head>
    <body><script>alert(1)</script><h1>Hello</h1><p>World&nbsp;today</p></body></html>
    """

    assert _extract_title(raw) == "Example & Test"
    text = _html_to_text(raw)
    assert "Hello" in text
    assert "World today" in text
    assert "alert" not in text
    assert ".x" not in text


def test_local_provider_capabilities() -> None:
    provider = LocalWebExtractProvider()

    assert provider.name == "local"
    assert provider.is_available() is True
    assert provider.supports_search() is False
    assert provider.supports_extract() is True
    assert provider.get_setup_schema()["env_vars"] == []


@pytest.mark.asyncio
async def test_local_provider_extract_reports_pdf_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        headers = {"content-type": "application/pdf"}
        text = "%PDF-1.7"
        url = "https://example.com/file.pdf"

        def raise_for_status(self) -> None:
            return None

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args) -> None:
            return None

        async def get(self, url: str):
            return FakeResponse()

    monkeypatch.setattr("plugins.web.local.provider.httpx.AsyncClient", FakeAsyncClient)

    result = await LocalWebExtractProvider().extract(["https://example.com/file.pdf"])

    assert result[0]["url"] == "https://example.com/file.pdf"
    assert "does not handle PDFs" in result[0]["error"]
