"""Unit tests for the real TrafilaturaExtractProvider (not the dispatcher).

Companion to test_trafilatura_extract_fallback.py, which exercises
web_extract_tool() end-to-end with fake providers. This file tests the
actual provider class's own safety/quality gates in isolation, mocking
only the network call (httpx.get) — is_safe_url / check_website_access
run for real so a regression in either shows up here.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from plugins.web.trafilatura.provider import TrafilaturaExtractProvider


def _make_response(
    status_code=200,
    content=b"<html><body><p>hello</p></body></html>",
    headers=None,
    url="https://example.com/page",
    history=None,
):
    return SimpleNamespace(
        status_code=status_code,
        content=content,
        headers=headers or {"content-type": "text/html"},
        url=url,
        history=history or [],
    )


class TestRedirectSSRFRecheck:
    """The provider must re-validate EVERY redirect hop's URL, not just the
    originally-requested one — a page can 200 at a safe URL and redirect
    to a private/internal address."""

    def test_redirect_to_private_address_is_blocked(self, monkeypatch):
        provider = TrafilaturaExtractProvider()

        # Simulate: requested URL is public, but the response's redirect
        # history includes a hop to a cloud-metadata address.
        private_hop = SimpleNamespace(url="http://169.254.169.254/latest/meta-data/")
        response = _make_response(
            url="http://169.254.169.254/latest/meta-data/",
            history=[private_hop],
        )
        monkeypatch.setattr(
            "plugins.web.trafilatura.provider.httpx.get",
            MagicMock(return_value=response),
        )
        # check_website_access is a real call (no block configured) —
        # the SSRF check must catch this even if policy doesn't.
        monkeypatch.setattr(
            "tools.website_policy.check_website_access", lambda url: None
        )

        results = provider.extract(["https://example.com/redirect-me"])
        assert len(results) == 1
        assert results[0]["error"] is not None
        assert "private address" in results[0]["error"] or "blocked" in results[0]["error"]
        # The requested URL must be echoed back, not the post-redirect one.
        assert results[0]["url"] == "https://example.com/redirect-me"

    def test_no_redirect_safe_url_proceeds(self, monkeypatch):
        provider = TrafilaturaExtractProvider()
        response = _make_response(
            content=b"<html><body>" + (b"word " * 100) + b"</body></html>",
        )
        monkeypatch.setattr(
            "plugins.web.trafilatura.provider.httpx.get",
            MagicMock(return_value=response),
        )
        monkeypatch.setattr(
            "tools.website_policy.check_website_access", lambda url: None
        )

        results = provider.extract(["https://example.com/page"])
        assert results[0].get("error") is None


class TestQualityGate:
    """Thin/blocked content must be surfaced as an error (not fabricated
    success) so the dispatcher falls back to the next provider."""

    def test_thin_content_no_structured_signal_is_rejected(self, monkeypatch):
        provider = TrafilaturaExtractProvider()
        # Very short body, no title/metadata -> should trip the thin-content gate.
        response = _make_response(content=b"<html><body><p>hi</p></body></html>")
        monkeypatch.setattr(
            "plugins.web.trafilatura.provider.httpx.get",
            MagicMock(return_value=response),
        )
        monkeypatch.setattr(
            "tools.website_policy.check_website_access", lambda url: None
        )

        results = provider.extract(["https://example.com/thin"])
        assert results[0]["error"] is not None
        assert "thin content" in results[0]["error"] or "no main content" in results[0]["error"]

    def test_block_page_signature_is_rejected(self, monkeypatch):
        provider = TrafilaturaExtractProvider()
        response = _make_response(
            content=b"<html><body><p>Please enable javascript and reload.</p></body></html>",
        )
        monkeypatch.setattr(
            "plugins.web.trafilatura.provider.httpx.get",
            MagicMock(return_value=response),
        )
        monkeypatch.setattr(
            "tools.website_policy.check_website_access", lambda url: None
        )

        results = provider.extract(["https://example.com/blocked"])
        assert results[0]["error"] is not None


class TestContentTypeShortCircuit:
    def test_image_content_type_short_circuits_before_extraction(self, monkeypatch):
        provider = TrafilaturaExtractProvider()
        response = _make_response(headers={"content-type": "image/png"})
        monkeypatch.setattr(
            "plugins.web.trafilatura.provider.httpx.get",
            MagicMock(return_value=response),
        )
        monkeypatch.setattr(
            "tools.website_policy.check_website_access", lambda url: None
        )

        results = provider.extract(["https://example.com/image.png"])
        assert results[0]["error"] is not None
        assert "non-extractable" in results[0]["error"]


class TestPolicyGate:
    def test_pre_fetch_policy_block_short_circuits_before_network_call(self, monkeypatch):
        provider = TrafilaturaExtractProvider()
        get_mock = MagicMock()
        monkeypatch.setattr(
            "plugins.web.trafilatura.provider.httpx.get", get_mock
        )
        monkeypatch.setattr(
            "tools.website_policy.check_website_access",
            lambda url: {"reason": "denylisted domain"},
        )

        results = provider.extract(["https://blocked-example.com/page"])
        assert results[0]["error"] is not None
        assert "website policy" in results[0]["error"]
        get_mock.assert_not_called()


class TestMetadataSerialization:
    """fc3dbe321 fix: lxml _Element fields (body, commentsbody) must never
    reach the JSON response — they aren't serializable."""

    def test_lxml_element_fields_stripped_from_metadata(self, monkeypatch):
        provider = TrafilaturaExtractProvider()
        response = _make_response(
            content=b"<html><head><title>Real Article</title></head><body>"
            + (b"word " * 100) + b"</body></html>",
        )
        monkeypatch.setattr(
            "plugins.web.trafilatura.provider.httpx.get",
            MagicMock(return_value=response),
        )
        monkeypatch.setattr(
            "tools.website_policy.check_website_access", lambda url: None
        )

        results = provider.extract(["https://example.com/article"])
        assert results[0].get("error") is None
        metadata = results[0].get("metadata", {})
        assert "body" not in metadata
        assert "commentsbody" not in metadata
        # Must be JSON-serializable — this is the whole point of the fix.
        import json
        json.dumps(results[0])
