"""http_get_json passes on what a search backend said about an HTTP error, not just its code."""

import logging

import httpx

from plugins.web._common import http_error_message, http_get_json


def _response(status, *, json=None, text=None, headers=None):
    request = httpx.Request("GET", "http://searx.local/search")
    if json is not None:
        return httpx.Response(status, json=json, headers=headers, request=request)
    return httpx.Response(status, text=text or "", headers=headers, request=request)


def test_json_error_and_detail_and_retry_after():
    resp = _response(
        429,
        json={"error": "rate_limited", "retry_after_s": 1, "detail": "Wait and retry."},
        headers={"Retry-After": "1"},
    )
    assert http_error_message("SearXNG", resp) == (
        "SearXNG returned HTTP 429: rate_limited: Wait and retry. (retry after 1)"
    )


def test_nested_error_message():
    resp = _response(401, json={"error": {"message": "Invalid API key", "code": "auth"}})
    assert http_error_message("Brave", resp) == "Brave returned HTTP 401: Invalid API key"


def test_plain_text_body_is_capped():
    resp = _response(400, text="x" * 500, headers={"content-type": "text/plain"})
    message = http_error_message("SearXNG", resp)
    assert message == "SearXNG returned HTTP 400: " + "x" * 300


def test_html_body_is_not_quoted():
    resp = _response(502, text="<html><body>Bad Gateway</body></html>", headers={"content-type": "text/html"})
    assert http_error_message("SearXNG", resp) == "SearXNG returned HTTP 502"


def test_http_get_json_failure_carries_the_reason(monkeypatch):
    resp = _response(400, json={"error": "language_required", "detail": "Send language=<code>."})
    monkeypatch.setattr(httpx, "get", lambda *a, **k: resp)
    data, failure = http_get_json(
        "SearXNG", "http://searx.local/search", params={}, headers={}, timeout=5,
        logger=logging.getLogger("test"),
    )
    assert data is None
    assert "language_required: Send language=<code>." in str(failure)
