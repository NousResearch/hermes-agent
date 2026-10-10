"""Web providers preserve failure facts before formatting model diagnostics."""

import json
import logging
from types import SimpleNamespace

import httpx
import pytest
import requests

from plugins.web import _common
from plugins.web.firecrawl import provider as firecrawl
from agent.web_acquisition_errors import (
    WebInvalidResponseError,
    WebRateLimitedError,
    WebTimeoutError,
    failure_from_data,
)


@pytest.mark.parametrize(
    "failure",
    [
        {"kind": "timeout", "retry": "never", "scope": "provider"},
        {
            "kind": "rate-limited",
            "retry": "transient",
            "scope": "provider",
            "status": True,
        },
        {
            "kind": "timeout",
            "retry": "transient",
            "scope": "provider",
            "timeout_seconds": float("inf"),
        },
        {
            "kind": "connection",
            "retry": "transient",
            "scope": "provider",
            "message": "synthetic-secret",
        },
    ],
)
def test_invalid_failure_facts_are_rejected(failure):
    with pytest.raises(WebInvalidResponseError):
        failure_from_data(failure)


@pytest.mark.parametrize("wrapped", ["direct", "sdk", "cause"])
@pytest.mark.parametrize("kind", ["rate-limited", "timeout", "invalid-response"])
def test_ddgs_worker_preserves_typed_failure_without_inventing_http_status(
    monkeypatch, wrapped, kind
):
    from io import StringIO
    from ddgs.ddgs import DDGS
    from ddgs.exceptions import RatelimitException, TimeoutException
    from plugins.web.ddgs import _search_worker, provider

    error = {
        "rate-limited": RatelimitException("Bearer synthetic-secret"),
        "timeout": TimeoutException("Bearer synthetic-secret"),
        "invalid-response": json.JSONDecodeError("Bearer synthetic-secret", "{", 1),
    }[kind]

    def fail_engine(*args, **kwargs):
        raise error

    client = DDGS()
    monkeypatch.setattr(
        client,
        "_get_engines",
        lambda *args: [
            SimpleNamespace(provider="fake", name="fake", search=fail_engine)
        ],
    )

    def fail(*args):
        if wrapped == "sdk":
            return client.text("news", max_results=1)
        if wrapped == "cause":
            raise RuntimeError("outer") from error
        raise error

    monkeypatch.setattr(provider, "_run_ddgs_search", fail)
    monkeypatch.setattr(
        _search_worker.sys, "stdin", StringIO('{"query":"news","safe_limit":5}')
    )
    output = StringIO()
    monkeypatch.setattr(_search_worker.sys, "stdout", output)
    assert _search_worker.main() == 1
    envelope = json.loads(output.getvalue())
    expected = {
        "kind": kind,
        "retry": "never" if kind == "invalid-response" else "transient",
        "scope": "provider",
    }
    assert envelope == {
        "ok": False,
        "error": f"Web acquisition failed: {kind}",
        "failure": expected,
    }
    native_class = {
        "rate-limited": WebRateLimitedError,
        "timeout": WebTimeoutError,
        "invalid-response": WebInvalidResponseError,
    }[kind]
    with pytest.raises(native_class) as raised:
        provider._parse_envelope(output.getvalue())
    assert raised.value.to_failure() == expected


@pytest.mark.parametrize(
    ("exception", "expected"),
    [
        (
            TimeoutError("Bearer synthetic-secret"),
            {"kind": "timeout", "retry": "transient", "scope": "provider"},
        ),
        (
            requests.ConnectionError("Bearer synthetic-secret"),
            {"kind": "connection", "retry": "transient", "scope": "provider"},
        ),
        (
            json.JSONDecodeError("Bearer synthetic-secret", "{", 1),
            {"kind": "invalid-response", "retry": "never", "scope": "provider"},
        ),
        (
            ValueError("HTTP 429: Bearer synthetic-secret"),
            {"kind": "unclassified", "retry": "unknown", "scope": "provider"},
        ),
        (
            httpx.HTTPStatusError(
                "Bearer synthetic-secret",
                request=httpx.Request("GET", "https://provider.example/search"),
                response=httpx.Response(429, headers={"retry-after": "3"}),
            ),
            {
                "kind": "rate-limited",
                "retry": "transient",
                "scope": "provider",
                "status": 429,
                "retry_after_ms": 3000,
            },
        ),
    ],
)
def test_common_guard_preserves_safe_failure_facts(monkeypatch, exception, expected):
    monkeypatch.setattr(_common, "_interrupted", lambda *args: False)

    def fail():
        raise exception

    result = _common.run_search("Test", logging.getLogger(__name__), fail)
    assert result["success"] is False
    assert result["failure"] == expected


def test_firecrawl_independent_catch_preserves_timeout(monkeypatch):
    monkeypatch.setattr("tools.interrupt.is_interrupted", lambda *args: False)
    monkeypatch.setattr(firecrawl, "_use_keyless_ring", lambda *args: False)

    def fail(**kwargs):
        raise TimeoutError("Bearer synthetic-secret")

    monkeypatch.setattr(
        firecrawl, "_get_firecrawl_client", lambda *args: SimpleNamespace(search=fail)
    )
    result = firecrawl.FirecrawlWebSearchProvider().search("news", 5)
    assert result["success"] is False
    assert result["failure"] == {
        "kind": "timeout",
        "retry": "transient",
        "scope": "provider",
    }


@pytest.mark.parametrize("kind", ["timeout", "connection", "invalid-response"])
def test_parallel_preserves_sdk_causes_without_a_standard_library_cause(
    monkeypatch, kind
):
    from parallel import APIConnectionError, APIResponseValidationError, APITimeoutError
    from plugins.web.parallel import provider

    request = httpx.Request("POST", "https://provider.example/search")
    errors = {
        "timeout": APITimeoutError(request),
        "connection": APIConnectionError(request=request),
        "invalid-response": APIResponseValidationError(
            httpx.Response(200, request=request), {"unexpected": True}
        ),
    }

    def fail(**kwargs):
        raise errors[kind]

    monkeypatch.setattr(_common, "_interrupted", lambda *args: False)
    monkeypatch.setattr(provider, "use_keyless", lambda *args: False)
    monkeypatch.setattr(
        provider,
        "_get_sync_client",
        lambda: SimpleNamespace(beta=SimpleNamespace(search=fail)),
    )
    result = provider.ParallelWebSearchProvider().search("news", 5)
    assert result["failure"] == {
        "kind": kind,
        "retry": "never" if kind == "invalid-response" else "transient",
        "scope": "provider",
    }


@pytest.mark.parametrize(
    ("sdk_kind", "expected"),
    [
        (
            "WebsiteNotSupportedError",
            {
                "kind": "website-unsupported",
                "retry": "never",
                "scope": "page",
                "status": 403,
            },
        ),
        (
            "PaymentRequiredError",
            {
                "kind": "payment-required",
                "retry": "never",
                "scope": "provider",
                "status": 402,
            },
        ),
    ],
)
def test_firecrawl_preserves_sdk_meaning_when_http_status_is_ambiguous(
    monkeypatch, sdk_kind, expected
):
    from firecrawl.v2.utils import error_handler

    def fail(**kwargs):
        raise getattr(error_handler, sdk_kind)(
            "Bearer synthetic-secret", status_code=expected["status"]
        )

    monkeypatch.setattr("tools.interrupt.is_interrupted", lambda *args: False)
    monkeypatch.setattr(firecrawl, "_use_keyless_ring", lambda *args: False)
    monkeypatch.setattr(
        firecrawl, "_get_firecrawl_client", lambda *args: SimpleNamespace(search=fail)
    )
    result = firecrawl.FirecrawlWebSearchProvider().search("news", 5)
    assert result["failure"] == expected


def test_keyless_firecrawl_catch_preserves_http_rate_limit(monkeypatch):
    from plugins.web import keyless_mcp

    error = httpx.HTTPStatusError(
        "Bearer synthetic-secret",
        request=httpx.Request("POST", "https://provider.example/search"),
        response=httpx.Response(429, headers={"retry-after": "3"}),
    )

    def fail(**kwargs):
        raise error

    monkeypatch.setattr(
        firecrawl, "_KeylessFirecrawlClient", lambda *args: SimpleNamespace(search=fail)
    )
    result = keyless_mcp.firecrawl_search_keyless("news", 5)
    assert result["failure"] == {
        "kind": "rate-limited",
        "retry": "transient",
        "scope": "provider",
        "status": 429,
        "retry_after_ms": 3000,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("expired", [False, True])
async def test_firecrawl_extract_timeout_reports_only_its_own_deadline(
    monkeypatch, expired
):
    monkeypatch.setattr(firecrawl, "check_website_access", lambda url: None)

    class ExpiredDeadline:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            raise TimeoutError()

        def expired(self):
            return True

    if expired:

        def deadline(seconds):
            assert seconds == 60
            return ExpiredDeadline()

        monkeypatch.setattr(firecrawl.asyncio, "timeout", deadline)

    def scrape(**kwargs):
        assert kwargs == {
            "url": "https://example.test/page",
            "formats": ["markdown"],
            "timeout": 60_000,
        }
        if not expired:
            raise TimeoutError("provider timed out")
        return {"markdown": "page"}

    monkeypatch.setattr(
        firecrawl,
        "_get_firecrawl_client",
        lambda capability: SimpleNamespace(scrape=scrape),
    )
    result = await firecrawl._scrape_one(
        "https://example.test/page", ["markdown"], "markdown"
    )
    failure = {"kind": "timeout", "retry": "transient", "scope": "provider"}
    if expired:
        failure["timeout_seconds"] = 60
    assert result == {
        "url": "https://example.test/page",
        "title": "",
        "content": "",
        **({} if expired else {"raw_content": ""}),
        "error": firecrawl._SCRAPE_TIMEOUT_MSG if expired else "provider timed out",
        "failure": failure,
    }


@pytest.mark.parametrize("vendor", ["parallel", "keenable"])
@pytest.mark.parametrize(
    "payload", [{}, {"results": None}, {"results": [None]}, {"results": []}]
)
def test_keyless_structured_search_requires_explicit_results(
    monkeypatch, vendor, payload
):
    from plugins.web import keyless_mcp

    monkeypatch.setattr(
        keyless_mcp, "mcp_call", lambda *args, **kwargs: json.dumps(payload)
    )
    monkeypatch.setattr(
        keyless_mcp, "_keenable_request", lambda *args, **kwargs: payload
    )
    result = getattr(keyless_mcp, f"{vendor}_search_keyless")("news", 5)
    if payload == {"results": []}:
        assert result == {"success": True, "data": {"web": []}}
        return
    assert result["success"] is False
    assert result["failure"] == {
        "kind": "invalid-response",
        "retry": "never",
        "scope": "provider",
    }


@pytest.mark.parametrize(
    "payload",
    [{"success": True}, {"success": True, "data": {}}, {"data": {"web": None}}],
)
def test_raw_firecrawl_search_requires_a_result_container(payload):
    with pytest.raises(WebInvalidResponseError):
        firecrawl._extract_web_search_results(payload)


def test_firecrawl_typed_empty_search_remains_valid():
    from firecrawl.v2.types import SearchData

    assert firecrawl._extract_web_search_results(SearchData()) == []
    assert (
        firecrawl._extract_web_search_results({"success": True, "data": {"web": []}})
        == []
    )
