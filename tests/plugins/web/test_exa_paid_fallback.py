"""Exa free-first policy: an explicit opt-in buys one retry on HTTP 429 only."""
import json
import logging
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import requests

from plugins.web import keyless_mcp
from plugins.web.exa.provider import ExaWebSearchProvider


@pytest.fixture
def exa_config(monkeypatch):
    home = Path(os.environ["HERMES_HOME"])
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        "web:\n  search_backend: exa\n  extract_backend: exa\n"
        "  provider_tier:\n    exa: free\n  exa_paid_fallback: true\n"
    )
    monkeypatch.setenv("EXA_API_KEY", "test-only-exa-credential")
    return home


def response(status=200, text="Title: Example\nURL: https://example.com\nHighlights:\nEvidence"):
    result = requests.Response()
    result.status_code = status
    result.headers["Content-Type"] = "application/json"
    result._content = json.dumps({"result": {"content": [{"type": "text", "text": text}]}}).encode()
    return result


def test_opted_in_search_retries_429_once_then_returns_to_free(exa_config, caplog):
    with patch.object(requests, "post", side_effect=[response(429), response(), response()]) as post:
        with caplog.at_level(logging.WARNING):
            provider = ExaWebSearchProvider()
            first = provider.search("official documentation", 1)
            second = provider.search("another question", 1)
    assert first["success"] and second["success"]
    assert post.call_count == 3
    headers = [call.kwargs["headers"] for call in post.call_args_list]
    assert [h.get("x-api-key") for h in headers] == [None, "test-only-exa-credential", None]
    assert "authenticated retry" in caplog.text
    assert "test-only-exa-credential" not in caplog.text


def test_search_supplies_required_mcp_objective(exa_config):
    def serve(url, *, json, **kwargs):
        assert json["params"]["arguments"]["objective"]
        return response()
    with patch.object(requests, "post", side_effect=serve):
        result = keyless_mcp.exa_search_keyless("official documentation", 1)
    assert result["success"]


def test_extract_retries_only_the_throttled_url(exa_config):
    urls = ["https://example.com/a", "https://example.com/b"]
    with patch.object(requests, "post", side_effect=[response(), response(429), response()]) as post:
        results = ExaWebSearchProvider().extract(urls)
    assert [r["url"] for r in results] == urls
    assert all(not r.get("error") for r in results)
    assert [c.kwargs["json"]["params"]["arguments"]["urls"] for c in post.call_args_list] == [urls[:1], urls[1:], urls[1:]]
    assert [c.kwargs["headers"].get("x-api-key") for c in post.call_args_list] == [None, None, "test-only-exa-credential"]


@pytest.mark.parametrize("setting", ["false", "'true'", "null"])
def test_no_paid_retry_without_boolean_opt_in(exa_config, setting):
    path = exa_config / "config.yaml"
    path.write_text(path.read_text().replace("exa_paid_fallback: true", f"exa_paid_fallback: {setting}"))
    with patch.object(requests, "post", return_value=response(429)) as post:
        result = keyless_mcp.exa_search_keyless("q", 1)
    assert not result["success"]
    assert post.call_count == 1


def test_missing_key_does_not_retry(exa_config, monkeypatch):
    monkeypatch.delenv("EXA_API_KEY")
    with patch.object(requests, "post", return_value=response(429)) as post:
        result = keyless_mcp.exa_search_keyless("q", 1)
    assert not result["success"]
    assert post.call_count == 1


@pytest.mark.parametrize("status", [400, 401, 403, 500, 503])
def test_other_status_with_rate_limit_text_cannot_trigger_spending(exa_config, status):
    with patch.object(requests, "post", return_value=response(status, "429 rate limit quota exceeded")) as post:
        result = keyless_mcp.exa_search_keyless("q", 1)
    assert not result["success"]
    assert post.call_count == 1


def test_mcp_error_text_is_not_an_http_429(exa_config):
    reply = response()
    reply._content = json.dumps({"result": {"isError": True, "content": [{"type": "text", "text": "429 too many requests"}]}}).encode()
    with patch.object(requests, "post", return_value=reply) as post:
        result = keyless_mcp.exa_search_keyless("q", 1)
    assert not result["success"]
    assert post.call_count == 1


@pytest.mark.parametrize("failure", [response(429), response(503), response(302), requests.Timeout("timed out")])
def test_authenticated_failure_is_terminal_and_bounded(exa_config, failure):
    with patch.object(requests, "post", side_effect=[response(429), failure]) as post:
        result = keyless_mcp.exa_search_keyless("q", 1)
    assert not result["success"]
    assert post.call_count == 2
    assert post.call_args.kwargs["allow_redirects"] is False


def test_transport_failure_never_enables_paid_retry(exa_config):
    with patch.object(requests, "post", side_effect=requests.Timeout("timed out")) as post:
        result = keyless_mcp.exa_search_keyless("q", 1)
    assert not result["success"]
    assert post.call_count == 1


def test_other_mcp_endpoint_never_receives_exa_key(exa_config):
    with patch.object(requests, "post", return_value=response(429)) as post:
        with pytest.raises(keyless_mcp.KeylessMCPError):
            keyless_mcp.mcp_call(keyless_mcp.PARALLEL_MCP_URL, "web_search", {"query": "q"})
    assert post.call_count == 1
    assert "x-api-key" not in post.call_args.kwargs["headers"]


def test_authenticated_error_redacts_echoed_key(exa_config, caplog):
    failure = response()
    failure._content = json.dumps({"error": {"code": -32000, "message": "bad test-only-exa-credential"}}).encode()
    with patch.object(requests, "post", side_effect=[response(429), failure]):
        result = keyless_mcp.exa_search_keyless("q", 1)
    assert not result["success"]
    assert "test-only-exa-credential" not in json.dumps(result) + caplog.text
    assert "[REDACTED]" in result["error"]


def test_anonymous_redirect_cannot_move_the_spending_trigger(exa_config):
    with patch.object(requests, "post", return_value=response(302)) as post:
        result = keyless_mcp.exa_search_keyless("q", 1)
    assert not result["success"]
    assert post.call_count == 1
    assert post.call_args.kwargs.get("allow_redirects") is False
