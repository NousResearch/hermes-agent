"""Live Anthropic context windows: ``max_input_tokens`` from ``/v1/models``, OAuth included.

Bug: a Claude login (an ``sk-ant-oat*`` token) never reached the live lookup. The probe sent
``x-api-key`` only, which Anthropic answers with 401 for an OAuth token, so the probe skipped
OAuth outright. Every model whose family lacked a hardcoded row then fell to the generic
``"claude": 200000`` default. ``claude-mythos-5-2`` (1,000,000 on the live endpoint) was capped at
200K. An OAuth bearer plus the ``oauth-2025-04-20`` beta is accepted by ``/v1/models``, as the
model picker's own fetcher (``hermes_cli.models._fetch_anthropic_models``) already proves.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent import model_metadata as mm
from agent import model_metadata_anthropic as ma

OAUTH = "sk-ant-oat01-test-token"
API_KEY = "sk-ant-api03-test-key"
NATIVE = "https://api.anthropic.com"

CATALOG = {
    "data": [
        {"id": "claude-mythos-5-2", "max_input_tokens": 1_000_000, "max_tokens": 128_000},
        {"id": "claude-quasar-6-1", "max_input_tokens": 1_000_000, "max_tokens": 128_000},
        {"id": "claude-opus-4-5-20251101", "max_input_tokens": 200_000, "max_tokens": 64_000},
        {"id": "claude-no-limit-published"},
    ],
    "has_more": False,
}


@pytest.fixture(autouse=True)
def _clear_live_catalog_memo():
    ma._anthropic_context_limits.clear()
    yield
    ma._anthropic_context_limits.clear()


@pytest.fixture
def http(monkeypatch):
    """Record every probe request; answer with ``responses`` in order (the last one repeats)."""
    calls: list[dict] = []
    responses: list = [SimpleNamespace(status_code=200, json=lambda: CATALOG)]

    def fake_get(url, headers=None, timeout=None, verify=None):
        calls.append({"url": url, "headers": dict(headers or {})})
        response = responses[min(len(calls), len(responses)) - 1]
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(ma.model_metadata_http, "get", fake_get)
    return SimpleNamespace(calls=calls, responses=responses)


def test_oauth_token_reads_live_window_with_bearer_and_oauth_beta(http):
    assert ma.query_anthropic_context_length("claude-mythos-5-2", NATIVE, OAUTH) == 1_000_000

    (call,) = http.calls
    assert call["url"] == "https://api.anthropic.com/v1/models?limit=1000"
    assert call["headers"]["Authorization"] == f"Bearer {OAUTH}"
    assert "oauth-2025-04-20" in call["headers"]["anthropic-beta"].split(",")
    assert call["headers"]["anthropic-version"] == "2023-06-01"
    assert "x-api-key" not in call["headers"]


def test_api_key_still_uses_x_api_key(http):
    assert ma.query_anthropic_context_length("claude-opus-4-5-20251101", NATIVE, API_KEY) == 200_000

    (call,) = http.calls
    assert call["headers"]["x-api-key"] == API_KEY
    assert "Authorization" not in call["headers"]


def test_callable_key_source_is_never_minted(http):
    minted = []

    def key_cmd():
        minted.append(True)
        return API_KEY

    assert ma.query_anthropic_context_length("claude-mythos-5-2", NATIVE, key_cmd) is None
    assert minted == [] and http.calls == []


def test_missing_key_resolves_the_native_credential(http, monkeypatch):
    monkeypatch.setattr("agent.anthropic_credentials.resolve_anthropic_token", lambda **_: OAUTH)

    assert ma.query_anthropic_context_length("claude-mythos-5-2", NATIVE, "") == 1_000_000
    assert http.calls[0]["headers"]["Authorization"] == f"Bearer {OAUTH}"


def test_missing_key_never_resolves_a_credential_for_a_foreign_host(http, monkeypatch):
    # An API key, not an OAuth token: the OAuth host guard must not be what stops this request.
    monkeypatch.setattr("agent.anthropic_credentials.resolve_anthropic_token", lambda **_: API_KEY)

    assert ma.query_anthropic_context_length("claude-mythos-5-2", "https://relay.example.com", "") is None
    assert http.calls == []


def test_oauth_token_is_never_sent_to_a_foreign_host(http):
    assert ma.query_anthropic_context_length("claude-mythos-5-2", "https://relay.example.com", OAUTH) is None
    assert http.calls == []


def test_unlisted_or_unpublished_model_falls_through(http):
    assert ma.query_anthropic_context_length("claude-not-in-catalog", NATIVE, OAUTH) is None
    assert ma.query_anthropic_context_length("claude-no-limit-published", NATIVE, OAUTH) is None


def test_catalog_is_memoized_per_credential(http):
    assert ma.query_anthropic_context_length("claude-mythos-5-2", NATIVE, OAUTH) == 1_000_000
    assert ma.query_anthropic_context_length("claude-opus-4-5-20251101", NATIVE, OAUTH) == 200_000
    assert len(http.calls) == 1

    # A different credential is a different catalog (entitlements are per account).
    assert ma.query_anthropic_context_length("claude-mythos-5-2", NATIVE, API_KEY) == 1_000_000
    assert len(http.calls) == 2


def test_memo_expires_so_a_later_switch_reads_live_again(http, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(ma.time, "monotonic", lambda: clock[0])

    ma.query_anthropic_context_length("claude-mythos-5-2", NATIVE, OAUTH)
    clock[0] += ma._ANTHROPIC_CONTEXT_LIMITS_TTL - 1
    ma.query_anthropic_context_length("claude-mythos-5-2", NATIVE, OAUTH)
    assert len(http.calls) == 1

    clock[0] += 2
    ma.query_anthropic_context_length("claude-mythos-5-2", NATIVE, OAUTH)
    assert len(http.calls) == 2


def test_failure_is_retried_after_the_short_failure_window(http, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(ma.time, "monotonic", lambda: clock[0])
    http.responses[:] = [SimpleNamespace(status_code=401, json=dict),
                         SimpleNamespace(status_code=200, json=lambda: CATALOG)]

    assert ma.query_anthropic_context_length("claude-mythos-5-2", NATIVE, OAUTH) is None
    assert ma.query_anthropic_context_length("claude-mythos-5-2", NATIVE, OAUTH) is None
    assert len(http.calls) == 1  # a failing endpoint is not re-probed on every lookup

    clock[0] += ma._ANTHROPIC_CONTEXT_LIMITS_FAILURE_TTL + 1
    assert ma.query_anthropic_context_length("claude-mythos-5-2", NATIVE, OAUTH) == 1_000_000
    assert len(http.calls) == 2


def test_network_error_falls_through(http):
    http.responses[:] = [OSError("unreachable")]
    assert ma.query_anthropic_context_length("claude-mythos-5-2", NATIVE, OAUTH) is None


def test_get_model_context_length_uses_live_window_for_a_claude_login(http, monkeypatch):
    monkeypatch.setattr(mm, "get_cached_context_length", lambda *a, **k: None)
    # A family with no row of its own: without the live window it lands on the generic claude row.
    static = mm._longest_key_match(mm.DEFAULT_CONTEXT_LENGTHS, "claude-quasar-6-1")
    assert static is not None and static[1] == 200_000

    ctx = mm.get_model_context_length("claude-quasar-6-1", base_url=NATIVE, api_key=OAUTH, provider="anthropic")

    assert ctx == 1_000_000
