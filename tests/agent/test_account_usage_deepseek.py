"""DeepSeek balance support in ``agent.account_usage``.

DeepSeek exposes a prepaid balance on the API ROOT (``GET /user/balance``) while the configured
base_url carries a route suffix (``https://api.deepseek.com/v1``), and every amount comes back as a
decimal STRING. Rendering it is fail-open: a rejected key is reported, any other failure renders the
shared ``Unavailable`` line on the switch surfaces (and stays silent on ``/usage``, which opts out via
``render_unavailable=False``), and a hung provider API can never stall the caller.
"""

import json
import time

import httpx
import pytest

from agent.account_usage import (
    _DEPLETED_LINE,
    _USAGE_FETCHERS,
    _deepseek_amount,
    _deepseek_balance_url,
    _fetch_deepseek_account_usage,
    _snapshot,
    account_usage_lines,
    fetch_account_usage,
    render_account_usage_lines,
)

_BALANCE = {
    "is_available": True,
    "balance_infos": [
        {
            "currency": "CNY",
            "total_balance": "110.00",
            "granted_balance": "10.00",
            "topped_up_balance": "100.00",
        }
    ],
}


class _Response:
    """Minimal ``httpx.Response`` stand-in; a 4xx/5xx raises a real ``HTTPStatusError``."""

    def __init__(self, payload=None, *, status_code=200, json_error=None):
        self._payload = payload
        self._json_error = json_error
        self.status_code = status_code
        self.request = httpx.Request("GET", "https://api.deepseek.com/user/balance")

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=self.request,
                response=httpx.Response(self.status_code, request=self.request),
            )

    def json(self):
        if self._json_error is not None:
            raise self._json_error
        return self._payload


class _RecordingClient:
    """``httpx.Client`` stand-in: records every GET and replays one scripted outcome."""

    def __init__(self, response=None, *, error=None):
        self._response = response
        self._error = error
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get(self, url, headers=None):
        self.calls.append({"url": url, "headers": headers or {}})
        if self._error is not None:
            raise self._error
        return self._response

    @property
    def urls(self):
        return [call["url"] for call in self.calls]


def _wire(monkeypatch, client, *, base_url="https://api.deepseek.com/v1", api_key="sk-deepseek-test"):
    """Point the DeepSeek fetcher at a scripted client + runtime resolver."""
    monkeypatch.setattr(
        "agent.account_usage.resolve_runtime_provider",
        lambda requested=None, explicit_base_url=None, explicit_api_key=None: {
            "provider": "deepseek", "base_url": base_url, "api_key": api_key,
        },
    )
    monkeypatch.setattr("agent.account_usage.httpx.Client", lambda timeout=10.0: client)
    return client


# --- endpoint ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        ("https://api.deepseek.com/v1", "https://api.deepseek.com/user/balance"),
        ("https://api.deepseek.com", "https://api.deepseek.com/user/balance"),
        ("https://api.deepseek.com/v1/", "https://api.deepseek.com/user/balance"),
        ("https://api.deepseek.com/beta", "https://api.deepseek.com/user/balance"),
        ("https://api.deepseek.com/anthropic", "https://api.deepseek.com/user/balance"),
        ("https://api.deepseek.com/v1?trace=1", "https://api.deepseek.com/user/balance"),
        ("api.deepseek.com/v1", "https://api.deepseek.com/user/balance"),
        ("  https://api.deepseek.com/v1  ", "https://api.deepseek.com/user/balance"),
        (None, "https://api.deepseek.com/user/balance"),
        ("", "https://api.deepseek.com/user/balance"),
    ],
)
def test_deepseek_balance_url_is_rebuilt_from_the_api_root(base_url, expected):
    """``{base_url}/user/balance`` 404s: the route suffix must not leak into the balance request."""
    assert _deepseek_balance_url(base_url) == expected


def test_deepseek_balance_url_keeps_a_self_hosted_origin():
    assert _deepseek_balance_url("https://proxy.internal:8443/v1") == "https://proxy.internal:8443/user/balance"
    assert _deepseek_balance_url("ftp://api.deepseek.com/v1") == "https://api.deepseek.com/user/balance"


# --- amount parsing ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("110.00", "110.00"),
        ("0", "0.00"),
        ("12.5", "12.50"),
        ("1000000", "1,000,000.00"),
        ("1,234.5", "1,234.5"),  # unparseable → shown verbatim, never silently dropped
        ("not-a-number", "not-a-number"),
        (7, "7.00"),
        (0.5, "0.50"),
        (None, None),
        ("", None),
        ("   ", None),
        (True, None),
        (float("nan"), None),
        (float("inf"), None),
    ],
)
def test_deepseek_amount_formats_decimal_strings(raw, expected):
    assert _deepseek_amount(raw) == expected


# --- fetching ------------------------------------------------------------------------------


def test_fetch_account_usage_deepseek_reads_the_balance_off_the_api_root(monkeypatch):
    client = _wire(monkeypatch, _RecordingClient(_Response(_BALANCE)))

    snapshot = fetch_account_usage("deepseek")

    assert client.urls == ["https://api.deepseek.com/user/balance"]
    assert client.calls[0]["headers"]["Authorization"] == "Bearer sk-deepseek-test"
    assert client.calls[0]["headers"]["Accept"] == "application/json"
    assert snapshot is not None
    assert snapshot.provider == "deepseek"
    assert snapshot.title == "DeepSeek balance"
    assert snapshot.windows == ()  # prepaid balance: no quota window
    assert snapshot.details == ("Balance: 110.00 CNY (granted 10.00 • topped up 100.00)",)
    assert render_account_usage_lines(snapshot)[:2] == ["📈 DeepSeek balance", "Provider: deepseek"]


def test_fetch_account_usage_dispatches_to_deepseek_for_any_slug_case(monkeypatch):
    assert _USAGE_FETCHERS["deepseek"] is _fetch_deepseek_account_usage
    _wire(monkeypatch, _RecordingClient(_Response(_BALANCE)))

    assert fetch_account_usage("DeepSeek").provider == "deepseek"
    assert fetch_account_usage(" deepseek ").provider == "deepseek"


def test_fetch_account_usage_deepseek_renders_every_currency(monkeypatch):
    _wire(monkeypatch, _RecordingClient(_Response({
        "is_available": True,
        "balance_infos": [
            {"currency": "CNY", "total_balance": "110.00", "granted_balance": "10.00",
             "topped_up_balance": "100.00"},
            {"currency": "USD", "total_balance": "3.50", "granted_balance": "0.00",
             "topped_up_balance": "3.50"},
        ],
    })))

    assert fetch_account_usage("deepseek").details == (
        "Balance: 110.00 CNY (granted 10.00 • topped up 100.00)",
        "Balance: 3.50 USD (topped up 3.50)",
    )


def test_fetch_account_usage_deepseek_flags_a_depleted_account(monkeypatch):
    _wire(monkeypatch, _RecordingClient(_Response({
        "is_available": False,
        "balance_infos": [{"currency": "USD", "total_balance": "0.00", "granted_balance": "0.00",
                           "topped_up_balance": "0.00"}],
    })))

    snapshot = fetch_account_usage("deepseek")

    assert snapshot.details == ("Balance: 0.00 USD", _DEPLETED_LINE)
    assert snapshot.available


@pytest.mark.parametrize(
    "payload",
    [
        {"is_available": True, "balance_infos": []},
        {"is_available": True},
        {"is_available": True, "balance_infos": None},
        {"is_available": True, "balance_infos": [None, "nope", {}, {"currency": "CNY"}]},
        "unexpected-non-object",
    ],
)
def test_fetch_account_usage_deepseek_returns_nothing_to_show(monkeypatch, payload):
    """A header with no numbers under it is worse than no block at all."""
    _wire(monkeypatch, _RecordingClient(_Response(payload)))

    assert fetch_account_usage("deepseek") is None


# --- error cases ---------------------------------------------------------------------------


def test_fetch_account_usage_deepseek_makes_no_request_without_a_key(monkeypatch):
    client = _wire(monkeypatch, _RecordingClient(_Response(_BALANCE)), api_key="")

    assert fetch_account_usage("deepseek") is None
    assert client.calls == []


def test_fetch_account_usage_deepseek_treats_a_blank_key_as_missing(monkeypatch):
    client = _wire(monkeypatch, _RecordingClient(_Response(_BALANCE)), api_key="   ")

    assert fetch_account_usage("deepseek") is None
    assert client.calls == []


@pytest.mark.parametrize("status", [400, 402, 429, 500, 503])
def test_fetch_account_usage_deepseek_degrades_on_a_provider_error(monkeypatch, status):
    """Rate limits and provider faults add nothing — and must never raise into the UI."""
    _wire(monkeypatch, _RecordingClient(_Response(None, status_code=status)))

    assert fetch_account_usage("deepseek") is None


@pytest.mark.parametrize("status", [401, 403])
def test_fetch_account_usage_deepseek_reports_a_rejected_key(monkeypatch, status):
    """A rejected key is durable: silence would read as an empty account."""
    _wire(monkeypatch, _RecordingClient(_Response(None, status_code=status)))

    snapshot = fetch_account_usage("deepseek")

    assert snapshot is not None
    assert snapshot.available is False
    assert f"HTTP {status}" in snapshot.unavailable_reason
    rendered = render_account_usage_lines(snapshot)
    assert rendered[0] == "📈 DeepSeek balance"
    assert any(line.startswith("Unavailable:") and "DEEPSEEK_API_KEY" in line for line in rendered)


def test_fetch_account_usage_deepseek_survives_malformed_json(monkeypatch):
    _wire(monkeypatch, _RecordingClient(_Response(json_error=json.JSONDecodeError("bad", "{", 0))))

    assert fetch_account_usage("deepseek") is None


@pytest.mark.parametrize(
    "error", [httpx.ConnectTimeout("timed out"), httpx.ConnectError("connection refused")]
)
def test_fetch_account_usage_deepseek_survives_a_network_failure(monkeypatch, error):
    _wire(monkeypatch, _RecordingClient(error=error))

    assert fetch_account_usage("deepseek") is None


# --- shared display entry point ------------------------------------------------------------


def test_account_usage_lines_renders_the_deepseek_block(monkeypatch):
    _wire(monkeypatch, _RecordingClient(_Response(_BALANCE)))

    assert account_usage_lines("deepseek", markdown=True) == [
        "📈 **DeepSeek balance**",
        "Provider: deepseek",
        "Balance: 110.00 CNY (granted 10.00 • topped up 100.00)",
    ]
    assert account_usage_lines("deepseek")[0] == "📈 DeepSeek balance"


def test_account_usage_lines_is_empty_for_a_provider_without_a_limits_api(monkeypatch):
    requested = []
    monkeypatch.setattr(
        "agent.account_usage.httpx.Client", lambda timeout=10.0: requested.append(timeout)
    )

    assert account_usage_lines("ollama") == []
    assert requested == []


@pytest.mark.parametrize("provider", [None, "", "   "])
def test_account_usage_lines_is_empty_without_a_provider(monkeypatch, provider):
    def _unreachable(*args, **kwargs):
        raise AssertionError("must not fetch account usage without a provider")

    monkeypatch.setattr("agent.account_usage.fetch_account_usage", _unreachable)

    assert account_usage_lines(provider) == []


def test_account_usage_lines_bounds_a_hung_provider(monkeypatch):
    """Display surfaces render inline: a wedged provider API costs the timeout, not the fetch."""

    def _hang(provider, **kwargs):
        time.sleep(30)

    monkeypatch.setattr("agent.account_usage.fetch_account_usage", _hang)

    started = time.monotonic()
    lines = account_usage_lines("deepseek", timeout=0.2)
    elapsed = time.monotonic() - started

    # Reconciled contract (t_61c70482): a switch confirmation must not look like it silently kept
    # the previous route's numbers, so the failure is rendered. ``/usage`` asks for silence
    # instead via ``render_unavailable=False`` — same fetch, same bound, no fabricated line.
    assert any("Unavailable: could not read the deepseek account balance" in line for line in lines)
    assert elapsed < 2.0, f"bounded fetch waited {elapsed:.1f}s on a hung provider API"

    started = time.monotonic()
    assert account_usage_lines("deepseek", timeout=0.2, render_unavailable=False) == []
    assert time.monotonic() - started < 2.0


def test_account_usage_lines_fails_open_when_the_fetch_raises(monkeypatch):
    def _boom(provider, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("agent.account_usage.fetch_account_usage", _boom)

    lines = account_usage_lines("deepseek")

    assert any("Unavailable: could not read the deepseek account balance" in line for line in lines)
    assert account_usage_lines("deepseek", render_unavailable=False) == []


def test_render_unavailable_false_still_shows_a_fetcher_supplied_reason(monkeypatch):
    """A durable reason the fetcher itself reported (e.g. a rejected key) renders on every surface —
    ``render_unavailable=False`` suppresses only the fabricated transport-failure line."""
    monkeypatch.setattr(
        "agent.account_usage.fetch_account_usage",
        lambda provider, **kwargs: _snapshot(
            "deepseek", "balance_api", [], [], title="DeepSeek balance",
            unavailable_reason="DeepSeek rejected the request (HTTP 401) — check DEEPSEEK_API_KEY."))

    lines = account_usage_lines("deepseek", render_unavailable=False)

    assert any("HTTP 401" in line for line in lines)
    assert account_usage_lines("deepseek") == lines
