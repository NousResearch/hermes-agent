"""DeepSeek's account balance reaches /usage through its provider profile hook (``GET /user/balance``)."""

from __future__ import annotations

import httpx
import pytest

from agent.account_usage import AccountBalance, fetch_account_usage, render_account_usage_lines


@pytest.fixture(autouse=True)
def deepseek_profile():
    import model_tools  # registers the provider plugins
    import providers

    profile = providers.get_provider_profile("deepseek")
    assert profile is not None, "deepseek provider profile must be registered"
    return profile


class _Response:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def _serve(monkeypatch, *, payload=None, status_code=200, error=None, base_url="https://api.deepseek.com/v1"):
    """Fake the runtime credentials and the HTTP client; return the list of (url, headers) requested."""
    requests: list[tuple[str, dict]] = []

    class _Client:
        def __init__(self, timeout=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get(self, url, headers=None):
            requests.append((url, headers or {}))
            if error:
                raise error
            return _Response(payload, status_code)

    monkeypatch.setattr("httpx.Client", _Client)
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda requested, explicit_base_url=None, explicit_api_key=None: {
            "provider": "deepseek", "base_url": base_url, "api_key": "sk-test"},
    )
    return requests


def test_balance_in_each_currency_reaches_usage(monkeypatch):
    requests = _serve(monkeypatch, payload={"is_available": True, "balance_infos": [
        {"currency": "CNY", "total_balance": "110.00", "granted_balance": "10.00", "topped_up_balance": "100.00"},
        {"currency": "USD", "total_balance": "4.5", "granted_balance": "0", "topped_up_balance": "4.5"},
    ]})

    snapshot = fetch_account_usage("deepseek")

    assert requests[0][0] == "https://api.deepseek.com/user/balance"
    assert requests[0][1]["Authorization"] == "Bearer sk-test"
    assert snapshot is not None and snapshot.provider == "deepseek"
    assert snapshot.balances == (AccountBalance("Balance", 110.0, "CNY"), AccountBalance("Balance", 4.5, "USD"))
    assert render_account_usage_lines(snapshot)[2:] == ["Balance: ¥110.00", "Balance: $4.50"]


def test_an_insufficient_balance_says_so(monkeypatch):
    _serve(monkeypatch, payload={"is_available": False, "balance_infos": [
        {"currency": "CNY", "total_balance": "0.00", "granted_balance": "0.00", "topped_up_balance": "0.00"}]})

    snapshot = fetch_account_usage("deepseek")

    assert snapshot is not None
    assert snapshot.balances == (AccountBalance("Balance", 0.0, "CNY"),)
    assert render_account_usage_lines(snapshot)[2:] == [
        "Balance: ¥0.00", "Status: balance too low for API calls — top up to restore"]


@pytest.mark.parametrize("payload", [
    {"is_available": True, "balance_infos": "110"},
    {"is_available": True, "balance_infos": [{"currency": "CNY", "total_balance": "lots"}]},
    {"is_available": True, "balance_infos": [{"currency": "yuan", "total_balance": "1.00"}]},
    {"is_available": True},
])
def test_a_body_without_a_readable_balance_reports_nothing(monkeypatch, payload):
    _serve(monkeypatch, payload=payload)

    assert fetch_account_usage("deepseek") is None


def test_a_rejected_key_or_timeout_reports_nothing(monkeypatch):
    _serve(monkeypatch, payload={"error": {"message": "Authentication Fails"}}, status_code=401)
    assert fetch_account_usage("deepseek") is None

    _serve(monkeypatch, error=httpx.TimeoutException("timed out"))
    assert fetch_account_usage("deepseek") is None


def test_the_key_only_goes_to_the_configured_host(monkeypatch):
    """A DeepSeek slot pointed at a proxy asks that proxy, never api.deepseek.com, with the proxy's key."""
    requests = _serve(monkeypatch, payload={"is_available": True, "balance_infos": [
        {"currency": "USD", "total_balance": "1.00", "granted_balance": "0", "topped_up_balance": "1.00"}]},
        base_url="https://proxy.example/deepseek/v1/")

    fetch_account_usage("deepseek")

    assert [url for url, _ in requests] == ["https://proxy.example/deepseek/user/balance"]
