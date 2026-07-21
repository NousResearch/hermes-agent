"""Tests for the shared OpenRouter credit probe (agent/openrouter_credits.py).

Covers the four properties the PR #57321 review asked for: correct parsing /
thresholds, per-account cache scoping (no prior account's balance leaks),
failure handling, and the non-blocking background refresh.
"""

import threading
import time
from types import SimpleNamespace

from agent.openrouter_credits import (
    OpenRouterCreditsProbe,
    account_identity,
    parse_credits,
)


def _snap(balance_str="Credits balance: $42.50", quota_pct=None):
    windows = []
    if quota_pct is not None:
        windows.append(SimpleNamespace(label="API key quota", used_percent=quota_pct))
    return SimpleNamespace(details=[balance_str], windows=windows)


def _wait_for_balance(probe, provider, base_url, api_key, timeout=2.0):
    """Poll snapshot() until the background refresh lands (or time out)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = probe.snapshot(provider, base_url, api_key)
        if result:
            return result
        time.sleep(0.01)
    return {}


class TestParseCredits:
    def test_extracts_balance_and_label(self):
        balance, label, quota = parse_credits(_snap("Credits balance: $42.50"))
        assert balance == 42.5
        assert label == "$42.50"
        assert quota is None

    def test_thousands_formatting(self):
        balance, label, _ = parse_credits(_snap("Credits balance: $1234.5"))
        assert balance == 1234.5
        assert label == "$1,234.50"

    def test_quota_percent_from_window(self):
        _, _, quota = parse_credits(_snap(quota_pct=37.0))
        assert quota == 37.0

    def test_none_snapshot(self):
        assert parse_credits(None) == (None, None, None)

    def test_no_details(self):
        assert parse_credits(SimpleNamespace(details=[], windows=[])) == (None, None, None)

    def test_unparseable_balance(self):
        assert parse_credits(_snap("no dollar amount here")) == (None, None, None)


class TestAccountIdentity:
    def test_trailing_slash_normalized(self):
        a = account_identity("openrouter", "https://x/api/", "sk-secret")
        b = account_identity("openrouter", "https://x/api", "sk-secret")
        assert a == b

    def test_different_key_yields_different_identity(self):
        a = account_identity("openrouter", "https://x", "sk-a")
        b = account_identity("openrouter", "https://x", "sk-b")
        assert a != b

    def test_different_base_url_yields_different_identity(self):
        a = account_identity("openrouter", "https://x", "sk-a")
        b = account_identity("openrouter", "https://y", "sk-a")
        assert a != b

    def test_api_key_never_appears_in_cleartext(self):
        secret = "sk-super-secret-value-12345"
        identity = account_identity("openrouter", "https://x", secret)
        assert secret not in identity

    def test_missing_key(self):
        assert "no-key" in account_identity("openrouter", "https://x", None)


class TestProbeCaching:
    def test_refresh_now_returns_balance(self):
        calls = []

        def fetcher(provider, *, base_url, api_key):
            calls.append((provider, base_url, api_key))
            return _snap("Credits balance: $10.00")

        probe = OpenRouterCreditsProbe(fetcher=fetcher)
        result = probe.refresh_now("openrouter", "https://x", "sk-a")
        assert result["balance"] == 10.0
        assert result["label"] == "$10.00"
        assert len(calls) == 1

    def test_snapshot_is_non_blocking_then_caches(self):
        calls = []

        def fetcher(provider, *, base_url, api_key):
            calls.append(1)
            return _snap("Credits balance: $5.00")

        probe = OpenRouterCreditsProbe(ttl=300.0, fetcher=fetcher)
        # First call returns immediately with nothing (fetch runs in background).
        assert probe.snapshot("openrouter", "https://x", "sk-a") == {}
        # Once it lands, the value is cached and not refetched within the TTL.
        landed = _wait_for_balance(probe, "openrouter", "https://x", "sk-a")
        assert landed["balance"] == 5.0
        probe.snapshot("openrouter", "https://x", "sk-a")
        assert len(calls) == 1

    def test_account_switch_never_shows_prior_balance(self):
        def fetcher(provider, *, base_url, api_key):
            amount = "100.00" if api_key == "sk-a" else "2.00"
            return _snap(f"Credits balance: ${amount}")

        probe = OpenRouterCreditsProbe(fetcher=fetcher)
        probe.refresh_now("openrouter", "https://x", "sk-a")  # $100 cached for A
        # A snapshot for a DIFFERENT account must never surface A's balance,
        # even before B's own fetch has completed.
        immediate = probe.snapshot("openrouter", "https://x", "sk-b")
        assert immediate.get("balance") != 100.0
        landed = _wait_for_balance(probe, "openrouter", "https://x", "sk-b")
        assert landed["balance"] == 2.0
        # A's value is still independently cached (per-account, not clobbered).
        assert probe.snapshot("openrouter", "https://x", "sk-a")["balance"] == 100.0

    def test_fetch_failure_is_swallowed(self):
        def fetcher(provider, *, base_url, api_key):
            raise RuntimeError("network down")

        probe = OpenRouterCreditsProbe(fetcher=fetcher)
        assert probe.refresh_now("openrouter", "https://x", "sk-a") == {}

    def test_on_update_fires_after_successful_refresh(self):
        fired = threading.Event()
        probe = OpenRouterCreditsProbe(
            fetcher=lambda p, *, base_url, api_key: _snap("Credits balance: $9.00")
        )
        probe.snapshot("openrouter", "https://x", "sk-a", on_update=fired.set)
        assert fired.wait(2.0)
