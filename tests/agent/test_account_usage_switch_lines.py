"""``account_usage_lines`` — the bounded, fail-open block a model switch shows for its NEW route.

A switch confirmation must (a) describe the route it was handed, (b) never raise and never hang
the confirmation, and (c) say ``Unavailable`` rather than render nothing when the fetch cannot
answer — a silently missing block reads as "the previous route's numbers still apply", which is
the stale-balance bug this exists to remove.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import pytest

from agent import account_usage
from agent.account_usage import (
    AccountUsageSnapshot,
    AccountUsageWindow,
    account_usage_lines,
    render_account_usage_lines,
)


def _snapshot(provider: str, amount: str, **kw) -> AccountUsageSnapshot:
    window = AccountUsageWindow(label="API key quota", used_percent=25.0)
    return AccountUsageSnapshot(
        provider=provider, source="test", fetched_at=datetime.now(timezone.utc),
        title=f"{provider} balance", windows=(window,), details=(f"Balance: {amount}",), **kw)


def _register(monkeypatch, slug: str, fetcher) -> None:
    monkeypatch.setitem(account_usage._USAGE_FETCHERS, slug, fetcher)


def test_provider_without_a_fetcher_renders_nothing(monkeypatch):
    """No fetcher means no claim: a custom/Ollama route must not get an "Unavailable" nag."""
    assert account_usage_lines("") == []
    assert account_usage_lines(None) == []
    assert account_usage_lines("ollama") == []
    assert account_usage_lines("some-custom-endpoint") == []


def test_renders_the_route_it_was_handed(monkeypatch):
    _register(monkeypatch, "route-a", lambda _b, _k: _snapshot("route-a", "12.00 USD"))

    lines = account_usage_lines("route-a", base_url="https://a.example/v1", api_key="key-a")

    assert lines == render_account_usage_lines(_snapshot("route-a", "12.00 USD"))
    assert any("Balance: 12.00 USD" in line for line in lines)
    # The provider slug is part of the block, so a stale header is visible even at a glance.
    assert "Provider: route-a" in lines


def test_fetcher_credentials_are_passed_through_verbatim(monkeypatch):
    seen = {}

    def _fetcher(base_url, api_key):
        seen["base_url"], seen["api_key"] = base_url, api_key
        return _snapshot("route-b", "3.00 USD")

    _register(monkeypatch, "route-b", _fetcher)

    account_usage_lines("route-b", base_url="https://b.example/v1", api_key="key-b")

    assert seen == {"base_url": "https://b.example/v1", "api_key": "key-b"}


def test_provider_slug_is_normalized_before_dispatch(monkeypatch):
    _register(monkeypatch, "route-c", lambda _b, _k: _snapshot("route-c", "5.00 USD"))
    assert any("Balance: 5.00 USD" in line for line in account_usage_lines("  ROUTE-C "))


def test_raising_fetch_renders_unavailable(monkeypatch):
    def _boom(_base_url, _api_key):
        raise RuntimeError("provider API exploded")

    _register(monkeypatch, "route-d", _boom)

    lines = account_usage_lines("route-d")

    assert any(line.startswith("Unavailable:") for line in lines), lines
    assert any("route-d" in line for line in lines)


def test_empty_fetch_renders_unavailable(monkeypatch):
    """A fetcher with no credentials (or no data) returns None — say so, don't print nothing."""
    _register(monkeypatch, "route-e", lambda _b, _k: None)

    lines = account_usage_lines("route-e")

    assert any(line.startswith("Unavailable:") for line in lines), lines


def test_contentless_snapshot_renders_unavailable(monkeypatch):
    """A snapshot with no windows/details would render as a bare header — report that instead."""
    _register(monkeypatch, "route-f", lambda _b, _k: AccountUsageSnapshot(
        provider="route-f", source="test", fetched_at=datetime.now(timezone.utc)))

    lines = account_usage_lines("route-f")

    assert any(line.startswith("Unavailable:") for line in lines), lines


def test_fetchers_own_unavailable_reason_wins(monkeypatch):
    """A fetcher that knows WHY it has nothing (e.g. non-OAuth Claude) keeps its explanation."""
    reason = "Anthropic account limits are only available for OAuth-backed Claude accounts."
    _register(monkeypatch, "route-g", lambda _b, _k: AccountUsageSnapshot(
        provider="route-g", source="test", fetched_at=datetime.now(timezone.utc),
        unavailable_reason=reason))

    lines = account_usage_lines("route-g")

    assert f"Unavailable: {reason}" in lines, lines


@pytest.mark.parametrize("timeout", [0.2, 0.5])
def test_hung_fetch_is_bounded_and_reported(monkeypatch, timeout):
    """A provider usage API that never answers costs the switch at most ``timeout`` seconds."""
    _register(monkeypatch, "route-h", lambda _b, _k: time.sleep(30))

    started = time.monotonic()
    lines = account_usage_lines("route-h", timeout=timeout)
    elapsed = time.monotonic() - started

    assert elapsed < 3.0, f"bounded fetch took {elapsed:.2f}s for a {timeout}s budget"
    assert any(line.startswith("Unavailable:") for line in lines), lines


def test_markdown_rendering(monkeypatch):
    _register(monkeypatch, "route-i", lambda _b, _k: _snapshot("route-i", "7.00 USD"))

    lines = account_usage_lines("route-i", markdown=True)

    assert lines[0] == "📈 **route-i balance**", lines
