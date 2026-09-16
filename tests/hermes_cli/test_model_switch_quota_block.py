"""CLI ``/model`` switch summary: the quota/balance block belongs to the NEW route.

Regression guard for the stale-balance shape of the bug: a switch summary that reuses the live
session's provider (or its credentials) prints the route the user just LEFT. The block is fetched
from ``result.target_provider`` with the pick's own endpoint/credentials, and a fetch that fails
degrades to an ``Unavailable:`` line instead of breaking the summary.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

import cli as cli_mod
from agent import account_usage
from agent.account_usage import AccountUsageSnapshot
from hermes_cli import cli_model_switch_mixin as mixin
from hermes_cli.model_switch import ModelSwitchResult


def _snapshot(provider: str, amount: str) -> AccountUsageSnapshot:
    return AccountUsageSnapshot(
        provider=provider, source="test", fetched_at=datetime.now(timezone.utc),
        title=f"{provider} balance", details=(f"Balance: {amount}",))


def _register(monkeypatch, slug: str, fetcher) -> None:
    monkeypatch.setitem(account_usage._USAGE_FETCHERS, slug, fetcher)


@pytest.fixture
def printed(monkeypatch):
    """Capture ``_cprint`` output; context resolution is stubbed (it can hit the network)."""
    lines: list[str] = []
    monkeypatch.setattr(cli_mod, "_cprint", lambda text="", *a, **kw: lines.append(text), raising=False)
    monkeypatch.setattr("hermes_cli.model_switch.resolve_display_context_length",
                        lambda *a, **kw: None, raising=False)
    return lines


def _cli(**kw) -> SimpleNamespace:
    """Bare stub in the shape the mixin's module-level helpers expect (``cli.agent`` may be None)."""
    fields = {
        "agent": None, "model": "old-model", "provider": "route-a", "requested_provider": "route-a",
        "base_url": "https://a.example/v1", "api_key": "key-a", "api_mode": "",
        "_pending_model_switch_note": "", "_pending_one_turn_model_restore": None,
    }
    fields.update(kw)
    return SimpleNamespace(**fields)


def _summary(printed, result, cli=None) -> str:
    mixin._print_switch_summary(
        cli if cli is not None else _cli(), result, "old-model", one_turn=False, strict_context=True)
    return "\n".join(printed)


def test_block_reports_the_route_being_switched_to(monkeypatch, printed):
    """The pick's route and credentials win — the ambient (previous) route's balance never prints."""
    _register(monkeypatch, "route-a", lambda _b, _k: _snapshot("route-a", "12.00 USD"))
    _register(monkeypatch, "route-b", lambda _b, _k: _snapshot("route-b", "41.00 USD"))
    result = ModelSwitchResult(
        success=True, new_model="b-model", target_provider="route-b",
        api_key="key-b", base_url="https://b.example/v1", provider_changed=True)

    text = _summary(printed, result)

    assert "Balance: 41.00 USD" in text
    assert "12.00 USD" not in text, "the previous route's balance leaked into the new route's block"
    assert "Provider: route-b" in text
    # The rest of the summary is untouched.
    assert "  ✓ Model switched: b-model" in text


def test_credentials_reach_the_target_fetcher(monkeypatch, printed):
    seen = {}

    def _fetcher(base_url, api_key):
        seen["base_url"], seen["api_key"] = base_url, api_key
        return _snapshot("route-b", "41.00 USD")

    _register(monkeypatch, "route-b", _fetcher)
    result = ModelSwitchResult(success=True, new_model="b-model", target_provider="route-b",
                               api_key="key-b", base_url="https://b.example/v1")

    _summary(printed, result)

    assert seen == {"base_url": "https://b.example/v1", "api_key": "key-b"}


def test_credential_less_pick_falls_back_to_the_live_route(monkeypatch, printed):
    """A pick that carries no endpoint/secret of its own resolves against the live route."""
    seen = {}

    def _fetcher(base_url, api_key):
        seen["base_url"], seen["api_key"] = base_url, api_key
        return _snapshot("route-a", "9.00 USD")

    _register(monkeypatch, "route-a", _fetcher)
    result = ModelSwitchResult(success=True, new_model="a-model", target_provider="route-a")

    _summary(printed, result)

    assert seen == {"base_url": "https://a.example/v1", "api_key": "key-a"}


def test_failed_fetch_prints_unavailable_and_keeps_the_summary(monkeypatch, printed):
    def _boom(_base_url, _api_key):
        raise RuntimeError("balance API down")

    _register(monkeypatch, "route-b", _boom)
    result = ModelSwitchResult(success=True, new_model="b-model", target_provider="route-b",
                               api_key="key-b", base_url="https://b.example/v1")

    text = _summary(printed, result)

    assert "Unavailable:" in text
    assert "  ✓ Model switched: b-model" in text
    assert "Provider: route-b" in text


def test_unfetchable_provider_adds_no_block_for_either_route(monkeypatch, printed):
    """Nothing is claimed about a route with no fetcher — and the previous route stays absent."""
    _register(monkeypatch, "route-a", lambda _b, _k: _snapshot("route-a", "12.00 USD"))
    result = ModelSwitchResult(success=True, new_model="local", target_provider="ollama",
                               base_url="http://localhost:11434/v1")

    text = _summary(printed, result)

    assert "📈" not in text and "Unavailable" not in text and "12.00 USD" not in text
    assert "  ✓ Model switched: local" in text


def test_a_broken_account_block_never_breaks_the_summary(monkeypatch, printed):
    monkeypatch.setattr("agent.account_usage.account_usage_lines",
                        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom")), raising=True)
    result = ModelSwitchResult(success=True, new_model="m", target_provider="openrouter",
                               api_key="or-key", base_url="https://openrouter.ai/api/v1")

    text = _summary(printed, result)

    assert "  ✓ Model switched: m" in text


def test_picker_path_prints_the_same_block(monkeypatch, printed):
    """The picker path commits through the same summariser, so it refreshes the same way."""
    _register(monkeypatch, "route-b", lambda _b, _k: _snapshot("route-b", "41.00 USD"))
    result = ModelSwitchResult(success=True, new_model="b-model", target_provider="route-b",
                               api_key="key-b", base_url="https://b.example/v1")

    text = _summary(printed, result, cli=_cli(agent=None))

    assert "Balance: 41.00 USD" in text
