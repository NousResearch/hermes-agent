"""Messaging-gateway ``/model`` confirmation: the quota/balance block is the NEW route's.

Two properties matter and are pinned here: the block is built from the pick's own
provider/endpoint/credentials (never the ambient route the user just left), and the fetch is
offloaded off the event loop and fail-open so a slow or broken provider usage API can neither
stall the gateway nor break the confirmation.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from agent import account_usage
from agent.account_usage import AccountUsageSnapshot
from gateway.slash_commands_model import GatewayModelCommandsMixin, _ModelSwitchContext
from hermes_cli.model_switch import ModelSwitchResult


def _snapshot(provider: str, amount: str) -> AccountUsageSnapshot:
    return AccountUsageSnapshot(
        provider=provider, source="test", fetched_at=datetime.now(timezone.utc),
        title=f"{provider} balance", details=(f"Balance: {amount}",))


def _register(monkeypatch, slug: str, fetcher) -> None:
    monkeypatch.setitem(account_usage._USAGE_FETCHERS, slug, fetcher)


@pytest.fixture
def offloaded(monkeypatch):
    """Record every ``asyncio.to_thread`` hop (and keep running the real one)."""
    calls: list[tuple] = []
    real = asyncio.to_thread

    async def _spy(func, *args, **kwargs):
        calls.append((func, args, kwargs))
        return await real(func, *args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", _spy)
    monkeypatch.setattr("hermes_cli.model_switch.resolve_display_context_length_async",
                        _no_context_length, raising=False)
    return calls


async def _no_context_length(*_args, **_kwargs):
    return None

def _ctx(**kw) -> _ModelSwitchContext:
    fields = dict(session_key="telegram:c1", source=None, config_path=None, persist_global=False,
                  current_model="old-model", current_provider="route-a",
                  current_base_url="https://a.example/v1", current_api_key="key-a")
    fields.update(kw)
    return _ModelSwitchContext(**fields)


async def _confirm(result, ctx) -> str:
    return await GatewayModelCommandsMixin._model_switch_confirmation(
        SimpleNamespace(), result, ctx, one_turn=False, picker=False)


@pytest.mark.asyncio
async def test_confirmation_shows_the_new_routes_balance(monkeypatch, offloaded):
    _register(monkeypatch, "route-a", lambda _b, _k: _snapshot("route-a", "12.00 USD"))
    _register(monkeypatch, "route-b", lambda _b, _k: _snapshot("route-b", "41.00 USD"))
    result = ModelSwitchResult(success=True, new_model="b-model", target_provider="route-b",
                               api_key="key-b", base_url="https://b.example/v1")

    reply = await _confirm(result, _ctx())

    assert "Balance: 41.00 USD" in reply
    assert "12.00 USD" not in reply, "the previous route's balance leaked into the confirmation"
    assert "Provider: route-b" in reply
    # The block is a block: separated from the summary and ahead of the persistence footer.
    assert "\n\n📈 **route-b balance**" in reply
    assert reply.index("Balance: 41.00 USD") < reply.index("session only")


@pytest.mark.asyncio
async def test_fetch_is_offloaded_with_the_picks_credentials(monkeypatch, offloaded):
    """Off the event loop, with the target route's own base_url/api_key and markdown rendering."""
    _register(monkeypatch, "route-b", lambda _b, _k: _snapshot("route-b", "41.00 USD"))
    result = ModelSwitchResult(success=True, new_model="b-model", target_provider="route-b",
                               api_key="key-b", base_url="https://b.example/v1")

    await _confirm(result, _ctx())

    func, args, kwargs = offloaded[0]
    assert func is account_usage.account_usage_lines
    assert args == ("route-b",)
    assert kwargs == {"base_url": "https://b.example/v1", "api_key": "key-b", "markdown": True}


@pytest.mark.asyncio
async def test_credential_less_pick_falls_back_to_the_current_route(monkeypatch, offloaded):
    _register(monkeypatch, "route-a", lambda _b, _k: _snapshot("route-a", "9.00 USD"))
    result = ModelSwitchResult(success=True, new_model="a-model", target_provider="route-a")

    reply = await _confirm(result, _ctx())

    assert offloaded[0][2] == {"base_url": "https://a.example/v1", "api_key": "key-a",
                               "markdown": True}
    assert "Balance: 9.00 USD" in reply


@pytest.mark.asyncio
async def test_provider_without_a_fetcher_adds_nothing(monkeypatch, offloaded):
    _register(monkeypatch, "route-a", lambda _b, _k: _snapshot("route-a", "12.00 USD"))
    result = ModelSwitchResult(success=True, new_model="local", target_provider="ollama",
                               base_url="http://localhost:11434/v1")

    reply = await _confirm(result, _ctx())

    assert "📈" not in reply and "Unavailable" not in reply and "12.00 USD" not in reply
    assert reply.count("\n\n") == 0, "an empty block must not leave a stray blank separator"


@pytest.mark.asyncio
async def test_failed_fetch_still_confirms_the_switch(monkeypatch, offloaded):
    def _boom(_base_url, _api_key):
        raise RuntimeError("balance API down")

    _register(monkeypatch, "route-b", _boom)
    result = ModelSwitchResult(success=True, new_model="b-model", target_provider="route-b",
                               api_key="key-b", base_url="https://b.example/v1")

    reply = await _confirm(result, _ctx())

    assert "Unavailable:" in reply
    assert "b-model" in reply


@pytest.mark.asyncio
async def test_a_broken_account_block_never_breaks_the_confirmation(monkeypatch, offloaded):
    monkeypatch.setattr("agent.account_usage.account_usage_lines",
                        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom")), raising=True)
    result = ModelSwitchResult(success=True, new_model="b-model", target_provider="route-b")

    reply = await _confirm(result, _ctx())

    assert "b-model" in reply
    assert "📈" not in reply
