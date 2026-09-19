"""The /model switch confirmation must surface the NEW route's remaining quota / balance.

Telegram and the other messaging platforms switch models through
``GatewayModelCommandsMixin._commit_model_switch`` → ``_model_switch_confirmation``, so that
helper is where the block lives. Invariants pinned here:

  * the block describes the route the session switched TO (provider + credentials of the pick),
    not the ambient one — a stale-route balance is worse than none;
  * the fetch rides ``asyncio.to_thread`` (network I/O never blocks the gateway event loop);
  * a route with no limits API adds NOTHING (no dangling divider, no empty header).
"""

from types import SimpleNamespace

import pytest

import gateway.slash_commands_model as model_cmd
from agent.i18n import t
from gateway.run import GatewayRunner
from gateway.slash_commands_model import _ModelSwitchContext


def _runner() -> GatewayRunner:
    return object.__new__(GatewayRunner)


def _ctx(**overrides) -> _ModelSwitchContext:
    ctx = _ModelSwitchContext(session_key="telegram:dm:1", source=None, config_path=None, persist_global=False)
    for key, value in overrides.items():
        setattr(ctx, key, value)
    return ctx


def _result(**overrides) -> SimpleNamespace:
    fields = dict(
        new_model="deepseek-v4-pro", target_provider="deepseek", provider_label="DeepSeek",
        base_url="https://api.deepseek.com/v1", api_key="sk-deepseek", api_mode="chat_completions",
        model_info=None, warning_message=None,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


@pytest.fixture(autouse=True)
def _offline_confirmation(monkeypatch):
    """Keep the confirmation hermetic: no config read, no models.dev context lookup."""
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda **kwargs: {})

    async def _no_context_length(*args, **kwargs):
        return None

    monkeypatch.setattr("hermes_cli.model_switch.resolve_display_context_length_async", _no_context_length)


class _ToThreadSpy:
    """Records what the offload doorway was asked to run (mirrors test_model_command_async_offload)."""

    def __init__(self):
        import asyncio
        self.calls = []
        self._real = asyncio.to_thread

    async def __call__(self, func, /, *args, **kwargs):
        self.calls.append((func, args, kwargs))
        return await self._real(func, *args, **kwargs)


@pytest.mark.asyncio
async def test_confirmation_shows_the_new_routes_quota_fetched_off_loop(monkeypatch):
    spy = _ToThreadSpy()
    monkeypatch.setattr(model_cmd.asyncio, "to_thread", spy)
    seen = {}

    def _lines(provider, **kwargs):
        seen.update(provider=provider, **kwargs)
        return ["📈 **DeepSeek balance**", "Provider: deepseek", "Balance: 110.00 CNY"]

    monkeypatch.setattr("agent.account_usage.account_usage_lines", _lines)

    reply = await _runner()._model_switch_confirmation(
        _result(), _ctx(current_base_url="https://openrouter.ai/api/v1", current_api_key="sk-old"),
        one_turn=False, picker=False,
    )

    # The block is the balance of the route the pick just moved to.
    assert seen["provider"] == "deepseek"
    assert seen["base_url"] == "https://api.deepseek.com/v1"
    assert seen["api_key"] == "sk-deepseek"
    assert seen["markdown"] is True
    assert "Balance: 110.00 CNY" in reply
    # ...and it sits above the trailing scope hint (footer stays last).
    assert reply.index("Balance: 110.00 CNY") < reply.index(t("gateway.model.session_only_hint"))
    assert _lines in [call[0] for call in spy.calls], (
        "the usage fetch must ride asyncio.to_thread — it is network I/O on the gateway loop"
    )


@pytest.mark.asyncio
async def test_confirmation_falls_back_to_the_ambient_route_when_the_pick_carries_no_credentials(monkeypatch):
    """A pick with no own base_url/api_key (a pooled/OAuth route) reports the session's route."""
    seen = {}

    def _lines(provider, **kwargs):
        seen.update(provider=provider, **kwargs)
        return []

    monkeypatch.setattr("agent.account_usage.account_usage_lines", _lines)

    await _runner()._model_switch_confirmation(
        _result(target_provider="anthropic", base_url="", api_key=""), 
        _ctx(current_base_url="https://api.anthropic.com", current_api_key="sk-ambient"),
        one_turn=False, picker=False,
    )

    assert seen["provider"] == "anthropic"
    assert seen["base_url"] == "https://api.anthropic.com"
    assert seen["api_key"] == "sk-ambient"


@pytest.mark.asyncio
async def test_confirmation_adds_nothing_for_a_route_without_a_limits_api():
    """``ollama`` has no usage fetcher: the real chain must contribute no lines and no divider."""
    reply = await _runner()._model_switch_confirmation(
        _result(new_model="llama3.1:8b", target_provider="ollama", provider_label="Ollama",
                base_url="http://localhost:11434", api_key=""),
        _ctx(), one_turn=False, picker=False,
    )

    assert "Balance" not in reply
    assert "\n\n" not in reply
    assert reply.endswith(t("gateway.model.session_only_hint"))
