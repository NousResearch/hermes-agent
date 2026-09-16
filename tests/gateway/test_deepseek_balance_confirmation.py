"""Messaging-gateway ``/model`` confirmation: DeepSeek's balance for the NEW route.

Telegram and the other platforms switch models through
``GatewayModelCommandsMixin._commit_model_switch`` → ``_model_switch_confirmation``, so that helper is
where the block lives. These tests drive the real chain (``account_usage_lines`` → DeepSeek fetcher →
scripted HTTP client), plus the off-loop doorway, so a slow or broken balance API can never block the
gateway event loop or break the switch.
"""

from types import SimpleNamespace

import httpx
import pytest

import gateway.slash_commands_model as model_cmd
from agent.i18n import t
from gateway.run import GatewayRunner
from gateway.slash_commands_model import _ModelSwitchContext

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


def _runner() -> GatewayRunner:
    return object.__new__(GatewayRunner)


def _ctx(**overrides) -> _ModelSwitchContext:
    ctx = _ModelSwitchContext(session_key="telegram:dm:1", source=None, config_path=None, persist_global=False)
    for key, value in overrides.items():
        setattr(ctx, key, value)
    return ctx


def _result(**overrides) -> SimpleNamespace:
    fields = dict(
        new_model="deepseek-v4-pro",
        target_provider="deepseek",
        provider_label="DeepSeek",
        base_url="https://api.deepseek.com/v1",
        api_key="sk-deepseek",
        api_mode="chat_completions",
        model_info=None,
        warning_message=None,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


class _Response:
    def __init__(self, payload=None, *, status_code=200):
        self._payload = payload
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
        return self._payload


class _RecordingClient:
    def __init__(self, response):
        self._response = response
        self.urls: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get(self, url, headers=None):
        self.urls.append(url)
        return self._response


class _ToThreadSpy:
    """Records what the offload doorway was asked to run."""

    def __init__(self):
        import asyncio

        self.calls = []
        self._real = asyncio.to_thread

    async def __call__(self, func, /, *args, **kwargs):
        self.calls.append((func, args, kwargs))
        return await self._real(func, *args, **kwargs)


@pytest.fixture(autouse=True)
def _offline_confirmation(monkeypatch):
    """Keep the confirmation hermetic: no config read, no models.dev context lookup."""
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda **kwargs: {})

    async def _no_context_length(*args, **kwargs):
        return None

    monkeypatch.setattr("hermes_cli.model_switch.resolve_display_context_length_async", _no_context_length)


def _wire_deepseek(monkeypatch, client):
    seen: list[tuple] = []

    def _resolve(requested=None, explicit_base_url=None, explicit_api_key=None):
        seen.append((requested, explicit_base_url, explicit_api_key))
        return {"provider": "deepseek", "base_url": explicit_base_url, "api_key": explicit_api_key}

    monkeypatch.setattr("agent.account_usage.resolve_runtime_provider", _resolve)
    monkeypatch.setattr("agent.account_usage.httpx.Client", lambda timeout=10.0: client)
    return seen


@pytest.mark.asyncio
async def test_confirmation_shows_the_deepseek_balance_fetched_off_loop(monkeypatch):
    spy = _ToThreadSpy()
    monkeypatch.setattr(model_cmd.asyncio, "to_thread", spy)
    client = _RecordingClient(_Response(_BALANCE))
    seen = _wire_deepseek(monkeypatch, client)

    reply = await _runner()._model_switch_confirmation(
        _result(),
        _ctx(current_base_url="https://openrouter.ai/api/v1", current_api_key="sk-old"),
        one_turn=False,
        picker=False,
    )

    # The block describes the route the pick moved TO, not the ambient one.
    assert seen == [("deepseek", "https://api.deepseek.com/v1", "sk-deepseek")]
    assert client.urls == ["https://api.deepseek.com/user/balance"]
    assert "**DeepSeek balance**" in reply
    assert "Balance: 110.00 CNY (granted 10.00 • topped up 100.00)" in reply
    # ...above the trailing scope hint, which stays last.
    assert reply.index("Balance: 110.00 CNY") < reply.index(t("gateway.model.session_only_hint"))
    assert any(getattr(call[0], "__name__", "") == "account_usage_lines" for call in spy.calls), (
        "the usage fetch must ride asyncio.to_thread — it is network I/O on the gateway loop"
    )


@pytest.mark.asyncio
async def test_confirmation_falls_back_to_the_ambient_route_when_the_pick_carries_no_credentials(monkeypatch):
    """A pick with no own base_url/api_key (pooled/OAuth route) reports the session's route."""
    client = _RecordingClient(_Response(_BALANCE))
    seen = _wire_deepseek(monkeypatch, client)

    reply = await _runner()._model_switch_confirmation(
        _result(base_url="", api_key=""),
        _ctx(current_base_url="https://api.deepseek.com/v1", current_api_key="sk-pooled"),
        one_turn=False,
        picker=False,
    )

    assert seen == [("deepseek", "https://api.deepseek.com/v1", "sk-pooled")]
    assert "Balance: 110.00 CNY" in reply


@pytest.mark.asyncio
async def test_confirmation_adds_nothing_for_a_route_without_a_limits_api(monkeypatch):
    """``ollama`` has no usage fetcher: the real chain contributes no lines and no divider."""
    def _unreachable(*args, **kwargs):
        raise AssertionError("a provider without a limits API must not be fetched")

    monkeypatch.setattr("agent.account_usage.httpx.Client", _unreachable)

    reply = await _runner()._model_switch_confirmation(
        _result(
            new_model="llama3.1:8b",
            target_provider="ollama",
            provider_label="Ollama",
            base_url="http://localhost:11434",
            api_key="",
        ),
        _ctx(),
        one_turn=False,
        picker=False,
    )

    assert "Balance" not in reply
    assert "\n\n" not in reply
    assert reply.endswith(t("gateway.model.session_only_hint"))


@pytest.mark.asyncio
async def test_confirmation_survives_a_deepseek_provider_outage(monkeypatch):
    _wire_deepseek(monkeypatch, _RecordingClient(_Response(None, status_code=503)))

    reply = await _runner()._model_switch_confirmation(_result(), _ctx(), one_turn=False, picker=False)

    assert "Balance" not in reply
    assert reply.endswith(t("gateway.model.session_only_hint"))
