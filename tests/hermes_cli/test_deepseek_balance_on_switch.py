"""CLI ``/model`` switch summary: DeepSeek's balance must appear for the route just picked.

``hermes_cli.cli_model_switch_mixin._print_switch_summary`` is the single CLI place both switch paths
(typed ``/model <name>`` and the picker) print through. These tests drive the REAL chain — switch
summary → ``account_usage_lines`` → the DeepSeek fetcher → a scripted HTTP client — so they prove the
balance reaches the printed block for the NEW route, and that a route without a limits API (or with a
rejected key) never breaks the switch.
"""

from types import SimpleNamespace

import httpx

import hermes_cli.cli_model_switch_mixin as switch_mixin

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


class _StubCLI:
    """Bare CLI stub: ``_print_switch_summary`` reads only the live route + agent."""

    model = "old/model"
    provider = "openrouter"
    base_url = "https://openrouter.ai/api/v1"
    api_key = "sk-ambient-openrouter"
    agent = None


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


def _print_summary(cli, result) -> str:
    import cli as cli_module

    printed: list[str] = []
    original = cli_module._cprint
    cli_module._cprint = lambda text: printed.append(text)
    try:
        switch_mixin._print_switch_summary(cli, result, cli.model, one_turn=False, strict_context=False)
    finally:
        cli_module._cprint = original
    return "\n".join(printed)


def _no_context_length(monkeypatch) -> None:
    monkeypatch.setattr("hermes_cli.model_switch.resolve_display_context_length", lambda *a, **k: None)


def _wire_deepseek(monkeypatch, client, *, api_key="sk-deepseek"):
    seen: list[tuple] = []

    def _resolve(requested=None, explicit_base_url=None, explicit_api_key=None):
        seen.append((requested, explicit_base_url, explicit_api_key))
        return {"provider": "deepseek", "base_url": explicit_base_url, "api_key": explicit_api_key or api_key}

    monkeypatch.setattr("agent.account_usage.resolve_runtime_provider", _resolve)
    monkeypatch.setattr("agent.account_usage.httpx.Client", lambda timeout=10.0: client)
    return seen


def test_switch_summary_prints_the_deepseek_balance_of_the_route_switched_to(monkeypatch):
    _no_context_length(monkeypatch)
    client = _RecordingClient(_Response(_BALANCE))
    seen = _wire_deepseek(monkeypatch, client)

    out = _print_summary(_StubCLI(), _result())

    # Fetched for the route the session moved TO — the ambient OpenRouter route is never consulted.
    assert seen == [("deepseek", "https://api.deepseek.com/v1", "sk-deepseek")]
    assert client.urls == ["https://api.deepseek.com/user/balance"]
    assert "Model switched: deepseek-v4-pro" in out
    assert "📈 DeepSeek balance" in out
    assert "Provider: deepseek" in out
    assert "Balance: 110.00 CNY (granted 10.00 • topped up 100.00)" in out
    # The balance belongs under the route metadata, not before it.
    assert out.index("Provider: DeepSeek") < out.index("Balance: 110.00 CNY")


def test_switch_summary_adds_nothing_for_a_route_without_a_limits_api(monkeypatch):
    """``ollama`` has no usage fetcher: no block, no dangling divider, no HTTP call."""
    _no_context_length(monkeypatch)

    def _unreachable(*args, **kwargs):
        raise AssertionError("a provider without a limits API must not be fetched")

    monkeypatch.setattr("agent.account_usage.httpx.Client", _unreachable)

    out = _print_summary(
        _StubCLI(),
        _result(
            new_model="llama3.1:8b",
            target_provider="ollama",
            provider_label="Ollama",
            base_url="http://localhost:11434",
            api_key="",
        ),
    )

    assert "Model switched: llama3.1:8b" in out
    assert "Balance" not in out
    assert "\n\n" not in out


def test_switch_summary_surfaces_a_rejected_deepseek_key_instead_of_failing(monkeypatch):
    _no_context_length(monkeypatch)
    _wire_deepseek(monkeypatch, _RecordingClient(_Response(None, status_code=401)))

    out = _print_summary(_StubCLI(), _result())

    assert "Model switched: deepseek-v4-pro" in out
    assert "Unavailable:" in out and "DEEPSEEK_API_KEY" in out


def test_switch_summary_survives_a_deepseek_provider_outage(monkeypatch):
    """A 5xx/rate-limited balance API must not touch the switch output beyond omitting the block."""
    _no_context_length(monkeypatch)
    _wire_deepseek(monkeypatch, _RecordingClient(_Response(None, status_code=503)))

    out = _print_summary(_StubCLI(), _result())

    assert "Model switched: deepseek-v4-pro" in out
    assert "Balance" not in out
