"""OpenAI Ultrafast (``service_tier: "ultrafast"``): one tier word table, a per-model request gate,
and pricing from the tier the response was SERVED at. Relationship tests, no catalog snapshots."""

from decimal import Decimal
from types import SimpleNamespace

import pytest

from agent.fast_mode import SERVICE_TIER_WORDS, STATIC_TIERS, parse_service_tier
from agent.usage_pricing import (
    _OPENAI_ULTRAFAST_PRICING,
    CanonicalUsage,
    estimate_usage_cost,
    with_served_service_tier,
)
from hermes_cli.models import model_supports_ultrafast, resolve_fast_mode_overrides

ASTRA_SPELLINGS = ("gpt-6-astra", "openai/gpt-6-astra", "gpt-6-astra-900k", "openai/gpt-6-astra-900k")
SOL_SPELLINGS = ("gpt-6.1-sol", "openai/gpt-6.1-sol", "gpt-6.1-sol-900k", "openai/gpt-6.1-sol-900k")


def test_every_config_loader_parses_tiers_through_the_same_table(monkeypatch):
    from gateway.run import GatewayRunner
    from hermes_cli.cli_config_load import _parse_service_tier_config
    import tui_gateway.server as tui

    for word, tier in {**SERVICE_TIER_WORDS, "normal": None, "off": None, "bogus": None}.items():
        monkeypatch.setattr(GatewayRunner, "_cfg_str", classmethod(lambda cls, *_k, _w=word: _w))
        monkeypatch.setattr(tui, "_load_cfg", lambda _w=word: {"agent": {"service_tier": _w}})
        assert parse_service_tier(word) == tier
        assert _parse_service_tier_config(word) == tier, word
        assert GatewayRunner._load_service_tier() == tier, word
        assert tui._load_service_tier() == tier, word
    assert "ultrafast" in STATIC_TIERS


@pytest.mark.parametrize("provider,base_url", [("openai", "https://api.openai.com/v1"),
                                               ("openai-api", "https://api.openai.com/v1"),
                                               ("openai-codex", "https://chatgpt.com/backend-api/codex")])
def test_ultrafast_is_requested_only_for_ultrafast_models_on_first_party_routes(provider, base_url):
    for model in (*ASTRA_SPELLINGS, *SOL_SPELLINGS):
        assert model_supports_ultrafast(model), model
        assert resolve_fast_mode_overrides(model, provider=provider, base_url=base_url, tier="ultrafast") == {
            "service_tier": "ultrafast"}
    # A Priority-capable model without Ultrafast gets nothing, never a silent swap to another paid tier.
    assert resolve_fast_mode_overrides("gpt-6-sol", provider=provider, base_url=base_url) == {"service_tier": "priority"}
    for model in ("gpt-6-sol", "gpt-6.1-sol-pro", "gpt-6.1-sol-mini", "gpt-6.1-sol-codex", "gpt-6.1-sol-preview"):
        assert not model_supports_ultrafast(model), model
        assert resolve_fast_mode_overrides(model, provider=provider, base_url=base_url, tier="ultrafast") is None
    # Proxies never see the tier, even under a first-party provider name.
    for model in (*ASTRA_SPELLINGS, *SOL_SPELLINGS):
        assert resolve_fast_mode_overrides(model, provider="openrouter",
                                           base_url="https://openrouter.ai/api/v1", tier="ultrafast") is None
        assert resolve_fast_mode_overrides(model, provider="openai-api",
                                           base_url="https://proxy.example/v1", tier="ultrafast") is None


def test_desktop_surfaces_carry_the_exact_tier_ultrafast_is_never_plain_fast():
    from hermes_cli.inventory import _apply_capabilities
    from tui_gateway.methods_session_model_guard import create_overrides

    rows = [{"slug": "openai-codex", "models": ["gpt-6-astra-900k", *SOL_SPELLINGS[::2], "gpt-6-sol", "gpt-daybreak-blue-latest-900k"]},
            {"slug": "openrouter", "models": ["openai/gpt-6-astra", "openai/gpt-6.1-sol"]}]
    _apply_capabilities(rows)
    caps = {m: c for row in rows for m, c in row["capabilities"].items()}
    assert caps["gpt-6-astra-900k"]["fast"] and caps["gpt-6-astra-900k"].get("ultrafast")
    for model in SOL_SPELLINGS[::2]:
        assert caps[model]["fast"] and caps[model].get("ultrafast"), model
    assert caps["gpt-6-sol"]["fast"] and not caps["gpt-6-sol"].get("ultrafast")
    assert caps["gpt-daybreak-blue-latest-900k"]["fast"] and not caps["gpt-daybreak-blue-latest-900k"].get("ultrafast")
    assert not caps["openai/gpt-6-astra"]["fast"] and not caps["openai/gpt-6-astra"].get("ultrafast")  # proxy route
    assert not caps["openai/gpt-6.1-sol"]["fast"] and not caps["openai/gpt-6.1-sol"].get("ultrafast")
    for params, tier in (({"fast": True, "service_tier": "ultrafast"}, "ultrafast"),
                         ({"fast": True, "service_tier": "normal"}, ""), ({"fast": True}, "priority"), ({}, None)):
        assert create_overrides(params)[2] == tier, params
    with pytest.raises(ValueError):
        create_overrides({"service_tier": "turbo"})


@pytest.mark.parametrize("provider", ["openai", "openai-api"])
def test_cli_and_gateway_turn_routes_send_the_static_tier(provider):
    import cli as cli_mod
    from gateway.run import GatewayRunner

    stub = SimpleNamespace(model="gpt-6-astra", api_key="k", base_url="https://api.openai.com/v1", provider=provider,
                           api_mode="codex_responses", acp_command=None, acp_args=[], _credential_pool=None,
                           service_tier="ultrafast")
    assert cli_mod.HermesCLI._resolve_turn_agent_config(stub, "hi")["request_overrides"] == {"service_tier": "ultrafast"}
    runner = object.__new__(GatewayRunner)
    runner._service_tier = "ultrafast"
    rk = {"api_key": "k", "base_url": "https://api.openai.com/v1", "provider": provider, "api_mode": "codex_responses",
          "command": None, "args": [], "credential_pool": None, "max_tokens": None}
    assert runner._resolve_turn_agent_config("hi", "gpt-6-astra", rk)["request_overrides"] == {"service_tier": "ultrafast"}


def test_cli_refuses_ultrafast_on_a_model_without_it(monkeypatch):
    import cli as cli_mod
    from unittest.mock import MagicMock

    stub = SimpleNamespace(service_tier="priority", model="gpt-6-sol", agent=MagicMock(model="gpt-6-sol"),
                           _fast_command_available=lambda: True)
    monkeypatch.setattr(cli_mod, "_cprint", lambda *a, **k: None)
    cli_mod.HermesCLI._handle_fast_command(stub, "/fast ultrafast")
    assert stub.service_tier == "priority"
    stub.model = stub.agent.model = "gpt-6-astra"
    cli_mod.HermesCLI._handle_fast_command(stub, "/fast ultrafast")
    assert stub.service_tier == "ultrafast"


def _usage(prompt_uncached: int, served_tier=None) -> CanonicalUsage:
    usage = CanonicalUsage(input_tokens=prompt_uncached, output_tokens=10_000, cache_read_tokens=20_000,
                           cache_write_tokens=5_000)
    return with_served_service_tier(usage, SimpleNamespace(service_tier=served_tier))


@pytest.mark.parametrize("prompt_uncached", [50_000, 400_000])  # below / above the 272K whole-request tier
@pytest.mark.parametrize("model", [*ASTRA_SPELLINGS, *SOL_SPELLINGS])
def test_served_ultrafast_bills_at_the_ultrafast_row_and_requested_only_does_not(prompt_uncached, model):
    standard = estimate_usage_cost(model, _usage(prompt_uncached), provider="openai-api")
    served_default = estimate_usage_cost(model, _usage(prompt_uncached, "default"), provider="openai-api")
    ultra = estimate_usage_cost(model, _usage(prompt_uncached, "ultrafast"), provider="openai-api")
    assert served_default.amount_usd == standard.amount_usd  # asked for Ultrafast, served at Standard
    assert ultra.status == "estimated"
    wire_model = model.removeprefix("openai/").removesuffix("-900k")
    assert ultra.pricing_version == _OPENAI_ULTRAFAST_PRICING[wire_model].pricing_version
    assert ultra.amount_usd == standard.amount_usd * 6  # every bucket, both context tiers


@pytest.mark.parametrize("model", ["gpt-6-sol", "gpt-6.1-sol-pro", "gpt-6.1-sol-preview-900k"])
def test_served_ultrafast_on_a_model_without_a_published_rate_is_unknown(model):
    assert estimate_usage_cost(model, _usage(1_000, "ultrafast"), provider="openai-api").status == "unknown"


@pytest.mark.parametrize("provider", ["openai", "openai-api"])
@pytest.mark.parametrize("base_url,amount,source", [
    ("https://api.openai.com/v1", "0.06", "official_docs_snapshot"),
    ("https://proxy.example/v1", "0.007", "provider_models_api"),
    ("https://api.openai.com.proxy.example/v1", "0.007", "provider_models_api"),
])
def test_context_alias_pricing_preserves_custom_endpoint_rates(monkeypatch, provider, base_url, amount, source):
    import agent.usage_pricing as pricing

    monkeypatch.setattr(pricing, "fetch_endpoint_model_metadata", lambda *a, **kw: {
        "gpt-6-astra-900k": {"pricing": {"prompt": "0.000003", "completion": "0.000004"}},
    })
    cost = estimate_usage_cost(
        "gpt-6-astra-900k", CanonicalUsage(input_tokens=1000, output_tokens=1000),
        provider=provider, base_url=base_url,
    )
    assert (cost.amount_usd, cost.source) == (Decimal(amount), source)


@pytest.mark.parametrize("model", SOL_SPELLINGS)
def test_sol_ultrafast_codex_request_uses_the_wire_slug_and_keeps_subscription_accounting(model):
    from agent.fast_mode import effective_request_overrides
    from agent.transports.codex import ResponsesApiTransport
    from hermes_cli.model_normalize import normalize_model_for_provider

    base_url = "https://chatgpt.com/backend-api/codex"
    runtime_model = normalize_model_for_provider(model, "openai-codex")
    overrides = resolve_fast_mode_overrides(runtime_model, provider="openai-codex", base_url=base_url, tier="ultrafast")
    agent = SimpleNamespace(service_tier="ultrafast", request_overrides=overrides)
    transport = ResponsesApiTransport()
    kwargs = transport.build_kwargs(
        runtime_model, [{"role": "user", "content": "hi"}], provider="openai-codex", base_url=base_url,
        is_codex_backend=True, request_overrides=effective_request_overrides(agent),
    )
    wire = transport.preflight_kwargs(kwargs, sanitize_harmony_tokens=True)
    assert wire["service_tier"] == "ultrafast"
    assert wire["model"] == "gpt-6.1-sol"
    cost = estimate_usage_cost(model, _usage(1_000, "ultrafast"), provider="openai-codex", base_url=base_url)
    assert cost.status == "included" and cost.pricing_version == "included-route"


def test_codex_stream_assembler_keeps_the_served_tier():
    from agent.codex_runtime import _consume_codex_event_stream

    done = {"type": "response.completed", "response": {"id": "r1", "status": "completed", "service_tier": "default",
                                                       "usage": {"input_tokens": 1, "output_tokens": 1}}}
    events = [{"type": "response.output_text.delta", "delta": "PASS"}, done]
    assert _consume_codex_event_stream(iter(events), model="gpt-6-astra").service_tier == "default"
