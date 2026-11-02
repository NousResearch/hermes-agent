"""Daybreak follows the signed-in account's Codex catalog, per model and per request (no live accounts)."""
import base64
import json
from types import SimpleNamespace

import pytest

from agent import codex_service_tiers as tiers
from agent import model_metadata as mm
from agent.fast_mode import effective_request_overrides
from agent.transports.codex import ResponsesApiTransport

BASE = "https://chatgpt.com/backend-api/codex"
# Shapes taken from the live ChatGPT Codex /models catalog.
CATALOG = {
    "A": [  # Daybreak Blue account
        {"slug": "gpt-6-sol", "context_window": 272000,
         "available_access_programs": {"cyber": ["standard", "daybreak_blue"]}},
        {"slug": "gpt-6.1-sol", "context_window": 272000,
         "available_access_programs": {"cyber": ["standard"]}},
        {"slug": "gpt-6-luna", "context_window": 272000},  # field absent: unknown
    ],
    "B": [  # same model, account without Daybreak
        {"slug": "gpt-6-sol", "context_window": 272000,
         "available_access_programs": {"cyber": ["standard"]}},
    ],
}


def token(account):
    payload = {"https://api.openai.com/auth": {"chatgpt_account_id": account}}
    return "test." + base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=") + ".test"


@pytest.fixture
def catalog(monkeypatch):
    monkeypatch.setattr(mm, "_codex_oauth_context_cache", {})
    monkeypatch.setattr(tiers, "_catalogs", {})
    calls = []

    def get(url, **kwargs):
        account = kwargs["headers"]["ChatGPT-Account-ID"]
        calls.append(account)
        return SimpleNamespace(status_code=200, json=lambda: {"models": CATALOG[account]})

    monkeypatch.setattr(mm.model_metadata_http, "get", get)
    return calls


def agent(account="A", model="gpt-6-sol", daybreak=True, **overrides):
    values = dict(provider="openai-codex", api_mode="codex_responses", base_url=BASE, api_key=token(account),
                  model=model, daybreak=daybreak, request_overrides={})
    values.update(overrides)
    return SimpleNamespace(**values)


def wire(a):
    return ResponsesApiTransport().build_kwargs(
        model=a.model, messages=[{"role": "user", "content": "hello"}], tools=[],
        request_overrides=effective_request_overrides(a), provider=a.provider, base_url=a.base_url,
        is_codex_backend=True,
    )


def test_enabled_on_a_listed_model_sends_daybreak_blue(catalog):
    a = agent(request_overrides={"extra_body": {"kept": 1}})
    assert wire(a)["extra_body"] == {"kept": 1, "access_programs": {"cyber": "daybreak_blue"}}
    assert a.request_overrides == {"extra_body": {"kept": 1}}  # the user's overrides are never mutated


@pytest.mark.parametrize("change", [
    {"daybreak": False},  # switch off (the default)
    {"model": "gpt-6.1-sol"},  # model lists only standard
    {"model": "gpt-6-luna"},  # catalog silent: unknown is never Daybreak
    {"api_key": token("B")},  # same model, account without the program
    {"provider": "openai-api", "base_url": "https://api.openai.com/v1"},  # not the subscription route
    {"base_url": "https://proxy.example/codex"},
])
def test_daybreak_is_never_sent_without_an_explicit_catalog_listing(catalog, change):
    assert "access_programs" not in (wire(agent(**change)).get("extra_body") or {})


def test_checked_per_request_so_a_model_switch_drops_it(catalog):
    a = agent()
    assert "access_programs" in wire(a)["extra_body"]
    a.model = "gpt-6.1-sol"
    assert "access_programs" not in (wire(a).get("extra_body") or {})
    a.model = "gpt-6-sol"
    assert "access_programs" in wire(a)["extra_body"]
    assert catalog == ["A"]  # one catalog read, then the cache


def test_unreachable_catalog_hides_and_never_sends(monkeypatch):
    monkeypatch.setattr(mm, "_codex_oauth_context_cache", {})
    monkeypatch.setattr(tiers, "_catalogs", {})
    monkeypatch.setattr(mm.model_metadata_http, "get", lambda url, **kw: SimpleNamespace(status_code=503, json=dict))
    assert not tiers.daybreak_offered("gpt-6-sol", provider="openai-codex", base_url=BASE, access_token=token("A"))
    assert "access_programs" not in (wire(agent()).get("extra_body") or {})


def test_offered_only_where_the_account_lists_it(catalog):
    codex = {"provider": "openai-codex", "base_url": BASE}
    assert tiers.daybreak_offered("gpt-6-sol", **codex, access_token=token("A"))
    assert not tiers.daybreak_offered("gpt-6.1-sol", **codex, access_token=token("A"))
    assert not tiers.daybreak_offered("gpt-6-sol", **codex, access_token=token("B"))
    assert not tiers.daybreak_offered("gpt-6-sol", provider="openai-api", base_url="https://api.openai.com/v1",
                                      access_token=token("A"))


def test_gateway_reads_the_profile_switch_each_turn(tmp_path, monkeypatch):
    """Every chat turn (Desktop, TUI, ``hermes chat``) runs in the gateway, which copies the switch onto
    the turn's agent from the profile's config.yaml, so a flip reaches the next turn of a running chat."""
    from gateway.run_config_loaders import GatewayConfigLoadersMixin

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    seen = {}
    for value in ("true", "false", '"yes"', None):
        (home / "config.yaml").write_text("model: {}\n" if value is None else f"agent:\n  daybreak: {value}\n",
                                          encoding="utf-8")
        seen[value] = GatewayConfigLoadersMixin._load_daybreak()
    assert seen == {"true": True, "false": False, '"yes"': False, None: False}


def test_model_menu_capability_marks_only_listed_subscription_models(catalog, monkeypatch):
    from hermes_cli import auth_codex
    from hermes_cli.inventory import _apply_capabilities

    monkeypatch.setattr(auth_codex, "resolve_codex_runtime_credentials", lambda **kw: {"api_key": token("A")})

    def caps(metadata_config=None):
        rows = [{"slug": "openai-codex", "models": ["gpt-6-sol", "gpt-6.1-sol", "gpt-6-luna"]},
                {"slug": "openrouter", "models": ["openai/gpt-6-sol"]}]
        _apply_capabilities(rows, metadata_config=metadata_config)
        return {m: c.get("daybreak", False) for row in rows for m, c in row["capabilities"].items()}

    assert caps() == {"gpt-6-sol": True, "gpt-6.1-sol": False, "gpt-6-luna": False, "openai/gpt-6-sol": False}
    # The opt-in codex app-server runtime never sends the Responses field, so it never offers the switch.
    assert not any(caps({"model": {"openai_runtime": "codex_app_server"}}).values())


def test_a_refreshed_catalog_publishes_programs_with_its_timestamp(catalog):
    """One record per credential: a revoked listing can never be read next to a fresh timestamp."""
    assert "access_programs" in wire(agent())["extra_body"]
    (key,) = tiers._catalogs
    tiers.remember_catalog(key, [{"slug": "gpt-6-sol", "available_access_programs": {"cyber": ["standard"]}}])
    _tiers, _fetched_at, programs = tiers._catalogs[key]
    assert programs == {"gpt-6-sol": frozenset({"standard"})}
    assert "access_programs" not in (wire(agent()).get("extra_body") or {})
