"""A model that fits an existing request contract is driven correctly from the remote catalog alone.

GPT-6.2 needed a release although nothing about it was new behaviour: its ids only had to join sets
the code already read (the effort ladder, the request sanitizer, Codex account gating, the fast
tiers). The cached manifest can now add an id to those sets, so the next such model works on an
install that has never heard of it. The manifest cannot add a set, and it cannot remove an id.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agent.bedrock_adapter import is_openai_bedrock_model
from agent.reasoning_effort import CODEX_ASTRA_EFFORTS, codex_supported_efforts
from agent.transports import get_transport
from hermes_cli import model_catalog
from hermes_cli.codex_models import codex_model_requires_account_discovery
from hermes_cli.models import provider_model_ids, resolve_fast_mode_overrides

NEW_OPENAI = "gpt-9-test"
NEW_CLAUDE = "claude-test-9"
API, US, EU = "https://api.openai.com/v1", "https://us.api.openai.com/v1", "https://eu.api.openai.com/v1"


def _manifest() -> dict:
    return {
        "version": 1,
        "providers": {
            "anthropic": {"models": [{"id": NEW_CLAUDE}]},
            "openai-api": {"models": [{"id": NEW_OPENAI, "requires": ["openai_gpt62_contract"]}]},
        },
        "model_sets": {
            "openai_gpt62_contract": [NEW_OPENAI],
            "openai_ultrafast": [NEW_OPENAI],
            "openai_fast_eu": [NEW_OPENAI],
            "anthropic_fast_mode": [NEW_CLAUDE],
            "bedrock_openai_responses": [f"openai.{NEW_OPENAI}"],
            "a_set_added_by_a_later_release": [NEW_OPENAI],
        },
    }


@pytest.fixture
def cached_manifest():
    """Write the manifest into this test's ``HERMES_HOME`` cache, as an earlier fetch would have."""
    model_catalog.reset_cache()
    path = Path(os.environ["HERMES_HOME"]) / "cache" / "model_catalog.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_manifest()), encoding="utf-8")
    yield
    model_catalog.reset_cache()


def _wire_kwargs(base_url: str) -> dict:
    """What the Responses transport would send for the new model with every rejected knob requested."""
    transport = get_transport("codex_responses")
    overrides = {"temperature": 0.5, "top_p": 0.5, "reasoning": {"effort": "none"}, "prompt_cache_retention": "24h"}
    kwargs = transport.build_kwargs(
        NEW_OPENAI, [{"role": "user", "content": "Hi"}], provider="openai", base_url=base_url,
        request_overrides=overrides)
    return transport.preflight_kwargs(kwargs, sanitize_harmony_tokens=True)


def test_a_model_added_to_the_gpt62_contract_takes_its_effort_ladder(cached_manifest):
    assert codex_supported_efforts(NEW_OPENAI) == CODEX_ASTRA_EFFORTS
    assert codex_supported_efforts(f"openai/{NEW_OPENAI}") == CODEX_ASTRA_EFFORTS


@pytest.mark.parametrize("base_url", [API, US, EU])
def test_a_model_added_to_the_gpt62_contract_is_sanitized_on_official_routes(cached_manifest, base_url):
    wire = _wire_kwargs(base_url)

    assert wire["reasoning"]["effort"] == "low"
    assert not ({"temperature", "top_p", "prompt_cache_retention"} & wire.keys())


def test_a_model_added_to_the_gpt62_contract_needs_codex_account_discovery(cached_manifest):
    assert codex_model_requires_account_discovery(NEW_OPENAI)
    assert codex_model_requires_account_discovery(f"openai/{NEW_OPENAI}")


def test_fast_tiers_follow_the_sets_and_the_code_owned_hosts(cached_manifest):
    def tier(base_url, name):
        return resolve_fast_mode_overrides(NEW_OPENAI, provider="openai", base_url=base_url, tier=name)

    assert tier(API, "ultrafast") == {"service_tier": "ultrafast"}
    assert tier(US, "ultrafast") == {"service_tier": "ultrafast"}
    assert tier(EU, "ultrafast") is None  # the GPT-6.2 contract keeps Ultrafast out of the EU
    assert tier(EU, None) == {"service_tier": "priority"}
    assert tier("https://au.api.openai.com/v1", "ultrafast") is None  # no set can add a host


def test_a_claude_model_added_to_the_fast_mode_set_takes_the_speed_param(cached_manifest):
    assert resolve_fast_mode_overrides(NEW_CLAUDE, provider="anthropic") == {"speed": "fast"}
    assert resolve_fast_mode_overrides(f"anthropic/{NEW_CLAUDE}-20270101", provider="anthropic") == {"speed": "fast"}


def test_a_bedrock_openai_model_added_to_its_set_is_recognised(cached_manifest):
    assert is_openai_bedrock_model(f"openai.{NEW_OPENAI}")
    assert is_openai_bedrock_model(f"us.openai.{NEW_OPENAI}")
    assert not is_openai_bedrock_model("openai.gpt-oss-test")


def test_a_picker_entry_whose_required_set_this_build_reads_is_listed(cached_manifest):
    assert provider_model_ids("anthropic")[0] == NEW_CLAUDE
    assert model_catalog.manifest_provider_models("openai-api") == [NEW_OPENAI]


def _recache(data: dict) -> None:
    Path(os.environ["HERMES_HOME"], "cache", "model_catalog.json").write_text(json.dumps(data), encoding="utf-8")


def test_the_publisher_tags_a_row_with_the_sets_it_cannot_be_driven_without():
    from scripts.build_model_catalog import build_catalog

    rows = build_catalog()["providers"]
    openai = {row["id"]: row for row in rows["openai-api"]["models"]}
    bedrock = {row["id"]: row for row in rows["bedrock"]["models"]}

    assert openai["gpt-6.2-sol"].get("requires") == ["openai_gpt62_contract"]
    assert "requires" not in openai["gpt-6.1-sol"]  # a fast tier is optional, so its set gates nothing
    assert bedrock["openai.gpt-6.1-sol"].get("requires") == ["bedrock_openai_responses"]


def test_codex_account_discovery_ignores_the_casing_of_a_configured_id(cached_manifest):
    assert codex_model_requires_account_discovery(f"openai/{NEW_OPENAI.upper()}")
    assert codex_model_requires_account_discovery("openai/GPT-6.2-SOL")


@pytest.mark.parametrize("requires", [
    ["a_set_added_by_a_later_release"], {"openai_gpt62_contract": True}, [["openai_gpt62_contract"]],
    False, {}, 0, "", None,
])
def test_a_requirement_this_build_cannot_meet_keeps_the_entry_out(cached_manifest, requires):
    data = _manifest()
    data["providers"]["openai-api"]["models"].append({"id": "gpt-9-falsy", "requires": requires})
    _recache(data)

    assert "gpt-9-falsy" not in model_catalog.manifest_provider_models("openai-api")


def test_the_openai_ultrafast_set_cannot_offer_the_tier_on_another_vendor(cached_manifest):
    data = _manifest()
    data["model_sets"]["openai_ultrafast"] += ["grok-4.6", "claude-opus-5-5"]
    _recache(data)

    assert resolve_fast_mode_overrides("grok-4.6", provider="xai", base_url="https://api.x.ai/v1", tier="ultrafast") is None
    assert resolve_fast_mode_overrides(
        "claude-opus-5-5", provider="anthropic", base_url="https://api.anthropic.com", tier="ultrafast") is None
    assert resolve_fast_mode_overrides(NEW_OPENAI, provider="openai", base_url=API, tier="ultrafast") == {
        "service_tier": "ultrafast"}


def test_a_published_mantle_model_joins_the_bedrock_picker_after_discovery(cached_manifest):
    from agent.bedrock_adapter import merge_bedrock_openai_model_ids

    data = _manifest()
    mantle = f"openai.{NEW_OPENAI}"
    data["providers"]["bedrock"] = {"models": [
        {"id": mantle, "requires": ["bedrock_openai_responses"]},
        {"id": f"us.{mantle}", "requires": ["bedrock_openai_responses"]},  # a regional profile rides Converse
        {"id": "openai.gpt-9-unknown", "requires": ["a_set_added_by_a_later_release"]},
        {"id": "openai.gpt-9-not-in-the-set"},
    ]}
    _recache(data)

    merged = merge_bedrock_openai_model_ids(["anthropic.claude-test"])

    assert merged.count(mantle) == 1
    assert {f"us.{mantle}", "openai.gpt-9-unknown", "openai.gpt-9-not-in-the-set"}.isdisjoint(merged)


def test_the_hermes_model_iam_wizard_offers_a_published_mantle_model_after_discovery(cached_manifest, monkeypatch):
    from agent import bedrock_adapter
    from hermes_cli import model_setup_flows_bedrock as flow

    data = _manifest()
    mantle = f"openai.{NEW_OPENAI}"
    data["providers"]["bedrock"] = {"models": [{"id": mantle, "requires": ["bedrock_openai_responses"]}]}
    _recache(data)
    monkeypatch.setattr(bedrock_adapter, "has_aws_credentials", lambda: True)
    monkeypatch.setattr(bedrock_adapter, "resolve_aws_auth_env_var", lambda: "AWS_PROFILE")
    monkeypatch.setattr(bedrock_adapter, "resolve_bedrock_region", lambda: "us-east-1")
    monkeypatch.setattr(bedrock_adapter, "discover_bedrock_models", lambda region: [
        {"id": "anthropic.claude-test-v1:0"}, {"id": "eu.anthropic.claude-test-v1:0"}])
    monkeypatch.setattr(flow, "_ask", lambda *a, **kw: "")  # default region, IAM chain
    offered: list[str] = []

    def pick(models, *a, **kw):
        offered.extend(models)

    monkeypatch.setattr(flow, "_pick_model_or_prompt", pick)
    monkeypatch.setattr(flow, "_finish_model", lambda *a, **kw: None)

    flow._model_flow_bedrock({})

    assert offered.count(mantle) == 1
    assert "eu.anthropic.claude-test-v1:0" not in offered  # the regional filter still applies


def test_a_published_mantle_row_counts_as_manifest_supplied_in_the_picker_cache(cached_manifest):
    from agent.bedrock_adapter import merge_bedrock_openai_model_ids
    from hermes_cli.models_detect import with_manifest_provenance

    data = _manifest()
    mantle = f"openai.{NEW_OPENAI}"
    data["providers"]["bedrock"] = {"models": [{"id": mantle, "requires": ["bedrock_openai_responses"]}]}
    _recache(data)

    with model_catalog.recorded_listings() as listings:
        rows = merge_bedrock_openai_model_ids(model_catalog.provider_listing(lambda: ["anthropic.claude-test"]))
    entry = with_manifest_provenance("bedrock", {"fp": "x", "at": 0, "models": rows}, listings)

    assert entry["manifest_only"] == [mantle]  # so disabling the catalog or dropping the row removes it


def test_with_nothing_cached_the_config_is_never_asked_whether_the_catalog_is_on(monkeypatch):
    """A refused config read (a remote config backend that is down) must not break a model the
    install already drives: with no manifest on disk there is nothing the setting could switch off."""
    model_catalog.reset_cache()

    def refuse() -> bool:
        raise SystemExit("config backend unavailable")

    monkeypatch.setattr(model_catalog, "_catalog_enabled", refuse)

    assert codex_supported_efforts("gpt-6.1-sol") == CODEX_ASTRA_EFFORTS
