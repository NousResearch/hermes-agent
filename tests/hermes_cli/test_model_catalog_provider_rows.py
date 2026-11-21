"""The remote model catalog adds picker rows for every built-in provider, and only adds.

A model published after a release reaches that release's pickers through the cached manifest. The
manifest never removes a row the install ships, and a row only the manifest supplied never decides
which model an install runs, where a bare model name goes, or what onboarding recommends.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

import providers
from hermes_cli import model_catalog, model_setup_flows, model_switch_providers, models, models_detect
from hermes_cli.auth import PROVIDER_REGISTRY
from hermes_cli.models import (
    _PROVIDER_MODELS,
    detect_provider_for_model,
    provider_model_ids,
)
from hermes_cli.models_validate import validate_requested_model
from hermes_cli.web_routers.models import get_recommended_default_model

NEW_ANTHROPIC = "claude-test-9"
NEW_MINIMAX = "MiniMax-Test-9"
SHIPPED_CLAUDE = _PROVIDER_MODELS["anthropic"][0]


def _manifest(**providers: list[dict]) -> dict:
    return {
        "version": 1,
        "updated_at": "2026-11-18T00:00:00Z",
        "providers": {name: {"models": models} for name, models in providers.items()},
    }


def _later_release_manifest() -> dict:
    """What main publishes after a release: each row leads with a model this install has not heard of."""
    return _manifest(
        anthropic=[{"id": NEW_ANTHROPIC}, *({"id": mid} for mid in _PROVIDER_MODELS["anthropic"])],
        minimax=[{"id": NEW_MINIMAX}, *({"id": mid} for mid in _PROVIDER_MODELS["minimax"])],
    )


@pytest.fixture
def cached(monkeypatch):
    """Write a manifest into this test's ``HERMES_HOME`` cache, as an earlier fetch would have."""
    model_catalog.reset_cache()

    def write(manifest: dict) -> None:
        path = Path(os.environ["HERMES_HOME"]) / "cache" / "model_catalog.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest), encoding="utf-8")

    yield write
    model_catalog.reset_cache()


@pytest.mark.parametrize("provider,new_id", [("anthropic", NEW_ANTHROPIC), ("minimax", NEW_MINIMAX)])
def test_a_model_published_after_the_release_is_listed(cached, provider, new_id):
    builtin = list(_PROVIDER_MODELS[provider])
    assert new_id not in provider_model_ids(provider)

    cached(_later_release_manifest())

    listed = provider_model_ids(provider)
    assert listed[0] == new_id
    assert listed[1:] == builtin


def test_an_entry_this_build_cannot_drive_is_not_listed(cached):
    cached(_manifest(anthropic=[{"id": NEW_ANTHROPIC, "requires": ["a-contract-added-later"]}, {"id": "claude-test-8"}]))

    listed = provider_model_ids("anthropic")
    assert NEW_ANTHROPIC not in listed
    assert listed[0] == "claude-test-8"


def test_a_shipped_row_keeps_its_order_and_spelling(cached):
    builtin = list(_PROVIDER_MODELS["anthropic"])
    cached(_manifest(anthropic=[{"id": builtin[-1].upper()}, {"id": NEW_ANTHROPIC}, {"id": builtin[0]}]))

    assert provider_model_ids("anthropic") == [NEW_ANTHROPIC, *builtin]


@pytest.mark.parametrize("provider", ["openai-codex", "xai", "xai-oauth", "moa"])
def test_a_row_with_its_own_source_takes_nothing_from_a_generic_block(cached, provider):
    cached(_manifest(**{provider: [{"id": "not-a-provider-owned-row"}]}))
    builtin = _PROVIDER_MODELS[provider]

    assert model_catalog.merge_remote_models(provider, builtin) == builtin


def test_the_manifest_cannot_advertise_a_model_only_the_account_listing_may(cached, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cached(_manifest(**{"openai-api": [{"id": "gpt-6-astra"}, {"id": "gpt-test-9"}]}))

    listed = provider_model_ids("openai-api")
    assert "gpt-6-astra" not in listed
    assert listed[0] == "gpt-test-9"


def test_a_cache_that_is_not_utf8_leaves_the_shipped_row(cached):
    cached(_later_release_manifest())
    model_catalog._cache_path().write_bytes(b"\xff\xfe")

    assert provider_model_ids("minimax") == list(_PROVIDER_MODELS["minimax"])


@pytest.mark.parametrize("labeled_default", [False, True])
def test_onboarding_does_not_recommend_a_row_the_manifest_added(cached, monkeypatch, labeled_default):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fixture")
    monkeypatch.setattr(models, "_fetch_anthropic_models", lambda **_: [])
    monkeypatch.setattr(model_catalog, "_fetch_manifest", lambda *_: None)
    before = get_recommended_default_model("anthropic")["model"]
    assert before in _PROVIDER_MODELS["anthropic"]
    models._provider_models_cache_path().unlink(missing_ok=True)  # the picker row is rebuilt, manifest included
    candidate = "claude-opus-test-9" if labeled_default else NEW_ANTHROPIC

    cached(_manifest(anthropic=[{"id": candidate, "default": labeled_default}]))

    assert provider_model_ids("anthropic")[0] == candidate
    assert get_recommended_default_model("anthropic")["model"] == before


@pytest.fixture
def opencode_go_live(monkeypatch):
    """OpenCode Go with a key, whose ``/models`` answers with the ids the test sets."""
    live: list[str] = []
    monkeypatch.setattr(models, "_api_key_credentials", lambda *_: ("fixture", "https://opencode.ai/zen/go/v1"))
    monkeypatch.setattr(providers.get_provider_profile("opencode-go"), "fetch_models", lambda **_: list(live))
    monkeypatch.setattr(models_detect, "provider_has_credentials", lambda provider: provider == "anthropic")
    return live


def test_an_added_row_is_selectable_exactly_where_the_picker_offers_it(cached, monkeypatch):
    new = "step-fixture-next"
    cached(_manifest(stepfun=[{"id": new}], gmi=[{"id": new}], **{"openai-api": [{"id": new}]}))
    monkeypatch.setattr(models, "_api_key_provider_live", lambda *_: ["step-live"])
    monkeypatch.setattr(models, "fetch_api_models", lambda *_: ["step-live"])
    monkeypatch.setattr(models, "_api_key_credentials", lambda *_: ("fixture", "https://api.gmi-serving.com/v1"))

    assert new in provider_model_ids("stepfun")  # Step Plan joins curated rows to a partial listing
    assert validate_requested_model(new, "stepfun", api_key="fixture", base_url="https://api.stepfun.ai/v1")["accepted"]
    assert provider_model_ids("gmi") == ["step-live"]  # GMI's healthy listing stands alone
    assert not validate_requested_model(new, "gmi", api_key="fixture", base_url="https://api.gmi-serving.com/v1")["accepted"]
    assert not validate_requested_model(new, "openai-api", api_key="fixture", base_url="https://api.openai.com/v1")["accepted"]


def test_setup_still_probes_a_provider_whose_shipped_row_is_short(cached, monkeypatch):
    monkeypatch.setitem(_PROVIDER_MODELS, "deepseek", ["deepseek-shipped-a", "deepseek-shipped-b"])
    cached(_manifest(deepseek=[{"id": f"deepseek-test-{n}"} for n in range(9)]))
    monkeypatch.setattr(model_setup_flows, "_models_dev_merged", lambda *_: [])
    monkeypatch.setattr(models, "probe_profile_catalog", lambda *_: ["deepseek-live"])

    listed = model_setup_flows._api_key_provider_model_list(
        "deepseek", PROVIDER_REGISTRY["deepseek"], "fixture", "", "https://api.deepseek.com/v1")

    assert listed == ["deepseek-live"]


def _write_manifest(home: Path, manifest: dict) -> None:
    path = home / "cache" / "model_catalog.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest), encoding="utf-8")


def test_onboarding_reads_the_requested_profiles_manifest(cached, monkeypatch):
    from hermes_cli import profiles as profiles_mod

    monkeypatch.setenv("ANTHROPIC_API_KEY", "fixture")
    monkeypatch.setattr(models, "_fetch_anthropic_models", lambda **_: [])
    monkeypatch.setattr(model_catalog, "_fetch_manifest", lambda *_: None)
    before = get_recommended_default_model("anthropic")["model"]
    b_home = profiles_mod.get_profile_dir("b")
    b_home.mkdir(parents=True, exist_ok=True)
    (b_home / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    _write_manifest(b_home, _manifest(anthropic=[{"id": "claude-probe-new"}]))  # only B's manifest adds it

    assert get_recommended_default_model("anthropic", profile="b")["model"] == before
    with monkeypatch.context() as scope:
        scope.setenv("HERMES_HOME", str(b_home))
        model_catalog.reset_cache()
        assert "claude-probe-new" in provider_model_ids("anthropic")  # B's manifest was read


def test_validation_decides_on_the_listing_it_fetched_not_an_older_picker_row(cached, monkeypatch):
    gmi = "https://api.gmi-serving.com/v1"
    cached(_manifest(gmi=[{"id": "fixture/non-builtin"}]))
    monkeypatch.setattr(models, "_api_key_credentials", lambda *_: ("fixture", gmi))
    monkeypatch.setattr(models, "fetch_api_models", lambda *_: ["fixture/listed", "fixture/non-builtin"])
    assert "fixture/non-builtin" in models.cached_provider_model_ids("gmi")

    monkeypatch.setattr(models, "fetch_api_models", lambda *_: ["fixture/listed"])

    assert not validate_requested_model("fixture/non-builtin", "gmi", api_key="fixture", base_url=gmi)["accepted"]


def test_a_manifest_change_keeps_a_warm_account_listing(cached, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fixture")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setattr(models, "fetch_api_models", lambda *_: ["gpt-4o"])
    assert models.cached_provider_model_ids("openai-api") == ["gpt-4o"]
    monkeypatch.setattr(models, "_spawn_swr_refresh", lambda *_a, **_k: None)  # hold the background refresh

    cached(_manifest(**{"openai-api": [{"id": "gpt-fixture-next"}]}))

    assert models.cached_provider_model_ids("openai-api", non_blocking=True) == ["gpt-4o"]


def test_a_plugin_owned_offline_list_offers_the_same_rows_cold_and_warm(cached, monkeypatch):
    from hermes_cli.model_switch_providers import _build_curated_lists

    new = "deepseek-fixture-next"
    cached(_manifest(deepseek=[{"id": new}]))
    listing: list[str] = []
    monkeypatch.setattr(models, "_api_key_credentials", lambda *_: ("fixture", "https://api.deepseek.com/v1"))
    monkeypatch.setattr(providers.get_provider_profile("deepseek"), "fetch_models", lambda **_: list(listing) or None)
    monkeypatch.setattr(model_catalog, "_fetch_manifest", lambda *_: None)

    assert new not in _build_curated_lists("", "", "", non_blocking=True)["deepseek"]  # cold
    assert new not in models.cached_provider_model_ids("deepseek")  # the refresh found no listing

    listing[:] = ["deepseek-live"]  # a healthy, partial listing still gets the addition
    assert new in models.cached_provider_model_ids("deepseek", force_refresh=True)


@pytest.mark.parametrize("listing,offered,absent", [
    (["text-embedding-3-small"], None, "gpt-fixture-extra"),  # no curated match: the shipped row
    (["gpt-4o"], ["gpt-4o"], "gpt-fixture-extra"),
    (["gpt-fixture-extra"], ["gpt-fixture-extra"], None),
])
def test_an_official_openai_listing_decides_which_added_rows_are_offered(cached, monkeypatch, listing, offered, absent):
    monkeypatch.setenv("OPENAI_API_KEY", "fixture")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setattr(models, "fetch_api_models", lambda *_: list(listing))
    cached(_manifest(**{"openai-api": [{"id": "gpt-fixture-extra"}]}))

    rows = provider_model_ids("openai-api")

    assert rows == (offered if offered is not None else list(_PROVIDER_MODELS["openai-api"]))
    assert absent is None or absent not in rows


def test_an_outage_on_the_first_listing_is_never_retried_and_keeps_the_shipped_rows(cached, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fixture")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    calls: list[int] = []

    def listing(*_a, **_k):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("discovery down")
        return ["gpt-4o"]

    monkeypatch.setattr(models, "fetch_api_models", listing)
    cached(_manifest(**{"openai-api": [{"id": "gpt-fixture-added"}]}))

    models.cached_provider_model_ids("openai-api")
    second = models.cached_provider_model_ids("openai-api")

    assert len(calls) == 1
    assert set(_PROVIDER_MODELS["openai-api"]) <= set(second)


def _official_openai_serving_only(monkeypatch, cached, mid: str) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "fixture")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setattr(models, "fetch_api_models", lambda *_a, **_k: [mid])
    cached(_manifest(**{"openai-api": [{"id": mid}]}))


def test_disabling_the_catalog_after_a_row_held_only_a_manifest_id_restores_the_shipped_row(cached, monkeypatch):
    _official_openai_serving_only(monkeypatch, cached, "gpt-fixture-only")
    assert models.cached_provider_model_ids("openai-api") == ["gpt-fixture-only"]

    Path(os.environ["HERMES_HOME"], "config.yaml").write_text("model_catalog:\n  enabled: false\n", encoding="utf-8")

    assert models.cached_provider_model_ids("openai-api") == list(_PROVIDER_MODELS["openai-api"])


def test_a_cache_row_written_before_the_upgrade_routes_from_the_cache_during_an_outage(cached, opencode_go_live):
    name = _PROVIDER_MODELS["anthropic"][0]
    cached(_manifest(**{"opencode-go": [{"id": name}]}))
    fp = models._credential_fingerprint("opencode-go")
    models._provider_models_cache_path().write_text(
        json.dumps({"opencode-go": {"fp": fp, "at": time.time(), "models": ["go-live", name]}}), encoding="utf-8")
    opencode_go_live[:] = []  # discovery is down

    assert detect_provider_for_model(name, "opencode-go") is None


def test_onboarding_keeps_a_live_top_model_the_manifest_also_lists(cached, monkeypatch, opencode_go_live):
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "fixture")
    monkeypatch.setattr(model_catalog, "_fetch_manifest", lambda *_: None)
    opencode_go_live[:] = ["go-new-top", "go-second"]
    cached(_manifest(**{"opencode-go": [{"id": "go-new-top"}]}))

    assert get_recommended_default_model("opencode-go")["model"] == "go-new-top"


def test_onboarding_keeps_a_configured_pin_the_manifest_also_lists(cached, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fixture")
    monkeypatch.setattr(models, "_fetch_anthropic_models", lambda **_: [])
    monkeypatch.setattr(model_catalog, "_fetch_manifest", lambda *_: None)
    Path(os.environ["HERMES_HOME"], "config.yaml").write_text(
        "providers:\n  anthropic:\n    models: [claude-pin-new]\n", encoding="utf-8")
    cached(_manifest(anthropic=[{"id": "claude-pin-new"}]))

    assert get_recommended_default_model("anthropic")["model"] == "claude-pin-new"


def test_a_warm_row_keeps_a_model_the_upgraded_install_now_ships(cached, monkeypatch):
    monkeypatch.setattr(models, "_fetch_anthropic_models", lambda **_: [])
    cached(_later_release_manifest())
    assert models.cached_provider_model_ids("anthropic")[0] == NEW_ANTHROPIC

    monkeypatch.setitem(_PROVIDER_MODELS, "anthropic", [NEW_ANTHROPIC, *_PROVIDER_MODELS["anthropic"]])
    Path(os.environ["HERMES_HOME"], "config.yaml").write_text("model_catalog:\n  enabled: false\n", encoding="utf-8")

    assert models.cached_provider_model_ids("anthropic")[0] == NEW_ANTHROPIC
    assert models_detect.current_provider_catalog_match(NEW_ANTHROPIC, "anthropic") == NEW_ANTHROPIC


def test_disabling_the_catalog_keeps_a_model_the_provider_itself_listed(cached, opencode_go_live):
    opencode_go_live[:] = ["go-new-top", "go-second"]
    cached(_manifest(**{"opencode-go": [{"id": "go-new-top"}]}))
    assert "go-new-top" in models.cached_provider_model_ids("opencode-go")

    Path(os.environ["HERMES_HOME"], "config.yaml").write_text("model_catalog:\n  enabled: false\n", encoding="utf-8")

    assert "go-new-top" in models.cached_provider_model_ids("opencode-go")


def test_a_match_the_manifest_re_cased_is_answered_in_the_providers_spelling(cached, monkeypatch):
    monkeypatch.setattr(models, "_api_key_credentials", lambda *_: ("fixture", "https://api.deepseek.com/v1"))
    monkeypatch.setattr(providers.get_provider_profile("deepseek"), "fetch_models", lambda **_: ["DeepSeek-V5-Mixed"])
    cached(_manifest(deepseek=[{"id": "DEEPSEEK-V5-MIXED"}]))

    assert models_detect.current_provider_catalog_match("deepseek-v5-mixed", "deepseek") == "DeepSeek-V5-Mixed"


@pytest.mark.parametrize("live,expected", [
    (["go-live"], ("anthropic", SHIPPED_CLAUDE)),  # only the manifest lists it: not evidence Go serves it
    (["go-live", SHIPPED_CLAUDE], None),  # Go lists it too: stay
    (["go-live", f"vendor/{SHIPPED_CLAUDE}"], ("opencode-go", f"vendor/{SHIPPED_CLAUDE}")),  # Go's own spelling wins
])
def test_a_manifest_row_never_decides_where_a_bare_name_goes(cached, opencode_go_live, live, expected):
    opencode_go_live[:] = live
    cached(_manifest(**{"opencode-go": [{"id": SHIPPED_CLAUDE}]}))

    assert detect_provider_for_model(SHIPPED_CLAUDE, "opencode-go") == expected


@pytest.mark.parametrize("config", ["model:\n  provider: deepseek\n  base_url: {relay}\n", "model:\n  provider: anthropic\n"])
def test_a_relay_lists_its_own_ids_in_setup_and_validation(cached, monkeypatch, config):
    relay = "https://relay.example/v1"
    new = "deepseek-fixture-next"
    cached(_manifest(deepseek=[{"id": new}]))
    Path(os.environ["HERMES_HOME"], "config.yaml").write_text(config.format(relay=relay), encoding="utf-8")
    monkeypatch.setattr(model_setup_flows, "_models_dev_merged", lambda *_: [])
    monkeypatch.setattr(providers.get_provider_profile("deepseek"), "fetch_models", lambda **_: ["relay-live"])
    monkeypatch.setattr(models, "fetch_api_models", lambda *_a, **_k: ["relay-live"])

    listed = model_setup_flows._api_key_provider_model_list("deepseek", PROVIDER_REGISTRY["deepseek"], "fixture", "", relay)

    assert "relay-live" in listed and new not in listed
    assert not validate_requested_model(new, "deepseek", api_key="fixture", base_url=relay)["accepted"]


_FREE = {"pricing": {"input": "0", "output": "0"}}


@pytest.mark.parametrize("index,expected", [
    (None, "shipped+added"),  # unavailable: the offline list
    ({}, "shipped"),  # HTTP 200 with no models is an index, not an outage
    ({"vendor/unrelated": {}}, "shipped"),  # valid, matches nothing: no unlisted addition
    ({"vendor/fixture-extra": {}, "moonshotai/kimi-free": _FREE}, "added"),  # no shipped match: no promotion
])
def test_the_ai_gateway_offers_an_added_row_only_from_a_live_index(cached, monkeypatch, index, expected):
    shipped = list(models.VERCEL_AI_GATEWAY_MODELS)
    new = "vendor/fixture-extra"
    cached(_manifest(**{"ai-gateway": [{"id": new}]}))
    monkeypatch.setattr(models, "_ai_gateway_catalog_cache", None)
    monkeypatch.setattr(models, "_fetch_live_catalog_index", lambda *_: None if index is None else ([], index))

    rows = models.fetch_ai_gateway_models(force_refresh=True)

    assert rows == {"shipped+added": [*shipped, (new, "")], "shipped": shipped, "added": [(new, "")]}[expected]


def test_the_in_process_copy_follows_the_cache_files_path_and_mtime(cached, tmp_path, monkeypatch):
    cached(_manifest(anthropic=[{"id": NEW_ANTHROPIC}]))
    a_path = model_catalog._cache_path()
    stamp = int(a_path.stat().st_mtime) - 2
    os.utime(a_path, (stamp, stamp))
    assert model_catalog.manifest_provider_models("anthropic") == [NEW_ANTHROPIC]

    with monkeypatch.context() as scope:  # another profile's file with the same mtime
        scope.setenv("HERMES_HOME", str(tmp_path / "other-profile"))
        cached(_manifest(anthropic=[{"id": "claude-profile-b"}]))
        os.utime(model_catalog._cache_path(), (stamp, stamp))
        assert model_catalog.manifest_provider_models("anthropic") == ["claude-profile-b"]

    assert model_catalog.manifest_provider_models("anthropic") == [NEW_ANTHROPIC]  # re-pin the slot to A
    cached(_manifest(anthropic=[{"id": "claude-replaced-a"}]))  # the same file, replaced
    os.utime(a_path, (stamp + 1, stamp + 1))
    assert model_catalog.manifest_provider_models("anthropic") == ["claude-replaced-a"]


def test_the_publisher_leaves_out_own_source_rows_and_empty_rows(monkeypatch):
    from scripts.build_model_catalog import _provider_blocks

    monkeypatch.setitem(_PROVIDER_MODELS, "empty-provider", [])
    published = _provider_blocks()

    assert "anthropic" in published
    assert {"nous", "openai-codex", "xai", "xai-oauth", "moa", "empty-provider"}.isdisjoint(published)


def test_onboarding_never_recommends_an_id_only_the_manifest_put_in_the_row(cached, monkeypatch):
    _official_openai_serving_only(monkeypatch, cached, "gpt-fixture-only")
    monkeypatch.setattr(model_catalog, "_fetch_manifest", lambda *_: None)
    shipped = models.pick_silent_default_model(list(_PROVIDER_MODELS["openai-api"]), provider="openai-api")

    assert get_recommended_default_model("openai-api")["model"] == shipped


def test_publishing_a_model_the_provider_already_lists_does_not_move_it_ahead_of_the_default(cached, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fixture")
    monkeypatch.setattr(model_catalog, "_fetch_manifest", lambda *_: None)
    monkeypatch.setattr(models, "_api_key_credentials", lambda *_: ("fixture", "https://api.deepseek.com/v1"))
    monkeypatch.setattr(providers.get_provider_profile("deepseek"), "fetch_models", lambda **_: ["deepseek-live-new"])
    before = get_recommended_default_model("deepseek")["model"]
    models._provider_models_cache_path().unlink(missing_ok=True)
    cached(_manifest(deepseek=[{"id": "deepseek-live-new"}]))

    assert models.cached_provider_model_ids("deepseek")[0] != "deepseek-live-new"
    assert get_recommended_default_model("deepseek")["model"] == before


def test_a_models_dev_row_the_manifest_repeats_counts_as_the_providers(cached, monkeypatch):
    from agent import models_dev

    monkeypatch.setenv("GEMINI_API_KEY", "fixture")
    monkeypatch.setattr(model_catalog, "_fetch_manifest", lambda *_: None)
    monkeypatch.setattr(models_dev, "list_agentic_models", lambda provider: ["gemini-mdev-top", "gemini-mdev-second"])
    cached(_manifest(gemini=[{"id": "gemini-mdev-top"}]))
    assert get_recommended_default_model("gemini")["model"] == "gemini-mdev-top"

    Path(os.environ["HERMES_HOME"], "config.yaml").write_text("model_catalog:\n  enabled: false\n", encoding="utf-8")

    assert "gemini-mdev-top" in models.cached_provider_model_ids("gemini")


def test_an_env_configured_relay_lists_its_own_ids_in_the_picker(cached, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fixture")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://relay.example/v1")
    monkeypatch.setattr(models, "_api_key_credentials", lambda *_: ("fixture", "https://relay.example/v1"))
    monkeypatch.setattr(providers.get_provider_profile("deepseek"), "fetch_models", lambda **_: ["relay-live"])
    cached(_manifest(deepseek=[{"id": "deepseek-fixture-next"}]))

    assert "deepseek-fixture-next" not in provider_model_ids("deepseek")


def test_a_nous_override_url_still_warms_the_master_manifest_for_other_providers(cached, monkeypatch):
    Path(os.environ["HERMES_HOME"], "config.yaml").write_text(
        "model_catalog:\n  providers:\n    nous:\n      url: https://override.example/nous.json\n",
        encoding="utf-8",
    )
    override = _manifest(nous=[{"id": "nous-override-only"}])
    master = _manifest(nous=[{"id": "nous-master"}], anthropic=[{"id": NEW_ANTHROPIC}])
    monkeypatch.setattr(
        model_catalog, "_fetch_manifest",
        lambda url, timeout=8.0: override if "override.example" in url else master,
    )

    curated = model_switch_providers._build_curated_lists("", "", "", non_blocking=True)

    assert curated["nous"] == ["nous-override-only"]
    assert curated["anthropic"][0] == NEW_ANTHROPIC


def test_default_label_on_a_non_whole_list_provider_is_ignored(cached):
    shipped_last = _PROVIDER_MODELS["anthropic"][-1]
    cached(_manifest(anthropic=[{"id": shipped_last, "default": True}]))

    assert model_catalog.get_default_model_from_cache("anthropic") is None
    assert models.pick_silent_default_model(list(_PROVIDER_MODELS["anthropic"]), provider="anthropic") != shipped_last
