"""Bundled Token Plan discovery keeps entitlement and tier boundaries across pickers."""
import io
import json
from types import SimpleNamespace

import pytest

from providers.alibaba_token_plan_models import PERSONAL_MODELS, TEAM_MODELS


PLANS = (
    ("alibaba-token-plan", PERSONAL_MODELS),
    ("alibaba-token-plan-cn", PERSONAL_MODELS),
    ("alibaba-token-plan-team", TEAM_MODELS),
    ("alibaba-token-plan-cn-team", TEAM_MODELS),
)


@pytest.mark.parametrize("provider_id,catalog", PLANS)
@pytest.mark.parametrize("outcome", ["live", "empty", "failure", "no-key"])
@pytest.mark.parametrize("custom_base", [False, True])
def test_setup_and_picker_share_entitled_catalog(monkeypatch, provider_id, catalog, outcome, custom_base):
    import providers
    from hermes_cli import model_setup_flows as flows, models, urllib_security

    profile = providers.get_provider_profile(provider_id)
    assert profile is not None
    base = "https://plan.example.invalid/v1" if custom_base else profile.base_url
    key = "" if outcome == "no-key" else "synthetic-plan-key"
    monkeypatch.setattr(models, "_get_model_config_dict", lambda: {"provider": provider_id, "base_url": base})
    monkeypatch.setattr(models, "_api_key_credentials", lambda _: (key, base))
    # A registry shortcut must not defeat the account-owned catalogue.
    monkeypatch.setattr(flows, "_models_dev_merged", lambda *_: ["registry-only-model"])
    requests = []
    selected = [catalog[0], catalog[-1]]

    def respond(request, **kwargs):
        requests.append(request.full_url)
        assert request.full_url == base + "/models"
        assert request.get_header("Authorization") == "Bearer synthetic-plan-key"
        if outcome == "failure":
            raise OSError("catalog unavailable")
        items = [{"id": "unknown-future-model"}, {"id": "wan2.7-t2v-plus", "type": "video"}]
        if outcome == "live":
            items += [{"id": model.upper()} for model in reversed(selected)]
        return io.BytesIO(json.dumps({"data": items}).encode())

    monkeypatch.setattr(urllib_security, "open_credentialed_url", respond)
    expected = selected if outcome == "live" else [] if outcome == "empty" else list(catalog)
    setup = flows._api_key_provider_model_list(
        provider_id, SimpleNamespace(name=provider_id), key, "", base,
    )
    assert setup == expected
    assert models.provider_model_ids(provider_id) == expected
    assert len(requests) == (0 if outcome == "no-key" else 2)


def test_personal_catalog_does_not_promote_team_only_models(monkeypatch):
    import providers
    from hermes_cli import urllib_security

    team_only = set(TEAM_MODELS) - set(PERSONAL_MODELS)
    assert team_only
    payload = json.dumps({"data": [{"id": model} for model in team_only]}).encode()
    monkeypatch.setattr(urllib_security, "open_credentialed_url", lambda *_a, **_k: io.BytesIO(payload))
    for provider_id, _ in PLANS[:2]:
        profile = providers.get_provider_profile(provider_id)
        assert profile.fetch_models(api_key="synthetic-plan-key") == []


def test_token_plan_profiles_fold_into_qwen_without_standalone_rows():
    from hermes_cli.models_catalog_static import group_providers

    slugs = ["alibaba", *[provider_id for provider_id, _ in PLANS], "openai-api"]
    rows = group_providers(slugs)
    groups = [row for row in rows if row["kind"] == "group"]
    assert len(groups) == 1
    assert groups[0]["group_id"] == "qwen"
    assert set(groups[0]["members"]) == set(slugs) - {"openai-api"}
    assert [row["slug"] for row in rows if row["kind"] == "single"] == ["openai-api"]


def test_unrelated_empty_catalog_retains_fallback():
    from hermes_cli.models import merge_profile_catalog
    from providers.base import ProviderProfile

    profile = ProviderProfile(name="example-offline", fallback_models=("offline-model",))
    assert merge_profile_catalog(profile.name, profile, []) == ["offline-model"]


@pytest.mark.parametrize("provider_id,catalog", PLANS)
def test_token_plan_default_preserves_catalogue_order(provider_id, catalog):
    from hermes_cli.models import _PROVIDER_MODELS, get_default_model_for_provider
    from providers.alibaba_token_plan_models import DEFAULT_MODEL

    assert get_default_model_for_provider(provider_id) == DEFAULT_MODEL
    assert DEFAULT_MODEL in catalog
    assert _PROVIDER_MODELS[provider_id] == list(catalog)
