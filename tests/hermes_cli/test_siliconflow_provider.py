"""Focused tests for SiliconFlow (硅基流动) provider wiring.

SiliconFlow is a two-site vendor: the international platform
(api.siliconflow.com) and the mainland-China platform (api.siliconflow.cn) have
separate accounts, keys, model catalogs and base URLs. Both are registered from
one plugin and must stay independently addressable.
"""

from __future__ import annotations

from hermes_cli.auth import (
    PROVIDER_REGISTRY,
    resolve_api_key_provider_credentials,
)
from hermes_cli.model_normalize import normalize_model_for_provider
from hermes_cli.models import CANONICAL_PROVIDERS, provider_model_ids
from hermes_cli.models_catalog_static import _PROVIDER_LABELS


def test_siliconflow_profiles_load():
    from providers import get_provider_profile

    intl = get_provider_profile("siliconflow")
    assert intl is not None
    assert intl.name == "siliconflow"
    assert intl.base_url == "https://api.siliconflow.com/v1"
    assert intl.env_vars == ("SILICONFLOW_API_KEY", "SILICONFLOW_BASE_URL")
    assert intl.auth_type == "api_key"
    assert intl.default_aux_model == "deepseek-ai/DeepSeek-V4-Flash"

    cn = get_provider_profile("siliconflow-cn")
    assert cn is not None
    assert cn.name == "siliconflow-cn"
    assert cn.base_url == "https://api.siliconflow.cn/v1"
    assert cn.env_vars == ("SILICONFLOW_CN_API_KEY", "SILICONFLOW_CN_BASE_URL")
    assert cn.auth_type == "api_key"


def test_siliconflow_sites_are_independent():
    """The two platforms must never share a key or an endpoint — this is the
    single most common misconfiguration for this vendor."""
    from providers import get_provider_profile

    intl = get_provider_profile("siliconflow")
    cn = get_provider_profile("siliconflow-cn")

    assert intl.base_url != cn.base_url
    assert not (set(intl.env_vars) & set(cn.env_vars))
    assert intl.default_aux_model == cn.default_aux_model  # available on both


def test_siliconflow_fallback_models_are_vendor_prefixed():
    """SiliconFlow serves ``vendor/model`` ids; a bare name would 404."""
    from providers import get_provider_profile

    for slug in ("siliconflow", "siliconflow-cn"):
        profile = get_provider_profile(slug)
        assert profile.fallback_models
        for model_id in profile.fallback_models:
            assert "/" in model_id, f"{slug}: {model_id} is not a vendor/model id"


def test_siliconflow_vendor_prefixed_ids_pass_through_untouched():
    """``deepseek-ai/…`` must NOT be read as a ``siliconflow/`` style prefix."""
    model = "deepseek-ai/DeepSeek-V4-Flash"
    assert normalize_model_for_provider(model, "siliconflow") == model
    assert normalize_model_for_provider(model, "siliconflow-cn") == model
    # A pasted config-style prefix is repaired, not stacked.
    assert (
        normalize_model_for_provider(f"siliconflow/{model}", "siliconflow") == model
    )


def test_siliconflow_credential_resolution(monkeypatch):
    monkeypatch.setenv("SILICONFLOW_API_KEY", "sf-intl-secret")
    monkeypatch.setenv("SILICONFLOW_CN_API_KEY", "sf-cn-secret")
    monkeypatch.setenv("SILICONFLOW_CN_BASE_URL", "https://sf-proxy.internal/v1")

    intl = PROVIDER_REGISTRY["siliconflow"]
    assert intl.id == "siliconflow"
    assert intl.auth_type == "api_key"
    assert intl.inference_base_url == "https://api.siliconflow.com/v1"

    creds_intl = resolve_api_key_provider_credentials("siliconflow")
    assert creds_intl["api_key"] == "sf-intl-secret"

    creds_cn = resolve_api_key_provider_credentials("siliconflow-cn")
    assert creds_cn["api_key"] == "sf-cn-secret"
    assert creds_cn["base_url"] == "https://sf-proxy.internal/v1"


def test_siliconflow_reaches_canonical_providers_and_labels():
    """Regression guard: without the plugin the vendor is in neither table, so
    it never reaches ``hermes model`` / ``/model`` / ``--provider``."""
    slugs = [p.slug for p in CANONICAL_PROVIDERS]
    assert "siliconflow" in slugs
    assert "siliconflow-cn" in slugs
    assert _PROVIDER_LABELS["siliconflow"] == "SiliconFlow"
    assert _PROVIDER_LABELS["siliconflow-cn"] == "SiliconFlow (中文站)"


def test_siliconflow_model_catalog_falls_back_to_profile_models(monkeypatch):
    """No key / offline: the picker still shows the curated list."""
    from providers import get_provider_profile

    for slug in ("siliconflow", "siliconflow-cn"):
        profile = get_provider_profile(slug)
        assert profile is not None
        monkeypatch.setattr(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            lambda provider_id, _slug=slug: {
                "provider": _slug,
                "api_key": "",
                "base_url": profile.base_url,
                "source": "",
            },
        )
        monkeypatch.setattr(
            profile, "fetch_models", lambda *, api_key=None, base_url=None, timeout=8.0: None
        )
        assert list(provider_model_ids(slug)) == list(profile.fallback_models)
