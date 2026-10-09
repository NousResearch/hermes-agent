"""Provider+model token-budget policy contracts (feature 007)."""

from datetime import datetime, timedelta, timezone
import hashlib
from types import SimpleNamespace

import pytest

from agent import model_metadata
from agent.token_budget_policy import (
    ProviderContextEvidence,
    TokenBudgetPolicyError,
    apply_runtime_token_budget,
    detect_provider_context_evidence,
    resolve_runtime_token_budget,
    resolve_token_budget,
    validate_token_budget_policy_config,
)


NOW = datetime(2026, 7, 19, 18, 0, tzinfo=timezone.utc)


def _model_config(model: str, *, observed_at: str, observed: int):
    targets = {
        "gpt-5.6-sol": (
            1_000_000,
            128_000,
            700_000,
            [272_000, 500_000, 750_000, 950_000, 1_000_000],
        ),
        "gpt-6-astra": (
            872_000,
            128_000,
            610_400,
            [272_000, 500_000, 750_000, 872_000],
        ),
        "gpt-5.6-terra": (750_000, 96_000, 525_000, [272_000, 500_000, 750_000]),
        "gpt-5.6-luna": (500_000, 64_000, 350_000, [272_000, 500_000]),
    }
    target_context, target_output, soft_budget, stages = targets[model]
    return {
        "target_context": target_context,
        "official_context": 872_000 if model == "gpt-6-astra" else 1_050_000,
        "target_max_output": target_output,
        "official_max_output": 128_000,
        "target_soft_budget": soft_budget,
        "stage_threshold_ratio": 0.7 if model == "gpt-6-astra" else 0.8,
        "stages": stages,
        "evidence": {
            "observed_context": observed,
            "observed_at": observed_at,
            "source": "codex_route_canary",
            "account_key": "zeca-primary",
            "route_key": "chatgpt.com/backend-api/codex",
        },
    }


def _config(*, observed_at="2026-07-19T17:30:00Z", observed=272_000):
    models = {
        model: _model_config(model, observed_at=observed_at, observed=observed)
        for model in (
            "gpt-5.6-sol",
            "gpt-6-astra",
            "gpt-5.6-terra",
            "gpt-5.6-luna",
        )
    }
    return {
        "token_budget_policy": {
            "enabled": True,
            "evidence_ttl_seconds": 3_600,
            "safe_context_limit": 272_000,
            "approved_stage": 272_000,
            "providers": {"openai-codex": {"models": models}},
        }
    }


@pytest.mark.parametrize(
    ("model", "target_context", "target_output", "target_soft_budget"),
    [
        ("gpt-5.6-sol", 1_000_000, 128_000, 700_000),
        ("gpt-5.6-terra", 750_000, 96_000, 525_000),
        ("gpt-5.6-luna", 500_000, 64_000, 350_000),
    ],
)
def test_exact_provider_model_policy_caps_context_soft_budget_and_output(
    model, target_context, target_output, target_soft_budget
):
    result = resolve_token_budget(
        _config(),
        provider="openai-codex",
        model=model,
        account_key="zeca-primary",
        route_key="chatgpt.com/backend-api/codex",
        now=NOW,
    )

    assert result is not None
    assert result.target_context == target_context
    assert result.official_context == 1_050_000
    assert result.observed_context == 272_000
    assert result.effective_context == 272_000
    assert result.soft_budget == 217_600
    assert result.max_output == min(target_output, 54_400)
    assert target_soft_budget >= result.soft_budget
    assert result.compression_threshold == pytest.approx(0.8)
    assert result.stage == 272_000
    assert result.status == "capped_by_provider_route"
    assert result.evidence_fresh is True


def test_provider_output_limits_cap_runtime_output():
    cfg = _config()
    evidence = ProviderContextEvidence(
        provider="openai-codex",
        model="gpt-5.6-sol",
        observed_context=750_000,
        observed_at=NOW,
        source="route_canary",
        account_key="zeca-primary",
        route_key="chatgpt.com/backend-api/codex",
        observed_max_output=24_000,
    )

    result = resolve_token_budget(
        cfg,
        provider="openai-codex",
        model="gpt-5.6-sol",
        evidence=evidence,
        account_key="zeca-primary",
        route_key="chatgpt.com/backend-api/codex",
        now=NOW,
    )

    assert result is not None
    assert result.official_max_output == 128_000
    assert result.observed_max_output == 24_000
    assert result.max_output == 24_000


def test_missing_official_output_limit_fails_closed():
    cfg = _config()
    model_cfg = cfg["token_budget_policy"]["providers"]["openai-codex"]["models"][
        "gpt-5.6-sol"
    ]
    del model_cfg["official_max_output"]

    with pytest.raises(TokenBudgetPolicyError, match="official_max_output"):
        resolve_token_budget(
            cfg,
            provider="openai-codex",
            model="gpt-5.6-sol",
            now=NOW,
        )


def test_runtime_evidence_can_promote_only_to_an_approved_stage():
    cfg = _config()
    cfg["token_budget_policy"]["approved_stage"] = 750_000
    evidence = ProviderContextEvidence(
        provider="openai-codex",
        model="gpt-5.6-sol",
        observed_context=800_000,
        observed_at=NOW,
        source="codex_oauth_catalog",
        account_key="zeca-primary",
        route_key="chatgpt.com/backend-api/codex",
    )

    result = resolve_token_budget(
        cfg,
        provider="openai-codex",
        model="gpt-5.6-sol",
        evidence=evidence,
        account_key="zeca-primary",
        route_key="chatgpt.com/backend-api/codex",
        now=NOW,
    )

    assert result is not None
    assert result.effective_context == 750_000
    assert result.stage == 750_000
    assert result.soft_budget == 600_000
    assert result.max_output == 128_000
    assert result.status == "capped_by_approved_stage"


def test_live_evidence_cannot_promote_beyond_current_approved_stage():
    evidence = ProviderContextEvidence(
        provider="openai-codex",
        model="gpt-5.6-sol",
        observed_context=800_000,
        observed_at=NOW,
        source="codex_oauth_catalog",
        account_key="zeca-primary",
        route_key="chatgpt.com/backend-api/codex",
    )

    result = resolve_token_budget(
        _config(),
        provider="openai-codex",
        model="gpt-5.6-sol",
        evidence=evidence,
        account_key="zeca-primary",
        route_key="chatgpt.com/backend-api/codex",
        now=NOW,
    )

    assert result is not None
    assert result.effective_context == 272_000
    assert result.stage == 272_000
    assert result.approved_stage == 272_000
    assert result.status == "capped_by_approved_stage"


def test_stale_evidence_blocks_promotion_and_uses_safe_cap():
    cfg = _config(observed_at="2026-07-19T15:00:00Z", observed=750_000)

    result = resolve_token_budget(
        cfg,
        provider="openai-codex",
        model="gpt-5.6-sol",
        account_key="zeca-primary",
        route_key="chatgpt.com/backend-api/codex",
        now=NOW,
    )

    assert result is not None
    assert result.evidence_fresh is False
    assert result.status == "blocked_stale_evidence"
    assert result.effective_context == 272_000
    assert result.stage == 272_000


def test_missing_evidence_blocks_promotion_and_uses_safe_cap():
    cfg = _config()
    del cfg["token_budget_policy"]["providers"]["openai-codex"]["models"][
        "gpt-5.6-sol"
    ]["evidence"]

    result = resolve_token_budget(
        cfg, provider="openai-codex", model="gpt-5.6-sol", now=NOW
    )

    assert result is not None
    assert result.evidence_fresh is False
    assert result.status == "blocked_missing_evidence"
    assert result.effective_context == 272_000


def test_astra_missing_evidence_blocks_promotion_and_uses_safe_cap():
    cfg = _config()
    del cfg["token_budget_policy"]["providers"]["openai-codex"]["models"][
        "gpt-6-astra"
    ]["evidence"]

    result = resolve_token_budget(
        cfg,
        provider="openai-codex",
        model="gpt-6-astra",
        account_key="zeca-primary",
        route_key="chatgpt.com/backend-api/codex",
        now=NOW,
    )

    assert result is not None
    assert result.evidence_fresh is False
    assert result.status == "blocked_missing_evidence"
    assert result.effective_context == 272_000
    assert result.max_output == 81_600


def test_astra_stale_evidence_blocks_promotion_and_uses_safe_cap():
    cfg = _config()
    cfg["token_budget_policy"]["approved_stage"] = 1_000_000
    evidence = ProviderContextEvidence(
        provider="openai-codex",
        model="gpt-6-astra",
        observed_context=1_000_000,
        observed_at=NOW - timedelta(hours=3),
        source="codex_oauth_catalog",
        account_key="zeca-primary",
        route_key="chatgpt.com/backend-api/codex",
    )

    result = resolve_token_budget(
        cfg,
        provider="openai-codex",
        model="gpt-6-astra",
        evidence=evidence,
        account_key="zeca-primary",
        route_key="chatgpt.com/backend-api/codex",
        now=NOW,
    )

    assert result is not None
    assert result.evidence_fresh is False
    assert result.status == "blocked_stale_evidence"
    assert result.effective_context == 272_000


@pytest.mark.parametrize(
    ("account_key", "route_key", "status"),
    [
        (
            "different-codex-account",
            "chatgpt.com/backend-api/codex",
            "blocked_account_mismatch",
        ),
        ("zeca-primary", "proxy.example.com/backend-api/codex", "blocked_route_mismatch"),
    ],
)
def test_astra_identity_mismatch_blocks_promotion_and_uses_safe_cap(
    account_key, route_key, status
):
    cfg = _config()
    cfg["token_budget_policy"]["approved_stage"] = 1_000_000
    evidence = ProviderContextEvidence(
        provider="openai-codex",
        model="gpt-6-astra",
        observed_context=1_000_000,
        observed_at=NOW,
        source="codex_oauth_catalog",
        account_key="zeca-primary",
        route_key="chatgpt.com/backend-api/codex",
    )

    result = resolve_token_budget(
        cfg,
        provider="openai-codex",
        model="gpt-6-astra",
        evidence=evidence,
        account_key=account_key,
        route_key=route_key,
        now=NOW,
    )

    assert result is not None
    assert result.evidence_fresh is False
    assert result.status == status
    assert result.effective_context == 272_000


def test_astra_promotes_to_872k_only_with_fresh_exact_codex_oauth_evidence():
    cfg = _config()
    cfg["token_budget_policy"]["approved_stage"] = 1_000_000
    evidence = ProviderContextEvidence(
        provider="openai-codex",
        model="gpt-6-astra",
        observed_context=872_000,
        observed_at=NOW,
        source="codex_oauth_catalog",
        account_key="zeca-primary",
        route_key="chatgpt.com/backend-api/codex",
    )

    result = resolve_token_budget(
        cfg,
        provider="openai-codex",
        model="gpt-6-astra",
        evidence=evidence,
        account_key="zeca-primary",
        route_key="chatgpt.com/backend-api/codex",
        now=NOW,
    )

    assert result is not None
    assert result.evidence_fresh is True
    assert result.target_context == 872_000
    assert result.official_context == 872_000
    assert result.official_max_output == 128_000
    assert result.max_output == 128_000
    assert result.effective_context == 872_000
    assert result.stage == 872_000
    assert result.status == "target_active"


def test_astra_rejects_non_oauth_evidence_even_when_identity_is_fresh():
    cfg = _config()
    cfg["token_budget_policy"]["approved_stage"] = 1_000_000
    evidence = ProviderContextEvidence(
        provider="openai-codex",
        model="gpt-6-astra",
        observed_context=1_000_000,
        observed_at=NOW,
        source="codex_route_canary",
        account_key="zeca-primary",
        route_key="chatgpt.com/backend-api/codex",
    )

    result = resolve_token_budget(
        cfg,
        provider="openai-codex",
        model="gpt-6-astra",
        evidence=evidence,
        account_key="zeca-primary",
        route_key="chatgpt.com/backend-api/codex",
        now=NOW,
    )

    assert result is not None
    assert result.evidence_fresh is False
    assert result.status == "blocked_untrusted_evidence"
    assert result.effective_context == 272_000


def test_astra_runtime_ignores_static_evidence_when_catalog_is_unavailable(monkeypatch):
    cfg = _config(observed=1_000_000)
    cfg["token_budget_policy"]["approved_stage"] = 1_000_000
    monkeypatch.setattr(
        "agent.token_budget_policy.detect_provider_context_evidence",
        lambda **_kwargs: None,
    )

    result = resolve_runtime_token_budget(
        cfg,
        provider="openai-codex",
        model="gpt-6-astra",
        base_url="https://chatgpt.com/backend-api/codex",
        api_key="opaque-codex-oauth-token",
        account_key="zeca-primary",
        now=NOW,
    )

    assert result is not None
    assert result.evidence_fresh is False
    assert result.status == "blocked_missing_evidence"
    assert result.effective_context == 272_000


def test_route_mismatch_cannot_promote_configured_evidence():
    cfg = _config(observed=800_000)
    cfg["token_budget_policy"]["approved_stage"] = 750_000

    result = resolve_token_budget(
        cfg,
        provider="openai-codex",
        model="gpt-5.6-sol",
        account_key="zeca-primary",
        route_key="proxy.example.com/backend-api/codex",
        now=NOW,
    )

    assert result is not None
    assert result.status == "blocked_route_mismatch"
    assert result.effective_context == 272_000
    assert result.evidence_fresh is False


def test_unknown_account_identity_cannot_promote_configured_evidence():
    cfg = _config(observed=800_000)
    cfg["token_budget_policy"]["approved_stage"] = 750_000

    result = resolve_token_budget(
        cfg,
        provider="openai-codex",
        model="gpt-5.6-sol",
        route_key="chatgpt.com/backend-api/codex",
        now=NOW,
    )

    assert result is not None
    assert result.status == "blocked_account_mismatch"
    assert result.effective_context == 272_000
    assert result.evidence_fresh is False


def test_evidence_expires_exactly_at_ttl_boundary():
    cfg = _config(observed_at="2026-07-19T17:00:00Z", observed=750_000)

    result = resolve_token_budget(
        cfg,
        provider="openai-codex",
        model="gpt-5.6-sol",
        account_key="zeca-primary",
        route_key="chatgpt.com/backend-api/codex",
        now=NOW,
    )

    assert result is not None
    assert result.status == "blocked_stale_evidence"
    assert result.evidence_fresh is False


def test_stale_lower_output_observation_remains_a_conservative_cap():
    cfg = _config()
    evidence = ProviderContextEvidence(
        provider="openai-codex",
        model="gpt-5.6-sol",
        observed_context=750_000,
        observed_at=NOW - timedelta(hours=2),
        source="codex_oauth_catalog",
        account_key="zeca-primary",
        route_key="chatgpt.com/backend-api/codex",
        observed_max_output=12_000,
    )

    result = resolve_token_budget(
        cfg,
        provider="openai-codex",
        model="gpt-5.6-sol",
        evidence=evidence,
        account_key="zeca-primary",
        route_key="chatgpt.com/backend-api/codex",
        now=NOW,
    )

    assert result is not None
    assert result.status == "blocked_stale_evidence"
    assert result.max_output == 12_000


def test_untrusted_evidence_source_is_sanitized_from_status():
    evidence = ProviderContextEvidence(
        provider="openai-codex",
        model="gpt-5.6-sol",
        observed_context=272_000,
        observed_at=NOW,
        source="Bearer test-secret",
        account_key="zeca-primary",
        route_key="chatgpt.com/backend-api/codex",
    )

    result = resolve_token_budget(
        _config(),
        provider="openai-codex",
        model="gpt-5.6-sol",
        evidence=evidence,
        account_key="zeca-primary",
        route_key="chatgpt.com/backend-api/codex",
        now=NOW,
    )

    assert result is not None
    assert result.evidence_source == "configured_evidence"
    assert "test-secret" not in str(result.as_dict())


def test_lower_fresh_route_observation_wins_over_safe_cap():
    evidence = ProviderContextEvidence(
        provider="openai-codex",
        model="gpt-5.6-sol",
        observed_context=128_000,
        observed_at=NOW,
        source="codex_oauth_catalog",
        account_key="zeca-primary",
        route_key="chatgpt.com/backend-api/codex",
    )

    result = resolve_token_budget(
        _config(),
        provider="openai-codex",
        model="gpt-5.6-sol",
        evidence=evidence,
        account_key="zeca-primary",
        route_key="chatgpt.com/backend-api/codex",
        now=NOW,
    )

    assert result is not None
    assert result.effective_context == 128_000
    assert result.stage == 128_000
    assert result.soft_budget == 102_400
    assert result.max_output == 25_600


def test_policy_is_exact_and_does_not_cross_provider_or_model_boundaries():
    cfg = _config()
    assert (
        resolve_token_budget(cfg, provider="openai", model="gpt-5.6-sol", now=NOW)
        is None
    )
    assert (
        resolve_token_budget(
            cfg, provider="openai-codex", model="gpt-5.6-unknown", now=NOW
        )
        is None
    )
    assert (
        resolve_token_budget(
            cfg, provider="openai-codex", model="vendor/gpt-5.6-sol", now=NOW
        )
        is None
    )


def test_enabled_policy_rejects_routes_outside_feature_allowlist():
    cfg = _config()
    providers = cfg["token_budget_policy"]["providers"]
    providers["openai"] = providers.pop("openai-codex")

    with pytest.raises(TokenBudgetPolicyError, match="provider key must be canonical"):
        validate_token_budget_policy_config(cfg)


def test_enabled_policy_accepts_exact_four_model_allowlist():
    validate_token_budget_policy_config(_config())


def test_enabled_policy_rejects_partial_feature_allowlist():
    cfg = _config()
    models = cfg["token_budget_policy"]["providers"]["openai-codex"]["models"]
    del models["gpt-5.6-luna"]

    with pytest.raises(TokenBudgetPolicyError, match="must configure exactly"):
        validate_token_budget_policy_config(cfg)


def test_enabled_policy_rejects_missing_astra_from_four_model_allowlist():
    cfg = _config()
    models = cfg["token_budget_policy"]["providers"]["openai-codex"]["models"]
    del models["gpt-6-astra"]

    with pytest.raises(TokenBudgetPolicyError, match="gpt-6-astra"):
        validate_token_budget_policy_config(cfg)


def test_enabled_policy_rejects_models_outside_feature_allowlist():
    cfg = _config()
    models = cfg["token_budget_policy"]["providers"]["openai-codex"]["models"]
    models["gpt-5.6-sol-preview"] = models.pop("gpt-5.6-sol")

    with pytest.raises(TokenBudgetPolicyError, match="must configure exactly"):
        validate_token_budget_policy_config(cfg)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("target_context", 999_999),
        ("official_context", 999_999),
        ("target_max_output", 127_999),
        ("official_max_output", 127_999),
        ("target_soft_budget", 699_999),
        ("stage_threshold_ratio", 0.75),
        ("stages", [272_000, 500_000]),
    ],
)
def test_enabled_policy_rejects_noncanonical_model_budget_values(
    field, invalid_value
):
    cfg = _config()
    model_cfg = cfg["token_budget_policy"]["providers"]["openai-codex"]["models"][
        "gpt-5.6-sol"
    ]
    model_cfg[field] = invalid_value

    with pytest.raises(TokenBudgetPolicyError, match=field):
        validate_token_budget_policy_config(cfg)


def test_runtime_resolution_rejects_partial_policy_before_matching_model():
    cfg = _config()
    models = cfg["token_budget_policy"]["providers"]["openai-codex"]["models"]
    del models["gpt-5.6-luna"]

    with pytest.raises(TokenBudgetPolicyError, match="must configure exactly"):
        resolve_token_budget(
            cfg,
            provider="openai-codex",
            model="gpt-5.6-sol",
            now=NOW,
        )


def test_enabled_policy_rejects_noncanonical_provider_key():
    cfg = _config()
    providers = cfg["token_budget_policy"]["providers"]
    providers[" OpenAI-Codex "] = providers.pop("openai-codex")

    with pytest.raises(TokenBudgetPolicyError, match="provider key must be canonical"):
        validate_token_budget_policy_config(cfg)


def test_global_approved_stage_clamps_to_each_models_canonical_target():
    cfg = _config()
    cfg["token_budget_policy"]["approved_stage"] = 750_000
    evidence = ProviderContextEvidence(
        provider="openai-codex",
        model="gpt-5.6-luna",
        observed_context=750_000,
        observed_at=NOW,
        source="codex_oauth_catalog",
        account_key="zeca-primary",
        route_key="chatgpt.com/backend-api/codex",
    )

    result = resolve_token_budget(
        cfg,
        provider="openai-codex",
        model="gpt-5.6-luna",
        evidence=evidence,
        account_key="zeca-primary",
        route_key="chatgpt.com/backend-api/codex",
        now=NOW,
    )

    assert result is not None
    assert result.approved_stage == 750_000
    assert result.effective_context == 500_000
    assert result.stage == 500_000


def test_invalid_policy_fails_closed_instead_of_ignoring_unsafe_values():
    cfg = _config()
    model_cfg = cfg["token_budget_policy"]["providers"]["openai-codex"]["models"][
        "gpt-5.6-sol"
    ]
    model_cfg["target_context"] = "1M"

    with pytest.raises(TokenBudgetPolicyError, match="target_context"):
        resolve_token_budget(cfg, provider="openai-codex", model="gpt-5.6-sol", now=NOW)


def test_codex_detector_accepts_only_exact_live_model_metadata(monkeypatch):
    monkeypatch.setattr(
        "agent.model_metadata._fetch_codex_oauth_context_lengths_with_source",
        lambda _token: (
            {"gpt-5.6-sol": 500_000, "gpt-5.6-sol-preview": 750_000},
            True,
        ),
    )

    evidence = detect_provider_context_evidence(
        provider="openai-codex",
        model="gpt-5.6-sol",
        base_url="https://chatgpt.com/backend-api/codex",
        api_key="opaque-runtime-token",
        account_key="pool-entry-1",
        now=NOW,
    )

    assert evidence is not None
    assert evidence.observed_context == 500_000
    assert evidence.observed_at == NOW
    assert evidence.source == "codex_oauth_catalog"
    assert evidence.account_key == "pool-entry-1"
    assert evidence.route_key == "chatgpt.com/backend-api/codex"


def test_codex_detector_accepts_exact_astra_catalog_entry(monkeypatch):
    monkeypatch.setattr(
        "agent.model_metadata._fetch_codex_oauth_context_lengths_with_source",
        lambda _token: (
            {"gpt-6-astra": 272_000, "gpt-6-astra-preview": 872_000},
            True,
        ),
    )

    evidence = detect_provider_context_evidence(
        provider="openai-codex",
        model="gpt-6-astra",
        base_url="https://chatgpt.com/backend-api/codex",
        api_key="opaque-codex-oauth-token",
        account_key="pool-entry-1",
        now=NOW,
    )

    assert evidence is not None
    assert evidence.model == "gpt-6-astra"
    assert evidence.observed_context == 272_000
    assert evidence.source == "codex_oauth_catalog"


def test_codex_detector_preserves_cached_observation_timestamp(monkeypatch):
    token = "synthetic-cache-token"
    cached_at = NOW - timedelta(minutes=30)
    cache_key = hashlib.sha256(token.encode("utf-8")).hexdigest()[:24]
    catalog = {"gpt-5.6-sol": 272_000}
    monkeypatch.setitem(
        model_metadata._codex_oauth_context_cache,
        cache_key,
        (cached_at.timestamp(), catalog),
    )
    monkeypatch.setattr(
        model_metadata,
        "_fetch_codex_oauth_context_lengths_with_source",
        lambda _token: (catalog, False),
    )

    evidence = detect_provider_context_evidence(
        provider="openai-codex",
        model="gpt-5.6-sol",
        base_url="https://chatgpt.com/backend-api/codex",
        api_key=token,
        account_key="synthetic-account",
        now=NOW,
    )

    assert evidence is not None
    assert evidence.observed_at == cached_at


def test_codex_detector_rejects_cache_hit_without_matching_timestamp(monkeypatch):
    monkeypatch.setattr(
        "agent.model_metadata._fetch_codex_oauth_context_lengths_with_source",
        lambda _token: ({"gpt-6-astra": 872_000}, False),
    )

    evidence = detect_provider_context_evidence(
        provider="openai-codex",
        model="gpt-6-astra",
        base_url="https://chatgpt.com/backend-api/codex",
        api_key="cache-without-proof",
        account_key="pool-entry-1",
        now=NOW,
    )

    assert evidence is None


def test_astra_cache_without_timestamp_stays_at_fail_closed_272k(monkeypatch):
    monkeypatch.setattr(
        "agent.model_metadata._fetch_codex_oauth_context_lengths_with_source",
        lambda _token: ({"gpt-6-astra": 872_000}, False),
    )

    result = resolve_runtime_token_budget(
        _config(),
        provider="openai-codex",
        model="gpt-6-astra",
        base_url="https://chatgpt.com/backend-api/codex",
        api_key="cache-without-proof",
        account_key="zeca-primary",
        now=NOW,
    )

    assert result is not None
    assert result.evidence_fresh is False
    assert result.status == "blocked_missing_evidence"
    assert result.effective_context == 272_000
    assert result.max_output == 81_600


def test_codex_detector_rejects_noncanonical_endpoint_without_fetch(monkeypatch):
    fetch = monkeypatch.setattr(
        "agent.model_metadata._fetch_codex_oauth_context_lengths_with_source",
        lambda _token: pytest.fail("catalog must not be called for an untrusted route"),
    )
    del fetch

    evidence = detect_provider_context_evidence(
        provider="openai-codex",
        model="gpt-5.6-sol",
        base_url="https://proxy.example.com/backend-api/codex",
        api_key="opaque-runtime-token",
        account_key="pool-entry-1",
        now=NOW,
    )

    assert evidence is None


def test_resolution_status_fingerprints_account_identity():
    result = resolve_token_budget(
        _config(observed=272_000),
        provider="openai-codex",
        model="gpt-5.6-sol",
        account_key="zeca-primary",
        now=NOW,
    )

    assert result is not None
    status = result.as_dict()
    assert "account_key" not in status
    assert status["account_fingerprint"]
    assert "zeca-primary" not in str(status)


@pytest.mark.parametrize(
    ("config", "provider", "model"),
    [
        (None, "openai-codex", "gpt-5.6-sol"),
        ({"token_budget_policy": {"enabled": False}}, "openai-codex", "gpt-5.6-sol"),
        (_config(), "anthropic", "gpt-5.6-sol"),
        (_config(), "openai-codex", "not-allowlisted"),
    ],
)
def test_runtime_resolution_skips_live_detection_when_policy_does_not_apply(
    monkeypatch, config, provider, model
):
    def _unexpected_detector(**_kwargs):
        raise AssertionError(
            "live evidence detector must not run for an inactive policy"
        )

    monkeypatch.setattr(
        "agent.token_budget_policy.detect_provider_context_evidence",
        _unexpected_detector,
    )

    result = resolve_runtime_token_budget(
        config,
        provider=provider,
        model=model,
        base_url="https://chatgpt.com/backend-api/codex",
        api_key="opaque-runtime-token",
        now=NOW,
    )

    assert result is None


def test_runtime_resolution_derives_opaque_account_identity_from_api_key(monkeypatch):
    captured = {}

    def _capture_detector(**kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr(
        "agent.token_budget_policy.detect_provider_context_evidence",
        _capture_detector,
    )
    api_key = "opaque-runtime-token"

    result = resolve_runtime_token_budget(
        _config(),
        provider="openai-codex",
        model="gpt-5.6-sol",
        base_url="https://chatgpt.com/backend-api/codex",
        api_key=api_key,
        now=NOW,
    )

    expected = (
        "credential-fingerprint-"
        + hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]
    )
    assert captured["account_key"] == expected
    assert api_key not in captured["account_key"]
    assert result is not None
    assert api_key not in str(result.as_dict())


def test_codex_detector_returns_none_when_live_catalog_lacks_exact_slug(monkeypatch):
    monkeypatch.setattr(
        "agent.model_metadata._fetch_codex_oauth_context_lengths_with_source",
        lambda _token: ({"gpt-5": 272_000}, True),
    )

    evidence = detect_provider_context_evidence(
        provider="openai-codex",
        model="gpt-5.6-sol",
        base_url="https://chatgpt.com/backend-api/codex",
        api_key="opaque-runtime-token",
        now=NOW,
    )

    assert evidence is None


def test_apply_runtime_policy_budgets_agent_output_and_surfaces_status(monkeypatch):
    monkeypatch.setattr(
        "agent.token_budget_policy.detect_provider_context_evidence",
        lambda **_kwargs: None,
    )
    agent = SimpleNamespace(
        provider="openai-codex",
        model="gpt-5.6-sol",
        base_url="https://chatgpt.com/backend-api/codex",
        api_key="not-a-real-key",
        _credential_pool=None,
        _configured_max_tokens=80_000,
        max_tokens=80_000,
    )

    resolution = apply_runtime_token_budget(agent, _config())

    assert resolution is not None
    assert agent.max_tokens == 54_400
    assert agent._token_budget_status["effective_context"] == 272_000
    assert agent._token_budget_status["runtime_max_output"] == 54_400
    assert agent._token_budget_status["output_cap_enforcement"] == "provider_default"
    assert agent._token_budget_status["request_output_cap"] is None
    assert agent._token_budget_policy_config == {
        "token_budget_policy": _config()["token_budget_policy"],
    }


def test_apply_runtime_policy_keeps_astra_at_272k_and_uses_provider_default(monkeypatch):
    monkeypatch.setattr(
        "agent.token_budget_policy.detect_provider_context_evidence",
        lambda **_kwargs: None,
    )
    agent = SimpleNamespace(
        provider="openai-codex",
        model="gpt-6-astra",
        base_url="https://chatgpt.com/backend-api/codex",
        api_mode="codex_responses",
        api_key="not-a-real-key",
        _credential_pool=None,
        _configured_max_tokens=None,
        max_tokens=None,
    )

    resolution = apply_runtime_token_budget(agent, _config())

    assert resolution is not None
    assert resolution.effective_context == 272_000
    assert resolution.soft_budget == 190_400
    assert resolution.max_output == 81_600
    assert agent.max_tokens == 81_600
    assert agent._token_budget_status["output_cap_enforcement"] == "provider_default"
    assert agent._token_budget_status["request_output_cap"] is None


def test_noncanonical_codex_lookalike_route_uses_request_cap(monkeypatch):
    monkeypatch.setattr(
        "agent.token_budget_policy.detect_provider_context_evidence",
        lambda **_kwargs: None,
    )
    agent = SimpleNamespace(
        provider="openai-codex",
        model="gpt-5.6-sol",
        api_mode="chat_completions",
        base_url="https://chatgpt.com.evil/backend-api/codex",
        api_key="not-a-real-key",
        _credential_pool=None,
        _configured_max_tokens=80_000,
        max_tokens=80_000,
    )

    resolution = apply_runtime_token_budget(agent, _config())

    assert resolution is not None
    assert agent._token_budget_status["output_cap_enforcement"] == "request_cap"
    assert agent._token_budget_status["request_output_cap"] == 54_400


def test_successful_policy_removal_clears_last_known_good_snapshot(monkeypatch):
    monkeypatch.setattr(
        "agent.token_budget_policy.detect_provider_context_evidence",
        lambda **_kwargs: None,
    )
    agent = SimpleNamespace(
        provider="openai-codex",
        model="gpt-5.6-sol",
        base_url="https://chatgpt.com/backend-api/codex",
        api_key="not-a-real-key",
        _credential_pool=None,
        _configured_max_tokens=80_000,
        max_tokens=80_000,
    )

    assert apply_runtime_token_budget(agent, _config()) is not None
    assert agent._token_budget_policy_config

    assert apply_runtime_token_budget(agent, {}) is None
    assert agent._token_budget_policy_config == {}
    assert agent.max_tokens == 80_000


def test_apply_runtime_policy_does_not_leak_to_unmatched_provider(monkeypatch):
    monkeypatch.setattr(
        "agent.token_budget_policy.detect_provider_context_evidence",
        lambda **_kwargs: None,
    )
    agent = SimpleNamespace(
        provider="openai",
        model="gpt-5.6-sol",
        base_url="https://api.openai.com/v1",
        api_key="not-a-real-key",
        _credential_pool=None,
        _configured_max_tokens=16_000,
        max_tokens=54_400,
        _token_budget_status={"effective_context": 272_000},
    )

    resolution = apply_runtime_token_budget(agent, _config())

    assert resolution is None
    assert agent.max_tokens == 16_000
    assert agent._token_budget_status is None


@pytest.mark.parametrize(
    "policy_config", [{}, {"token_budget_policy": {"enabled": False}}]
)
def test_absent_or_disabled_policy_preserves_resolved_config_max_tokens(
    monkeypatch, policy_config
):
    monkeypatch.setattr(
        "agent.token_budget_policy.detect_provider_context_evidence",
        lambda **_kwargs: None,
    )
    agent = SimpleNamespace(
        provider="openai-codex",
        model="gpt-5.6-sol",
        base_url="https://chatgpt.com/backend-api/codex",
        api_key="not-a-real-key",
        _credential_pool=None,
        _configured_max_tokens=None,
        max_tokens=16_000,
    )

    assert apply_runtime_token_budget(agent, policy_config) is None
    assert agent.max_tokens == 16_000


def test_captured_provider_default_survives_policy_then_unmatched_route(monkeypatch):
    """A policy-derived cap must not become the baseline when baseline is None."""
    monkeypatch.setattr(
        "agent.token_budget_policy.detect_provider_context_evidence",
        lambda **_kwargs: None,
    )
    agent = SimpleNamespace(
        provider="openai-codex",
        model="gpt-5.6-sol",
        base_url="https://chatgpt.com/backend-api/codex",
        api_key="not-a-real-key",
        api_mode="codex_responses",
        _credential_pool=None,
        _configured_max_tokens=None,
        _configured_max_tokens_captured=True,
        max_tokens=None,
    )

    assert apply_runtime_token_budget(agent, _config()) is not None
    assert agent.max_tokens == 54_400

    agent.provider = "openai"
    agent.base_url = "https://api.openai.com/v1"
    agent.api_mode = "chat_completions"
    assert apply_runtime_token_budget(agent, _config()) is None
    assert agent.max_tokens is None
    assert agent._configured_max_tokens is None
    assert agent._configured_max_tokens_captured is True
