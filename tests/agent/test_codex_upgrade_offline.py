"""Offline regression contract for the Codex 6.x policy upgrade.

No provider calls, credentials, or live configuration are used here.
"""
from copy import deepcopy
from datetime import datetime, timezone

import pytest

from agent.token_budget_policy import (
    ProviderContextEvidence,
    TokenBudgetPolicyError,
    resolve_token_budget,
    validate_token_budget_policy_config,
)

NOW = datetime(2026, 10, 7, 3, 17, 19, tzinfo=timezone.utc)
ROUTE = "chatgpt.com/backend-api/codex"
ACCOUNT = "offline-synthetic-account"
NEW_MODELS = ("gpt-6.1-sol", "gpt-6-luna")
LEGACY_MODELS = ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-6-astra")


def _model(
    *,
    target_context: int,
    official_context: int,
    target_max_output: int,
    official_max_output: int,
    target_soft_budget: int,
    ratio: float,
    stages: list[int],
    approved_stage: int | None = None,
    evidence: dict | None = None,
) -> dict:
    value = {
        "target_context": target_context,
        "official_context": official_context,
        "target_max_output": target_max_output,
        "official_max_output": official_max_output,
        "target_soft_budget": target_soft_budget,
        "stage_threshold_ratio": ratio,
        "stages": stages,
    }
    if approved_stage is not None:
        value["approved_stage"] = approved_stage
    if evidence is not None:
        value["evidence"] = evidence
    return value


def _config() -> dict:
    """Explicit candidate config: all four legacy entries plus optional new ones."""
    models = {
        "gpt-5.6-sol": _model(
            target_context=1_000_000, official_context=1_050_000,
            target_max_output=128_000, official_max_output=128_000,
            target_soft_budget=700_000, ratio=0.8,
            stages=[272_000, 500_000, 750_000, 950_000, 1_000_000],
        ),
        "gpt-5.6-terra": _model(
            target_context=750_000, official_context=1_050_000,
            target_max_output=96_000, official_max_output=128_000,
            target_soft_budget=525_000, ratio=0.8,
            stages=[272_000, 500_000, 750_000],
        ),
        "gpt-5.6-luna": _model(
            target_context=500_000, official_context=1_050_000,
            target_max_output=64_000, official_max_output=128_000,
            target_soft_budget=350_000, ratio=0.8,
            stages=[272_000, 500_000],
        ),
        "gpt-6-astra": _model(
            target_context=872_000, official_context=872_000,
            target_max_output=128_000, official_max_output=128_000,
            target_soft_budget=610_400, ratio=0.7,
            stages=[272_000, 500_000, 750_000, 872_000],
        ),
        # Official public capacity is documented, but OAuth route capability
        # is only the observed 272K. The operational cap is its 54,400 reserve.
        "gpt-6.1-sol": _model(
            target_context=272_000, official_context=1_050_000,
            target_max_output=54_400, official_max_output=128_000,
            target_soft_budget=217_600, ratio=0.8, stages=[272_000],
            approved_stage=272_000,
        ),
        "gpt-6-luna": _model(
            target_context=272_000, official_context=1_050_000,
            target_max_output=54_400, official_max_output=128_000,
            target_soft_budget=217_600, ratio=0.8, stages=[272_000],
            approved_stage=272_000,
        ),
    }
    return {
        "token_budget_policy": {
            "enabled": True,
            "evidence_ttl_seconds": 86_400,
            "safe_context_limit": 272_000,
            "approved_stage": 1_000_000,
            "providers": {"openai-codex": {"models": models}},
        }
    }


@pytest.mark.parametrize("model", NEW_MODELS)
def test_optional_new_models_keep_required_legacy_allowlist_and_fixed_272k_cap(model):
    cfg = _config()
    validate_token_budget_policy_config(cfg)
    assert set(LEGACY_MODELS).issubset(
        cfg["token_budget_policy"]["providers"]["openai-codex"]["models"]
    )

    result = resolve_token_budget(
        cfg, provider="openai-codex", model=model,
        account_key=ACCOUNT, route_key=ROUTE, now=NOW,
    )

    assert result is not None
    assert result.effective_context == 272_000
    assert result.max_output == 54_400
    assert result.official_max_output == 128_000


@pytest.mark.parametrize("model", NEW_MODELS)
def test_new_models_require_explicit_per_model_approved_stage(model):
    cfg = _config()
    del cfg["token_budget_policy"]["providers"]["openai-codex"]["models"][model]["approved_stage"]

    with pytest.raises(TokenBudgetPolicyError, match="approved_stage"):
        validate_token_budget_policy_config(cfg)


def test_resolution_uses_minimum_of_global_model_and_fresh_route_ceiling():
    cfg = _config()
    sol = cfg["token_budget_policy"]["providers"]["openai-codex"]["models"]["gpt-5.6-sol"]
    cfg["token_budget_policy"]["approved_stage"] = 750_000
    sol["approved_stage"] = 500_000  # optional and backward-compatible for legacy entries
    evidence = ProviderContextEvidence(
        provider="openai-codex", model="gpt-5.6-sol", observed_context=750_000,
        observed_at=NOW, source="codex_oauth_catalog", account_key=ACCOUNT, route_key=ROUTE,
    )

    result = resolve_token_budget(
        cfg, provider="openai-codex", model="gpt-5.6-sol", evidence=evidence,
        account_key=ACCOUNT, route_key=ROUTE, now=NOW,
    )

    assert result is not None
    assert result.effective_context == 500_000
    assert result.approved_stage == 500_000


@pytest.mark.parametrize("model", NEW_MODELS)
def test_new_models_ignore_configured_evidence_and_do_not_promote_from_it(model):
    cfg = _config()
    entry = cfg["token_budget_policy"]["providers"]["openai-codex"]["models"][model]
    entry["evidence"] = {
        "observed_context": 872_000,
        "observed_at": NOW.isoformat(),
        "source": "codex_oauth_catalog",
        "account_key": ACCOUNT,
        "route_key": ROUTE,
    }

    result = resolve_token_budget(
        cfg, provider="openai-codex", model=model,
        account_key=ACCOUNT, route_key=ROUTE, now=NOW,
    )

    assert result is not None
    assert result.evidence_fresh is False
    assert result.status == "blocked_missing_evidence"
    assert result.effective_context == 272_000


@pytest.mark.parametrize("model", NEW_MODELS)
def test_new_model_safe_fallback_cannot_grow_when_global_stage_is_larger(model):
    cfg = _config()
    cfg["token_budget_policy"]["approved_stage"] = 1_000_000

    result = resolve_token_budget(cfg, provider="openai-codex", model=model, now=NOW)

    assert result is not None
    assert result.effective_context == 272_000
    assert result.soft_budget == 217_600
