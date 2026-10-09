"""Adversarial offline cases for the upgrade candidate; no network."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest
from agent.token_budget_policy import ProviderContextEvidence, TokenBudgetPolicyError, resolve_token_budget, validate_token_budget_policy_config

NOW = datetime(2026, 10, 7, 3, 17, 19, tzinfo=timezone.utc)
ROUTE = "chatgpt.com/backend-api/codex"
ACCOUNT = "offline-synthetic-account"
NEW_MODELS = ("gpt-6.1-sol", "gpt-6-luna")


def _model(
    target_context,
    official_context,
    target_max_output,
    official_max_output,
    target_soft_budget,
    ratio,
    stages,
    approved_stage=None,
):
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
    return value


def _config():
    models = {
        "gpt-5.6-sol": _model(
            1_000_000, 1_050_000, 128_000, 128_000, 700_000, 0.8,
            [272_000, 500_000, 750_000, 950_000, 1_000_000],
        ),
        "gpt-5.6-terra": _model(
            750_000, 1_050_000, 96_000, 128_000, 525_000, 0.8,
            [272_000, 500_000, 750_000],
        ),
        "gpt-5.6-luna": _model(
            500_000, 1_050_000, 64_000, 128_000, 350_000, 0.8,
            [272_000, 500_000],
        ),
        "gpt-6-astra": _model(
            872_000, 872_000, 128_000, 128_000, 610_400, 0.7,
            [272_000, 500_000, 750_000, 872_000],
        ),
        "gpt-6.1-sol": _model(
            272_000, 1_050_000, 54_400, 128_000, 217_600, 0.8,
            [272_000], approved_stage=272_000,
        ),
        "gpt-6-luna": _model(
            272_000, 1_050_000, 54_400, 128_000, 217_600, 0.8,
            [272_000], approved_stage=272_000,
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

@pytest.mark.parametrize('model', NEW_MODELS)
@pytest.mark.parametrize('case,expected_status,expected_cap', [
    ('untrusted','blocked_untrusted_evidence',272000),
    ('stale','blocked_stale_evidence',272000),
    ('wrong_account','blocked_account_mismatch',272000),
    ('wrong_route','blocked_route_mismatch',272000),
    ('future','blocked_future_evidence',272000),
    ('lower_route','capped_by_provider_route',128000),
])
def test_route_evidence_cannot_self_authorize(model, case, expected_status, expected_cap):
    ev = ProviderContextEvidence(provider='openai-codex', model=model,
        observed_context=872000, observed_at=NOW, source='codex_oauth_catalog',
        account_key=ACCOUNT, route_key=ROUTE)
    changes = {
        'untrusted': {'source':'editable_local_receipt'},
        'stale': {'observed_at':NOW-timedelta(days=2)},
        'wrong_account': {'account_key':'OTHER-SYNTHETIC-ACCOUNT'},
        'wrong_route': {'route_key':'other.invalid/codex'},
        'future': {'observed_at':NOW+timedelta(days=1)},
        'lower_route': {'observed_context':128000},
    }
    result = resolve_token_budget(_config(),provider='openai-codex',model=model,
        evidence=replace(ev,**changes[case]),account_key=ACCOUNT,route_key=ROUTE,now=NOW)
    assert result.status == expected_status
    assert result.evidence_fresh == (case=='lower_route')
    assert result.effective_context == expected_cap

@pytest.mark.parametrize('model',NEW_MODELS)
@pytest.mark.parametrize('bad',[None,True,False,0,-1,'272000',272000.0,500000,872000])
def test_invalid_or_expanded_model_approval_is_rejected(model,bad):
    cfg = _config()
    cfg['token_budget_policy']['providers']['openai-codex']['models'][model]['approved_stage']=bad
    with pytest.raises(TokenBudgetPolicyError):
        validate_token_budget_policy_config(cfg)
