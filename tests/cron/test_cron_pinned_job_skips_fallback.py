"""Pinned cron jobs must not walk fallback_providers (#100437)."""

from unittest.mock import MagicMock, patch

import pytest

from cron.scheduler import _CronJobConfig, _resolve_job_runtime
from hermes_cli.auth import AuthError


def _jc(**overrides):
    cfg = {
        "fallback_providers": [
            {"provider": "ollama", "model": "qwen3:8b"},
        ],
    }
    cfg.update(overrides.pop("cfg", {}))
    return _CronJobConfig(
        cfg=cfg,
        model=overrides.get("model", "claude-sonnet"),
        model_cfg=overrides.get("model_cfg") or {},
        cron_default_provider=overrides.get("cron_default_provider", ""),
    )


def test_pinned_model_skips_fallback_entries_with_a_different_model():
    job = {"id": "pin-job", "provider": "nous", "model": "claude-sonnet"}
    resolve = MagicMock(side_effect=AuthError("no credentials", provider="nous"))
    with patch("hermes_cli.runtime_provider.resolve_runtime_provider", resolve):
        with pytest.raises(RuntimeError, match="no credentials"):
            _resolve_job_runtime(job, "pin-job", _jc())
    assert resolve.call_count == 1
    assert resolve.call_args.kwargs["requested"] == "nous"


def test_pinned_model_still_allows_same_model_provider_swap():
    job = {"id": "dns-pin", "provider": "xai-oauth", "model": "grok-4.5"}
    xai_runtime = {
        "api_key": "k",
        "base_url": "https://api.x.ai/v1",
        "provider": "xai",
        "api_mode": "chat_completions",
    }

    def fake_resolve(**kwargs):
        if kwargs.get("requested") == "xai":
            assert kwargs["target_model"] == "grok-4.5"
            return xai_runtime
        raise AuthError("oauth down", provider="xai-oauth")

    jc = _jc(model="grok-4.5", cfg={
        "fallback_providers": [
            {"provider": "xai", "model": "grok-4.5"},
            {"provider": "ollama", "model": "qwen3:8b"},
        ],
    })
    with patch("hermes_cli.runtime_provider.resolve_runtime_provider", side_effect=fake_resolve):
        runtime, model, _drift = _resolve_job_runtime(job, "dns-pin", jc)
    assert runtime["provider"] == "xai"
    assert model == "grok-4.5"


def test_unpinned_job_still_walks_fallback_on_auth_error():
    job = {"id": "free-job"}
    ollama_runtime = {
        "api_key": "k",
        "base_url": "http://127.0.0.1:11434/v1",
        "provider": "ollama",
        "api_mode": "chat_completions",
    }

    def fake_resolve(**kwargs):
        if kwargs.get("requested") == "ollama":
            return ollama_runtime
        raise AuthError("no credentials", provider="openrouter")

    with patch("hermes_cli.runtime_provider.resolve_runtime_provider", side_effect=fake_resolve):
        runtime, model, _drift = _resolve_job_runtime(job, "free-job", _jc(model="openrouter/auto"))
    assert runtime["provider"] == "ollama"
    assert model == "qwen3:8b"
