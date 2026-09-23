"""Routing-policy rejections are terminal in cron's provider fallback loop."""

import pytest

import cron.scheduler as scheduler


def test_primary_failure_then_denied_fallback_does_not_try_second_fallback(monkeypatch):
    """A policy-denied fallback is not an availability failure and stops the chain immediately."""
    from hermes_cli.auth import AuthError
    from hermes_cli.routing_policy import RoutingPolicyError

    calls = []

    def resolve_runtime_provider(**kwargs):
        calls.append(kwargs["requested"])
        if kwargs["requested"] == "primary":
            raise AuthError("primary unavailable", provider="primary", code="invalid_api_key")
        if kwargs["requested"] == "denied-fallback":
            raise RoutingPolicyError("denied fallback")
        raise AssertionError("second fallback must not be attempted")

    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", resolve_runtime_provider)
    monkeypatch.setattr(
        scheduler, "get_fallback_chain",
        lambda _cfg: [
            {"provider": "denied-fallback", "model": "denied-model"},
            {"provider": "second-fallback", "model": "second-model"},
        ],
    )

    job = {"provider": "primary"}
    jc = scheduler._CronJobConfig(cfg={}, model="primary-model", cron_default_provider=None, model_cfg={})

    with pytest.raises(RoutingPolicyError, match="denied fallback"):
        scheduler._resolve_job_runtime(job, "job-1", jc)

    assert calls == ["primary", "denied-fallback"]