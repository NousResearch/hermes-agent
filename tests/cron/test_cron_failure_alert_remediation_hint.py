"""The empty-chain failure alert must tell the operator how to fix it.

Field report: a user whose cron died with "No fallback
chain configured." still cannot self-serve — the alert names the problem but
not the remedy. The empty-chain branch of _fallback_chain_phrase() must name
the exact commands: `hermes fallback add` for the chain, and the
cron.model / cron.model_provider fleet-default keys as the operator-level
alternative. The exhausted branch stays terse — the chain is intact and the
problem is provider-side, so no config command applies.
"""

import cron.scheduler as scheduler
from cron.scheduler import _summarize_cron_failure_for_delivery


def test_empty_chain_alert_names_the_remediation_commands(monkeypatch):
    monkeypatch.setattr(scheduler, "load_config", lambda: {"fallback_providers": []})
    monkeypatch.setattr(scheduler, "get_fallback_chain", lambda cfg: [])
    job = {"name": "semi-analyst-radar", "id": "aaa111"}
    msg = _summarize_cron_failure_for_delivery(job, "Request timed out.")
    assert "No fallback chain configured" in msg
    assert "hermes fallback add" in msg
    assert "cron.model_provider" in msg


def test_exhausted_chain_alert_does_not_carry_the_config_hint(monkeypatch):
    monkeypatch.setattr(
        scheduler, "load_config",
        lambda: {"fallback_providers": [{"provider": "openrouter", "model": "x"}]},
    )
    monkeypatch.setattr(
        scheduler, "get_fallback_chain",
        lambda cfg: [{"provider": "openrouter", "model": "x"}],
    )
    job = {"name": "semi-analyst-radar", "id": "aaa111"}
    msg = _summarize_cron_failure_for_delivery(job, "Request timed out.")
    assert "Fallback chain was exhausted or unavailable." in msg
    assert "hermes fallback add" not in msg


def test_rate_limit_empty_chain_also_carries_the_hint(monkeypatch):
    monkeypatch.setattr(scheduler, "load_config", lambda: {"fallback_providers": []})
    monkeypatch.setattr(scheduler, "get_fallback_chain", lambda cfg: [])
    job = {"name": "kz-coverage", "id": "bbb222"}
    msg = _summarize_cron_failure_for_delivery(job, "HTTP 429: rate limit exceeded")
    assert "No fallback chain configured" in msg
    assert "hermes fallback add" in msg
