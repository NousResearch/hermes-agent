from __future__ import annotations

from datetime import datetime, timezone


def test_cron_preflight_uses_healthy_fallback_without_primary_attempt(tmp_path):
    from agent.provider_health import ProviderHealthStore, ProviderRoute
    from cron.provider_health_circuit import resolve_cron_route

    now = datetime(2026, 9, 5, tzinfo=timezone.utc)
    ProviderHealthStore(tmp_path).record_failure(
        ProviderRoute("zai", "glm-5.3-flash"),
        "Weekly Limit Exhausted; reset_at=2026-09-11T00:00:00Z",
        source="cron:eng-completion",
        now=now,
    )
    cfg = {
        "fallback_providers": [
            {
                "provider": "openai-codex",
                "model": "gpt-5.6-sol",
                "reasoning_effort": "medium",
            }
        ]
    }
    job = {
        "id": "eng-completion",
        "provider": "zai",
        "model": "glm-5.3-flash",
    }

    resolved = resolve_cron_route(job, cfg, hermes_home=tmp_path, now=now)

    assert resolved.deferred_until is None
    assert resolved.job["provider"] == "openai-codex"
    assert resolved.job["model"] == "gpt-5.6-sol"
    assert resolved.reasoning_effort == "medium"
    assert resolved.fallback_chain == []
    assert resolved.skipped == (("zai", "glm-5.3-flash"),)


def test_cron_preflight_defers_when_every_allowed_route_is_open(tmp_path):
    from agent.provider_health import ProviderHealthStore, ProviderRoute
    from cron.provider_health_circuit import resolve_cron_route

    now = datetime(2026, 9, 5, tzinfo=timezone.utc)
    routes = [
        ProviderRoute("zai", "glm-5.3-flash"),
        ProviderRoute("openai-codex", "gpt-5.6-sol"),
    ]
    store = ProviderHealthStore(tmp_path)
    for route in routes:
        store.record_failure(
            route,
            "quota exhausted; reset_at=2026-09-11T00:00:00Z",
            source="cron:eng-completion",
            now=now,
        )

    resolved = resolve_cron_route(
        {"id": "eng-completion", "provider": "zai", "model": "glm-5.3-flash"},
        {"fallback_providers": [{"provider": "openai-codex", "model": "gpt-5.6-sol"}]},
        hermes_home=tmp_path,
        now=now,
    )

    assert resolved.job is None
    assert resolved.deferred_until == datetime(2026, 9, 11, tzinfo=timezone.utc)
