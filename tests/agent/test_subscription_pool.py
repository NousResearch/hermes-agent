from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path

import pytest

from agent.subscription_pool import (
    PoolConfigurationError,
    PoolRouteConfig,
    PoolVerification,
    select_model_from_roster,
    select_subscription_route,
)


def _verified(
    *,
    cloud: bool = True,
    cloud_model: bool = True,
    auth: bool = True,
    models: bool = True,
    quota: bool = True,
) -> PoolVerification:
    return PoolVerification(
        installed=True,
        authenticated=auth,
        models_available=models,
        headless=True,
        workspace=True,
        tools=True,
        quota_visible=quota,
        billing_verified=True,
        cloud_verified=cloud,
        cloud_model_verified=cloud_model,
    )


def _route(
    route_id: str,
    priority: int,
    *,
    provider: str = "openai-codex",
    model: str = "runtime-model",
    capabilities: tuple[str, ...] = ("coding",),
    billing: str = "subscription",
    quota: str = "available",
    verification: PoolVerification | None = None,
    paid: bool = False,
    surface: str = "hermes",
    host: str = "chatgpt.com",
    remaining: float | None = 80.0,
    reserve: float | None = 10.0,
) -> PoolRouteConfig:
    runtime_base_url = f"https://{host}/v1" if provider in {"ollama", "ollama-cloud"} else ""
    return PoolRouteConfig(
        route_id=route_id,
        provider=provider,
        model=model,
        execution_surface=surface,
        subscription_pool=f"{route_id}:authenticated",
        billing_mode=billing,
        capabilities=capabilities,
        priority=priority,
        verification=verification or _verified(),
        paid_usage_possible=paid,
        quota_state=quota,
        remaining_percent=remaining,
        soft_reserve_percent=reserve,
        endpoint_host=host,
        runtime_base_url=runtime_base_url,
    )


@pytest.mark.parametrize("host", ("localhost", "127.0.0.1", "::1", "0.0.0.0"))
def test_local_and_loopback_ollama_endpoints_are_never_selected(host):
    local = _route("local", 1, provider="ollama-cloud", host=host)
    cloud = _route("cloud", 2)

    selected = select_subscription_route((local, cloud), capability="coding")

    assert selected.selected.route_id == "cloud"
    assert selected.rejected[0]["code"] == "local_model_forbidden"


def test_ollama_requires_verified_cloud_route():
    unverified = _route(
        "ollama-unverified",
        1,
        provider="ollama-cloud",
        host="ollama.com",
        verification=_verified(cloud=False),
    )
    fallback = _route("openai", 3)

    selected = select_subscription_route((unverified, fallback), capability="coding")

    assert selected.selected.route_id == "openai"
    assert any(item["code"] == "unverified_cloud_route" for item in selected.rejected)


def test_configured_ollama_cloud_pool_with_unknown_quota_is_preferred_and_explained():
    ollama = replace(
        _route(
            "ollama",
            50,
            provider="ollama",
            model="qwen3-coder:cloud",
            host="ollama.com",
            quota="unknown",
            verification=_verified(quota=False),
        ),
        subscription_pool="ollama-api-key:pro",
    )
    openai = _route("openai", 1)

    selected = select_subscription_route((openai, ollama), capability="coding")

    assert selected.selected.route_id == "ollama"
    assert any(
        item["route_id"] == "ollama" and item["code"] == "quota_unknown"
        for item in selected.rejected
    )
    assert "pool ollama-api-key:pro" in selected.selection_reason
    assert "quota is unknown" in selected.selection_reason


def test_ollama_rejects_unapproved_endpoints_and_non_cloud_models():
    unapproved = _route(
        "ollama-unapproved",
        1,
        provider="ollama",
        host="unapproved.example",
    )
    non_cloud_model = _route(
        "ollama-local-model",
        2,
        provider="ollama",
        host="ollama.com",
        verification=_verified(cloud_model=False),
    )
    openai = _route("openai", 3)

    selected = select_subscription_route(
        (unapproved, non_cloud_model, openai), capability="coding"
    )

    assert selected.selected.route_id == "openai"
    assert {item["code"] for item in selected.rejected} >= {
        "unapproved_ollama_runtime_endpoint",
        "non_cloud_model_forbidden",
    }


def test_ollama_selection_trusts_authenticated_runtime_not_endpoint_host_metadata():
    ollama = replace(
        _route("ollama", 1, provider="ollama", host="metadata.example"),
        runtime_base_url="https://ollama.com/v1",
    )
    openai = _route("openai", 2)

    selected = select_subscription_route((ollama, openai), capability="coding")

    assert selected.selected.route_id == "ollama"


def test_cheapest_capable_subscription_route_wins_and_incapable_is_skipped():
    incapable = _route("cheap-research", 1, capabilities=("research",))
    cheapest = _route("cheap-coding", 2)
    expensive = _route("expensive-coding", 3)

    selected = select_subscription_route((incapable, expensive, cheapest), capability="coding")

    assert selected.selected.route_id == "cheap-coding"
    assert selected.fallback_order == ("expensive-coding",)
    assert any(item["code"] == "missing_capability" for item in selected.rejected)


def test_ollama_exhaustion_auth_failure_and_model_unavailability_advance_to_openai():
    exhausted = _route(
        "ollama-exhausted", 1, provider="ollama-cloud", host="ollama.com", quota="exhausted"
    )
    auth_failed = _route(
        "ollama-auth", 2, provider="ollama-cloud", host="ollama.com",
        quota="auth_failed", verification=_verified(auth=False),
    )
    unavailable = _route(
        "ollama-unavailable", 3, provider="ollama-cloud", host="ollama.com",
        verification=_verified(models=False),
    )
    openai = _route("openai", 4)

    selected = select_subscription_route(
        (exhausted, auth_failed, unavailable, openai), capability="coding"
    )

    assert selected.selected.route_id == "openai"
    assert [item["code"] for item in selected.rejected[:3]] == [
        "quota_exhausted",
        "authentication_failed",
        "model_unavailable",
    ]
    assert all(item["next_route"] == "openai" for item in selected.rejected)


def test_soft_reserve_influences_routing_without_disabling_pool():
    reserved = _route("cheap-reserved", 1, remaining=5.0, reserve=10.0)
    available = _route("next-pool", 2, remaining=80.0)

    selected = select_subscription_route((reserved, available), capability="coding")

    assert selected.selected.route_id == "next-pool"
    assert selected.fallback_order == ("cheap-reserved",)
    assert "soft reserve" in selected.selection_reason


def test_metered_and_paid_overage_routes_remain_disabled_without_authorization():
    metered = _route("metered", 1, billing="metered")
    overage = _route("overage", 2, paid=True)
    subscription = _route("subscription", 3)

    selected = select_subscription_route((metered, overage, subscription), capability="coding")

    assert selected.selected.route_id == "subscription"
    assert {item["code"] for item in selected.rejected} >= {
        "metered_not_authorized",
        "paid_usage_not_authorized",
    }


def test_selection_reason_reports_selected_metered_and_overage_authorizations():
    authorized = _route("authorized", 1, billing="metered", paid=True)

    selected = select_subscription_route(
        (authorized,), capability="coding", allow_metered=True, allow_paid_usage=True
    )

    assert selected.selected.route_id == "authorized"
    assert "metered billing" in selected.selection_reason
    assert "metered usage explicitly authorized" in selected.selection_reason
    assert "paid overage explicitly authorized" in selected.selection_reason


def test_provider_pool_and_billing_attribution_are_required_before_execution():
    ambiguous = replace(_route("bad", 1), subscription_pool="")

    with pytest.raises(PoolConfigurationError, match="authenticated pool attribution"):
        select_subscription_route((ambiguous,), capability="coding")


def test_configured_pool_only_bypasses_request_billing_metadata_for_ollama():
    generic = _route(
        "generic", 1,
        verification=replace(_verified(), billing_verified=False),
    )
    fallback = _route("openai", 2)

    selected = select_subscription_route((generic, fallback), capability="coding")

    assert selected.selected.route_id == "openai"
    assert any(item["code"] == "billing_unverified" for item in selected.rejected)


def test_sonnet_five_is_default_anthropic_family_not_opus():
    roster = (
        "claude-opus-4-8",
        "claude-sonnet-5",
        "claude-sonnet-4-6",
    )

    assert select_model_from_roster(
        roster, family="sonnet", minimum_generation=5
    ) == "claude-sonnet-5"


def test_unknown_quota_is_reported_without_disabling_the_pool():
    unavailable = _route("first", 1, verification=_verified(quota=False), quota="unknown")
    fallback = _route("second", 2)

    selected = select_subscription_route((unavailable, fallback), capability="coding")
    reason = next(item for item in selected.rejected if item["code"] == "quota_unknown")

    assert reason == {
        "route_id": "first",
        "code": "quota_unknown",
        "reason": "remaining quota is not visible",
        "next_route": "first",
    }
    assert selected.selected.route_id == "first"
    assert unavailable.verification.complete is True


def test_read_only_ollama_evaluation_reports_configured_subscription_pool(monkeypatch):
    script_path = Path(__file__).parents[2] / "scripts" / "evaluate_subscription_routes.py"
    spec = importlib.util.spec_from_file_location("subscription_route_evaluation", script_path)
    assert spec is not None and spec.loader is not None
    evaluation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluation)

    from hermes_cli import config, models, runtime_provider

    monkeypatch.setattr(
        config,
        "load_config",
        lambda: {
            "task_routing": {
                "pools": [{
                    "provider": "ollama",
                    "subscription_pool": "ollama-api-key:pro",
                    "billing_mode": "subscription",
                    "paid_usage_possible": False,
                }]
            }
        },
    )
    monkeypatch.setattr(
        runtime_provider,
        "resolve_runtime_provider",
        lambda **_: {"api_key": "test-key", "base_url": "https://ollama.com/v1"},
    )
    monkeypatch.setattr(models, "fetch_api_models_strict", lambda *_: ["qwen3-coder:cloud"])

    report = evaluation._ollama_probe()

    assert evaluation.POOL_ORDER[:2] == ("ollama-cloud", "openai-subscription")
    assert report["eligible"] is True
    assert report["subscription_pool"] == "ollama-api-key:pro"
    assert report["quota"] == {"visible": False, "state": "unknown"}
    assert report["blocked_by"] == []


def test_read_only_ollama_evaluation_requires_successful_authentication(monkeypatch):
    script_path = Path(__file__).parents[2] / "scripts" / "evaluate_subscription_routes.py"
    spec = importlib.util.spec_from_file_location("subscription_route_auth_evaluation", script_path)
    assert spec is not None and spec.loader is not None
    evaluation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluation)

    from hermes_cli import config, models, runtime_provider

    monkeypatch.setattr(
        config,
        "load_config",
        lambda: {
            "task_routing": {
                "pools": [{
                    "provider": "ollama-cloud",
                    "subscription_pool": "ollama-subscription",
                    "billing_mode": "subscription",
                    "paid_usage_possible": False,
                }]
            }
        },
    )
    monkeypatch.setattr(
        runtime_provider,
        "resolve_runtime_provider",
        lambda **_: {"api_key": "configured-key", "base_url": "https://ollama.com/v1"},
    )
    monkeypatch.setattr(models, "fetch_api_models_strict", lambda *_: None)

    report = evaluation._ollama_probe()

    assert report["authenticated"] is False
    assert report["eligible"] is False
    assert "authentication or model-roster probe failed" in "; ".join(report["blocked_by"])


def test_disabled_integration_exposes_configured_blockers():
    disabled = replace(
        _route("antigravity", 1),
        enabled=False,
        billing_mode="unverified",
        blockers=("billing unavailable", "worker bridge missing"),
    )
    fallback = _route("openai", 2)

    selected = select_subscription_route((disabled, fallback), capability="coding")

    assert selected.selected.route_id == "openai"
    assert selected.rejected[0]["code"] == "route_disabled"
    assert selected.rejected[0]["reason"] == "billing unavailable; worker bridge missing"


def test_unverified_billing_cannot_be_enabled():
    route = replace(_route("ambiguous", 1), billing_mode="unverified")

    with pytest.raises(PoolConfigurationError, match="cannot be enabled"):
        select_subscription_route((route,), capability="coding")


@pytest.mark.parametrize(
    "change,match",
    [
        ({"provider": "auto"}, "ambiguous provider"),
        ({"billing_mode": "maybe"}, "ambiguous billing mode"),
        ({"quota_state": "stale"}, "invalid quota state"),
        ({"execution_surface": "mystery"}, "unsupported execution surface"),
    ],
)
def test_invalid_or_ambiguous_configuration_fails_visibly(change, match):
    route = replace(_route("bad", 1), **change)

    with pytest.raises(PoolConfigurationError, match=match):
        select_subscription_route((route,), capability="coding")
