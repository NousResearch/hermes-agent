from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agent.task_routing import (
    ProfileResolutionError,
    ProfileRouteConfig,
    _configured_pool_routes,
    resolve_route_runtime,
    resolve_task_route,
    routing_receipt,
)
from agent.subscription_pool import PoolRouteConfig, PoolVerification
from hermes_cli import profiles


@pytest.fixture(autouse=True)
def _isolate_pool_policy(monkeypatch):
    """Legacy profile tests must not depend on the operator's live pool config."""
    monkeypatch.setattr("agent.task_routing._configured_pool_routes", lambda profile: ())


def _candidate(profile: str, capabilities: tuple[str, ...] = ("general",)) -> ProfileRouteConfig:
    return ProfileRouteConfig(
        profile, f"/p/{profile}", "openai-codex", f"{profile}-model", capabilities,
        1000, "openai-codex:pool", "oauth", "included",
    )


def _canonical_candidates() -> tuple[ProfileRouteConfig, ...]:
    return tuple(_candidate(name) for name in (
        "default", "intelligence", "engineering", "assurance", "operations", "localcred", "martial",
    ))


def test_routes_coding_research_and_verifier_to_configured_profiles(monkeypatch):
    candidates = (
        ProfileRouteConfig("engineering", "/p/engineering", "zai", "glm-4.7", ("coding",), 12000,
                           "zai:pool", "oauth", "included"),
        ProfileRouteConfig("intelligence", "/p/intelligence", "kimi-coding", "kimi-k2.5", ("research",), 9000,
                           "kimi-coding:pool", "oauth", "included"),
        ProfileRouteConfig("assurance", "/p/assurance", "custom:verify", "verify-1", ("verification",), 7000,
                           "custom:verify:pool", "api_key", "metered"),
    )
    monkeypatch.setattr("agent.task_routing._profile_configs", lambda profile=None: candidates)

    assert resolve_task_route("implement parser", role="implementation").profile == "engineering"
    assert resolve_task_route("research providers", role="research").profile == "intelligence"
    verifier = resolve_task_route("verify parser", role="verification")
    assert verifier.profile == "assurance"
    assert verifier.delegated is True


def test_included_capacity_preferred_and_fallback_visible(monkeypatch):
    candidates = (
        ProfileRouteConfig("api", "/p/api", "custom:api", "api-model", ("coding",), 100,
                           "custom:api:configured", "configured", "metered"),
        ProfileRouteConfig("oauth", "/p/oauth", "zai", "glm-4.7", ("coding",), 200,
                           "zai:pool", "oauth", "included"),
    )
    monkeypatch.setattr("agent.task_routing._profile_configs", lambda profile=None: candidates)
    route = resolve_task_route("code", role="coding")
    assert (route.profile, route.cost_mode) == ("oauth", "included")
    assert route.fallback_reason == "preferred profile 'engineering' is unavailable"
    assert {"profile": "api", "reason": "lower-ranked policy fallback"} in route.rejected


@pytest.mark.parametrize(("role", "task", "expected"), [
    ("orchestrator", "coordinate the local disposable acceptance", "default"),
    ("implementation", "repair the disposable parser fixture", "engineering"),
    ("coding", "implement a harmless fixture", "engineering"),
    ("research", "synthesize the acceptance evidence", "intelligence"),
    ("strategic-synthesis", "compare the acceptance evidence", "intelligence"),
    ("verification", "review the fixture result", "assurance"),
    ("testing", "test the fixture result", "assurance"),
    ("operations", "maintain continuity records", "operations"),
    ("generic", "format this tiny generic note", "default"),
])
def test_role_preferences_are_deterministic(monkeypatch, role, task, expected):
    monkeypatch.setattr("agent.task_routing._profile_configs", lambda profile=None: _canonical_candidates())
    assert resolve_task_route(task, role=role).profile == expected


@pytest.mark.parametrize(("task", "role", "expected"), [
    ("Prepare the LocalCred customer acceptance note", "implementation", "localcred"),
    ("Prepare the Martial OS test checklist", "verification", "martial"),
])
def test_explicit_domain_work_outranks_generic_role_preference(monkeypatch, task, role, expected):
    monkeypatch.setattr("agent.task_routing._profile_configs", lambda profile=None: _canonical_candidates())
    route = resolve_task_route(task, role=role)
    assert (route.role, route.profile, route.delegated) == (expected, expected, True)


def test_explicit_profile_request_is_authoritative(monkeypatch):
    candidates = _canonical_candidates()
    monkeypatch.setattr(
        "agent.task_routing._profile_configs",
        lambda profile=None: tuple(item for item in candidates if profile in {None, item.profile}),
    )
    route = resolve_task_route("implement a harmless fixture", role="implementation", profile="assurance")
    assert route.profile == "assurance"
    assert route.fallback_reason is None


def test_explicit_profile_identity_remains_authoritative_with_worker_pool(monkeypatch):
    candidates = _canonical_candidates()
    monkeypatch.setattr(
        "agent.task_routing._profile_configs",
        lambda profile=None: tuple(item for item in candidates if profile in {None, item.profile}),
    )
    pool = PoolRouteConfig(
        route_id="openai-subscription",
        provider="openai-codex",
        model="runtime-model",
        execution_surface="hermes",
        subscription_pool="chatgpt-team",
        billing_mode="subscription",
        capabilities=("coding",),
        priority=30,
        verification=PoolVerification(True, True, True, True, True, True, True, True),
        quota_state="available",
    )

    route = resolve_task_route(
        "implement fixture", role="implementation", profile="assurance", pool_routes=(pool,)
    )

    assert route.profile == "assurance"
    assert route.provider == "openai-codex"
    assert route.subscription_pool == "chatgpt-team"


def test_configured_ollama_api_key_pool_is_authoritative_without_request_billing_metadata():
    raw = {
        "id": "ollama-cloud",
        "provider": "ollama",
        "model": "qwen3-coder:cloud",
        "execution_surface": "hermes",
        "subscription_pool": "ollama-api-key:pro",
        "billing_mode": "subscription",
        "capabilities": ["coding"],
        "priority": 1,
        "endpoint_host": "ollama.com",
        "paid_usage_possible": False,
        "quota_state": "unknown",
        "verification": {
            "installed": True,
            "authenticated": True,
            "models_available": True,
            "headless": True,
            "workspace": True,
            "tools": True,
            "quota_visible": False,
            "cloud_verified": True,
            "cloud_model_verified": True,
        },
    }

    (pool,) = _configured_pool_routes(_candidate("engineering", ("coding",)), (raw,))

    assert pool.subscription_pool == "ollama-api-key:pro"
    assert pool.verification.billing_verified is True


def test_unavailable_preferred_profile_has_visible_default_fallback(monkeypatch):
    candidates = tuple(item for item in _canonical_candidates() if item.profile != "engineering")
    monkeypatch.setattr("agent.task_routing._profile_configs", lambda profile=None: candidates)
    route = resolve_task_route("implement a harmless fixture", role="implementation")
    assert route.profile == "default"
    assert route.fallback_reason == "preferred profile 'engineering' is unavailable"
    assert {"profile": "engineering", "reason": route.fallback_reason} in route.rejected


def test_incapable_preferred_profile_has_visible_default_fallback(monkeypatch):
    candidates = tuple(
        _candidate(item.profile, ("research",) if item.profile == "engineering" else item.capabilities)
        for item in _canonical_candidates()
    )
    monkeypatch.setattr("agent.task_routing._profile_configs", lambda profile=None: candidates)
    route = resolve_task_route("implement a harmless fixture", role="implementation")
    assert route.profile == "default"
    assert route.fallback_reason == "preferred profile 'engineering' is missing coding capability"
    assert {"profile": "engineering", "reason": route.fallback_reason} in route.rejected


def test_specialist_selection_does_not_depend_on_registry_order(monkeypatch):
    candidates = tuple(reversed(_canonical_candidates()))
    monkeypatch.setattr("agent.task_routing._profile_configs", lambda profile=None: candidates)
    assert resolve_task_route("implement a harmless fixture", role="implementation").profile == "engineering"


def test_receipt_preserves_child_route_and_lineage(monkeypatch):
    candidate = ProfileRouteConfig("engineering", "/p/engineering", "zai", "glm-4.7", ("coding",), 12000,
                                   "zai:pool", "oauth", "included")
    monkeypatch.setattr("agent.task_routing._profile_configs", lambda profile=None: (candidate,))
    route = resolve_task_route("implement", role="implementation")
    receipt = routing_receipt(route, task_id="task-1", parent_session_id="parent", child_session_id="child",
                              token_usage={"input": 3, "output": 2})
    assert receipt["profile"] == "engineering"
    assert receipt["model"] == "glm-4.7"
    assert receipt["pool_alias"] == "zai:pool"
    assert receipt["parent_session_id"] == "parent"
    assert receipt["child_session_id"] == "child"
    assert receipt["token_usage"] == {"input": 3, "output": 2}


def test_receipt_records_attempt_attribution_and_structured_handoff(monkeypatch):
    candidate = _candidate("engineering", ("coding",))
    monkeypatch.setattr("agent.task_routing._profile_configs", lambda profile=None: (candidate,))
    route = resolve_task_route("implement", role="implementation")

    receipt = routing_receipt(
        route,
        task_id="task-2",
        parent_profile="default",
        elapsed_time=1.25,
        acceptance_evidence=("pytest: passed",),
        escalation_reason={"code": "quota_exhausted", "detail": "pool one exhausted"},
        next_route="openai-subscription",
        status="failed",
    )

    assert receipt["task_class"] == "coding"
    assert receipt["execution_surface"] == "hermes"
    assert receipt["billing_mode"] in {"subscription", "metered"}
    assert receipt["parent_profile"] == "default"
    assert receipt["child_delegation"] is True
    assert receipt["elapsed_time"] == 1.25
    assert receipt["escalation_reason"]["code"] == "quota_exhausted"
    assert receipt["next_route"] == "openai-subscription"


def _ollama_pool_raw() -> dict[str, object]:
    return {
        "id": "ollama-cloud",
        "provider": "ollama-cloud",
        "model": "qwen3-coder:cloud",
        "model_source": "ollama-cloud",
        "execution_surface": "hermes",
        "subscription_pool": "ollama-subscription",
        "billing_mode": "subscription",
        "capabilities": ["coding"],
        "priority": 1,
        "paid_usage_possible": False,
        "quota_state": "available",
        "verification": {
            "installed": True,
            "authenticated": True,
            "models_available": True,
            "headless": True,
            "workspace": True,
            "tools": True,
            "quota_visible": False,
            "cloud_verified": True,
            "cloud_model_verified": True,
        },
    }


def test_configured_ollama_route_uses_live_authenticated_roster_and_binds_runtime_url(monkeypatch):
    from hermes_cli import models, runtime_provider

    candidate = _candidate("engineering", ("coding",))
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        runtime_provider,
        "resolve_runtime_provider",
        lambda **_: {"api_key": "live-key", "base_url": "https://ollama.com/v1"},
    )
    monkeypatch.setattr(
        models,
        "fetch_api_models_strict",
        lambda api_key, base_url: calls.append((api_key, base_url)) or ["qwen3-coder:cloud"],
    )

    (route,) = _configured_pool_routes(candidate, (_ollama_pool_raw(),))
    monkeypatch.setattr("agent.task_routing._profile_configs", lambda profile=None: (candidate,))
    task_route = resolve_task_route("implement fixture", role="implementation", pool_routes=(route,))
    runtime = resolve_route_runtime(task_route)

    assert calls == [
        ("live-key", "https://ollama.com/v1"),
        ("live-key", "https://ollama.com/v1"),
    ]
    assert route.model == "qwen3-coder:cloud"
    assert route.runtime_base_url == "https://ollama.com/v1"
    assert task_route.runtime_base_url == "https://ollama.com/v1"
    assert runtime["base_url"] == task_route.runtime_base_url


@pytest.mark.parametrize(("runtime", "expected_code"), (
    ({"api_key": "live-key", "base_url": "http://ollama.com/v1"}, "unapproved_ollama_runtime_endpoint"),
    ({"api_key": "live-key", "base_url": "https://ollama.com/v1"}, "model_unavailable"),
    ({"api_key": "", "base_url": "https://ollama.com/v1"}, "authentication_failed"),
))
def test_live_ollama_endpoint_auth_or_roster_failure_advances_to_openai(
    monkeypatch, runtime, expected_code
):
    from hermes_cli import models, runtime_provider

    candidate = _candidate("engineering", ("coding",))
    openai = PoolRouteConfig(
        route_id="openai-subscription", provider="openai-codex", model="gpt-5",
        execution_surface="hermes", subscription_pool="openai-subscription",
        billing_mode="subscription", capabilities=("coding",), priority=2,
        verification=PoolVerification(True, True, True, True, True, True, True, True),
    )
    monkeypatch.setattr(runtime_provider, "resolve_runtime_provider", lambda **_: runtime)
    monkeypatch.setattr(models, "fetch_api_models_strict", lambda *_: None)

    pools = _configured_pool_routes(candidate, (_ollama_pool_raw(),)) + (openai,)
    monkeypatch.setattr("agent.task_routing._profile_configs", lambda profile=None: (candidate,))
    route = resolve_task_route("implement fixture", role="implementation", pool_routes=pools)

    assert route.provider == "openai-codex"
    assert route.fallback_order == ()
    rejected = next(item for item in route.rejected if item["route_id"] == "ollama-cloud")
    assert rejected["code"] == expected_code
    assert rejected["next_route"] == "openai-subscription"


@pytest.mark.parametrize(("ollama_runtime", "roster", "expected_code"), (
    ({"api_key": "live-key", "base_url": "http://ollama.com/v1"}, ["qwen3-coder:cloud"],
     "endpoint_validation_failed"),
    ({"api_key": "", "base_url": "https://ollama.com/v1"}, ["qwen3-coder:cloud"],
     "authentication_failed"),
    ({"api_key": "live-key", "base_url": "https://ollama.com/v1"}, [],
     "model_unavailable"),
))
def test_post_selection_ollama_runtime_failure_uses_frozen_openai_fallback(
    monkeypatch, ollama_runtime, roster, expected_code
):
    from hermes_cli import models, runtime_provider

    candidate = _candidate("engineering", ("coding",))
    ollama = PoolRouteConfig(
        route_id="ollama-cloud", provider="ollama-cloud", model="qwen3-coder:cloud",
        execution_surface="hermes", subscription_pool="ollama-subscription",
        billing_mode="subscription", capabilities=("coding",), priority=1,
        verification=PoolVerification(True, True, True, True, True, True, False, True),
        quota_state="unknown", endpoint_host="ollama.com",
        runtime_base_url="https://ollama.com/v1",
    )
    openai = PoolRouteConfig(
        route_id="openai-subscription", provider="openai-codex", model="gpt-5",
        execution_surface="hermes", subscription_pool="chatgpt-team",
        billing_mode="subscription", capabilities=("coding",), priority=2,
        verification=PoolVerification(True, True, True, True, True, True, True, True),
    )
    monkeypatch.setattr("agent.task_routing._profile_configs", lambda profile=None: (candidate,))
    route = resolve_task_route(
        "implement fixture", role="implementation", pool_routes=(ollama, openai)
    )

    def resolve_runtime_provider(*, requested, **_):
        if requested == "ollama-cloud":
            return ollama_runtime
        return {
            "provider": "openai-codex", "api_key": "team-token",
            "base_url": "https://chatgpt.com/backend-api/codex",
        }

    monkeypatch.setattr(runtime_provider, "resolve_runtime_provider", resolve_runtime_provider)
    monkeypatch.setattr(models, "fetch_api_models_strict", lambda *_: roster)

    runtime = resolve_route_runtime(route)
    effective = runtime.pop("_task_route")

    assert effective.provider == "openai-codex"
    assert effective.model == "gpt-5"
    assert effective.fallback_order == ()
    assert effective.rejected[-1] == {
        "route_id": "ollama-cloud",
        "code": expected_code,
        "reason": effective.rejected[-1]["reason"],
        "next_route": "openai-subscription",
    }
    assert runtime["provider"] == "openai-codex"


def test_strict_ollama_roster_probe_never_falls_back_from_v1(monkeypatch):
    from hermes_cli import models

    requested: list[str] = []

    def fail(request, **_):
        requested.append(request.full_url)
        raise OSError("unavailable")

    monkeypatch.setattr(models, "_urlopen_model_catalog_request", fail)

    assert models.fetch_api_models_strict("live-key", "https://ollama.com/v1") is None
    assert models.fetch_api_models_strict("live-key", "https://ollama.com") is None
    assert requested == ["https://ollama.com/v1/models"]
