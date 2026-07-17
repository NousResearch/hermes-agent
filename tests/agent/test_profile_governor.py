from __future__ import annotations

from pathlib import Path

import yaml

from agent.agent_init import _resolve_compression_threshold
from agent.auxiliary_client import _compression_threshold_for_model
from agent.task_routing import (
    ProfileRouteConfig,
    TaskRoute,
    _profile_configs,
    _configured_pool_routes,
    profile_runtime_scope,
    resolve_profile_governor,
)
from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override
from hermes_cli.config import load_config
from hermes_cli import profiles


def _write_profile(
    home: Path,
    *,
    turns: int,
    request_tokens: int,
    threshold: float,
    route_id: str,
    codex_autoraise: bool = True,
) -> None:
    home.mkdir(parents=True, exist_ok=True)
    config = {
        "agent": {"max_turns": turns},
        "model": {"provider": "openai-codex", "default": "gpt-5.6-terra"},
        "compression": {
            "threshold": threshold,
            "codex_gpt55_autoraise": codex_autoraise,
        },
        "usage_guardrails": {
            "operator_authorized": True,
            "interactive": {
                "max_model_calls": 512,
                "max_subscription_weighted_input_tokens": 3_000_000,
                "max_request_input_tokens": request_tokens,
                "max_consecutive_large_requests": 24,
                "large_request_tokens": 150_000,
            },
        },
        "cron": {"script_timeout_seconds": 60},
        "delegation": {"max_concurrent_children": 2},
        "task_routing": {
            "pools": [
                {
                    "id": route_id,
                    "enabled": False,
                    "priority": 10,
                    "provider": "openai-codex",
                    "model": "profile",
                    "execution_surface": "hermes",
                    "subscription_pool": route_id,
                    "billing_mode": "subscription",
                    "capabilities": ["general"],
                    "quota_state": "available",
                    "paid_usage_possible": False,
                    "blockers": ["test fixture is disabled"],
                    "verification": {
                        "installed": True,
                        "authenticated": True,
                        "models_available": True,
                        "headless": True,
                        "workspace": True,
                        "tools": True,
                        "quota_visible": True,
                        "billing_verified": True,
                    },
                }
            ]
        },
    }
    (home / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")


def _route(profile: str, home: Path) -> TaskRoute:
    return TaskRoute(
        task="fixture",
        role="generic",
        profile=profile,
        provider="openai-codex",
        model="runtime-model",
        pool_alias="subscription",
        pool_type="subscription",
        cost_mode="included",
        budget_tokens=1_000,
        delegated=True,
        home=str(home),
        subscription_pool="subscription",
        billing_mode="subscription",
    )


def _profile(route: TaskRoute) -> ProfileRouteConfig:
    return ProfileRouteConfig(
        route.profile,
        route.home,
        route.provider,
        route.model,
        ("general",),
        1_000,
        route.pool_alias,
        route.pool_type,
        route.cost_mode,
    )


def test_profile_governors_load_effectively_and_preserve_other_limits(tmp_path):
    root = tmp_path / "hermes"
    engineering = root / "profiles" / "engineering"
    assurance = root / "profiles" / "assurance"
    _write_profile(root, turns=80, request_tokens=200_000, threshold=0.50, route_id="dru-pool")
    _write_profile(
        engineering,
        turns=120,
        request_tokens=220_000,
        threshold=0.55,
        route_id="brok-pool",
        codex_autoraise=False,
    )
    _write_profile(assurance, turns=80, request_tokens=200_000, threshold=0.50, route_id="tyr-pool")
    active_profile = root / "active_profile"
    active_profile.write_text("default\n", encoding="utf-8")

    token = set_hermes_home_override(root)
    try:
        for profile, home, expected in (
            ("default", root, (80, 200_000, 0.50)),
            ("engineering", engineering, (120, 220_000, 0.55)),
            ("assurance", assurance, (80, 200_000, 0.50)),
        ):
            route = _route(profile, home)
            governor = resolve_profile_governor(route)
            assert (
                governor.max_turns,
                governor.max_request_input_tokens,
                governor.compression_threshold,
            ) == expected
            with profile_runtime_scope(route):
                config = load_config()
                assert config["cron"]["script_timeout_seconds"] == 60
                assert config["delegation"]["max_concurrent_children"] == 2
                interactive = config["usage_guardrails"]["interactive"]
                assert {
                    "max_model_calls": interactive["max_model_calls"],
                    "max_subscription_weighted_input_tokens": interactive[
                        "max_subscription_weighted_input_tokens"
                    ],
                    "max_consecutive_large_requests": interactive[
                        "max_consecutive_large_requests"
                    ],
                    "large_request_tokens": interactive["large_request_tokens"],
                } == {
                    "max_model_calls": 512,
                    "max_subscription_weighted_input_tokens": 3_000_000,
                    "max_consecutive_large_requests": 24,
                    "large_request_tokens": 150_000,
                }
        assert get_hermes_home() == root
        assert active_profile.read_text(encoding="utf-8") == "default\n"
    finally:
        reset_hermes_home_override(token)


def test_pool_configuration_is_profile_scoped_without_leakage(tmp_path):
    root = tmp_path / "hermes"
    engineering = root / "profiles" / "engineering"
    _write_profile(root, turns=80, request_tokens=200_000, threshold=0.50, route_id="dru-pool")
    _write_profile(engineering, turns=120, request_tokens=220_000, threshold=0.55, route_id="brok-pool")
    dru = _route("default", root)
    brok = _route("engineering", engineering)

    token = set_hermes_home_override(root)
    try:
        with profile_runtime_scope(dru):
            dru_pools = _configured_pool_routes(_profile(dru))
        with profile_runtime_scope(brok):
            brok_pools = _configured_pool_routes(_profile(brok))
    finally:
        reset_hermes_home_override(token)

    assert [item.route_id for item in dru_pools] == ["dru-pool"]
    assert [item.route_id for item in brok_pools] == ["brok-pool"]


def test_brok_profile_threshold_disables_codex_autoraise(tmp_path):
    root = tmp_path / "hermes"
    engineering = root / "profiles" / "engineering"
    _write_profile(
        engineering,
        turns=120,
        request_tokens=220_000,
        threshold=0.55,
        route_id="brok-pool",
        codex_autoraise=False,
    )
    route = _route("engineering", engineering)

    with profile_runtime_scope(route):
        config = load_config()
    governor = resolve_profile_governor(route)
    codex_autoraise = config["compression"]["codex_gpt55_autoraise"]
    model_threshold = _compression_threshold_for_model(
        "gpt-5.6-terra",
        "openai-codex",
        allow_codex_gpt55_autoraise=codex_autoraise,
    )
    effective_threshold, notice = _resolve_compression_threshold(
        governor.compression_threshold,
        model_threshold,
        model="gpt-5.6-terra",
        is_codex_autoraise=True,
    )

    assert codex_autoraise is False
    assert model_threshold is None
    assert effective_threshold == 0.55
    assert notice is None


def test_authoritative_profile_registry_feeds_task_routing_loader(tmp_path, monkeypatch):
    root = tmp_path / "hermes"
    engineering = root / "profiles" / "engineering"
    _write_profile(
        root,
        turns=80,
        request_tokens=200_000,
        threshold=0.50,
        route_id="dru-pool",
    )
    _write_profile(
        engineering,
        turns=120,
        request_tokens=220_000,
        threshold=0.55,
        route_id="brok-pool",
        codex_autoraise=False,
    )
    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: root)

    candidates = {candidate.profile: candidate for candidate in _profile_configs()}

    assert candidates["default"].home == str(root)
    assert candidates["default"].model == "gpt-5.6-terra"
    assert candidates["engineering"].home == str(engineering)
    assert candidates["engineering"].model == "gpt-5.6-terra"
