from __future__ import annotations

import pytest

from agent.model_performance_router import (
    REQUIRED_SURFACES,
    RouteCompilationError,
    compile_profile_routes,
    resolve_route,
)


def _route(provider: str = "nous", model: str = "fast-model", **overrides):
    row = {
        "provider": provider,
        "model": model,
        "reasoning_effort": None,
        "context_window": 131_072,
        "max_output_tokens": 4096,
        "no_output_timeout_seconds": 20.0,
        "total_timeout_seconds": 90.0,
        "concurrency_weight": 1,
        "privacy_class": "sanitized_remote",
        "destination_class": "remote",
        "benchmark_case_ids": ["case-1"],
    }
    row.update(overrides)
    return row


def _artifact(*, profiles=("default",), missing: str | None = None):
    profile_rows = {}
    for profile in profiles:
        rows = {}
        for surface in REQUIRED_SURFACES:
            if surface == missing:
                continue
            rows[surface] = {
                "primary": _route(),
                "privacy_fallback": _route(
                    provider="ollama",
                    model="local-fast",
                    privacy_class="private_local",
                    destination_class="loopback",
                ),
            }
        profile_rows[profile] = rows
    return {
        "artifact_digest": "a" * 64,
        "policy_digest": "b" * 64,
        "profiles": profile_rows,
    }


def test_compiler_requires_every_profile_surface_and_explicit_policy():
    compiled = compile_profile_routes(["default", "reviewer"], _artifact(profiles=("default", "reviewer")))

    assert set(compiled) == {"default", "reviewer"}
    assert set(compiled["default"]) == REQUIRED_SURFACES
    for rows in compiled.values():
        for row in rows.values():
            assert row.primary.context_window >= 65_536
            assert row.primary.no_output_timeout_seconds > 0
            assert row.primary.total_timeout_seconds >= row.primary.no_output_timeout_seconds
            assert row.primary.artifact_digest == "a" * 64
            assert row.primary.policy_digest == "b" * 64
            assert row.privacy_fallback.destination_class == "loopback"


def test_missing_auxiliary_surface_fails_closed():
    with pytest.raises(RouteCompilationError, match="skills_hub"):
        compile_profile_routes(["default"], _artifact(missing="skills_hub"))


def test_missing_profile_fails_closed_instead_of_inheriting_default():
    with pytest.raises(RouteCompilationError, match="reviewer"):
        compile_profile_routes(["default", "reviewer"], _artifact())


def test_remote_privacy_fallback_is_rejected():
    artifact = _artifact()
    artifact["profiles"]["default"]["review"]["privacy_fallback"] = _route()

    with pytest.raises(RouteCompilationError, match="privacy_fallback"):
        compile_profile_routes(["default"], artifact)


def test_unbenchmarked_or_undersized_route_is_rejected():
    artifact = _artifact()
    artifact["profiles"]["default"]["compression"]["primary"]["benchmark_case_ids"] = []
    with pytest.raises(RouteCompilationError, match="benchmark_case_ids"):
        compile_profile_routes(["default"], artifact)

    artifact = _artifact()
    artifact["profiles"]["default"]["compression"]["primary"]["context_window"] = 16_384
    with pytest.raises(RouteCompilationError, match="context_window"):
        compile_profile_routes(["default"], artifact)


def test_resolver_enforces_privacy_and_call_context_floor():
    compiled = compile_profile_routes(["default"], _artifact())

    private = resolve_route(
        compiled,
        profile="default",
        surface="review",
        privacy="private",
        required_context=65_536,
    )
    assert private.provider == "ollama"

    with pytest.raises(RouteCompilationError, match="required context"):
        resolve_route(
            compiled,
            profile="default",
            surface="review",
            privacy="sanitized",
            required_context=200_000,
        )
