from __future__ import annotations

import json

import pytest

from agent.model_performance_router import REQUIRED_SURFACES, RouteCompilationError
from hermes_cli.profile_route_compiler import compile_profile_configs


def _route(provider: str, *, destination: str):
    return {
        "provider": provider,
        "model": "model",
        "reasoning_effort": None,
        "context_window": 65_536,
        "max_output_tokens": 2048,
        "no_output_timeout_seconds": 10,
        "total_timeout_seconds": 30,
        "concurrency_weight": 1,
        "privacy_class": "private_local" if destination == "loopback" else "sanitized_remote",
        "destination_class": destination,
        "benchmark_case_ids": ["case"],
    }


def _artifact():
    rows = {
        surface: {
            "primary": _route("nous", destination="remote"),
            "privacy_fallback": _route("ollama", destination="loopback"),
        }
        for surface in REQUIRED_SURFACES
    }
    return {
        "artifact_digest": "a" * 64,
        "policy_digest": "b" * 64,
        "profiles": {"default": rows},
    }


def test_compile_writes_complete_digest_bound_profile_config(tmp_path):
    destination = tmp_path / "routes.json"
    compiled = compile_profile_configs(["default"], _artifact(), destination=destination)

    written = json.loads(destination.read_text())
    assert set(written["profiles"]["default"]) == REQUIRED_SURFACES
    assert written["artifact_digest"] == "a" * 64
    assert compiled["default"]["title_generation"].primary.provider == "nous"


def test_compile_does_not_replace_existing_file_when_validation_fails(tmp_path):
    destination = tmp_path / "routes.json"
    destination.write_text('{"previous": true}\n')
    artifact = _artifact()
    del artifact["profiles"]["default"]["vision"]

    with pytest.raises(RouteCompilationError, match="vision"):
        compile_profile_configs(["default"], artifact, destination=destination)

    assert destination.read_text() == '{"previous": true}\n'
