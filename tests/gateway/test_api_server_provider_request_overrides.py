"""A per-request ``provider`` swap must replace ``request_overrides``, not keep the default
provider's ``extra_body`` on the agent (the provider docs promise extra_body "never leaks
to another provider"). Regression for #134587."""

import json


def test_provider_swap_replaces_request_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = {
        "model": {"default": "fixture", "provider": "default-custom"},
        "providers": {
            "default-custom": {
                "api": "http://127.0.0.1:1/v1",
                "api_key": "default-key",
                "extra_body": {"max_tokens": 4096, "include_reasoning": True},
            },
            "other-custom": {
                "api": "http://127.0.0.1:2/v1",
                "api_key": "other-key",
            },
        },
    }
    (tmp_path / "config.yaml").write_text(json.dumps(config))

    from gateway.platforms.api_server import (
        _apply_runtime_agent_overrides,
        _resolve_request_runtime_agent_kwargs,
    )

    default_runtime = _resolve_request_runtime_agent_kwargs(provider="default-custom", target_model="fixture")
    assert default_runtime["request_overrides"] == {
        "extra_body": {"max_tokens": 4096, "include_reasoning": True},
    }

    other_runtime = _resolve_request_runtime_agent_kwargs(provider="other-custom", target_model="fixture")
    assert other_runtime["request_overrides"] == {}

    # The agent is built from the default provider's kwargs; the per-request
    # swap must overwrite request_overrides (empty for a provider without
    # extra_body) rather than leaving the default's value in place.
    agent_kwargs = _apply_runtime_agent_overrides(dict(default_runtime), other_runtime)
    assert agent_kwargs["request_overrides"] == {}
    assert agent_kwargs["base_url"] == "http://127.0.0.1:2/v1"

    # Naming the default provider keeps its own extra_body.
    same = _apply_runtime_agent_overrides(dict(default_runtime), default_runtime)
    assert same["request_overrides"] == {
        "extra_body": {"max_tokens": 4096, "include_reasoning": True},
    }
