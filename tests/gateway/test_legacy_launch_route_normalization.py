"""Legacy adoption keeps automatic resets and provider-owned pseudo-addresses compatible."""

import json

import pytest


@pytest.mark.parametrize("storage", ["gateway_runtime", "top_level", "billing"])
@pytest.mark.parametrize(
    "provider, base_url, expected_url",
    [
        ("claude-code", "process://claude", None),
        ("moa", "moa://default", None),
        ("custom:work", "process://worker", None),
        ("custom", "process://worker", "process://worker"),
        ("custom", "moa://default", "moa://default"),
        ("custom:work", "http://127.0.0.1:9876/v1", "http://127.0.0.1:9876/v1"),
        ("custom:work", "https://models.example/v1", "https://models.example/v1"),
    ],
)
def test_legacy_launch_preserves_routes_without_provider_pseudo_endpoints(storage, provider, base_url, expected_url):
    from gateway.session_local_migration import legacy_launch_params

    route = {"model": "saved-model", "provider": provider, "base_url": base_url}
    row = {"source": "cli", "model": "saved-model"}
    if storage == "billing":
        row.update(billing_provider=provider, billing_base_url=base_url)
    else:
        saved = {"gateway_runtime": route} if storage == "gateway_runtime" else route
        row["model_config"] = json.dumps(saved)

    params, api_mode = legacy_launch_params(row)

    expected = {"model": "saved-model", "provider": provider}
    if expected_url is not None:
        expected["base_url"] = expected_url
    assert params == expected
    assert api_mode is None


@pytest.mark.parametrize("storage", ["gateway_runtime", "top_level"])
@pytest.mark.parametrize("model", ["saved-model", ""])
@pytest.mark.parametrize("base_url", [None, "process://claude", "moa://default"])
def test_legacy_auto_reset_inherits_profile_route_after_pseudo_endpoint_cleanup(tmp_path, storage, model, base_url):
    from gateway.session_local_migration import legacy_launch_params
    from gateway.session_policy import build_policy

    route = {"model": model, "provider": "auto"}
    if base_url is not None:
        route["base_url"] = base_url
    saved = {"gateway_runtime": route} if storage == "gateway_runtime" else route
    row = {
        "source": "cli",
        "model": model,
        "model_config": json.dumps(saved),
        "billing_provider": "custom:old",
        "billing_base_url": "http://127.0.0.1:9876/v1",
    }
    profile = {
        "model": {
            "default": "profile-model",
            "provider": "custom:current",
            "base_url": "http://127.0.0.1:9877/v1",
            "api_mode": "chat_completions",
        },
    }

    params, api_mode = legacy_launch_params(row)
    policy = build_policy({**params, "cwd": str(tmp_path)}, profile, profile_terminal=False)

    assert params == ({"model": model} if model else {})
    assert api_mode is None
    assert policy.config()["model"]["provider"] == "custom:current"
    assert policy.config()["model"]["base_url"] == "http://127.0.0.1:9877/v1"
    assert policy.config()["model"]["api_mode"] == "chat_completions"
