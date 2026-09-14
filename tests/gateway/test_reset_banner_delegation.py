"""Compare banner claims with real delegation routing, without network access."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from gateway.session_banner import format_reset_settings
from tools.delegate_tool_config import _resolve_child_runtime, _resolve_delegation_credentials


@pytest.mark.parametrize("style", ["providers", "custom_providers"])
@pytest.mark.parametrize("explicit_model", [None, "explicit-worker"])
@pytest.mark.parametrize("effort", [None, False, "low", "invalid"])
def test_provider_default_is_not_main_inheritance(tmp_path, monkeypatch, style, explicit_model, effort):
    import hermes_cli.runtime_provider as runtime_provider

    def forbidden(*args, **kwargs):
        pytest.fail("Banner/resolver must not probe network or real credentials")

    monkeypatch.setattr("socket.socket.connect", forbidden)
    monkeypatch.setattr("socket.socket.connect_ex", forbidden)
    monkeypatch.setattr("socket.getaddrinfo", forbidden)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    delegation = {"provider": "custom:review-worker", "reasoning_effort": effort}
    if explicit_model:
        delegation["model"] = explicit_model
    endpoint = {"name": "review-worker", "base_url": "https://worker.example.invalid/v1",
                "api_key": "synthetic-test-key"}
    if style == "providers":
        endpoint["default_model"] = "worker-default"
        providers = {"review-worker": endpoint}
    else:
        endpoint["model"] = "worker-default"
        providers = [endpoint]
    config = {style: providers, "delegation": delegation,
              "model": {"default": "main-model"}}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    # Exercise actual config lookup and runtime/delegation resolution. Only the
    # credential environment/pool boundaries are replaced with empty test stores.
    monkeypatch.setattr(runtime_provider, "_getenv", lambda *args: "")
    monkeypatch.setattr(runtime_provider, "load_pool", lambda *args, **kwargs:
                        SimpleNamespace(has_credentials=lambda: False))
    parent = SimpleNamespace(model="main-model", provider="openrouter",
                             base_url="https://main.example.invalid/v1",
                             reasoning_config={"enabled": True, "effort": "high"})
    credentials = _resolve_delegation_credentials(delegation, parent)
    assert credentials["model"] == (explicit_model or endpoint.get("default_model") or endpoint["model"])
    child = _resolve_child_runtime(
        parent, delegation, "synthetic-parent-key", model=credentials["model"],
        override_provider=credentials["provider"], override_base_url=credentials["base_url"],
        override_api_key=credentials["api_key"], override_api_mode=credentials["api_mode"],
        override_acp_command=None, override_acp_args=None,
    )
    runner = SimpleNamespace(_load_reasoning_config=lambda model: parent.reasoning_config,
                             _load_service_tier=lambda: None)
    # Banner collection must not reuse the credential-resolving path above.
    with patch.object(runtime_provider, "resolve_runtime_provider", side_effect=forbidden), \
         patch.object(runtime_provider, "_get_named_custom_provider", side_effect=forbidden), \
         patch.object(runtime_provider, "_getenv", side_effect=forbidden), \
         patch.object(runtime_provider, "load_pool", side_effect=forbidden):
        banner = format_reset_settings(runner, model=parent.model)
    if explicit_model:
        assert f"model: {child['model']} (configured)" in banner
    else:
        assert child["model"] != parent.model
        assert "model: inherited from main" not in banner
        assert "model: unknown (provider override)" in banner
    if effort is None or effort == "invalid":
        assert child["reasoning_config"] == parent.reasoning_config
        assert "effort: inherited from main" in banner
    elif effort is False:
        assert child["reasoning_config"]["enabled"] is False
        assert "effort: off (configured)" in banner
    else:
        assert child["reasoning_config"]["effort"] == effort
        assert f"effort: {effort} (configured)" in banner
    assert "synthetic-test-key" not in banner
    assert "example.invalid" not in banner


@pytest.mark.parametrize("delegation,expected", [
    ({"provider": "unknown-provider"}, "unknown (provider override)"),
    ({"provider": "auto"}, "unknown (provider override)"),
    ({"provider": "  ", "model": "  "}, "inherited from main"),
    ({"provider": "custom:worker", "model": "  "}, "unknown (provider override)"),
    ({"base_url": "https://worker.example.invalid/v1"}, "inherited from main"),
])
def test_unresolved_delegation_labels(tmp_path, monkeypatch, delegation, expected):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({"delegation": delegation}), encoding="utf-8")
    runner = SimpleNamespace(_load_reasoning_config=lambda model: None, _load_service_tier=lambda: None)
    assert f"model: {expected};" in format_reset_settings(runner, model="main-model")
