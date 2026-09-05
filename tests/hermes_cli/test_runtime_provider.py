"""Provider-neutral whole-turn runtime resolution contracts."""

from __future__ import annotations

from types import SimpleNamespace

from hermes_cli import runtime_provider as rp


def test_agent_runtime_resolution_skips_credentials_and_pool(monkeypatch):
    config = {
        "model": {
            "provider": "custom:synthetic-runtime",
            "default": "synthetic-model",
        },
        "providers": {
            "synthetic-runtime": {
                "api": "runtime://synthetic",
                "api_mode": "agent_runtime",
            }
        },
    }
    monkeypatch.setattr(rp, "load_config", lambda: config)

    def fail(*_args, **_kwargs):
        raise AssertionError("agent_runtime must not enter credential resolution")

    monkeypatch.setattr(rp, "_try_resolve_from_custom_pool", fail)
    monkeypatch.setattr(rp, "resolve_api_key_provider_credentials", fail)
    monkeypatch.setattr(rp, "resolve_external_process_provider_credentials", fail)

    resolved = rp.resolve_runtime_provider(requested="custom:synthetic-runtime")

    assert resolved["provider"] == "custom"
    assert resolved["api_mode"] == "agent_runtime"
    assert resolved["base_url"] == "runtime://synthetic"
    assert resolved["api_key"] == ""
    assert resolved["requested_provider"] == "custom:synthetic-runtime"


def test_agent_runtime_mode_is_a_valid_runtime_provider_mode():
    assert rp._parse_api_mode("agent_runtime") == "agent_runtime"


def test_agent_runtime_profile_uses_structural_endpoint_without_credentials(monkeypatch):
    config = {
        "model": {
            "provider": "synthetic-profile",
            "default": "synthetic-model",
        }
    }
    monkeypatch.setattr(rp, "load_config", lambda: config)

    def fail(*_args, **_kwargs):
        raise AssertionError("agent_runtime must not enter credential resolution")

    monkeypatch.setattr(rp, "resolve_api_key_provider_credentials", fail)
    monkeypatch.setattr(rp, "resolve_external_process_provider_credentials", fail)

    import providers

    monkeypatch.setattr(
        providers,
        "get_provider_profile",
        lambda _name: SimpleNamespace(
            name="synthetic-profile",
            api_mode="agent_runtime",
            base_url="",
        ),
    )

    resolved = rp.resolve_runtime_provider(requested="synthetic-profile")

    assert resolved["provider"] == "synthetic-profile"
    assert resolved["api_mode"] == "agent_runtime"
    assert resolved["base_url"] == "runtime://synthetic-profile"
    assert resolved["api_key"] == ""
