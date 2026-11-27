import pytest

from hermes_cli import providers
from hermes_cli.model_switch import list_authenticated_providers
from hermes_cli.providers import resolve_provider_full


@pytest.mark.parametrize("separator", [" - ", " — "])
def test_legacy_picker_group_slug_resolves_to_its_configured_provider(monkeypatch, separator):
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", dict)
    monkeypatch.setattr(providers, "HERMES_OVERLAYS", {})
    endpoint = {
        "name": f"Lab{separator}Local Qwen",
        "base_url": "http://127.0.0.1:18300/v1",
        "model": "local-qwen",
        "discover_models": False,
    }
    rows = list_authenticated_providers(
        user_providers={},
        custom_providers=[endpoint],
        probe_custom_providers=False,
    )
    slug = next(row["slug"] for row in rows if row["name"] == "Lab")
    resolved = resolve_provider_full(slug, user_providers={}, custom_providers=[endpoint])
    assert resolved is not None
    assert resolved.name == endpoint["name"]
    assert resolved.base_url == endpoint["base_url"]


def test_legacy_picker_group_slug_does_not_shadow_an_exact_provider():
    endpoint = {"name": "Lab", "base_url": "http://127.0.0.1:18301/v1"}
    resolved = resolve_provider_full(
        "custom:lab",
        user_providers={},
        custom_providers=[
            {"name": "Lab - Local Qwen", "base_url": "http://127.0.0.1:18300/v1"},
            endpoint,
        ],
    )
    assert resolved is not None
    assert resolved.base_url == endpoint["base_url"]
