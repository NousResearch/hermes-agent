"""Picker admission for out-of-tree provider plugins across auth types."""

import os
import stat

import pytest


def _register(monkeypatch, profile):
    import providers

    monkeypatch.setattr(providers.registry, "_REGISTRY", dict(providers.registry._REGISTRY))
    monkeypatch.setattr(providers.registry, "_ALIASES", dict(providers.registry._ALIASES))
    monkeypatch.setattr(providers.registry, "_SOURCES", dict(providers.registry._SOURCES))
    monkeypatch.setattr(providers.registry, "_PROVIDER_LIST_CACHE", None)
    providers.register_provider(profile)


@pytest.mark.parametrize(
    "auth_type",
    ["external_process", "oauth_external", "oauth_device_code", "api_key"],
)
def test_plugin_profiles_are_admitted_by_slug_not_auth_type(auth_type, monkeypatch):
    from hermes_cli.provider_catalog import provider_catalog_by_slug, provider_slugs
    from providers.base import ProviderProfile

    slug = f"acme-plugin-{auth_type}"
    _register(
        monkeypatch,
        ProviderProfile(
            name=slug,
            display_name="Acme Plugin",
            description="Acme plugin fixture",
            auth_type=auth_type,
        ),
    )

    descriptor = provider_catalog_by_slug()[slug]
    assert descriptor.auth_type == auth_type
    assert provider_slugs().count(slug) == 1
    assert provider_slugs().count("bedrock") == 1


def test_external_process_plugin_authenticated_flag_tracks_binary_and_catalog_uses_fallback(
    monkeypatch, tmp_path
):
    """Authentication follows binary resolution and fallback models stay available."""
    from hermes_cli import models
    from providers.base import ProviderProfile

    exe = tmp_path / "acme-acp"
    exe.write_text("#!/bin/sh\nexit 0\n")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    profile = ProviderProfile(
        name="acme-acp",
        display_name="Acme ACP",
        description="Acme external-process provider",
        auth_type="external_process",
        base_url="acp://acme",
        process_command="acme-acp",
        fallback_models=("acme-acp",),
    )
    _register(monkeypatch, profile)

    def authenticated():
        return {
            row["id"]: row["authenticated"]
            for row in models.list_available_providers()
        }["acme-acp"]

    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}")
    assert authenticated() is True
    monkeypatch.setenv("PATH", str(tmp_path / "nowhere"))
    assert authenticated() is False
    assert models.provider_model_ids("acme-acp") == ["acme-acp"]
