"""Picker admission for out-of-tree provider plugins with non-api-key auth types.

Regression: the CANONICAL_PROVIDERS auto-extend skipped every plugin profile whose ``auth_type``
was not ``api_key``, so an out-of-tree external-process (ACP) or OAuth provider loaded, inferred and
was invisible to ``/model``, ``hermes model`` and ``list_available_providers()`` — while the in-tree
copilot-acp row (same auth_type, hand-written entry) appeared. Visibility is gated by credentials
downstream, so admission by slug is safe (#102421).
"""

import os
import stat

import pytest


def _register(monkeypatch, profile):
    import providers
    from hermes_cli import auth

    monkeypatch.setitem(providers._REGISTRY, profile.name, profile)
    monkeypatch.delitem(auth.PROVIDER_REGISTRY, profile.name, raising=False)
    # Mirror into the auth registry the way plugin discovery does; built from public types so the
    # helper is independent of the auth module's private mirroring function.
    pconfig = auth.ProviderConfig(
        profile.name, profile.display_name or profile.name, profile.auth_type, inference_base_url=profile.base_url)
    if profile.auth_type == "api_key" and profile.env_vars:
        pconfig = auth._api_key_provider(profile.name, profile.display_name or profile.name, profile.base_url, tuple(profile.env_vars), "")
    monkeypatch.setitem(auth.PROVIDER_REGISTRY, profile.name, pconfig)


@pytest.mark.parametrize("auth_type", ["external_process", "oauth_external", "oauth_device_code", "api_key"])
def test_plugin_profiles_are_admitted_by_slug_not_auth_type(auth_type):
    from providers.base import ProviderProfile
    from hermes_cli.models_catalog_static import CANONICAL_PROVIDERS, _plugin_provider_enters_picker

    assert _plugin_provider_enters_picker(ProviderProfile(name="acme-plugin", auth_type=auth_type)) is True
    # A plugin re-declaring a built-in slug is deduped, never doubled (bedrock is aws_sdk in-tree).
    assert _plugin_provider_enters_picker(ProviderProfile(name="bedrock", auth_type="aws_sdk")) is False
    assert [p.slug for p in CANONICAL_PROVIDERS].count("bedrock") == 1


def test_external_process_plugin_authenticated_flag_tracks_binary_and_catalog_uses_fallback(
        monkeypatch, tmp_path):
    """The row's authenticated flag is the real binary-resolves gate (not a hardcoded slug), and the
    profile's fallback_models is its catalog when the subprocess probe yields nothing."""
    from providers.base import ProviderProfile
    from hermes_cli import models, models_catalog_static

    exe = tmp_path / "acme-acp"
    exe.write_text("#!/bin/sh\nexit 0\n")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    profile = ProviderProfile(
        name="acme-acp", auth_type="external_process", base_url="acp://acme", process_command="acme-acp",
        fallback_models=("acme-acp",))
    _register(monkeypatch, profile)
    entry = models_catalog_static.ProviderEntry("acme-acp", "Acme ACP", "acme")
    monkeypatch.setattr(models, "CANONICAL_PROVIDERS", [*models.CANONICAL_PROVIDERS, entry])

    def authenticated():
        return {r["id"]: r["authenticated"] for r in models.list_available_providers()}["acme-acp"]

    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}")
    assert authenticated() is True
    monkeypatch.setenv("PATH", str(tmp_path / "nowhere"))
    assert authenticated() is False
    assert models.provider_model_ids("acme-acp") == ["acme-acp"]
