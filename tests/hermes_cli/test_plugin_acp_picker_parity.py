"""Picker parity for plugin external-process (ACP) providers.

Regression: the CANONICAL_PROVIDERS auto-extend skipped every plugin
`auth_type="external_process"` provider, so plugin ACP backends never reached
`/model` / `list_available_providers()` — while the in-tree copilot-acp
provider (same auth_type, hardcoded list entry) does appear. Selection already
works through the standard model_switch pipeline, and the authenticated flag
comes from auth.get_external_process_provider_status; only picker admission
was missing.
"""

from types import SimpleNamespace

import pytest


def _profile(name: str, auth_type: str) -> SimpleNamespace:
    return SimpleNamespace(
        name=name, auth_type=auth_type, display_name=name.title(),
        description=f"{name} test provider", aliases=(),
    )


def test_external_process_plugin_provider_admitted():
    from hermes_cli.models_catalog_static import _plugin_provider_enters_picker

    assert _plugin_provider_enters_picker(_profile("acme-acp", "external_process")) is True


@pytest.mark.parametrize("auth_type", ["oauth_device_code", "oauth_external", "aws_sdk", "copilot", "vertex"])
def test_bespoke_auth_classes_still_excluded(auth_type):
    from hermes_cli.models_catalog_static import _plugin_provider_enters_picker

    assert _plugin_provider_enters_picker(_profile("acme-" + auth_type.replace("_", ""), auth_type)) is False


def test_api_key_plugin_providers_unchanged():
    from hermes_cli.models_catalog_static import _plugin_provider_enters_picker

    assert _plugin_provider_enters_picker(_profile("acme-api", "api_key")) is True


def test_auto_extend_seats_external_process_provider(monkeypatch):
    """Integration through the real auto-extend loop (reloaded with a stubbed
    plugin registry, mirroring discovery output)."""
    import importlib
    import providers
    import hermes_cli.models_catalog_static as mcs

    fake = [_profile("acme-acp", "external_process"), _profile("acme-oauth", "oauth_external")]
    monkeypatch.setattr(providers, "list_providers", lambda: fake)
    reloaded = importlib.reload(mcs)
    try:
        ids = [p.slug for p in reloaded.CANONICAL_PROVIDERS]
        assert "acme-acp" in ids
        assert "acme-oauth" not in ids
    finally:
        importlib.reload(mcs)  # restore the real module state
