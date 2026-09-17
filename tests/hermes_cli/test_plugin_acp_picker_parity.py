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


def test_external_process_providers_in_model_options_payload(monkeypatch):
    """Layer 1 admits it into CANONICAL_PROVIDERS; layer 2 (inventory) seats a
    row with a self-named model, reachability as the credential."""
    import hermes_cli.auth as auth
    import hermes_cli.inventory as inv
    import hermes_cli.models as models

    fake = _make_registry_profile()
    monkeypatch.setattr("providers.list_providers", lambda: [fake])
    monkeypatch.setattr(auth, "get_external_process_provider_status", lambda slug: {"configured": True})
    entry = models.ProviderEntry("ext-proc", "Ext Proc", "external process test provider")
    monkeypatch.setattr(models, "CANONICAL_PROVIDERS", list(models.CANONICAL_PROVIDERS) + [entry])
    ctx = inv.ConfigContext(
        current_provider="zai", current_model="glm-5.3",
        current_base_url="https://api.z.ai/api/coding/paas/v4",
        user_providers={}, custom_providers=[])
    payload = inv.build_model_options_payload(ctx, explicit_only=False)
    rows = [p for p in payload["providers"] if p.get("slug") == "ext-proc"]
    assert rows, "external_process plugin provider missing from model.options payload"
    assert rows[0]["models"] == ["ext-proc"]
    assert rows[0]["authenticated"] is True


def test_unresolvable_external_process_binary_not_listed(monkeypatch):
    """Reachability is the credential: a binary that doesn't resolve stays hidden."""
    import hermes_cli.auth as auth
    import hermes_cli.inventory as inv
    import hermes_cli.models as models

    fake = _make_registry_profile()
    monkeypatch.setattr("providers.list_providers", lambda: [fake])
    monkeypatch.setattr(
        auth, "get_external_process_provider_status",
        lambda slug: {"configured": False})
    entry = models.ProviderEntry("ext-proc", "Ext Proc", "external process test provider")
    monkeypatch.setattr(models, "CANONICAL_PROVIDERS", list(models.CANONICAL_PROVIDERS) + [entry])
    ctx = inv.ConfigContext(
        current_provider="zai", current_model="glm-5.3", current_base_url="",
        user_providers={}, custom_providers=[])
    payload = inv.build_model_options_payload(ctx, explicit_only=False)
    assert not [p for p in payload["providers"] if p.get("slug") == "ext-proc"]


def _make_registry_profile():
    """A profile + matching ProviderConfig, registered in both registries."""
    from types import SimpleNamespace
    profile = SimpleNamespace(
        name="ext-proc", auth_type="external_process", display_name="Ext Proc",
        description="external process test provider", aliases=(),
        process_command="/fake/does-not-exist", process_args=("acp",),
    )
    import hermes_cli.auth as auth
    registry_entry = SimpleNamespace(
        name="ext-proc", display_name="Ext Proc", auth_type="external_process",
        api_key_env_vars=(), base_url="acp://ext", inference_base_url="acp://ext",
        command="ext-proc", args=("acp",), base_url_env_var=None,
    )
    auth.PROVIDER_REGISTRY.setdefault("ext-proc", registry_entry)
    return profile


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
