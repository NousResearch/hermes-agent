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


def test_explicit_client_kwargs_launches_any_external_process_profile(monkeypatch):
    """Launch kwargs are keyed on the provider profile's auth_type, not a hardcoded
    vendor slug: an out-of-tree external_process provider gets the same ACP launch
    path (command/args) as copilot-acp (#102421)."""
    import providers as providers_mod
    from agent.agent_init import _explicit_client_kwargs

    class _Agent:
        provider = "acme-acp"
        acp_command = ["acme"]
        acp_args = ["--stdio"]

    monkeypatch.setattr(providers_mod, "get_provider_profile",
                        lambda name: SimpleNamespace(auth_type="external_process"))
    kwargs = _explicit_client_kwargs(_Agent(), api_key=None, base_url=None, _provider_timeout=None)
    assert kwargs["command"] == ["acme"] and kwargs["args"] == ["--stdio"]


def test_explicit_client_kwargs_copilot_acp_still_launches(monkeypatch):
    """The built-in copilot-acp provider keeps its launch kwargs through the generic
    profile-keyed path — same contract, now without the vendor hardcode."""
    import providers as providers_mod
    from agent.agent_init import _explicit_client_kwargs

    class _Agent:
        provider = "copilot-acp"
        acp_command = ["copilot"]
        acp_args = []

    monkeypatch.setattr(providers_mod, "get_provider_profile",
                        lambda name: SimpleNamespace(auth_type="external_process"))
    kwargs = _explicit_client_kwargs(_Agent(), api_key=None, base_url=None, _provider_timeout=None)
    assert kwargs["command"] == ["copilot"]


@pytest.mark.parametrize("auth_type", ["oauth_device_code", "oauth_external", "aws_sdk", "copilot", "vertex"])
def test_bespoke_auth_classes_still_excluded(auth_type):
    from hermes_cli.models_catalog_static import _plugin_provider_enters_picker

    assert _plugin_provider_enters_picker(_profile("acme-" + auth_type.replace("_", ""), auth_type)) is False


def test_api_key_plugin_providers_unchanged():
    from hermes_cli.models_catalog_static import _plugin_provider_enters_picker

    assert _plugin_provider_enters_picker(_profile("acme-api", "api_key")) is True


def test_live_catalog_tolerates_credential_kwargs_fetch_models():
    """A signature-strict external_process profile (fetch_models requiring keyword-only
    api_key/base_url) still yields its catalog: probe no-args, fall back to credentials
    (#111194 hardened discovery the same way)."""
    from hermes_cli.models import _profile_live_catalog

    def _strict_fetch(*, api_key, base_url):
        assert api_key, "credential kwargs must be passed on the TypeError fallback"
        return ["acme-pro"]

    profile = SimpleNamespace(
        auth_type="external_process", fetch_models=_strict_fetch,
        base_url="https://acme.example/v1",
    )
    import providers as providers_mod
    import hermes_cli.models as models_mod
    orig = getattr(providers_mod, "get_provider_profile", None)
    providers_mod.get_provider_profile = lambda name: profile
    orig_creds = models_mod._api_key_credentials
    models_mod._api_key_credentials = lambda normalized: ("test-key", None)
    try:
        assert _profile_live_catalog("acme-acp") == ["acme-pro"]
    finally:
        if orig is not None:
            providers_mod.get_provider_profile = orig
        models_mod._api_key_credentials = orig_creds


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
