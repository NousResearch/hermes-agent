"""External-process provider plugins participate in the model picker."""

from __future__ import annotations

import importlib

from providers import register_provider
from providers.base import ProviderProfile


class _PickerACPProfile(ProviderProfile):
    def create_client(self, **kwargs):
        return kwargs

    def fetch_models(self, **kwargs):
        return ["picker-acp-live", "picker-acp-next"]


register_provider(
    _PickerACPProfile(
        name="picker-acp",
        aliases=("picker",),
        display_name="Picker ACP",
        description="An out-of-tree ACP provider for picker tests",
        base_url="acp://picker",
        auth_type="external_process",
        process_command="picker-cli",
        process_args=("acp",),
        fallback_models=("picker-acp-fallback",),
    )
)


def test_external_process_plugin_is_visible_and_uses_its_live_catalog():
    """A plugin profile needs neither an API key nor a bespoke core picker flow."""
    import hermes_cli.models as models
    import hermes_cli.models_catalog_static as catalog

    # Test collection can import these modules before this test module registers its profile.
    # Reloading makes their import-time plugin catalog extension deterministic.
    importlib.reload(catalog)
    importlib.reload(models)

    assert any(entry.slug == "picker-acp" for entry in catalog.CANONICAL_PROVIDERS)
    assert catalog._PROVIDER_ALIASES["picker"] == "picker-acp"
    assert models.provider_model_ids("picker-acp") == [
        "picker-acp-live", "picker-acp-next", "picker-acp-fallback"
    ]


def test_external_process_model_flow_persists_selected_model(monkeypatch):
    """The generic flow selects and persists a plugin-provided ACP model."""
    import hermes_cli.model_setup_flows as flows

    picked: dict[str, object] = {}
    monkeypatch.setattr(
        "hermes_cli.auth.get_external_process_provider_status",
        lambda provider_id: {"resolved_command": "picker-cli", "base_url": "acp://picker"},
    )
    monkeypatch.setattr(
        "hermes_cli.auth.resolve_external_process_provider_credentials",
        lambda provider_id: {"base_url": "acp://picker"},
    )
    monkeypatch.setattr(
        "hermes_cli.models.cached_provider_model_ids", lambda provider_id: ["picker-acp-live"]
    )
    monkeypatch.setattr(flows, "_pick_model_or_prompt", lambda *args, **kwargs: "picker-acp-live")
    monkeypatch.setattr(
        flows,
        "_finish_model",
        lambda model, provider, message, **kwargs: picked.update(
            model=model, provider=provider, base_url=kwargs["base_url"], api_mode=kwargs["api_mode"]
        ),
    )

    flows._model_flow_external_process({}, "picker-acp", "old-model")

    assert picked == {
        "model": "picker-acp-live",
        "provider": "picker-acp",
        "base_url": "acp://picker",
        "api_mode": "chat_completions",
    }


def test_select_provider_and_model_routes_external_process_plugin_to_generic_flow(monkeypatch):
    """The picker dispatches a plugin ACP provider instead of silently doing nothing."""
    import hermes_cli.main as main

    routed: dict[str, object] = {}
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"model": {"default": "old-model"}})
    monkeypatch.setattr(main, "_resolve_active_provider", lambda *args: None)
    monkeypatch.setattr(main, "_pick_provider", lambda *args: "picker-acp")
    monkeypatch.setattr(
        main,
        "_model_flow_external_process",
        lambda config, provider_id, current_model: routed.update(
            provider=provider_id, current_model=current_model
        ),
    )
    monkeypatch.setattr(main, "_offer_reasoning_after_pick", lambda *args: None)
    monkeypatch.setattr(main, "_clear_stale_openai_base_url", lambda: None)

    main.select_provider_and_model()

    assert routed == {"provider": "picker-acp", "current_model": "old-model"}


def test_in_session_picker_lists_configured_external_process_plugin(monkeypatch):
    """`/model`, TUI, and desktop pickers accept a configured plugin CLI as credentials."""
    import hermes_cli.model_switch_providers as switch_providers
    from hermes_cli.model_switch import list_authenticated_providers

    monkeypatch.setattr(
        "hermes_cli.auth.get_auth_status",
        lambda provider_id: {"configured": provider_id == "picker-acp"},
    )
    monkeypatch.setattr(switch_providers, "_auth_store_has_provider", lambda *args: False)
    monkeypatch.setattr(switch_providers, "_pool_usable", lambda *args: False)
    monkeypatch.setattr(
        switch_providers, "_live_or_curated_ids", lambda slug, *args, **kwargs: ["picker-acp-live"]
    )

    rows = list_authenticated_providers(current_provider="openrouter", max_models=50)
    row = next((row for row in rows if row["slug"] == "picker-acp"), None)

    assert row is not None
    assert row["models"] == ["picker-acp-live"]
