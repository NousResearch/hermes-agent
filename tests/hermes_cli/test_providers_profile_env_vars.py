"""A plugin provider profile keeps its own credential and endpoint env vars when models.dev lists the
same id under different names.

Regression: models.dev added ``commandcode`` keyed as ``CMD_API_KEY``. ``get_provider`` returns the
models.dev entry first, so ``COMMANDCODE_API_KEY`` (what ``hermes setup`` writes, what the plugin
profile declares) stopped being a credential of the provider and ``COMMANDCODE_BASE_URL`` stopped
being its endpoint override: an existing CommandCode user looked unconfigured.
"""

from agent.models_dev import ProviderInfo


def test_models_dev_entry_adds_names_but_never_drops_the_profiles(monkeypatch):
    import hermes_cli.providers as providers_mod
    from providers import get_provider_profile

    profile = get_provider_profile("commandcode")
    assert profile is not None
    catalog = ProviderInfo(id="commandcode", name="CommandCode", env=("CATALOG_ONLY_KEY",),
                           api="https://catalog.example/v1")
    monkeypatch.setattr(providers_mod, "_models_dev_info",
                        lambda canonical, allow_network=True: catalog if canonical == "commandcode" else None)

    pdef = providers_mod.get_provider("commandcode")

    assert pdef is not None and pdef.source == "models.dev"
    assert "CATALOG_ONLY_KEY" in pdef.api_key_env_vars
    assert "COMMANDCODE_API_KEY" in pdef.api_key_env_vars
    assert pdef.base_url_env_var == "COMMANDCODE_BASE_URL"
    # Every rung of the full resolver sees the same union.
    full = providers_mod.resolve_provider_full("commandcode", {}, [])
    assert full is not None and full.api_key_env_vars == pdef.api_key_env_vars
