"""Remote-only view changes invalidate setup and account catalog caches."""

pytest_plugins = ["tests.agent.test_secret_store_routing"]


def test_setup_reinventories_after_a_remote_key_appears(store, monkeypatch):
    provider, home = store
    from hermes_cli import free_tier_bootstrap as boot
    from hermes_constants import hermes_home_key
    boot.reset_for_tests()
    monkeypatch.setattr("hermes_constants.get_process_hermes_home", lambda: home)
    monkeypatch.setattr("hermes_constants.hermes_home_key", lambda path=None: str(path or home))
    assert hermes_home_key(home)
    monkeypatch.setattr(boot, "_broadcast", lambda record: None)
    monkeypatch.setattr(boot, "_due_changed", lambda: None)
    monkeypatch.setattr(boot, "_resolve_inference", lambda: "openrouter")
    monkeypatch.setattr(boot, "_record", boot.SetupRecord(False, "", False, False, False))
    monkeypatch.setattr(boot, "_inventory_stamp", boot._config_stamp())
    calls = []
    monkeypatch.setattr(boot, "_inventory_other_providers", lambda: calls.append(True) or True)
    provider.seed_external(home, {"OPENROUTER_API_KEY": "new-marker"})
    provider.refresh(home)
    assert boot.reconcile_record().provider_configured
    assert calls == [True]


def test_model_catalog_follows_remote_account_rotation_without_local_writes(store, monkeypatch):
    provider, home = store
    from agent.runtime_scope import bind_state, scope_for_profile
    from agent.secret_scope import get_secret
    from hermes_cli import models
    provider.seed_external(home, {"OPENROUTER_API_KEY": "account-a-marker"})
    provider.refresh(home)
    monkeypatch.setattr(models, "provider_model_ids", lambda *args, **kwargs: [
        "account-a-model" if get_secret("OPENROUTER_API_KEY") == "account-a-marker" else "account-b-model"])
    with bind_state(scope_for_profile(home), hydrate_secrets=False, terminal=False):
        assert models.cached_provider_model_ids("openrouter") == ["account-a-model"]
        provider.remote[provider._key(home)]["env"] = {"OPENROUTER_API_KEY": "account-b-marker"}
        provider.remote[provider._key(home)]["version"] += 1
        provider.refresh(home)
        assert get_secret("OPENROUTER_API_KEY") == "account-b-marker"
        assert models.cached_provider_model_ids("openrouter") == ["account-b-model"]
