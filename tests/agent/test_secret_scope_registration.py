"""Scope registration cannot strand a view built before a concurrent refresh."""

import threading

pytest_plugins = ["tests.agent.test_secret_store_routing"]


def test_a_poll_between_build_and_registration_cannot_retain_a_revoked_name(store, monkeypatch):
    provider, home = store
    from agent.runtime_scope import bind_state, scope_for_profile
    from agent.secret_scope import get_secret
    from tui_gateway import launch_profile_policy as policy

    provider.seed_external(home, {"BIND_WINDOW_KEY": "old-marker"})
    provider.refresh(home)
    built, polled = threading.Event(), threading.Event()
    original = policy.owner_secret_scope
    first = True

    def builder(path):
        nonlocal first
        mapping = original(path)
        if first:
            first = False
            built.set()
            assert polled.wait(10)
        return mapping

    def poll():
        assert built.wait(10)
        provider.remote[provider._key(home)]["env"] = {"BIND_WINDOW_KEY": "new-marker"}
        provider.remote[provider._key(home)]["version"] += 1
        provider.refresh(home)
        polled.set()

    monkeypatch.setattr(policy, "owner_secret_scope", builder)
    worker = threading.Thread(target=poll)
    worker.start()
    try:
        with bind_state(scope_for_profile(home), hydrate_secrets=False, terminal=False):
            assert get_secret("BIND_WINDOW_KEY") == "new-marker"
            provider.remote[provider._key(home)]["env"] = {}
            provider.remote[provider._key(home)]["version"] += 1
            provider.refresh(home)
            assert get_secret("BIND_WINDOW_KEY") is None
    finally:
        worker.join(10)
    assert not worker.is_alive()


def test_rebuild_drops_revoked_names_but_composes_managed_policy(store, monkeypatch):
    provider, home = store
    from agent.runtime_scope import bind_state, scope_for_profile
    from agent.secret_scope import get_secret
    from hermes_cli import managed_scope

    monkeypatch.setattr(managed_scope, "load_managed_env", lambda: {"POLICY_KEY": "managed-marker"})
    provider.seed_external(home, {"POLICY_KEY": "old-marker", "REVOKED_KEY": "old-marker"})
    provider.refresh(home)
    with bind_state(scope_for_profile(home), hydrate_secrets=False, terminal=False):
        provider.remote[provider._key(home)]["env"] = {"POLICY_KEY": "new-marker", "REVOKED_KEY": "new-marker"}
        provider.remote[provider._key(home)]["version"] += 1
        provider.refresh(home)
        provider.remote[provider._key(home)]["env"] = {}
        provider.remote[provider._key(home)]["version"] += 1
        provider.refresh(home)
        assert get_secret("POLICY_KEY") == "managed-marker"
        assert get_secret("REVOKED_KEY") is None
