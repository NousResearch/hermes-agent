"""Caller-owned setup values survive store registration and later revocations."""

from agent.runtime_scope import bind_state, scope_for_profile
from agent.secret_scope import build_profile_secret_scope, get_secret, overlay_secret_scope

pytest_plugins = ["tests.agent.test_secret_store_routing"]


def test_mcp_oauth_worker_keeps_unsaved_setup_credentials(store, monkeypatch):
    provider, home = store
    provider.seed_external(home, {"MCP_SETUP_KEY": "saved-marker"})
    provider.refresh(home)
    from tools.connectors import mcp_oauth

    observed = []
    monkeypatch.setattr(mcp_oauth, "probe_with_rollback", lambda *a, **kw: observed.append(
        (get_secret("MCP_SETUP_KEY"), get_secret("UNSAVED_KEY"))))
    mcp_oauth.run_worker(str(home), "example", {}, False, flow=None,
                         env={"MCP_SETUP_KEY": "setup-marker", "UNSAVED_KEY": "temporary-marker"})
    assert observed == [("setup-marker", "temporary-marker")]


def test_revocation_rebuilds_the_store_base_and_keeps_the_explicit_overlay(store):
    provider, home = store
    provider.seed_external(home, {"REVOKED_KEY": "saved-marker", "OVERRIDE_KEY": "saved-marker"})
    provider.refresh(home)
    explicit = overlay_secret_scope(build_profile_secret_scope(home),
                                    {"OVERRIDE_KEY": "setup-marker", "UNSAVED_KEY": "temporary-marker"})
    with bind_state(scope_for_profile(home), secrets=explicit, terminal=False):
        assert get_secret("OVERRIDE_KEY") == "setup-marker"
        provider.remote[provider._key(home)]["env"] = {}
        provider.remote[provider._key(home)]["version"] += 1
        provider.refresh(home)
        assert get_secret("REVOKED_KEY") is None
        assert get_secret("OVERRIDE_KEY") == "setup-marker"
        assert get_secret("UNSAVED_KEY") == "temporary-marker"


def test_an_explicit_mapping_without_a_store_base_is_not_rebuilt(store):
    provider, home = store
    provider.seed_external(home, {"STORE_KEY": "saved-marker"})
    provider.refresh(home)
    explicit = {"SETUP_KEY": "temporary-marker"}
    with bind_state(scope_for_profile(home), secrets=explicit, terminal=False):
        provider.seed_external(home, {"NEW_KEY": "new-marker"})
        provider.refresh(home)
        assert explicit == {"SETUP_KEY": "temporary-marker"}
