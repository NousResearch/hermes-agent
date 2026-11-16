"""Live store layers have no construction/registration window or rebuild budget."""

import pytest

from agent.runtime_scope import bind_state, scope_for_profile
from agent.secret_scope import build_profile_secret_scope, get_secret

pytest_plugins = ["tests.agent.test_secret_store_routing"]


def replace(provider, home, env):
    remote = provider._remote(home)
    remote["env"] = dict(env)
    remote["version"] += 1
    provider.refresh(home)


@pytest.mark.parametrize("interleave", [False, True])
def test_revocation_between_construction_and_bind(store, interleave):
    provider, home = store
    provider.seed_external(home, {"REVOKED_KEY": "old-marker", "ANCHOR_KEY": "anchor-marker"})
    provider.refresh(home)
    secrets = build_profile_secret_scope(home)
    if interleave:
        replace(provider, home, {"ANCHOR_KEY": "anchor-marker"})
    with bind_state(scope_for_profile(home), secrets=secrets, terminal=False):
        replace(provider, home, {"ANCHOR_KEY": "anchor-marker"})
        assert get_secret("REVOKED_KEY") is None


@pytest.mark.parametrize("interleave", [False, True])
def test_caller_overlay_survives_equal_store_value(store, interleave):
    from agent.secret_scope import overlay_secret_scope
    provider, home = store
    provider.seed_external(home, {"CALLER_KEY": "saved-marker"})
    provider.refresh(home)
    secrets = overlay_secret_scope(build_profile_secret_scope(home), {"CALLER_KEY": "caller-marker"})
    if interleave:
        replace(provider, home, {"CALLER_KEY": "caller-marker"})
    with bind_state(scope_for_profile(home), secrets=secrets, terminal=False):
        replace(provider, home, {})
        assert get_secret("CALLER_KEY") == "caller-marker"


def test_unbound_view_has_no_churn_budget(store):
    provider, home = store
    provider.seed_external(home, {"REVOKED_KEY": "old-marker"})
    provider.refresh(home)
    secrets = build_profile_secret_scope(home)
    for version in range(32):
        replace(provider, home, {"CHURN_KEY": str(version)})
    assert "REVOKED_KEY" not in secrets
    assert dict(secrets) == {"CHURN_KEY": "31"}


def test_snapshot_install_is_monotonic():
    from hermes_secret_port import SecretSnapshot, SecretSnapshotCell
    cell = SecretSnapshotCell()
    assert cell.install(SecretSnapshot(2, {"KEY": "new-marker"}, {}))
    assert not cell.install(SecretSnapshot(1, {"KEY": "old-marker"}, {}))
    assert cell.current.env == {"KEY": "new-marker"}
