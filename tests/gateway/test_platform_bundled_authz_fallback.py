"""Authorization env names of bundled platforms survive a registry entry that lacks them.

They live in each plugin's ``gateway_env.AUTHZ`` (read by core straight from the file) and reach the
registry through ``register()``. A user plugin overriding a bundled platform, safe mode or a
disabled plugin must still gate on them: core's own tables always did."""

import dataclasses

import pytest

from gateway.config import Platform
from gateway.platform_env import _BUNDLED_ENV_SIBLINGS, bundled_authz

_FIELDS = ("allowed_users_env", "allow_all_env", "allow_bots_env", "own_policy_env")


def _bundled_with_authz():
    names = set(Platform._scan_bundled_plugin_platforms()[0]) | set(_BUNDLED_ENV_SIBLINGS)
    return sorted(name for name in names if bundled_authz(name))


def test_registered_entries_carry_exactly_the_file_declaration():
    from gateway.platform_registry import platform_registry
    from hermes_cli.plugins import discover_plugins

    discover_plugins()
    names = _bundled_with_authz()
    assert {"telegram", "weixin", "wecom_callback", "qqbot"} <= set(names)
    for name in names:
        entry = platform_registry.get(name)
        declared = bundled_authz(name)
        assert entry is not None, name
        assert {f: getattr(entry, f) for f in declared} == declared, name


@pytest.mark.parametrize("name", ["telegram", "weixin"])
def test_override_without_fields_still_gates_on_bundled_names(name, monkeypatch):
    from gateway import pairing, run_startup
    from gateway.platform_registry import platform_registry
    from hermes_cli.plugins import discover_plugins

    discover_plugins()
    bare = dataclasses.replace(
        platform_registry.get(name), allowed_users_env="", allow_all_env="", allow_bots_env="",
        own_policy_env=None)
    monkeypatch.setattr(platform_registry, "get", lambda value: bare if value == name else None)

    declared = bundled_authz(name)
    assert pairing._allowlist_env_for_platform(name) == declared["allowed_users_env"]
    own = run_startup._own_policy_envs(Platform(name))
    if "own_policy_env" in declared:
        assert own == (*declared["own_policy_env"], declared["allow_all_env"])
    else:
        assert own is None
