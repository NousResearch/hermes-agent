"""gateway.config_env.builtin_platform_status: the gateway's (enabled, connected) for one built-in platform,
from a profile's config dict, without load_gateway_config(). Run here without a multiplexer, so the env pass's
explicit-disable rule is visible (the dashboard's scoped reads run as a multiplex secondary, where the shared-
listener guard keeps env-only api_server/webhook off anyway)."""

from __future__ import annotations

import json

import pytest

from gateway.config_env import builtin_platform_status

_KEY = "k" * 32


@pytest.fixture(autouse=True)
def _plain_process(monkeypatch):
    from agent import secret_scope
    from hermes_cli import managed_scope

    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", False)
    for name in ("GATEWAY_RELAY_URL", "GATEWAY_RELAY_ALLOW_DIRECT_PLATFORMS"):  # not in the suite's env filter
        monkeypatch.delenv(name, raising=False)
    managed_scope.invalidate_managed_cache()
    yield
    managed_scope.invalidate_managed_cache()


def test_plugin_only_ids_are_not_judged():
    assert builtin_platform_status("workchat", {"platforms": {"workchat": {"enabled": True}}}) is None


def test_the_callers_config_is_left_alone():
    config = {"platforms": {"webhook": {"extra": {}}}, "webhook": {"enabled": True, "port": 9}}
    before = json.dumps(config, sort_keys=True)

    builtin_platform_status("webhook", config)

    assert json.dumps(config, sort_keys=True) == before


@pytest.mark.parametrize("config", [
    pytest.param({"platforms": {"api_server": {"enabled": False}}}, id="platforms"),
    pytest.param({"gateway": {"platforms": {"api_server": {"enabled": False}}}}, id="gateway-platforms"),
    pytest.param({"api_server": {"enabled": False}}, id="toplevel"),
])
def test_explicit_disable_beats_an_env_key(monkeypatch, config):
    monkeypatch.setenv("API_SERVER_KEY", _KEY)

    assert builtin_platform_status("api_server", config) == (False, True)
    assert builtin_platform_status("api_server", {}) == (True, True)  # the same key alone enables it


def test_an_administrator_pinned_key_outranks_the_users_empty_one(monkeypatch, tmp_path):
    from hermes_cli import managed_scope

    managed = tmp_path / "managed"
    managed.mkdir()
    (managed / "config.yaml").write_text(json.dumps({"api_server": {"key": _KEY}}), encoding="utf-8")
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    managed_scope.invalidate_managed_cache()
    user = {"platforms": {"api_server": {"enabled": True, "extra": {"key": ""}}}}

    # load_config() hands the dashboard the user config with the managed overlay deep-merged in.
    assert builtin_platform_status("api_server", managed_scope.apply_managed_overlay(user)) == (True, True)

    # ... which is the gateway's own verdict for that profile.
    from gateway.config import Platform, load_gateway_config

    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text(json.dumps(user), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    gateway = load_gateway_config()
    api_server = gateway.platforms[Platform.API_SERVER]
    assert (api_server.enabled, gateway._is_platform_connected(Platform.API_SERVER, api_server)) == (True, True)
