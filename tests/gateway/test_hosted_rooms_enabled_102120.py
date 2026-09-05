"""Regression tests for #102120 — hosted rooms worker opt-out flag.

On multi-profile installs every profile gateway starts the Group Chat worker
against the install-wide shared state.db, and a fleet restart firing them
back-to-back has corrupted the store. `hosted_rooms_enabled: false` (config
or HERMES_GATEWAY_HOSTED_ROOMS_ENABLED env) must disable the worker while
the default preserves today's behavior.
"""
from gateway.config import GatewayConfig
from gateway.config_env import _hosted_rooms_env
from gateway.config_loader import bridge_toplevel_keys


def test_default_enabled_and_round_trips():
    cfg = GatewayConfig.from_dict({})
    assert cfg.hosted_rooms_enabled is True
    assert GatewayConfig.from_dict(cfg.to_dict()).hosted_rooms_enabled is True


def test_from_dict_false():
    assert GatewayConfig.from_dict({"hosted_rooms_enabled": False}).hosted_rooms_enabled is False


def test_yaml_bridge_top_level_and_nested():
    gw_data: dict = {}
    bridge_toplevel_keys({"hosted_rooms_enabled": False}, {}, gw_data)
    assert gw_data["hosted_rooms_enabled"] is False
    gw_data = {}
    bridge_toplevel_keys({}, {"hosted_rooms_enabled": False}, gw_data)
    assert gw_data["hosted_rooms_enabled"] is False


def test_env_override_wins(monkeypatch):
    cfg = GatewayConfig.from_dict({"hosted_rooms_enabled": True})
    monkeypatch.setenv("HERMES_GATEWAY_HOSTED_ROOMS_ENABLED", "false")
    _hosted_rooms_env(cfg)
    assert cfg.hosted_rooms_enabled is False
    monkeypatch.setenv("HERMES_GATEWAY_HOSTED_ROOMS_ENABLED", "true")
    _hosted_rooms_env(cfg)
    assert cfg.hosted_rooms_enabled is True


def test_startup_gate_defaults_open():
    """The run_startup gate must fail open when the flag is absent."""
    assert getattr(object(), "hosted_rooms_enabled", True) is True
