from unittest.mock import patch

from tools.computer_use import cua_backend


_VAR = "CUA_DRIVER_RS_ENABLE_WAYLAND"


def test_configured_native_wayland_reaches_linux_wayland_child():
    config = {"computer_use": {"native_wayland": True}}
    with patch("hermes_cli.config.load_config", return_value=config), \
         patch.object(cua_backend.sys, "platform", "linux"):
        env = cua_backend.cua_driver_child_env({"WAYLAND_DISPLAY": "wayland-1"})
    assert env[_VAR] == "1"


def test_configured_native_wayland_does_not_enable_x11_child():
    config = {"computer_use": {"native_wayland": True}}
    with patch("hermes_cli.config.load_config", return_value=config), \
         patch.object(cua_backend.sys, "platform", "linux"):
        env = cua_backend.cua_driver_child_env({"DISPLAY": ":0"})
    assert _VAR not in env


def test_default_preserves_manual_environment_opt_in():
    config = {"computer_use": {"native_wayland": False}}
    base_env = {"WAYLAND_DISPLAY": "wayland-1", _VAR: "1"}
    with patch("hermes_cli.config.load_config", return_value=config), \
         patch.object(cua_backend.sys, "platform", "linux"):
        env = cua_backend.cua_driver_child_env(base_env)
    assert env[_VAR] == "1"
