from unittest.mock import patch

from tools.computer_use import cua_backend


_VAR = "CUA_DRIVER_RS_ENABLE_WAYLAND"


def _child_env(base_env, native_wayland):
    config = {"computer_use": {"native_wayland": native_wayland}}
    with patch("hermes_cli.config.load_config", return_value=config), \
         patch.object(cua_backend.sys, "platform", "linux"):
        return cua_backend.cua_driver_child_env(base_env)


def test_configured_native_wayland_reaches_linux_wayland_child():
    assert _child_env({"WAYLAND_DISPLAY": "wayland-1"}, True)[_VAR] == "1"


def test_native_wayland_not_injected_without_wayland_display_or_opt_in():
    assert _VAR not in _child_env({"DISPLAY": ":0"}, True)
    assert _VAR not in _child_env({"WAYLAND_DISPLAY": "wayland-1"}, False)
