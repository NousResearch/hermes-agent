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



def test_exact_driver_routes_opt_in_through_safety_policy(monkeypatch):
    from tools.computer_use import linux_wayland

    observed = {}

    def policy(driver_cmd, config, env):
        observed["driver_cmd"] = driver_cmd
        observed["config"] = config
        observed["env"] = dict(env)
        out = dict(env)
        out[_VAR] = "0"
        return out

    config = {"computer_use": {"native_wayland": True}}
    monkeypatch.setattr(linux_wayland, "native_wayland_child_env", policy)
    with patch("hermes_cli.config.load_config", return_value=config), \
         patch.object(cua_backend.sys, "platform", "linux"):
        got = cua_backend.cua_driver_child_env(
            {"WAYLAND_DISPLAY": "wayland-1"},
            driver_cmd="/opt/cua-driver",
        )

    assert got[_VAR] == "0"
    assert observed["driver_cmd"] == "/opt/cua-driver"
    assert observed["config"]["native_wayland"] is True
    assert observed["env"][_VAR] == "1"


def test_exact_driver_policy_failure_fails_closed(monkeypatch):
    from tools.computer_use import linux_wayland

    config = {"computer_use": {"native_wayland": True}}

    def fail_policy(*_args, **_kwargs):
        raise RuntimeError("probe failed")

    monkeypatch.setattr(linux_wayland, "native_wayland_child_env", fail_policy)
    with patch("hermes_cli.config.load_config", return_value=config), \
         patch.object(cua_backend.sys, "platform", "linux"):
        got = cua_backend.cua_driver_child_env(
            {"WAYLAND_DISPLAY": "wayland-1"},
            driver_cmd="/opt/cua-driver",
        )

    assert got[_VAR] == "0"


def test_embedded_daemon_threads_exact_driver_to_child_env(monkeypatch):
    observed = {}

    def child_env(base_env=None, driver_cmd=None):
        observed["driver_cmd"] = driver_cmd
        return dict(base_env or {})

    monkeypatch.setattr(cua_backend, "cua_driver_child_env", child_env)
    daemon = cua_backend._EmbeddedCuaDaemon(
        "/opt/cua-driver",
        "unrestricted",
    )

    daemon.child_env()

    assert observed["driver_cmd"] == "/opt/cua-driver"
