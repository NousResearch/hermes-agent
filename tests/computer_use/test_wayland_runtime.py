"""Regression coverage for the current-main native-Wayland runtime seam."""

from __future__ import annotations

from types import SimpleNamespace

from tools.computer_use import linux_wayland
from tools.computer_use import wayland_runtime


def _dummy_backend(base_result=None):
    base_result = dict(base_result or {})

    def child_env(base_env=None):
        result = dict(base_result)
        result.update(dict(base_env or {}))
        result["UPSTREAM_ENV_BUILDER_RAN"] = "1"
        return result

    return SimpleNamespace(
        cua_driver_child_env=child_env,
        resolve_cua_driver_cmd=lambda: "/opt/cua-driver",
        _computer_use_cfg=lambda: {
            "linux": {"wayland": {"enabled": "auto"}},
            "permission_mode": "bounded",
        },
    )


def test_wrapper_preserves_upstream_env_and_routes_resolved_driver(monkeypatch):
    backend = _dummy_backend()
    observed = {}

    def fake_wayland_env(driver_cmd, config, env):
        observed["driver_cmd"] = driver_cmd
        observed["config"] = config
        observed["env"] = dict(env)
        out = dict(env)
        out[linux_wayland.WAYLAND_ENABLE_ENV] = "1"
        return out

    monkeypatch.setattr(linux_wayland, "native_wayland_child_env", fake_wayland_env)

    wrapped = wayland_runtime._wrap_child_env_for_wayland(backend)
    result = wrapped({"CALLER": "kept"})

    assert result["UPSTREAM_ENV_BUILDER_RAN"] == "1"
    assert result["CALLER"] == "kept"
    assert result[linux_wayland.WAYLAND_ENABLE_ENV] == "1"
    assert observed["driver_cmd"] == "/opt/cua-driver"
    assert observed["config"]["permission_mode"] == "bounded"
    assert observed["env"]["UPSTREAM_ENV_BUILDER_RAN"] == "1"


def test_wrapper_is_idempotent():
    backend = _dummy_backend()
    first = wayland_runtime._wrap_child_env_for_wayland(backend)
    second = wayland_runtime._wrap_child_env_for_wayland(backend)

    assert second is first
    assert backend.cua_driver_child_env is first


def test_wrapper_fails_closed_over_stale_inherited_opt_in(monkeypatch):
    backend = _dummy_backend({linux_wayland.WAYLAND_ENABLE_ENV: "1"})

    def fail_policy(*_args, **_kwargs):
        raise RuntimeError("manifest probe exploded")

    monkeypatch.setattr(linux_wayland, "native_wayland_child_env", fail_policy)
    wrapped = wayland_runtime._wrap_child_env_for_wayland(backend)

    result = wrapped()

    assert result["UPSTREAM_ENV_BUILDER_RAN"] == "1"
    assert result[linux_wayland.WAYLAND_ENABLE_ENV] == "0"


def test_feature_probe_scrubs_inherited_secrets(monkeypatch):
    observed = {}

    def probe(_driver_cmd, env):
        observed.update(env)
        return "features"

    wayland = SimpleNamespace(
        probe_driver_features=probe,
        CuaDriverFeatures=lambda: "closed",
    )
    monkeypatch.setattr(
        "tools.environments.local._sanitize_subprocess_env",
        lambda env: {key: value for key, value in env.items() if key != "OPENAI_API_KEY"},
    )

    wrapped = wayland_runtime._wrap_feature_probe_for_safety(wayland)
    result = wrapped("/opt/cua-driver", {"PATH": "/bin", "OPENAI_API_KEY": "secret"})

    assert result == "features"
    assert observed["PATH"] == "/bin"
    assert "OPENAI_API_KEY" not in observed


def test_feature_probe_fails_closed_when_sanitizer_is_unavailable(monkeypatch):
    wayland = SimpleNamespace(
        probe_driver_features=lambda *_args, **_kwargs: "must-not-run",
        CuaDriverFeatures=lambda: "closed",
    )

    def fail_sanitizer(_env):
        raise RuntimeError("sanitizer unavailable")

    monkeypatch.setattr(
        "tools.environments.local._sanitize_subprocess_env",
        fail_sanitizer,
    )
    wrapped = wayland_runtime._wrap_feature_probe_for_safety(wayland)

    assert wrapped("/opt/cua-driver", {"OPENAI_API_KEY": "secret"}) == "closed"
