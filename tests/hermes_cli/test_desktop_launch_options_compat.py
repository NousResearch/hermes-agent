"""External desktop launchers can still unpack the original four options."""

import argparse
from pathlib import Path

import pytest

from hermes_cli import main_desktop


@pytest.mark.parametrize("accessibility", [True, False])
def test_legacy_launcher_unpack_uses_real_config(monkeypatch, tmp_path, accessibility):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(
        "desktop:\n"
        "  electron_flags: [--disable-dev-shm-usage]\n"
        "  disable_gpu: true\n"
        "  password_store: kwallet6\n"
        "  ozone_platform_hint: wayland\n"
        f"  renderer_accessibility: {str(accessibility).lower()}\n",
        encoding="utf-8",
    )
    # The external wrapper's exact consumer operation, not a length snapshot.
    try:
        flags, gpu, store, ozone = main_desktop._desktop_launch_options()
    except ValueError as exc:
        pytest.fail(f"legacy desktop launcher cannot unpack its launch options: {exc}")
    assert (flags, gpu, store, ozone) == (
        ["--disable-dev-shm-usage"], "1", "kwallet6", "wayland"
    )

    for key in ("HERMES_DESKTOP_DISABLE_GPU", "ELECTRON_OZONE_PLATFORM_HINT",
                "HERMES_DESKTOP_RENDERER_ACCESSIBILITY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(main_desktop, "_detect_linux_password_store", lambda: None)
    env, native_flags = main_desktop._desktop_launch_env(argparse.Namespace(cwd=str(tmp_path)))
    assert native_flags == flags
    assert env["HERMES_DESKTOP_DISABLE_GPU"] == gpu
    assert env["ELECTRON_OZONE_PLATFORM_HINT"] == ozone
    assert env.get("HERMES_DESKTOP_RENDERER_ACCESSIBILITY") == (None if accessibility else "0")
    assert Path(env["HERMES_DESKTOP_CWD"]) == tmp_path


@pytest.mark.parametrize("config", [None, {}, {"desktop": {}}])
def test_empty_config_keeps_legacy_defaults(monkeypatch, config):
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: config)
    try:
        flags, gpu, store, ozone = main_desktop._desktop_launch_options()
    except ValueError as exc:
        pytest.fail(f"legacy desktop launcher cannot unpack defaults: {exc}")
    assert flags == []
    assert gpu == store == ozone == "auto"


def test_config_error_keeps_legacy_fallback(monkeypatch):
    def unavailable():
        raise OSError("fixture config unavailable")
    monkeypatch.setattr("hermes_cli.config.load_config", unavailable)
    try:
        flags, gpu, store, ozone = main_desktop._desktop_launch_options()
    except ValueError as exc:
        pytest.fail(f"legacy desktop launcher cannot unpack fallback: {exc}")
    assert flags == []
    assert gpu == store == ozone == "auto"
