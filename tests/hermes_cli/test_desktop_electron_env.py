"""GUI launches must not inherit Electron's Node-only mode (#135635)."""

import argparse
import json
import os
import sys
import time

import pytest

from hermes_cli import bundled_app, main_desktop


@pytest.mark.parametrize("node_mode", ["1", "0", ""])
def test_checkout_launch_env_keeps_gui_options_without_node_mode(monkeypatch, tmp_path, node_mode):
    monkeypatch.setenv("ELECTRON_RUN_AS_NODE", node_mode)
    monkeypatch.setenv("HERMES_DESKTOP_DISABLE_GPU", "1")
    monkeypatch.setenv("ELECTRON_OZONE_PLATFORM_HINT", "wayland")
    monkeypatch.setenv("HERMES_DESKTOP_PASSWORD_STORE", "basic")
    monkeypatch.setattr(main_desktop, "_desktop_launch_options",
                        lambda: (["--disable-dev-shm-usage"], "auto", "auto", "auto", True))

    env, flags = main_desktop._desktop_launch_env(argparse.Namespace(cwd=str(tmp_path)))

    assert "ELECTRON_RUN_AS_NODE" not in env
    assert env["HERMES_DESKTOP_DISABLE_GPU"] == "1"
    assert env["ELECTRON_OZONE_PLATFORM_HINT"] == "wayland"
    assert env["HERMES_DESKTOP_CWD"] == str(tmp_path.resolve())
    assert flags == ["--disable-dev-shm-usage"]
    assert os.environ["ELECTRON_RUN_AS_NODE"] == node_mode


@pytest.mark.parametrize("explicit_env", [False, True])
def test_detached_app_launch_cleans_inherited_and_explicit_env(monkeypatch, tmp_path, explicit_env):
    monkeypatch.setenv("ELECTRON_RUN_AS_NODE", "1")
    monkeypatch.setenv("HERMES_DESKTOP_CWD", str(tmp_path))
    supplied = dict(os.environ) if explicit_env else None
    marker = tmp_path / "child-env.json"
    script = (
        "import json, os; from pathlib import Path; "
        f"target = Path({str(marker)!r}); staged = target.with_suffix('.tmp'); "
        "staged.write_text(json.dumps({'node_mode': os.environ.get('ELECTRON_RUN_AS_NODE'), "
        "'desktop_cwd': os.environ['HERMES_DESKTOP_CWD'], 'cwd': os.getcwd()})); "
        "staged.replace(target)"
    )
    pid = bundled_app.launch_detached([sys.executable, "-c", script], env=supplied, cwd=tmp_path)
    assert pid > 0
    deadline = time.monotonic() + 10
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert marker.exists(), "the detached child did not publish its environment"
    observed = json.loads(marker.read_text(encoding="utf-8-sig"))
    assert observed == {"node_mode": None, "desktop_cwd": str(tmp_path), "cwd": str(tmp_path)}
    assert os.environ["ELECTRON_RUN_AS_NODE"] == "1"
    if supplied is not None:
        assert supplied["ELECTRON_RUN_AS_NODE"] == "1"
