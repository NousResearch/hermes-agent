"""Windows gateway interpreter-startup encoding contract (regression #127633)."""

import json
import os
import subprocess
import sys

import pytest

from hermes_cli import gateway_windows


@pytest.mark.platforms("windows")
@pytest.mark.parametrize("kind", ["launch", "restart"])
def test_gateway_overlay_enables_utf8_before_interpreter_start(kind, monkeypatch, tmp_path):
    import hermes_cli.gateway as gateway

    monkeypatch.setattr(gateway, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(gateway_windows, "_launcher_settings", lambda home=None: (
        sys.executable, str(tmp_path), str(tmp_path), ""))
    monkeypatch.setattr(gateway_windows, "_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(gateway_windows, "_stable_gateway_working_dir", lambda root: str(tmp_path))
    if kind == "launch":
        argv, cwd, overlay = gateway_windows._build_gateway_argv()
    else:
        argv, cwd, overlay = gateway_windows.windowless_gateway_restart_spec(
            [sys.executable, "-m", "hermes_cli.main", "gateway", "run"])
    env = os.environ.copy()
    env.pop("PYTHONUTF8", None)
    env.update(overlay)
    probe = (
        "import json,sys,subprocess; "
        "r=subprocess.run([sys.executable,'-c',"
        "\"import sys;sys.stdout.buffer.write(bytes.fromhex('e4b8ade69687'))\"],"
        "capture_output=True,text=True); "
        "print(json.dumps([sys.flags.utf8_mode,r.stdout]))"
    )
    result = subprocess.run([argv[0], "-c", probe], cwd=cwd, env=env,
                            capture_output=True, encoding="utf-8", timeout=15)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [1, "中文"]


@pytest.mark.platforms("windows")
def test_generated_launchers_enable_utf8_with_existing_environment(monkeypatch, tmp_path):
    monkeypatch.setattr(gateway_windows, "_resolve_detached_python", lambda path: (
        path, tmp_path, []))
    args = (sys.executable, str(tmp_path), str(tmp_path), "--profile work")
    cmd = gateway_windows._build_gateway_cmd_script(*args)
    vbs = gateway_windows._build_gateway_vbs_script(*args)
    assert 'set "PYTHONUTF8=1"' in cmd
    assert 'env.Item("PYTHONUTF8") = "1"' in vbs
    assert 'set "PYTHONIOENCODING=utf-8"' in cmd
    assert 'env.Item("PYTHONIOENCODING") = "utf-8"' in vbs
    assert "--profile work gateway run" in cmd
    assert "--profile work gateway run" in vbs
