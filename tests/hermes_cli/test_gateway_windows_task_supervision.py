"""Real Windows launcher lifetimes, without registering a task or starting Hermes."""

import json
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import psutil
import pytest

from gateway.restart import EXTERNAL_GATEWAY_SUPERVISOR_ENV
from hermes_cli import gateway_windows


def _launchers(monkeypatch, tmp_path, exit_code):
    root = tmp_path / "launcher artifacts"
    root.mkdir()
    child = root / "probe child.py"
    child.write_text(
        "import ctypes, ctypes.wintypes, json, os, sys, time\n"
        "from pathlib import Path\n"
        "from gateway.restart import EXTERNAL_GATEWAY_SUPERVISOR_ENV, is_gateway_supervisor_process\n"
        "root = Path(sys.argv[1])\n"
        "ctypes.windll.kernel32.GetConsoleWindow.restype = ctypes.wintypes.HWND\n"
        "ctypes.windll.user32.IsWindowVisible.argtypes = [ctypes.wintypes.HWND]\n"
        "window = ctypes.windll.kernel32.GetConsoleWindow()\n"
        "visible = bool(window and ctypes.windll.user32.IsWindowVisible(window))\n"
        "state = {'pid': os.getpid(), 'visible': visible, 'supervised': is_gateway_supervisor_process(), "
        "'supervisor_marker': os.environ.get(EXTERNAL_GATEWAY_SUPERVISOR_ENV, '')}\n"
        "(root / 'started.tmp').write_text(json.dumps(state), encoding='utf-8')\n"
        "(root / 'started.tmp').replace(root / 'started.json')\n"
        "deadline = time.monotonic() + 60\n"
        "while not (root / 'release').exists() and time.monotonic() < deadline:\n"
        "    time.sleep(0.05)\n"
        "sys.exit(int(sys.argv[2]))\n",
        encoding="utf-8",
    )
    script_path = root / "Hermes_Gateway_probe.cmd"
    monkeypatch.setattr(gateway_windows, "get_task_script_path", lambda: script_path)
    monkeypatch.setattr(
        gateway_windows, "_launcher_settings",
        lambda: (sys.executable, str(root), str(root), "--profile probe"),
    )
    monkeypatch.setattr(
        gateway_windows, "_gateway_run_argv",
        lambda *_args: [sys.executable, str(child), str(root), str(exit_code)],
    )
    gateway_windows._write_task_script()
    return script_path


def _exercise_launcher(command, root, exit_code, *, supervised):
    child = None
    wrapper = subprocess.Popen(command, creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        started = root / "started.json"
        deadline = time.monotonic() + 30
        while not started.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert started.exists(), "The generated launcher did not start its child"
        state = json.loads(started.read_text(encoding="utf-8"))
        child = psutil.Process(state["pid"])
        assert state["visible"] is False, "The launcher exposed a console window"
        if supervised:
            with pytest.raises(subprocess.TimeoutExpired):
                wrapper.wait(timeout=2)
            (root / "release").touch()
            assert wrapper.wait(timeout=30) == exit_code
        else:
            assert wrapper.wait(timeout=10) == 0
            assert child.is_running(), "The async launcher must leave its child running"
        assert state["supervised"] is supervised, "In-chat restart must preserve the launcher's ownership"
        assert state["supervisor_marker"] == ("1" if supervised else "")
    finally:
        (root / "release").touch()
        wrapper.wait(timeout=30)
        if child is not None:
            child.wait(timeout=30)


@pytest.mark.windows_only
@pytest.mark.parametrize(("exit_code", "task_result"), [(78, 0), (0, 0), (75, 75), (1, 1)])
def test_scheduled_action_waits_for_hidden_child_and_applies_restart_policy(monkeypatch, tmp_path, exit_code, task_result):
    monkeypatch.delenv(EXTERNAL_GATEWAY_SUPERVISOR_ENV, raising=False)
    script_path = _launchers(monkeypatch, tmp_path, exit_code)
    captured = {}

    def schtasks(args):
        if args[0] == "/Create":
            captured["xml"] = Path(args[args.index("/XML") + 1]).read_text(encoding="utf-16")
        return 0, "", ""

    monkeypatch.setattr(gateway_windows, "_exec_schtasks", schtasks)
    assert gateway_windows._install_scheduled_task("Hermes_Gateway_probe", script_path)[0]
    action = ET.fromstring(captured["xml"]).find("{*}Actions/{*}Exec")
    executable = action.findtext("{*}Command")
    arguments = action.findtext("{*}Arguments")
    _exercise_launcher(f'"{executable}" {arguments}', script_path.parent, task_result, supervised=True)
    assert str(script_path.with_suffix(".vbs")) not in arguments


@pytest.mark.windows_only
@pytest.mark.parametrize("inherited_supervisor", ["", "1"])
def test_shared_launcher_remains_detached_and_startup_uses_it(monkeypatch, tmp_path, inherited_supervisor):
    monkeypatch.setenv(EXTERNAL_GATEWAY_SUPERVISOR_ENV, inherited_supervisor)
    script_path = _launchers(monkeypatch, tmp_path, 75)
    shared = script_path.with_suffix(".vbs")
    _exercise_launcher(
        ["wscript.exe", "//B", "//Nologo", str(shared)], script_path.parent, 75, supervised=False,
    )
    startup = gateway_windows._build_startup_launcher(script_path)
    assert str(shared) in startup
    assert str(script_path.with_suffix(".task.vbs")) not in startup
