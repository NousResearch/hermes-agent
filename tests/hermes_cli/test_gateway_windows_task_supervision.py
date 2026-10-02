"""Real Windows launcher lifetimes, without registering a task or starting Hermes."""

import json
import os
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
        "ctypes.windll.kernel32.CreateFileW.restype = ctypes.wintypes.HANDLE\n"
        "ctypes.windll.user32.IsWindowVisible.argtypes = [ctypes.wintypes.HWND]\n"
        "window = ctypes.windll.kernel32.GetConsoleWindow()\n"
        "mode = ctypes.wintypes.DWORD()\n"
        "conin = ctypes.windll.kernel32.CreateFileW('CONIN$', 0x80000000, 3, None, 3, 0, None)\n"
        "invalid_handle = ctypes.c_void_p(-1).value\n"
        "console = bool(conin not in (None, invalid_handle) and ctypes.windll.kernel32.GetConsoleMode(conin, ctypes.byref(mode)))\n"
        "if conin not in (None, invalid_handle): ctypes.windll.kernel32.CloseHandle(conin)\n"
        "visible = bool(window and ctypes.windll.user32.IsWindowVisible(window))\n"
        "state = {'pid': os.getpid(), 'console': console, 'window': bool(window), 'visible': visible, "
        "'supervised': is_gateway_supervisor_process(), "
        "'supervisor_marker': os.environ.get(EXTERNAL_GATEWAY_SUPERVISOR_ENV, '')}\n"
        "(root / 'started.tmp').write_text(json.dumps(state), encoding='utf-8')\n"
        "(root / 'started.tmp').replace(root / 'started.json')\n"
        "deadline = time.monotonic() + 120\n"
        "while not (root / 'release').exists() and time.monotonic() < deadline:\n"
        "    time.sleep(0.05)\n"
        "sys.exit(int(sys.argv[2]))\n",
        encoding="utf-8",
    )
    script_path = root / "Hermes_Gateway_probe.cmd"
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("APPDATA", str(root / "AppData" / "Roaming"))
    monkeypatch.setattr(gateway_windows, "_hermes_home", lambda: root)
    monkeypatch.setattr(gateway_windows, "get_task_script_path", lambda: script_path)
    monkeypatch.setattr(
        gateway_windows, "_launcher_settings",
        lambda: (sys.executable, str(root), str(root), "--profile probe"),
    )
    gateway_windows._write_task_script()
    launcher = script_path.with_suffix(".ps1")
    content = launcher.read_text(encoding="utf-8-sig")
    production_code = (
        "from hermes_cli.gateway_windows import _run_generated_launcher as r; "
        "raise SystemExit(r('--supervised' in __import__('sys').argv))"
    )

    def test_code(supervised):
        # Only the gateway command is replaced. The overlay keeps _build_gateway_argv's PYTHONPATH
        # contract: the detached spawn's base env (served_profile_child_env) strips Hermes-owned
        # PYTHONPATH entries, and the test environment does not install the project itself.
        return (
            "import sys; from pathlib import Path; import hermes_cli.gateway_windows as g; "
            "from hermes_cli.gateway import PROJECT_ROOT; overlay = {}; "
            "g._prepend_pythonpath(overlay, [str(PROJECT_ROOT)]); "
            f"g._build_gateway_argv=lambda home=None: ([sys.executable, {str(child)!r}, "
            f"{str(root)!r}, {str(exit_code)!r}], {str(root)!r}, overlay); "
            f"g._hermes_home=lambda: Path({str(root)!r}); "
            f"raise SystemExit(g._run_generated_launcher({supervised!r}))"
        )

    replacements = (
        (
            subprocess.list2cmdline(["-c", production_code]),
            subprocess.list2cmdline(["-c", test_code(False)]),
        ),
        (
            subprocess.list2cmdline(["-c", production_code, "--supervised"]),
            subprocess.list2cmdline(["-c", test_code(True)]),
        ),
    )
    for production, harmless in replacements:
        needle = gateway_windows._powershell_utf8_expression(production)
        assert content.count(needle) == 1, "generated PowerShell bootstrap contract changed"
        content = content.replace(needle, gateway_windows._powershell_utf8_expression(harmless))
    launcher.write_bytes(content.encode("utf-8-sig"))
    return script_path


def _exercise_launcher(command, root, exit_code, *, supervised):
    child = None
    launcher_log = root / "launcher-probe.log"
    output = launcher_log.open("wb")
    wrapper = subprocess.Popen(command, creationflags=subprocess.CREATE_NO_WINDOW,
                               stdout=output, stderr=subprocess.STDOUT)
    try:
        started = root / "started.json"
        deadline = time.monotonic() + 60
        while not started.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        if not started.exists():
            # The child's own stderr goes to the gateway's stdio sidecar, not the launcher log.
            child_log = root / "logs" / "gateway-stdio.log"
            child_output = child_log.read_text(encoding="utf-8", errors="replace") if child_log.exists() else ""
            pytest.fail(
                "The generated launcher did not start its child: "
                + launcher_log.read_text(encoding="utf-8", errors="replace")[-4000:]
                + "\nchild output: " + child_output[-4000:]
            )
        state = json.loads(started.read_text(encoding="utf-8"))
        child = psutil.Process(state["pid"])
        assert state["visible"] is False, "The launcher exposed a console window"
        if supervised:
            assert state["console"] is True and state["window"] is True
            with pytest.raises(subprocess.TimeoutExpired):
                wrapper.wait(timeout=2)
            (root / "release").touch()
            assert wrapper.wait(timeout=60) == exit_code
        else:
            assert wrapper.wait(timeout=10) == 0
            assert child.is_running(), "The async launcher must leave its child running"
        assert state["supervised"] is supervised, "In-chat restart must preserve the launcher's ownership"
        assert state["supervisor_marker"] == ("1" if supervised else "")
    finally:
        (root / "release").touch()
        # Cleanup must not mask an earlier assertion or consume the whole
        # per-file CI deadline. These are only the two processes this fixture
        # created; every production lifetime assertion above stays in place.
        try:
            wrapper.wait(timeout=10)
        except subprocess.TimeoutExpired:
            wrapper.kill()
            wrapper.wait(timeout=10)
        if child is not None:
            try:
                child.wait(timeout=10)
            except psutil.TimeoutExpired:
                if child.is_running():
                    child.kill()
                child.wait(timeout=10)
        output.close()


@pytest.mark.platforms('windows')
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
    assert executable.lower().endswith("powershell.exe")
    assert str(script_path.with_suffix(".ps1")) in arguments
    assert "-Supervised" in arguments
    assert ".vbs" not in arguments.lower() and "wscript" not in executable.lower()


def _read_shortcut(entry):
    command = (
        "$s=(New-Object -ComObject WScript.Shell).CreateShortcut($env:HERMES_SHORTCUT_PATH); "
        "[pscustomobject]@{target=$s.TargetPath;arguments=$s.Arguments;working=$s.WorkingDirectory} "
        "| ConvertTo-Json -Compress"
    )
    result = subprocess.run(
        [gateway_windows._powershell_executable(), "-NoProfile", "-NonInteractive", "-Command", command],
        env={**os.environ, "HERMES_SHORTCUT_PATH": str(entry)},
        capture_output=True, text=True, timeout=60,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.platforms('windows')
@pytest.mark.parametrize("inherited_supervisor", ["", "1"])
def test_shared_launcher_remains_detached_and_startup_uses_it(monkeypatch, tmp_path, inherited_supervisor):
    monkeypatch.setenv(EXTERNAL_GATEWAY_SUPERVISOR_ENV, inherited_supervisor)
    script_path = _launchers(monkeypatch, tmp_path, 75)
    monkeypatch.setattr(gateway_windows, "_SCHTASKS_TIMEOUT_S", 60)
    entry = gateway_windows._install_startup_entry(script_path)
    shortcut = _read_shortcut(entry)
    shared = script_path.with_suffix(".ps1")
    assert entry.suffix.lower() == ".lnk" and entry.exists()
    assert Path(shortcut["target"]).name.lower() == "powershell.exe"
    assert shortcut["arguments"] == gateway_windows._powershell_launcher_arguments(shared)
    assert "-Supervised" not in shortcut["arguments"]
    assert Path(shortcut["working"]) == script_path.parent
    _exercise_launcher(
        f'"{shortcut["target"]}" {shortcut["arguments"]}', script_path.parent, 75, supervised=False,
    )
