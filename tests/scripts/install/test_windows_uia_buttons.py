"""The Windows packaged-update driver presses the button its patterns rank first.

On the About page the settings sidebar's "About" comes before "Check now" in UI
Automation order. When the lookup took the first button matching any pattern,
the driver re-opened About every 3 s and never re-checked after NEW was published,
so the windows-packaged arm reported "You're running the latest version" on every
stable claim (rehearsal rc.1-v0.23.2, v0.22.1 claims).

It also finds App Installer's Update prompt wherever App Installer put it. The update check
Windows runs when OLD launches leaves an App Installer window with no buttons open, and the
prompt can open in that process or in a second one. Looking only at the first App Installer
process and its first window, the driver missed the prompt on both runners and the arm
failed with "Native update did not automatically relaunch" (rehearsal rc.5-v0.23.2).
"""
from pathlib import Path
import shutil
import subprocess

import pytest

pytestmark = pytest.mark.platforms("windows")
HELPER = Path(__file__).resolve().parents[3] / "tests" / "install" / "e2e-assets" / "windows-uia-buttons.ps1"
NAVIGATION = ["^Check now$", "^Open settings$", "^About$", "choose a provider later"]


def _ps_array(items: list[str]) -> str:
    return "@(" + ", ".join("'" + item.replace("'", "''") + "'" for item in items) + ")"


def _pick(names: list[str], patterns: list[str]) -> str:
    """Name and 1-based tree position of the button Find-InvokableButton returns, or 'none'."""
    powershell = shutil.which("powershell")
    assert powershell, "Windows PowerShell 5.1 is part of every supported Windows"
    script = (
        "$ErrorActionPreference = 'Stop'; Add-Type -AssemblyName UIAutomationClient, UIAutomationTypes; "
        f". '{HELPER}'; "
        f"$names = {_ps_array(names)}; "
        "$buttons = @(for ($i = 0; $i -lt $names.Count; $i++) { "
        "[pscustomobject]@{ Id = $i + 1; Current = [pscustomobject]@{ Name = $names[$i] } } }); "
        "$root = [pscustomobject]@{ Buttons = $buttons }; "
        "$root | Add-Member -MemberType ScriptMethod -Name FindAll -Value { param($scope, $condition) $this.Buttons }; "
        f"$hit = Find-InvokableButton -Root $root -Patterns {_ps_array(patterns)}; "
        "if ($hit) { \"$($hit.Name)#$($hit.Element.Id)\" } else { 'none' }"
    )
    result = subprocess.run([powershell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
                            capture_output=True, text=True, timeout=120, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout.strip()


def test_about_page_presses_check_now_ahead_of_the_sidebar_about_entry():
    about_page = ["Close settings", "Plugins", "About", "Collapse: About", "Version & updates",
                  "Uninstall", "About", "Check now", "Search Ctrl K"]
    assert _pick(about_page, NAVIGATION) == "Check now#8"


def test_main_window_opens_settings_and_a_shared_pattern_keeps_tree_order():
    assert _pick(["Minimize", "New chat", "Open settings"], NAVIGATION) == "Open settings#3"
    assert _pick(["Close settings", "About", "Collapse: About", "About"], NAVIGATION) == "About#2"
    assert _pick(["Minimize", "Close"], NAVIGATION) == "none"


APP_INSTALLER = ["^Update$", "^Install$", "^(Reinstall|Update) .*"]


def _pick_window(processes: list[tuple[int, list[list[str]]]]) -> str:
    """Find-WindowButton over fake App Installer processes: (pid, windows), each window a list
    of button names. Returns 'name@pid/window' (1-based window) or 'none'."""
    powershell = shutil.which("powershell")
    assert powershell, "Windows PowerShell 5.1 is part of every supported Windows"
    windows = []
    for pid, pid_windows in processes:
        for number, names in enumerate(pid_windows, 1):
            windows.append(
                f"[pscustomobject]@{{ Pid = {pid}; Tag = '{pid}/{number}'; Names = {_ps_array(names)} }}")
    script = (
        "$ErrorActionPreference = 'Stop'; Add-Type -AssemblyName UIAutomationClient, UIAutomationTypes; "
        f". '{HELPER}'; "
        f"$windows = @({', '.join(windows)}); "
        "foreach ($w in $windows) { "
        "$w | Add-Member -MemberType ScriptMethod -Name FindAll -Value { param($scope, $condition) "
        "@(foreach ($n in $this.Names) { [pscustomobject]@{ Tag = $this.Tag; Current = [pscustomobject]@{ Name = $n } } }) } }; "
        "$root = [pscustomobject]@{ Windows = $windows }; "
        "$root | Add-Member -MemberType ScriptMethod -Name FindAll -Value { param($scope, $condition) "
        "@($this.Windows | Where-Object { $_.Pid -eq $condition.Value }) }; "
        f"$hit = Find-WindowButton -Root $root -ProcessIds @({', '.join(str(pid) for pid, _ in processes)}) "
        f"-Patterns {_ps_array(APP_INSTALLER)}; "
        "if ($hit) { \"$($hit.Name)@$($hit.Element.Tag)\" } else { 'none' }"
    )
    result = subprocess.run([powershell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
                            capture_output=True, text=True, timeout=120, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout.strip()


def test_app_installer_prompt_in_a_second_process_is_found():
    launch_check = (3600, [[]])
    prompt = (13444, [["Minimize", "Maximize", "Close", "Update", "Cancel"]])
    assert _pick_window([launch_check, prompt]) == "Update@13444/1"


def test_app_installer_prompt_beside_a_buttonless_window_is_found():
    one_process = (3948, [["Minimize", "Close"], ["Minimize", "Maximize", "Close", "Update", "Cancel"]])
    assert _pick_window([one_process]) == "Update@3948/2"


def test_no_prompt_is_reported_as_none():
    assert _pick_window([(3600, [[]]), (13444, [["Minimize", "Close"]])]) == "none"
