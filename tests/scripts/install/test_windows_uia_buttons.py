"""The Windows packaged-update driver presses the button its patterns rank first.

On the About page the settings sidebar's "About" comes before "Check now" in UI
Automation order. When the lookup took the first button matching any pattern,
the driver re-opened About every 3 s and never re-checked after NEW was published,
so the windows-packaged arm reported "You're running the latest version" on every
stable claim (rehearsal rc.1-v0.23.2, v0.22.1 claims).

It also finds App Installer's Update prompt wherever App Installer put it. Looking only at
the first App Installer process and its first window, the driver missed the prompt on both
runners and the arm failed with "Native update did not automatically relaunch" (rehearsal
rc.5-v0.23.2). App Installer 1.29 (arm64 runner) can open the prompt in a second process.
App Installer 1.26 (x64 runner) is a UWP app: its window is an ApplicationFrameHost frame
and only the frame's CoreWindow child belongs to App Installer, so a lookup by App
Installer's process id found no window at all (rehearsal rc.6-v0.23.2).
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
WINUI = "WinUIDesktopWin32WindowClass"
UWP_FRAME = "ApplicationFrameWindow"

# Fake UI Automation elements: FindAll(Children) filters an element's children by the
# condition's process id or class name; FindAll(Descendants) returns its own buttons and
# its children's, each tagged with the element that holds it.
FAKE_UIA = r"""
function Test-FakeCondition($Element, $Condition) {
    if ($Condition.Property.ProgrammaticName -match 'ClassName') { return $Element.Class -eq $Condition.Value }
    return $Element.Pid -eq $Condition.Value
}
function New-FakeElement($OwnerId, [string]$Class, [string]$Tag, [string[]]$Names, [object[]]$Children) {
    $element = [pscustomobject]@{ Pid = $OwnerId; Class = $Class; Tag = $Tag; Names = $Names; Children = $Children }
    $element | Add-Member -MemberType ScriptMethod -Name FindAll -Value {
        param($scope, $condition)
        if ($scope -eq [System.Windows.Automation.TreeScope]::Children) {
            return @($this.Children | Where-Object { Test-FakeCondition $_ $condition })
        }
        foreach ($n in $this.Names) { [pscustomobject]@{ Tag = $this.Tag; Current = [pscustomobject]@{ Name = $n } } }
        foreach ($child in $this.Children) { $child.FindAll($scope, $condition) }
    }
    $element
}
"""


def _element(pid: int, cls: str, tag: str, names: list[str], children: tuple[str, ...] = ()) -> str:
    return f"(New-FakeElement {pid} '{cls}' '{tag}' {_ps_array(names)} @({', '.join(children)}))"


def _find(top_level: list[str], installers: list[int]) -> str:
    """Find-WindowButton over fake top-level windows. Returns 'name@tag' or 'none'."""
    powershell = shutil.which("powershell")
    assert powershell, "Windows PowerShell 5.1 is part of every supported Windows"
    script = (
        "$ErrorActionPreference = 'Stop'; Add-Type -AssemblyName UIAutomationClient, UIAutomationTypes; "
        f". '{HELPER}'; {FAKE_UIA}; "
        f"$root = New-FakeElement 0 'Desktop' 'desktop' @() @({', '.join(top_level)}); "
        f"$hit = Find-WindowButton -Root $root -ProcessIds @({', '.join(map(str, installers))}) "
        f"-Patterns {_ps_array(APP_INSTALLER)}; "
        "if ($hit) { \"$($hit.Name)@$($hit.Element.Tag)\" } else { 'none' }"
    )
    result = subprocess.run([powershell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
                            capture_output=True, text=True, timeout=120, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout.strip()


def _pick_window(processes: list[tuple[int, list[list[str]]]]) -> str:
    """App Installer processes that own their windows (1.29): (pid, windows), each window a
    list of button names. Returns 'name@pid/window' (1-based window) or 'none'."""
    top_level = [_element(pid, WINUI, f"{pid}/{number}", names)
                 for pid, windows in processes for number, names in enumerate(windows, 1)]
    return _find(top_level, [pid for pid, _ in processes])


def test_app_installer_prompt_in_a_second_process_is_found():
    launch_check = (3600, [[]])
    prompt = (13444, [["Minimize", "Maximize", "Close", "Update", "Cancel"]])
    assert _pick_window([launch_check, prompt]) == "Update@13444/1"


def test_app_installer_prompt_beside_a_buttonless_window_is_found():
    one_process = (3948, [["Minimize", "Close"], ["Minimize", "Maximize", "Close", "Update", "Cancel"]])
    assert _pick_window([one_process]) == "Update@3948/2"


def test_no_prompt_is_reported_as_none():
    assert _pick_window([(3600, [[]]), (13444, [["Minimize", "Close"]])]) == "none"


def _uwp_frame(host: int, app: int, names: list[str]) -> str:
    """An ApplicationFrameHost frame as UI Automation shows it, around one app's CoreWindow."""
    return _element(host, UWP_FRAME, f"{host}/frame", ["Minimize", "Maximize", "Close"], (
        _element(host, "ApplicationFrameTitleBarWindow", f"{host}/title", []),
        _element(app, "Windows.UI.Core.CoreWindow", f"{app}/core", names),
    ))


def test_uwp_app_installer_prompt_inside_its_application_frame_is_found():
    hermes = _element(1444, "Chrome_WidgetWin_1", "1444/1", ["Open settings"])
    frame = _uwp_frame(7580, 2224, ["Update", "Cancel"])
    assert _find([hermes, frame], [2224]) == "Update@2224/core"


def test_another_uwp_apps_frame_is_never_pressed():
    other_app = _uwp_frame(7580, 5120, ["Update", "Cancel"])
    assert _find([other_app, _uwp_frame(7580, 2224, [])], [2224]) == "none"
