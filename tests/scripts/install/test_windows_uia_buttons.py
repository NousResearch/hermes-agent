"""The Windows packaged-update driver presses the button its patterns rank first.

On the About page the settings sidebar's "About" comes before "Check now" in UI
Automation order. When the lookup took the first button matching any pattern,
the driver re-opened About every 3 s and never re-checked after NEW was published,
so the windows-packaged arm reported "You're running the latest version" on every
stable claim (rehearsal rc.1-v0.23.2, v0.22.1 claims).
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
