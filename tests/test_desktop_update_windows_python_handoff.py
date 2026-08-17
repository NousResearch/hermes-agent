"""Regression: the Windows Desktop update hand-off must run through python.exe.

`scripts/desktop-update/windows.ps1` drives `hermes update` for the in-app
Desktop updater. It used to invoke the update through the venv's
`venv\\Scripts\\hermes.exe` console-script launcher. On Windows that launcher is
a real process that keeps `hermes.exe` mapped as its running image and spawns
`python.exe` as a child. The update ends in `uv pip install -e .`, which rewrites
the console-script shims -- including the `hermes.exe` the launcher still has
mapped -- and Windows refuses to replace a file mapped as a running image
("os error 32"). The rename fallback then defers to next reboot via
`MOVEFILE_DELAY_UNTIL_REBOOT`, which needs elevation a Desktop-driven update
does not have, so `uv pip install -e .` exits non-zero, the ZIP fallback repeats
the same sequence, the desktop build stage is never reached, and the pre-build
clean has already removed `apps/desktop/release` -- leaving an install whose
Start Menu shortcut points at a `Hermes.exe` that no longer exists.

Driving the update as `python.exe -m hermes_cli.main update` puts the inherited
image handle on `python.exe`, which uv never has to replace, so the shim is an
ordinary unlocked file when uv rewrites it.

This test is source-level because Linux CI cannot execute the PowerShell
hand-off. The invariant it guards is that every `Invoke-HermesStep` call site
(the update, its retry, and the desktop rebuild) drives `$pythonExe`, never the
`$hermesExe` shim. `hermes.exe` may still be *named* in the file for the
step-2 unlock preflight -- that is a lock probe, not an invocation -- so we
assert against the invocation sites specifically.
"""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
WINDOWS_PS1 = REPO_ROOT / "scripts" / "desktop-update" / "windows.ps1"


def _read() -> str:
    return WINDOWS_PS1.read_text(encoding="utf-8")


def test_invoke_hermes_step_calls_drive_python_not_the_shim() -> None:
    source = _read()

    invocations = re.findall(r"Invoke-HermesStep\s+(\$\w+)", source)
    assert invocations, (
        "Expected at least one Invoke-HermesStep call in "
        "scripts/desktop-update/windows.ps1; the update hand-off structure "
        "changed -- update this guard."
    )

    offenders = [exe for exe in invocations if exe != "$pythonExe"]
    assert not offenders, (
        "Every Invoke-HermesStep call in scripts/desktop-update/windows.ps1 "
        "must drive $pythonExe, not the hermes.exe shim. Driving the update "
        "through the shim keeps hermes.exe mapped as a running image, so uv's "
        "final shim rewrite fails with os error 32 and the Desktop update can "
        "never complete. Offending target(s): "
        f"{sorted(set(offenders))}."
    )


def test_update_invocation_uses_module_entrypoint() -> None:
    source = _read()

    assert '@("-m", "hermes_cli.main", "update"' in source, (
        "The update step must invoke `python.exe -m hermes_cli.main update ...` "
        "so the inherited image handle lands on python.exe, which uv never has "
        "to replace."
    )
    assert (
        '@("-m", "hermes_cli.main", "desktop", "--force-build", "--build-only")'
        in source
    ), (
        "The desktop rebuild step must also go through "
        "`python.exe -m hermes_cli.main desktop ...` for the same reason."
    )


def test_update_no_longer_invokes_the_hermes_exe_shim() -> None:
    source = _read()

    assert "Invoke-HermesStep $hermesExe" not in source, (
        "scripts/desktop-update/windows.ps1 still invokes the update through "
        "the hermes.exe shim (`Invoke-HermesStep $hermesExe`). That is the "
        "exact self-lock this fix removes -- route it through $pythonExe "
        "instead."
    )


def test_desktop_relaunch_waits_for_an_in_place_rebuild() -> None:
    source = _read()
    relaunch = re.search(
        r"function Start-DesktopRelaunch \{(?P<body>.*?)\n\}\n\nfunction Invoke-HermesStep",
        source,
        re.DOTALL,
    )
    assert relaunch, "Expected Start-DesktopRelaunch in the Windows hand-off script."

    body = relaunch.group("body")
    assert "if (-not $RelaunchExe) { return $false }" in body
    assert "$relaunchDeadline = (Get-Date).AddSeconds(120)" in body
    assert "while (-not (Test-Path -LiteralPath $RelaunchExe))" in body
    assert "if ((Get-Date) -ge $relaunchDeadline)" in body
    assert "Start-Sleep -Milliseconds 500" in body
    assert "[System.Windows.Forms.Application]::DoEvents()" in body
