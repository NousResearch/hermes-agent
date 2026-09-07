"""Windows Desktop hand-off must fail closed on missing prerequisites.

Regression for the antivirus-quarantine incident (#104689): AVG quarantined
the maintained updater script and later the managed ``venv\\Scripts\\python.exe``
mid-update. The hand-off still exited 0 and wrote ``ok=true / Update complete``
while the runtime and the desktop build were incomplete, so the user had a
false success receipt and a broken install.

Two invariants guard against that class of silent corruption:

1. The flat compat forwarder ``scripts/desktop-update.ps1`` (spawned by a
   Desktop whose asar is one update behind) must refuse to run when the
   maintained ``desktop-update\\windows.ps1`` is missing, instead of letting
   the failed native invocation collapse into whatever ``$LASTEXITCODE``
   happens to hold.
2. The maintained script must probe the venv python BEFORE the Desktop and
   gateways are torn down (fail closed, pre-teardown), and must verify the
   runtime actually imports AFTER ``hermes update`` reports success — exit 0
   is not proof the install works when an antivirus removed the interpreter
   mid-update.

The updater script is not executable on the Linux CI lane, so these tests
lock the source-level contract, following the pattern of
``test_desktop_update_windows_python_handoff.py``.
"""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
FORWARDER_PS1 = REPO_ROOT / "scripts" / "desktop-update.ps1"
WINDOWS_PS1 = REPO_ROOT / "scripts" / "desktop-update" / "windows.ps1"


def _read(path: Path) -> str:
    # Both scripts are eol=crlf in .gitattributes; normalize so anchors match
    # regardless of the working-copy line endings.
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")


def test_forwarder_refuses_to_run_when_maintained_script_is_missing() -> None:
    source = _read(FORWARDER_PS1)

    assert "Test-Path -LiteralPath $target" in source, (
        "scripts/desktop-update.ps1 must Test-Path the maintained "
        "desktop-update\\windows.ps1 before invoking it. When the maintained "
        "script is quarantined by antivirus software the forwarder otherwise "
        "invokes a missing file and its exit status collapses into a "
        "false-success hand-off (#104689)."
    )
    assert "exit 3" in source, (
        "The forwarder must exit non-zero when the maintained script is "
        "missing so the Desktop's dwell check and the result reader both see "
        "a failed hand-off."
    )


def test_handoff_probes_venv_python_before_tearing_down_the_desktop() -> None:
    source = _read(WINDOWS_PS1)

    preflight = re.search(
        r"-- 0\.5\. Prerequisite preflight.*?Write-HandoffLog \"prerequisite preflight passed",
        source,
        re.DOTALL,
    )
    assert preflight, (
        "Expected the step-0.5 prerequisite preflight in "
        "scripts/desktop-update/windows.ps1; the hand-off structure changed "
        "-- update this guard."
    )

    teardown = source.find("-- 1. Wait for the Desktop to exit")
    assert teardown > 0, "Expected the step-1 desktop teardown marker."
    assert preflight.start() < teardown, (
        "The prerequisite preflight must run BEFORE the Desktop exit wait: "
        "past that point an abort costs the user their running app for "
        "nothing (#104689)."
    )
    assert "venv\\Scripts\\python.exe" in preflight.group(0), (
        "The preflight must probe the venv python — the interpreter that "
        "drives every Invoke-HermesStep call."
    )


def test_handoff_verifies_runtime_before_reporting_success() -> None:
    source = _read(WINDOWS_PS1)

    verify = re.search(
        r"-- 5\. Runtime verification.*?exit \$finalCode",
        source,
        re.DOTALL,
    )
    assert verify, (
        "Expected the step-5 runtime verification block in "
        "scripts/desktop-update/windows.ps1; the hand-off structure changed "
        "-- update this guard."
    )

    block = verify.group(0)
    assert "import hermes_cli.main" in block, (
        "The post-update verification must exercise the managed runtime "
        "(python -c 'import hermes_cli.main'), not just an existence check: "
        "a quarantined-but-present file would pass Test-Path."
    )
    assert "Invoke-HermesStep $pythonExe" in block, (
        "The verification step must drive $pythonExe like every other step "
        "(see test_desktop_update_windows_python_handoff.py)."
    )
    assert "$finalCode = 8" in block, (
        "A failed runtime verification must produce its own failure code, "
        "never the success receipt."
    )

    success = source.find('"Update complete."')
    assert success > 0, "Expected the success message."
    assert verify.start() < success, (
        "The runtime verification must run BEFORE the success message is "
        "chosen — exit 0 from `hermes update` is not proof the install "
        "works (#104689)."
    )
