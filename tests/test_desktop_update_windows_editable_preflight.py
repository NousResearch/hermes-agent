"""Regression: the Windows Desktop update hand-off must repair a stale editable install.

`scripts/desktop-update/windows.ps1` drives `hermes update` as
`python.exe -m hermes_cli.main update`. The venv's editable install maps
top-level packages through a finder file that pins the checkout path it was
installed from. When that checkout was a worktree that has since been deleted
(a campaign lane, a PR branch), the finder still points at the dead path and
every `python -m hermes_cli.main ...` dies before argparse with
``ModuleNotFoundError: No module named 'hermes_cli'``. A shadow copy supplied
by ``PYTHONPATH`` or the inherited cwd can also make a bare import succeed
while the canonical editable install remains broken.

The hand-off now proves that ``hermes_cli.__file__`` resolves beneath the
canonical install root before the update step. When the proof fails, it first
rewrites only the editable finder (``pip install -e . --no-deps``), then
escalates to a dependency-aware editable reinstall when the narrow repair
fails or does not restore the canonical origin. It re-probes after each
successful repair and fails closed with dedicated exit code 7 only after both
repair rungs are exhausted.

This test is source-level because Linux CI cannot execute the PowerShell
hand-off. The invariants it guards are structural: the probe binds import
origin to ``$InstallRoot``; repair drives ``$pythonExe`` rather than the
``hermes.exe`` shim; repair runs from the install root; the dependency-aware
fallback exists; and exit 7 flows through the Desktop's generic, terminal
handoff-failure surface instead of the update retry.
"""

from __future__ import annotations

import re
from pathlib import Path, PureWindowsPath


REPO_ROOT = Path(__file__).resolve().parent.parent
WINDOWS_PS1 = REPO_ROOT / "scripts" / "desktop-update" / "windows.ps1"
DESKTOP_MAIN = REPO_ROOT / "apps" / "desktop" / "electron" / "main.ts"


def _read() -> str:
    return WINDOWS_PS1.read_text(encoding="utf-8")


def _preflight_block() -> str:
    """The editable-install preflight, from its banner to the --keep-stash probe."""
    source = _read()
    match = re.search(
        r"# -- 3a\. Editable-install preflight.*?# --keep-stash: never re-apply",
        source,
        re.DOTALL,
    )
    assert match, (
        "Expected the editable-install preflight block in "
        "scripts/desktop-update/windows.ps1; the hand-off structure changed -- "
        "update this guard."
    )
    return match.group(0)


def test_preflight_binds_import_to_the_canonical_install_root() -> None:
    block = _preflight_block()

    assert "function Test-HermesCliImportFromInstallRoot" in block
    assert "hermes_cli.__file__" in block
    assert "pathlib.Path(sys.argv[1]).resolve()" in block
    assert "root in module.parents" in block, (
        "The probe must use path-component ancestry, not a string-prefix check "
        "that would accept an install-root sibling such as Hermes-agent-old."
    )
    assert "__HERMES_CLI_ORIGIN_B64__=" in block, (
        "The probe must return a loggable origin receipt. Base64 keeps Unicode "
        "Windows paths intact across PowerShell 5.1 native-output decoding."
    )
    assert "editable install probe resolved outside install root" in block
    assert "editable install probe resolved canonical module" in block


def test_windows_component_ancestry_rejects_prefix_named_sibling() -> None:
    """WindowsPath ancestry is case-insensitive but component-boundary exact."""
    root = PureWindowsPath(r"C:\Hermes-agent")
    canonical_module = PureWindowsPath(
        r"c:\HERMES-AGENT\hermes_cli\__init__.py"
    )
    prefix_sibling_module = PureWindowsPath(
        r"C:\Hermes-agent-old\hermes_cli\__init__.py"
    )

    assert root in canonical_module.parents
    assert root not in prefix_sibling_module.parents


def test_repair_drives_python_not_the_shim() -> None:
    block = _preflight_block()

    assert "function Invoke-EditableInstallRepair" in block
    assert "Invoke-HermesStep $pythonExe $pipArgs $tag" in block, (
        "Editable repair must drive $pythonExe, not the hermes.exe shim. "
        "Driving the update through the shim keeps hermes.exe mapped as a "
        "running image, so uv's final shim rewrite fails with os error 32 "
        "(see test_desktop_update_windows_python_handoff.py)."
    )
    assert "Invoke-HermesStep $hermesExe" not in block


def test_repair_runs_from_the_install_root() -> None:
    block = _preflight_block()

    assert "Push-Location -LiteralPath $InstallRoot" in block, (
        "`pip install -e .` resolves the tree it installs from the working "
        "directory, and the hand-off inherits the Desktop's cwd (HERMES_HOME, "
        "not the install root). The repair must Push-Location to $InstallRoot "
        "first -- the same class posix.sh guards against."
    )
    assert "Pop-Location" in block, (
        "The repair must restore the original location so the update step "
        "runs from the same cwd the hand-off inherited."
    )


def test_repair_escalates_to_a_dependency_aware_second_rung() -> None:
    block = _preflight_block()

    assert '$pipArgs += "--no-deps"' in block, (
        "The first repair rung must remain the narrow finder rewrite."
    )
    assert "Invoke-EditableInstallRepair -WithDependencies" in block, (
        "A missing or mismatched dependency can leave the canonical import "
        "broken after --no-deps; the hand-off must try a full editable reinstall "
        "before requiring manual repair."
    )
    assert "finder-only repair did not restore a canonical import" in block
    assert 'Publish-UiProgress "Repairing Hermes dependencies"' in block


def test_each_successful_repair_is_reprobed_before_proceeding() -> None:
    block = _preflight_block()

    assert block.count("$probeOk = Test-HermesCliImportFromInstallRoot") == 3, (
        "The preflight needs one initial canonical-origin probe and one re-probe "
        "after each repair rung."
    )
    assert "editable install repaired; update proceeding" in block, (
        "The success path must log that a canonical import was restored before "
        "the update step runs."
    )


def test_exhausted_repairs_fail_closed_with_terminal_exit_7() -> None:
    block = _preflight_block()

    assert block.count("$finalCode = 7") == 1, (
        "The two repair rungs should converge on one dedicated terminal state."
    )
    assert "exit $finalCode" in block, (
        "An exhausted preflight must exit before the update step; falling "
        "through runs the update into the same import wall."
    )
    assert "finder-only and dependency-aware editable reinstalls" in block
    assert "hermes doctor" in block, (
        "The terminal message must point at a real repair path."
    )


def test_exit_7_flows_to_the_desktops_terminal_failure_surface() -> None:
    windows = _read()
    desktop = DESKTOP_MAIN.read_text(encoding="utf-8")

    assert "Write-Result ($finalCode -eq 0) $finalCode $finalMsg" in windows, (
        "Every preflight exit must be persisted for the relaunched Desktop."
    )
    assert "else if (result)" in desktop
    assert "detached update FAILED (exit ${result.exitCode})" in desktop
    assert "'Hermes update did not finish'" in desktop, (
        "Desktop consumes every non-ok result through one terminal error dialog; "
        "it does not classify unknown codes as retryable."
    )
