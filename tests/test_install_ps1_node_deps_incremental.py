"""hc-452: Node-deps stage (``Install-NodeDeps``) skip logic in install.ps1.

Windows mirror of ``tests/test_install_sh_node_deps_incremental.py``. Kael's
2026-07-08 real-machine report measured the node-deps stage (unconditional
``npm install`` at repo root for browser tools, then again in ``ui-tui\\``) as
the single slowest step of a desktop runtime bootstrap/update (43s+), with no
skip path even when nothing in package.json/package-lock.json had changed.

``Get-NodeDepsFingerprint`` / ``Test-NodeDepsUpToDate`` / ``Set-NodeDepsInstalled``
mirror ``scripts/install.sh``'s ``node_deps_fingerprint`` /
``node_deps_up_to_date`` / ``node_deps_mark_installed`` one-for-one (SHA256
hash of package.json + package-lock.json when present, stored in a
repo-local marker, verified against actual ``node_modules`` content -- never
trusted in isolation, per the Windows workspace-hoisting flake documented in
this same file's ``Install-Desktop`` npm-ci comment).

install.ps1 only runs on Windows, so -- like the other
``tests/test_install_ps1_*.py`` files in this repo -- these tests lock the
contract at the source level rather than executing the script (no pwsh
runner on Linux/macOS CI).
"""

from __future__ import annotations

import re
from pathlib import Path

_INSTALL_PS1 = Path(__file__).resolve().parents[1] / "scripts" / "install.ps1"


def _source() -> str:
    return _INSTALL_PS1.read_text(encoding="utf-8")


def _function_body(source: str, name: str) -> str:
    """Return the text of a PowerShell ``function <name> { ... }`` block,
    brace-depth aware so nested blocks (if/foreach/try) don't truncate early."""
    start = source.index(f"function {name}")
    brace = source.index("{", start)
    depth = 0
    for i in range(brace, len(source)):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                return source[brace : i + 1]
    raise AssertionError(f"unterminated function body for {name}")


# ---------------------------------------------------------------------------
# Helper functions exist with the expected fail-open contract
# ---------------------------------------------------------------------------


def test_helper_functions_are_defined() -> None:
    source = _source()
    for name in (
        "Get-NodeDepsFingerprint",
        "Get-NodeDepsMarkerPath",
        "Test-NodeDepsUpToDate",
        "Set-NodeDepsInstalled",
    ):
        assert f"function {name}" in source, f"expected a {name} helper function"


def test_fingerprint_covers_both_package_json_and_lockfile() -> None:
    fn = _function_body(_source(), "Get-NodeDepsFingerprint")
    assert '"package.json"' in fn
    assert '"package-lock.json"' in fn
    # Order matters -- must match install.sh's fixed cat order so a
    # package.json-only change and a lockfile-only change both invalidate.
    idx_pkg = fn.find('"package.json"')
    idx_lock = fn.find('"package-lock.json"')
    assert idx_pkg < idx_lock


def test_fingerprint_returns_null_when_no_input_files_exist() -> None:
    fn = _function_body(_source(), "Get-NodeDepsFingerprint")
    assert "if ($inputs.Count -eq 0) { return $null }" in fn, (
        "an empty project directory (nothing to fingerprint) must return $null, "
        "not fabricate a hash of nothing -- callers treat $null as 'needs install'"
    )


def test_up_to_date_checks_marker_before_anything_else() -> None:
    fn = _function_body(_source(), "Test-NodeDepsUpToDate")
    idx_marker_check = fn.find("if (-not (Test-Path $marker)) { return $false }")
    assert idx_marker_check != -1
    # It must be the very first real check (cheapest short-circuit).
    body_start = fn.index("{") + 1
    prefix = fn[body_start:idx_marker_check]
    # Only the marker-path resolution line should precede it.
    assert "Get-NodeDepsMarkerPath" in prefix
    assert "Test-Path" not in prefix.replace("$marker = Get-NodeDepsMarkerPath $Dir", "")


def test_up_to_date_verifies_node_modules_has_real_content() -> None:
    """Regression guard for the flake Install-Desktop's own npm-ci comment
    documents: node_modules\\.package-lock.json can be stale while
    node_modules is actually empty (Windows workspace-hoisting). The marker
    file alone must never be trusted -- node_modules content is checked too."""
    fn = _function_body(_source(), "Test-NodeDepsUpToDate")
    assert "node_modules" in fn
    assert '".package-lock.json"' in fn, (
        "must exclude npm's own bookkeeping file when counting real "
        "node_modules entries, otherwise a bookkeeping-file-only empty "
        "install would incorrectly count as 'has content'"
    )
    idx_nm_check = fn.find('Join-Path $Dir "node_modules"')
    idx_fingerprint = fn.find("Get-NodeDepsFingerprint")
    assert idx_nm_check != -1 and idx_fingerprint != -1
    assert idx_nm_check < idx_fingerprint, (
        "node_modules content must be verified before the (more expensive) "
        "fingerprint hash comparison -- cheap checks first"
    )


def test_set_installed_is_best_effort_and_never_throws_uncaught() -> None:
    fn = _function_body(_source(), "Set-NodeDepsInstalled")
    assert "try {" in fn and "} catch {" in fn, (
        "a marker-write failure (read-only FS, permissions) must be caught, "
        "never propagated as a stage failure -- writing the skip marker is "
        "an optimization, not a correctness requirement"
    )


# ---------------------------------------------------------------------------
# Install-NodeDeps actually wires the skip checks in front of both npm calls
# ---------------------------------------------------------------------------


def test_install_node_deps_consults_skip_check_before_browser_tools_install() -> None:
    fn = _function_body(_source(), "Install-NodeDeps")
    idx_check = fn.find("Test-NodeDepsUpToDate $InstallDir")
    idx_npm = fn.find('_Run-NpmInstall "Browser tools"')
    assert idx_check != -1, "must consult Test-NodeDepsUpToDate before the browser-tools npm install"
    assert idx_npm != -1
    assert idx_check < idx_npm


def test_install_node_deps_consults_skip_check_before_tui_install() -> None:
    fn = _function_body(_source(), "Install-NodeDeps")
    idx_check = fn.find("Test-NodeDepsUpToDate $tuiDir")
    idx_npm = fn.rfind('_Run-NpmInstall "TUI"')
    assert idx_check != -1, "must consult Test-NodeDepsUpToDate before the TUI npm install"
    assert idx_npm != -1
    assert idx_check < idx_npm


def test_install_node_deps_marks_installed_only_on_real_install_success() -> None:
    """Set-NodeDepsInstalled must only run after a REAL npm install succeeds,
    never in the skip branch (which would be a redundant no-op write) and
    never on a failed install (which would poison the marker with an
    incomplete/broken node_modules state)."""
    fn = _function_body(_source(), "Install-NodeDeps")
    assert "if ($browserNpmOk) { Set-NodeDepsInstalled $InstallDir }" in fn
    assert "if ($tuiNpmOk) { Set-NodeDepsInstalled $tuiDir }" in fn


def test_playwright_still_runs_when_browser_tools_npm_install_was_skipped() -> None:
    """Skipping the npm install must not skip the downstream Playwright
    Chromium install -- $browserNpmOk must be set to $true in the skip
    branch too, so the existing `if ($browserNpmOk)` gate below still fires.
    Playwright has its own idempotent skip (browser revision cache check) so
    this is safe and not redundant work."""
    fn = _function_body(_source(), "Install-NodeDeps")
    idx_skip_branch = fn.find("already up to date (package.json/package-lock.json unchanged)")
    assert idx_skip_branch != -1
    # The line immediately setting $browserNpmOk = $true should follow shortly.
    window = fn[idx_skip_branch : idx_skip_branch + 200]
    assert "$browserNpmOk = $true" in window


def test_stage_surfaces_skipped_reason_when_everything_was_up_to_date() -> None:
    fn = _function_body(_source(), "Install-NodeDeps")
    assert "$nodeDepsAnySkipped = $true" in fn, "must default to 'everything skipped' before any sub-check runs"
    assert '$nodeDepsAnySkipped = $false' in fn
    assert "$script:_StageSkippedReason" in fn, (
        "must surface skipped=true to Invoke-Stage via the same "
        "$script:_StageSkippedReason channel Stage-Node already uses"
    )
    # Both branches (browser tools install AND TUI install) must be able to
    # flip the flag to $false -- a real install in EITHER spot should count.
    assert fn.count("$nodeDepsAnySkipped = $false") == 2


def test_node_deps_stage_worker_is_still_wired_to_install_node_deps() -> None:
    source = _source()
    assert re.search(r'function\s+Stage-NodeDeps\s*\{\s*Install-NodeDeps\s*\}', source), (
        "Stage-NodeDeps must still delegate to Install-NodeDeps unchanged"
    )
