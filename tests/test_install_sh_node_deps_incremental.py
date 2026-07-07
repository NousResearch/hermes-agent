"""hc-452: Node-deps stage (``install_node_deps``) skip logic in install.sh.

Kael's 2026-07-08 real-machine report: every desktop runtime *update* re-ran
the full 10-stage bootstrap from scratch, including an unconditional
``npm install`` at both the repo root (browser tools) and ``ui-tui/`` — the
single slowest stage observed (43s+), with zero skip path even when nothing
in ``package.json``/``package-lock.json`` had changed since the last install.

These tests pin the fix: ``node_deps_fingerprint`` / ``node_deps_up_to_date`` /
``node_deps_mark_installed`` implement a hash-marker judge (cksum of
package.json + package-lock.json when present) stored in a repo-local marker
file, consulted before each of the two ``npm install`` call sites in
``install_node_deps``. The check is fail-open by construction: any ambiguous
state (no marker yet, empty ``node_modules``, unreadable fingerprint) reports
"needs install" rather than risking a false skip.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"

pytestmark = pytest.mark.skipif(
    subprocess.run(["bash", "--version"], capture_output=True).returncode != 0,
    reason="needs bash",
)


def _extract_function(name: str) -> str:
    text = INSTALL_SH.read_text()
    match = re.search(rf"^{re.escape(name)}\(\) \{{.*?\n\}}", text, re.DOTALL | re.MULTILINE)
    assert match is not None, f"{name}() not found in install.sh"
    return match.group(0)


def _harness() -> str:
    """Concatenate the three helper functions under test, in dependency order."""
    return "\n".join(
        _extract_function(name)
        for name in (
            "node_deps_fingerprint",
            "node_deps_marker_path",
            "node_deps_up_to_date",
            "node_deps_mark_installed",
        )
    )


def _run(script: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "-c", f"set -e\n{_harness()}\n{script}"],
        cwd=cwd,
        capture_output=True,
        text=True,
    )


# ---------------------------------------------------------------------------
# Static guards: the skip checks must actually be wired into install_node_deps,
# not just exist as unused helper functions.
# ---------------------------------------------------------------------------


def test_install_node_deps_calls_up_to_date_before_root_npm_install() -> None:
    text = INSTALL_SH.read_text()
    fn = _extract_function("install_node_deps")
    idx_check = fn.find('node_deps_up_to_date "$INSTALL_DIR"')
    idx_npm = fn.find("run_with_timeout \"$NODE_DEPS_TIMEOUT\" npm install --silent")
    assert idx_check != -1, "install_node_deps must consult node_deps_up_to_date for the root install"
    assert idx_npm != -1, "expected the root npm install call to still be present"
    assert idx_check < idx_npm, "the skip check must run BEFORE the npm install it's meant to skip"
    assert 'node_deps_mark_installed "$INSTALL_DIR"' in fn, (
        "a successful root npm install must record the marker so the next run can skip"
    )


def test_install_node_deps_calls_up_to_date_before_tui_npm_install() -> None:
    fn = _extract_function("install_node_deps")
    idx_check = fn.find('node_deps_up_to_date "$INSTALL_DIR/ui-tui"')
    idx_npm = fn.rfind("run_with_timeout \"$NODE_DEPS_TIMEOUT\" npm install --silent")
    assert idx_check != -1, "install_node_deps must consult node_deps_up_to_date for the ui-tui install"
    assert idx_npm != -1
    assert idx_check < idx_npm
    assert 'node_deps_mark_installed "$INSTALL_DIR/ui-tui"' in fn


def test_install_node_deps_marks_stage_skipped_when_nothing_ran() -> None:
    fn = _extract_function("install_node_deps")
    assert "mark_stage_skipped" in fn, (
        "when every npm install in this stage was a no-op, the stage must surface "
        "skipped=true via mark_stage_skipped (same protocol setup_venv/install_deps use) "
        "so the desktop bootstrap UI shows it as skipped rather than a fresh install"
    )


# ---------------------------------------------------------------------------
# Behavioral: root project (has package-lock.json)
# ---------------------------------------------------------------------------


def test_fresh_checkout_with_no_marker_needs_install(tmp_path: Path) -> None:
    (tmp_path / "package.json").write_text('{"name":"t"}\n')
    (tmp_path / "package-lock.json").write_text('{"lockfileVersion":3}\n')

    res = _run('node_deps_up_to_date "$PWD" && echo UP_TO_DATE || echo NEEDS_INSTALL', tmp_path)
    assert res.returncode == 0, res.stderr
    assert "NEEDS_INSTALL" in res.stdout


def test_marked_install_with_unchanged_lockfile_is_up_to_date(tmp_path: Path) -> None:
    (tmp_path / "package.json").write_text('{"name":"t"}\n')
    (tmp_path / "package-lock.json").write_text('{"lockfileVersion":3}\n')
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "somepkg").mkdir()

    res = _run(
        'node_deps_mark_installed "$PWD"\n'
        'node_deps_up_to_date "$PWD" && echo UP_TO_DATE || echo NEEDS_INSTALL',
        tmp_path,
    )
    assert res.returncode == 0, res.stderr
    assert "UP_TO_DATE" in res.stdout


def test_lockfile_change_invalidates_marker(tmp_path: Path) -> None:
    (tmp_path / "package.json").write_text('{"name":"t"}\n')
    (tmp_path / "package-lock.json").write_text('{"lockfileVersion":3}\n')
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "somepkg").mkdir()

    res = _run(
        'node_deps_mark_installed "$PWD"\n'
        'echo \'{"lockfileVersion":3,"changed":true}\' > package-lock.json\n'
        'node_deps_up_to_date "$PWD" && echo UP_TO_DATE || echo NEEDS_INSTALL',
        tmp_path,
    )
    assert res.returncode == 0, res.stderr
    assert "NEEDS_INSTALL" in res.stdout, (
        "changing package-lock.json after marking installed must invalidate the marker"
    )


def test_package_json_only_change_invalidates_marker(tmp_path: Path) -> None:
    """package.json can change (e.g. a version/script edit) with the lockfile
    catching up in the same commit, or drifting temporarily -- either way the
    fingerprint must cover both files, not just the lockfile."""
    (tmp_path / "package.json").write_text('{"name":"t","version":"1.0.0"}\n')
    (tmp_path / "package-lock.json").write_text('{"lockfileVersion":3}\n')
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "somepkg").mkdir()

    res = _run(
        'node_deps_mark_installed "$PWD"\n'
        'echo \'{"name":"t","version":"1.0.1"}\' > package.json\n'
        'node_deps_up_to_date "$PWD" && echo UP_TO_DATE || echo NEEDS_INSTALL',
        tmp_path,
    )
    assert res.returncode == 0, res.stderr
    assert "NEEDS_INSTALL" in res.stdout


def test_empty_node_modules_forces_install_despite_fresh_marker(tmp_path: Path) -> None:
    """Regression guard for the flake documented in install.ps1's Install-Desktop
    (npm's own node_modules/.package-lock.json marker can be stale while
    node_modules is actually empty -- a Windows workspace-hoisting flake).
    Our marker must not be trusted in isolation; node_modules content is
    checked too."""
    (tmp_path / "package.json").write_text('{"name":"t"}\n')
    (tmp_path / "package-lock.json").write_text('{"lockfileVersion":3}\n')
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "somepkg").mkdir()

    res = _run(
        'node_deps_mark_installed "$PWD"\n'
        'rm -rf node_modules/somepkg\n'
        'node_deps_up_to_date "$PWD" && echo UP_TO_DATE || echo NEEDS_INSTALL',
        tmp_path,
    )
    assert res.returncode == 0, res.stderr
    assert "NEEDS_INSTALL" in res.stdout, (
        "an empty node_modules must force a real install even with a fresh, "
        "matching-fingerprint marker on disk"
    )


def test_missing_node_modules_entirely_needs_install(tmp_path: Path) -> None:
    (tmp_path / "package.json").write_text('{"name":"t"}\n')
    (tmp_path / "package-lock.json").write_text('{"lockfileVersion":3}\n')

    res = _run(
        # Write a marker directly (simulating a corrupted/partial state) without
        # ever having created node_modules.
        'node_deps_fingerprint "$PWD" > .node-deps-installed\n'
        'node_deps_up_to_date "$PWD" && echo UP_TO_DATE || echo NEEDS_INSTALL',
        tmp_path,
    )
    assert res.returncode == 0, res.stderr
    assert "NEEDS_INSTALL" in res.stdout


# ---------------------------------------------------------------------------
# Behavioral: ui-tui-style project (package.json only, no lockfile)
# ---------------------------------------------------------------------------


def test_no_lockfile_project_falls_back_to_package_json_fingerprint(tmp_path: Path) -> None:
    (tmp_path / "package.json").write_text('{"name":"tui"}\n')
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "pkg").mkdir()

    res = _run(
        'node_deps_mark_installed "$PWD"\n'
        'node_deps_up_to_date "$PWD" && echo UP_TO_DATE || echo NEEDS_INSTALL',
        tmp_path,
    )
    assert res.returncode == 0, res.stderr
    assert "UP_TO_DATE" in res.stdout, (
        "a package.json-only project (no package-lock.json, e.g. ui-tui) must "
        "still get a working fingerprint/marker cycle"
    )


def test_no_lockfile_project_package_json_change_invalidates(tmp_path: Path) -> None:
    (tmp_path / "package.json").write_text('{"name":"tui","version":"1.0.0"}\n')
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "pkg").mkdir()

    res = _run(
        'node_deps_mark_installed "$PWD"\n'
        'echo \'{"name":"tui","version":"1.0.1"}\' > package.json\n'
        'node_deps_up_to_date "$PWD" && echo UP_TO_DATE || echo NEEDS_INSTALL',
        tmp_path,
    )
    assert res.returncode == 0, res.stderr
    assert "NEEDS_INSTALL" in res.stdout


# ---------------------------------------------------------------------------
# Fail-open guards
# ---------------------------------------------------------------------------


def test_no_package_json_at_all_fails_open_to_needs_install(tmp_path: Path) -> None:
    """An empty directory (nothing to fingerprint) must never report
    up-to-date -- there is nothing to compare against, so the honest answer
    is "run the real install and let it decide"."""
    res = _run('node_deps_up_to_date "$PWD" && echo UP_TO_DATE || echo NEEDS_INSTALL', tmp_path)
    assert res.returncode == 0, res.stderr
    assert "NEEDS_INSTALL" in res.stdout


def test_marker_write_failure_does_not_abort_install(tmp_path: Path) -> None:
    """node_deps_mark_installed must be best-effort: a read-only project dir
    (disk full, permissions) must not fail the calling stage."""
    (tmp_path / "package.json").write_text('{"name":"t"}\n')
    (tmp_path / "package-lock.json").write_text('{"lockfileVersion":3}\n')

    res = _run(
        'chmod 555 "$PWD"\n'
        'node_deps_mark_installed "$PWD"\n'
        'echo MARK_INSTALLED_DID_NOT_ABORT\n'
        'chmod 755 "$PWD"',
        tmp_path,
    )
    assert res.returncode == 0, res.stderr
    assert "MARK_INSTALLED_DID_NOT_ABORT" in res.stdout
