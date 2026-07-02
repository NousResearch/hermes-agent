"""Behavioral regression coverage for desktop installer npm tree self-repair."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"


def _desktop_install_functions() -> str:
    """Extract the installer functions needed for an isolated shell fixture.

    The assertions below execute the real functions with a stubbed ``npm``;
    this helper keeps the test from running install.sh's top-level installer.
    """
    source = INSTALL_SH.read_text()
    function_names = (
        "_desktop_run_logged",
        "_desktop_workspace_npm_install_attempt",
        "_desktop_npm_log_has_tree_corruption",
        "_purge_desktop_npm_artifacts",
        "install_desktop",
    )
    functions = []
    for name in function_names:
        match = re.search(
            rf"^{re.escape(name)}\(\) \{{.*?^\}}",
            source,
            re.MULTILINE | re.DOTALL,
        )
        assert match is not None, f"could not extract {name}() from install.sh"
        functions.append(match.group(0))
    return "\n\n".join(functions)


def _run_desktop_install(tmp_path: Path, npm_mode: str) -> dict[str, object]:
    """Run ``install_desktop`` against a controlled npm failure fixture."""
    install_dir = tmp_path / "install"
    desktop_dir = install_dir / "apps" / "desktop"
    desktop_dir.mkdir(parents=True)
    (desktop_dir / "package.json").write_text("{}")
    lockfile = install_dir / "package-lock.json"
    lockfile.write_text('{"lockfileVersion": 3}')

    artifacts = (
        install_dir / "node_modules",
        desktop_dir / "node_modules",
        desktop_dir / "dist",
        desktop_dir / "release",
    )
    for artifact in artifacts:
        artifact.mkdir()
        (artifact / "sentinel").write_text("generated")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    runlog = tmp_path / "npm-runs.log"
    eventlog = tmp_path / "installer-events.log"
    npm = bin_dir / "npm"
    npm.write_text(
        """#!/usr/bin/env bash
set -eu
printf '%s\\n' "$*" >> "$RUNLOG"
calls=$(wc -l < "$RUNLOG")

case "$NPM_MODE" in
  repaired)
    if [ "$calls" -lt 3 ]; then
      printf '%s\\n' 'npm ERR! code ENOTEMPTY' >&2
      printf '%s\\n' 'npm ERR! directory not empty, rename node_modules/pkg to node_modules/.pkg-abc' >&2
      exit 1
    fi
    if [ ! -e "$INSTALL_DIR/node_modules" ] \\
       && [ ! -e "$INSTALL_DIR/apps/desktop/node_modules" ] \\
       && [ ! -e "$INSTALL_DIR/apps/desktop/dist" ] \\
       && [ ! -e "$INSTALL_DIR/apps/desktop/release" ]; then
      printf '%s\\n' 'cleanup-observed' >> "$RUNLOG"
      exit 0
    fi
    printf '%s\\n' 'cleanup-not-observed' >> "$RUNLOG"
    exit 1
    ;;
  persistent-corruption)
    printf '%s\\n' 'npm ERR! code ENOTEMPTY' >&2
    printf '%s\\n' 'npm ERR! directory not empty, rename node_modules/pkg to node_modules/.pkg-abc' >&2
    exit 1
    ;;
  ordinary-failure)
    printf '%s\\n' 'npm ERR! code ECONNRESET' >&2
    exit 1
    ;;
esac
"""
    )
    npm.chmod(0o755)

    shell = f"""
set -u
INSTALL_DIR={shlex.quote(str(install_dir))}
DESKTOP_BUILD_TIMEOUT=60
OS=linux
RUNLOG={shlex.quote(str(runlog))}
EVENTLOG={shlex.quote(str(eventlog))}
NPM_MODE={shlex.quote(npm_mode)}
export INSTALL_DIR RUNLOG NPM_MODE
PATH={shlex.quote(str(bin_dir))}:$PATH

log_info() {{ printf 'INFO:%s\\n' "$*" >> "$EVENTLOG"; }}
log_warn() {{ printf 'WARN:%s\\n' "$*" >> "$EVENTLOG"; }}
log_error() {{ printf 'ERROR:%s\\n' "$*" >> "$EVENTLOG"; }}
log_success() {{ printf 'SUCCESS:%s\\n' "$*" >> "$EVENTLOG"; }}
check_node() {{ :; }}
run_with_timeout() {{ local _timeout="$1"; shift; "$@"; }}
_electron_pkg_staged_missing_dist() {{ return 1; }}
_restore_electron_dist_with_fallback() {{ return 1; }}
_electron_dist_ok() {{ return 1; }}
_desktop_pack() {{
    mkdir -p "$INSTALL_DIR/apps/desktop/release/linux-unpacked"
    : > "$INSTALL_DIR/apps/desktop/release/linux-unpacked/Hermes"
    chmod +x "$INSTALL_DIR/apps/desktop/release/linux-unpacked/Hermes"
    printf '%s\n' 'pack' >> "$EVENTLOG"
}}
clear_electron_build_cache() {{ :; }}
_restore_electron_dist() {{ return 1; }}
restore_dirty_lockfiles() {{ :; }}

{_desktop_install_functions()}

if install_desktop; then
    result=0
else
    result=$?
fi
printf 'RESULT=%s\n' "$result"
"""
    completed = subprocess.run(
        ["bash", "-c", shell],
        capture_output=True,
        text=True,
        env=dict(os.environ),
    )
    result_line = next(
        (line for line in completed.stdout.splitlines() if line.startswith("RESULT=")),
        None,
    )
    assert result_line is not None, (
        "desktop installer fixture did not report a result:\n"
        f"stdout={completed.stdout}\nstderr={completed.stderr}"
    )
    result = int(result_line.split("=", 1)[1])
    return {
        "returncode": completed.returncode,
        "result": result,
        "npm_runs": runlog.read_text().splitlines(),
        "events": eventlog.read_text().splitlines(),
        "artifacts": artifacts,
        "lockfile": lockfile,
        "stderr": completed.stderr,
    }


def test_enotempty_clears_generated_artifacts_then_retries_once(tmp_path: Path) -> None:
    result = _run_desktop_install(tmp_path, "repaired")

    assert result["returncode"] == 0, result["stderr"]
    assert result["result"] == 0
    assert result["npm_runs"] == ["ci", "install", "ci", "cleanup-observed"]
    assert all(not artifact.exists() for artifact in result["artifacts"][:3])
    assert result["artifacts"][3].is_dir(), (
        "the successful desktop build recreates release/"
    )
    assert result["lockfile"].read_text() == '{"lockfileVersion": 3}'
    assert any(
        "clearing generated npm/build artifacts and retrying once" in event
        for event in result["events"]
    )


def test_non_corruption_failure_does_not_clear_or_retry(tmp_path: Path) -> None:
    result = _run_desktop_install(tmp_path, "ordinary-failure")

    assert result["returncode"] == 0, result["stderr"]
    assert result["result"] == 1
    assert result["npm_runs"] == ["ci", "install"]
    assert all(artifact.exists() for artifact in result["artifacts"])
    assert not any(
        "clearing generated npm/build artifacts" in event for event in result["events"]
    )


def test_persistent_enotempty_reports_manual_repair_guidance(tmp_path: Path) -> None:
    result = _run_desktop_install(tmp_path, "persistent-corruption")

    assert result["returncode"] == 0, result["stderr"]
    assert result["result"] == 1
    assert result["npm_runs"] == ["ci", "install", "ci", "install"]
    assert all(not artifact.exists() for artifact in result["artifacts"])
    assert any(
        "already removed generated desktop dependency/build artifacts and retried once"
        in event
        for event in result["events"]
    )
    assert any(
        "quit all Hermes/Node/npm processes" in event for event in result["events"]
    )
