"""The hand-off result protocol's retryable classification (#64577 Cause 4).

A deterministic git failure — a stash/pull conflict, un-mergeable local
commits, a parked branch, a bricked venv — fails identically on every retry,
and each retry re-runs the (possibly multi-GB) pre-update backup. The result
file must therefore carry ``retryable:false`` for these classes so the Desktop
update-failure card can drop its Retry button; ordinary failures stay
retryable.

These tests drive the REAL ``posix.sh`` update-run + classification +
``write_result`` sequence (via ``--self-test-result``, the same self-test
pattern the TCC heal uses) against a stub installation launcher whose exit
code and output reproduce each failure class. No mocks of the script, no
source-text reading.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
POSIX_SH = REPO_ROOT / "scripts" / "desktop-update" / "posix.sh"

requires_bash = pytest.mark.skipif(
    not os.path.exists("/bin/bash"), reason="posix.sh needs /bin/bash"
)


def _write_exe(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def _run_handoff(home: Path, install_root: Path, launcher_body: str) -> dict:
    """Drive the real script; return the parsed result JSON."""
    # Bootable interpreter stubs: the legacy-install epilogue probes
    # venv/bin/python{,3} with `import encodings`; without these the probe
    # would classify the stub install as a bricked venv (#95759) and mask the
    # failure class under test.
    _write_exe(install_root / "venv" / "bin" / "python3", "#!/usr/bin/env bash\nexit 0\n")
    _write_exe(install_root / "venv" / "bin" / "python", "#!/usr/bin/env bash\nexit 0\n")
    _write_exe(install_root / "venv" / "bin" / "hermes", launcher_body)
    env = dict(os.environ)
    env["HERMES_HOME"] = str(home)
    env.pop("HERMES_UPDATE_STATUS_FILE", None)
    proc = subprocess.run(
        [
            "/bin/bash",
            str(POSIX_SH),
            "--self-test-result",
            "--install-root",
            str(install_root),
            "--branch",
            "main",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        env=env,
        cwd=str(install_root),
    )
    result_path = home / ".hermes-update-result.json"
    # The hand-off daemonizes: this invocation spawns a detached child (via
    # nohup) and returns immediately, so the result file lands asynchronously.
    deadline = time.monotonic() + 60
    while not result_path.exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    assert result_path.exists(), (
        f"the hand-off never wrote its result file (exit {proc.returncode}).\n"
        f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    return json.loads(result_path.read_text(encoding="utf-8-sig"))


def _launcher(exit_code: int, *lines: str) -> str:
    """A stub installation launcher: `--help` advertises --keep-stash; every
    other invocation prints the given lines and exits with exit_code."""
    printed = "".join(f'echo "{line}"\n' for line in lines)
    return (
        "#!/usr/bin/env bash\n"
        'if [ "$1" = "update" ] && [ "$2" = "--help" ]; then echo "  --keep-stash"; exit 0; fi\n'
        f"{printed}"
        f"exit {exit_code}\n"
    )


@requires_bash
def test_transient_failure_stays_retryable(tmp_path):
    home = tmp_path / "home"
    install_root = tmp_path / "install"
    home.mkdir()
    result = _run_handoff(
        home,
        install_root,
        _launcher(1, "some transient error"),
    )
    assert result["ok"] is False
    assert result["exit_code"] == 1
    assert result["retryable"] is True


@requires_bash
def test_stash_conflict_failure_is_not_retryable(tmp_path):
    home = tmp_path / "home"
    install_root = tmp_path / "install"
    home.mkdir()
    result = _run_handoff(
        home,
        install_root,
        _launcher(1, "✗ Could not stash local changes — update aborted."),
    )
    assert result["ok"] is False
    assert result["retryable"] is False, (
        "a stash/pull conflict fails identically on every retry — the result "
        "file must say retrying cannot help (#64577)"
    )


@requires_bash
def test_overwritten_by_merge_failure_is_not_retryable(tmp_path):
    home = tmp_path / "home"
    install_root = tmp_path / "install"
    home.mkdir()
    result = _run_handoff(
        home,
        install_root,
        _launcher(1, "error: Your local changes to the file would be overwritten by merge."),
    )
    assert result["retryable"] is False
