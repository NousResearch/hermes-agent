"""Installer regression tests for the SQLite WAL-reset runtime migration."""

from __future__ import annotations

import os
import re
import stat
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"
INSTALL_PS1 = REPO_ROOT / "scripts" / "install.ps1"


def _function_body(source: str, declaration: str) -> str:
    start = source.index(declaration)
    brace = source.index("{", start)
    depth = 0
    for index in range(brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    raise AssertionError(f"unterminated function: {declaration}")


def _make_executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def test_posix_installer_reinstalls_vulnerable_runtime_from_refreshed_catalog(
    tmp_path,
):
    """The shell helper must select the newly verified managed interpreter."""
    install_source = INSTALL_SH.read_text(encoding="utf-8")
    functions = "\n\n".join(
        _function_body(install_source, f"{name}()")
        for name in (
            "sqlite_wal_reset_vulnerable",
            "ensure_fixed_sqlite_python",
        )
    )

    old_python = tmp_path / "old-python"
    fixed_python = tmp_path / "fixed-python"
    fake_uv = tmp_path / "uv"
    state = tmp_path / "state"
    calls = tmp_path / "calls"

    _make_executable(
        old_python,
        """#!/bin/sh
case "$*" in
  *"raise SystemExit(42 if vulnerable else 0)"*) exit 42 ;;
  *"sqlite3.sqlite_version"*) echo 3.50.4; exit 0 ;;
  *"--version"*) echo "Python 3.11.15"; exit 0 ;;
esac
exit 1
""",
    )
    _make_executable(
        fixed_python,
        """#!/bin/sh
case "$*" in
  *"raise SystemExit(42 if vulnerable else 0)"*) exit 0 ;;
  *"sqlite3.sqlite_version"*) echo 3.53.1; exit 0 ;;
  *"--version"*) echo "Python 3.11.15"; exit 0 ;;
esac
exit 1
""",
    )
    _make_executable(
        fake_uv,
        f"""#!/bin/sh
echo "$*" >> "{calls}"
case "$1 $2" in
  "self update") exit 0 ;;
  "python install") printf fixed > "{state}"; exit 0 ;;
  "python find")
    if [ -f "{state}" ]; then
      printf '%s\\n' "{fixed_python}"
    else
      printf '%s\\n' "{old_python}"
    fi
    exit 0
    ;;
esac
exit 1
""",
    )

    script = f"""set -e
UV_CMD={fake_uv}
PYTHON_VERSION=3.11
PYTHON_PATH={old_python}
log_warn() {{ :; }}
log_success() {{ :; }}
{functions}
ensure_fixed_sqlite_python
printf '%s\\n' "$PYTHON_PATH"
"""
    result = subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PATH": os.environ.get("PATH", "")},
    )

    assert result.stdout.strip() == str(fixed_python)
    call_lines = calls.read_text(encoding="utf-8").splitlines()
    assert call_lines.index("self update") < call_lines.index(
        "python install 3.11 --reinstall"
    )
    assert "python find 3.11 --managed-python" in call_lines


def test_posix_venv_is_pinned_to_verified_python_path():
    source = INSTALL_SH.read_text(encoding="utf-8")
    setup_venv = _function_body(source, "setup_venv()")
    assert 'venv venv --python "$PYTHON_PATH"' in setup_venv
    assert "ensure_fixed_sqlite_python" in _function_body(source, "check_python()")


def test_windows_installer_refreshes_uv_before_force_reinstall():
    source = INSTALL_PS1.read_text(encoding="utf-8")
    resolver = _function_body(source, "function Resolve-FixedSqlitePythonPath")
    update_at = resolver.index("& $UvCmd self update")
    reinstall_at = resolver.index("& $UvCmd python install $Version --reinstall")
    verify_at = resolver.index(
        "Test-SqliteWalResetVulnerable -PythonPath $candidatePath"
    )

    assert update_at < reinstall_at < verify_at
    assert resolver.count("| Out-Host") >= 2
    assert "python find $Version --managed-python" in resolver
    assert re.search(r"\(3,\s*51,\s*3\)", source)
    assert re.search(r"\(3,\s*50,\s*7\)", source)
    assert re.search(r"\(3,\s*44,\s*6\)", source)


def test_windows_venv_retries_after_process_sweep_and_pins_verified_path():
    source = INSTALL_PS1.read_text(encoding="utf-8")
    install_venv = _function_body(source, "function Install-Venv")
    process_sweep_at = install_venv.index("Stop-Process")
    retry_at = install_venv.index("Resolve-FixedSqlitePythonPath")
    create_at = install_venv.index("& $UvCmd venv venv --python $venvPythonRequest")

    assert process_sweep_at < retry_at < create_at
