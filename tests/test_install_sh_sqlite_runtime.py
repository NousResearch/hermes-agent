"""Behavioral installer coverage for managed SQLite runtime preservation.

The tests execute the public ``--stage`` protocol.  Only external executables
(Python and uv) are replaced with disposable shims; installer shell functions
are never extracted, sourced, or stubbed.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
INSTALL_SH = REPO_ROOT / "scripts" / "install.sh"
REAL_PYTHON = Path(sys.executable).absolute()

_LIVE_PYTHON = r"""#!/bin/bash
set -e
if [ "$1" = "-I" ] && [ "$2" = "-c" ] && [[ "$3" == *"sqlite_source_id"* ]]; then
  if [ "${HERMES_TEST_REPAIR_OUTCOME:-safe}" = "safe" ]; then
    sqlite="3.53.1"
  else
    sqlite="3.50.4"
  fi
  printf '{"base_prefix":"/test","executable":"%s","python_version":[3,11,14],"sqlite_version":[%s],"sqlite_version_string":"%s","sqlite_source_id":"test-live"}\n' \
    "$0" "${sqlite//./,}" "$sqlite"
  exit 0
fi
exec __REAL_PYTHON__ "$@"
"""

_CANDIDATE_PYTHON = r"""#!/bin/bash
set -e
if [ "$1" = "-I" ] && [ "$2" = "-c" ] && [[ "$3" == *"sqlite_source_id"* ]]; then
  printf '{"base_prefix":"/test","executable":"%s","python_version":[3,11,15],"sqlite_version":[3,53,1],"sqlite_version_string":"3.53.1","sqlite_source_id":"test-fixed"}\n' "$0"
  exit 0
fi
# The repairer's smoke command validates imports.  Dependency installation is
# the uv shim's responsibility in this integration harness, so report success.
if [ "$1" = "-I" ] && [ "$2" = "-c" ]; then
  exit 0
fi
exec __REAL_PYTHON__ "$@"
"""

_MANAGED_UV = r"""#!/bin/bash
set -e
printf '%s\n' "$*" >> "${HERMES_TEST_UV_LOG}"
if [ "$1" = "--version" ]; then
  echo 'uv test'
  exit 0
fi
if [ "$1" = "python" ] && [ "$2" = "find" ]; then
  if [[ " $* " == *" --managed-python "* ]]; then
    printf '%s\n' "${UV_PYTHON_INSTALL_DIR}/cpython-3.11.15/bin/python"
  else
    printf '%s\n' "__REAL_PYTHON__"
  fi
  exit 0
fi
if [ "$1" = "python" ] && [ "$2" = "install" ]; then
  candidate="${UV_PYTHON_INSTALL_DIR}/cpython-3.11.15/bin/python"
  mkdir -p "$(dirname "$candidate")"
  cat > "$candidate" <<'PY'
__CANDIDATE_PYTHON__
PY
  chmod +x "$candidate"
  exit 0
fi
if [ "$1" = "venv" ]; then
  candidate="$2/bin/python"
  mkdir -p "$(dirname "$candidate")"
  cat > "$candidate" <<'PY'
__CANDIDATE_PYTHON__
PY
  chmod +x "$candidate"
  exit 0
fi
if [ "$1" = "sync" ] && [[ " $* " == *" --python "* ]] && \
   [ "${HERMES_TEST_REPAIR_OUTCOME}" = "failure" ]; then
  echo 'simulated candidate dependency sync failure' >&2
  exit 17
fi
exit 0
"""

_VENV_PYTHON = r"""#!/bin/bash
if [ "$1" = "--version" ]; then
  echo 'Python __VERSION__.0'
  exit 0
fi
if [ "$1" = "-I" ] && [ "$2" = "-c" ]; then
  if [ "__PROBE_FAILURE__" = "true" ]; then
    exit 9
  fi
  echo '__VERSION__'
  exit 0
fi
exit 0
"""

_VENV_UV = r"""#!/bin/bash
set -e
printf '%s\n' "$*" >> "${HERMES_TEST_UV_LOG}"
if [ "$1" = "--version" ]; then
  echo 'uv test'
elif [ "$1" = "python" ] && [ "$2" = "find" ]; then
  printf '%s\n' "${HERMES_INSTALL_DIR}/venv/bin/python"
elif [ "$1" = "venv" ]; then
  mkdir -p venv/bin
  printf '#!/bin/bash\nexit 0\n' > venv/bin/python
  chmod +x venv/bin/python
fi
exit 0
"""


def _write_executable(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _json_frame(stdout: str) -> dict[str, object]:
    frames: list[dict[str, object]] = []
    for line in stdout.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("stage"):
            frames.append(value)
    assert len(frames) == 1, stdout
    return frames[0]


def _run_stage(
    tmp_path: Path, stage: str, env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "bash",
            str(INSTALL_SH),
            "--stage",
            stage,
            "--json",
            "--non-interactive",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )


def _runtime_stage_fixture(
    tmp_path: Path, outcome: str
) -> tuple[Path, dict[str, str], Path]:
    root = tmp_path / "checkout"
    (root / "venv" / "bin").mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        '[project]\nname = "installer-runtime-test"\nversion = "0"\n',
        encoding="utf-8",
    )
    (root / "uv.lock").write_text(
        'version = 1\nrevision = 1\nrequires-python = ">=3.11"\n',
        encoding="utf-8",
    )
    live_python = _LIVE_PYTHON.replace("__REAL_PYTHON__", str(REAL_PYTHON))
    _write_executable(root / "venv" / "bin" / "python", live_python)
    (root / "venv" / "sentinel").write_text("live", encoding="utf-8")

    home = tmp_path / "hermes-home"
    candidate_python = _CANDIDATE_PYTHON.replace("__REAL_PYTHON__", str(REAL_PYTHON))
    managed_uv = _MANAGED_UV.replace("__REAL_PYTHON__", str(REAL_PYTHON)).replace(
        "__CANDIDATE_PYTHON__", candidate_python
    )
    _write_executable(home / "bin" / "uv", managed_uv)

    uv_log = tmp_path / "uv.log"
    env = os.environ | {
        "HERMES_HOME": str(home),
        "HERMES_INSTALL_DIR": str(root),
        "HERMES_TEST_REPAIR_OUTCOME": outcome,
        "HERMES_TEST_UV_LOG": str(uv_log),
    }
    return root, env, uv_log


@pytest.mark.live_system_guard_bypass
@pytest.mark.skipif(
    shutil.which("bash") is None or not REAL_PYTHON.is_file(),
    reason="needs bash and the repository test venv",
)
@pytest.mark.parametrize(
    ("outcome", "expected_ok"),
    [("safe", True), ("repaired", True), ("failure", False)],
)
def test_python_deps_stage_runs_real_runtime_repair(
    tmp_path: Path, outcome: str, expected_ok: bool
) -> None:
    root, env, uv_log = _runtime_stage_fixture(tmp_path, outcome)

    result = _run_stage(tmp_path, "python-deps", env)

    frame = _json_frame(result.stdout)
    assert frame["stage"] == "python-deps"
    assert frame["ok"] is expected_ok
    assert (result.returncode == 0) is expected_ok
    calls = uv_log.read_text(encoding="utf-8")
    assert "sync --extra all --locked" in calls

    sentinel = root / "venv" / "sentinel"
    if outcome == "safe":
        assert sentinel.read_text(encoding="utf-8") == "live"
        assert "python install" not in calls
        assert "Managed Python/SQLite runtime verified" in result.stdout
    elif outcome == "repaired":
        assert not sentinel.exists()
        assert "python install 3.11" in calls
        assert "Managed Python runtime repaired (SQLite 3.50.4" in result.stdout
    else:
        assert sentinel.read_text(encoding="utf-8") == "live"
        assert "sync --extra all --locked --python" in calls
        assert "Managed Python/SQLite runtime is not safe" in result.stdout
        assert "replacement environment did not pass" in result.stderr
        assert frame["reason"] == "exit code 1"


def _venv_stage_fixture(
    tmp_path: Path, *, version: str, probe_failure: bool = False
) -> tuple[Path, dict[str, str], Path]:
    root = tmp_path / "checkout"
    python = root / "venv" / "bin" / "python"
    _write_executable(
        python,
        _VENV_PYTHON.replace("__VERSION__", version).replace(
            "__PROBE_FAILURE__", str(probe_failure).lower()
        ),
    )
    (root / "venv" / "sentinel").write_text("keep", encoding="utf-8")

    home = tmp_path / "hermes-home"
    uv_log = tmp_path / "uv.log"
    _write_executable(home / "bin" / "uv", _VENV_UV)
    env = os.environ | {
        "HERMES_HOME": str(home),
        "HERMES_INSTALL_DIR": str(root),
        "HERMES_TEST_UV_LOG": str(uv_log),
    }
    return root, env, uv_log


@pytest.mark.live_system_guard_bypass
@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_venv_stage_preserves_compatible_existing_environment(tmp_path: Path) -> None:
    root, env, uv_log = _venv_stage_fixture(tmp_path, version="3.11")

    result = _run_stage(tmp_path, "venv", env)

    assert result.returncode == 0, result.stderr
    assert _json_frame(result.stdout)["ok"] is True
    assert (root / "venv" / "sentinel").read_text(encoding="utf-8") == "keep"
    assert (
        "Compatible virtual environment already exists; preserving it" in result.stdout
    )
    assert not any(
        line.startswith("venv ")
        for line in uv_log.read_text(encoding="utf-8").splitlines()
    )


@pytest.mark.live_system_guard_bypass
@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
@pytest.mark.parametrize(
    ("version", "probe_failure"), [("3.12", False), ("3.11", True)]
)
def test_venv_stage_recreates_unusable_existing_environment(
    tmp_path: Path, version: str, probe_failure: bool
) -> None:
    root, env, uv_log = _venv_stage_fixture(
        tmp_path, version=version, probe_failure=probe_failure
    )

    result = _run_stage(tmp_path, "venv", env)

    assert result.returncode == 0, result.stderr
    assert _json_frame(result.stdout)["ok"] is True
    assert not (root / "venv" / "sentinel").exists()
    assert any(
        line.startswith("venv ")
        for line in uv_log.read_text(encoding="utf-8").splitlines()
    )
