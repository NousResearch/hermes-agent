from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path


ISOLATED_RUNTIME_ENV = "MUNCHO_OWNER_GATE_ISOLATED_TEST_RUNTIME"


def _minimal_environment() -> dict[str, str]:
    return {
        "HOME": os.environ["HOME"],
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": os.environ["PATH"],
        "PYTHONHASHSEED": "0",
        "TZ": "UTC",
        "UV_NO_PROGRESS": "1",
    }


def test_passkey_v2_suites_run_under_exact_isolated_runtime(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[3]
    project = (
        repository
        / "tests"
        / "scripts"
        / "canary"
        / "isolated_passkey_runtime"
    )
    uv = shutil.which("uv")
    assert uv is not None, "uv is required for the isolated passkey test runtime"

    runtime = tmp_path / "runtime"
    sync_environment = {
        **_minimal_environment(),
        "UV_PROJECT_ENVIRONMENT": str(runtime),
    }
    synced = subprocess.run(
        [
            uv,
            "sync",
            "--no-config",
            "--project",
            str(project),
            "--locked",
            "--python",
            "3.11",
        ],
        cwd=repository,
        env=sync_environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=120,
        check=False,
    )
    assert synced.returncode == 0, synced.stdout

    isolated_python = runtime / "bin" / "python"
    assert isolated_python.is_file()
    test_environment = {
        **_minimal_environment(),
        ISOLATED_RUNTIME_ENV: "1",
    }
    completed = subprocess.run(
        [
            str(repository / "scripts/run_tests.sh"),
            "--python",
            str(isolated_python),
            str(project / "runtime_contract_suite.py"),
            str(repository / "tests/scripts/canary/test_passkey_v2_security.py"),
            str(
                repository
                / "tests/scripts/canary/test_passkey_v2_executor_e2e.py"
            ),
            "-q",
        ],
        cwd=repository,
        env=test_environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout
    for suite in (
        project / "runtime_contract_suite.py",
        repository / "tests/scripts/canary/test_passkey_v2_security.py",
        repository / "tests/scripts/canary/test_passkey_v2_executor_e2e.py",
    ):
        relative = suite.relative_to(repository)
        progress = [
            line
            for line in completed.stdout.splitlines()
            if f"✓ {relative} (" in line
        ]
        assert len(progress) == 1, completed.stdout
        # Behavioral invariant: every isolated suite executed at least one
        # test and none of its tests silently skipped. Do not freeze the
        # suite's enumeration count; new tests remain free to land.
        assert re.search(r"\([1-9][0-9]*✓(?:,| )", progress[0]) is not None
        assert re.search(r"[1-9][0-9]*s(?:,| )", progress[0]) is None
