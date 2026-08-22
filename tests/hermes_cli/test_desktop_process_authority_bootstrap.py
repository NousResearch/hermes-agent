from __future__ import annotations

import os
import runpy
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import pytest

BOOTSTRAP = (
    Path(__file__).resolve().parents[2]
    / "hermes_cli"
    / "desktop_bootstrap"
    / "sitecustomize.py"
)
AUTHORITY_ENV = "HERMES_DESKTOP_PROCESS_AUTHORITY"
DESCENDANT_GUARD_ENV = "_HERMES_DESKTOP_POSIX_DESCENDANT_GUARD"


def run_bootstrap(monkeypatch, mode: str | None):
    monkeypatch.delenv(DESCENDANT_GUARD_ENV, raising=False)
    if mode is None:
        monkeypatch.delenv(AUTHORITY_ENV, raising=False)
    else:
        monkeypatch.setenv(AUTHORITY_ENV, mode)
    return runpy.run_path(
        str(BOOTSTRAP),
        run_name="desktop_authority_sitecustomize_test",
    )


def test_unmarked_interpreter_is_untouched(monkeypatch):
    run_bootstrap(monkeypatch, None)


def test_windows_mode_installs_windows_authority(monkeypatch):
    install = Mock(return_value=None)
    module = types.ModuleType("hermes_cli.windows_process_authority")
    module.install_windows_process_authority = install
    monkeypatch.setitem(sys.modules, module.__name__, module)

    run_bootstrap(monkeypatch, "windows-job-v1")

    install.assert_called_once_with()


def test_posix_mode_installs_posix_authority(monkeypatch):
    install = Mock(return_value=None)
    module = types.ModuleType("hermes_cli.posix_process_authority")
    module.install_posix_process_authority = install
    monkeypatch.setitem(sys.modules, module.__name__, module)

    run_bootstrap(monkeypatch, "posix-session-v1")

    install.assert_called_once_with()


def test_unknown_marked_mode_fails_closed(monkeypatch):
    with pytest.raises(RuntimeError, match="unsupported Desktop process authority mode"):
        run_bootstrap(monkeypatch, "pid-v1")


def test_bootstrap_does_not_mutate_mode(monkeypatch):
    run_bootstrap(monkeypatch, None)
    assert os.environ.get(AUTHORITY_ENV) is None


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX bootstrap topology")
def test_scoped_main_shim_arms_authority_before_real_package_init(
    monkeypatch,
    tmp_path,
):
    import json
    import shutil
    import subprocess

    project_root = Path(__file__).resolve().parents[2]
    bootstrap_root = project_root / "hermes_cli" / "desktop_bootstrap"
    real_package = tmp_path / "real" / "hermes_cli"
    real_package.mkdir(parents=True)
    isolated_cwd = tmp_path / "cwd"
    isolated_cwd.mkdir()
    marker = tmp_path / "real-init-pids.txt"
    (real_package / "__init__.py").write_text(
        "import os\n"
        "from pathlib import Path\n"
        "Path(os.environ['BOOTSTRAP_INIT_MARKER']).open('a').write(str(os.getpid()) + '\\n')\n",
        encoding="utf-8",
    )
    (real_package / "main.py").write_text(
        "import json, os\n"
        "print(json.dumps({'backend': os.getpid(), 'pgid': os.getpgrp()}), flush=True)\n",
        encoding="utf-8",
    )
    for name in (
        "posix_process_authority.py",
        "_posix_process_authority_state.py",
        "_posix_process_transfer.py",
        "_posix_process_guard.py",
        "_subprocess_compat.py",
    ):
        shutil.copy2(project_root / "hermes_cli" / name, real_package / name)

    env = os.environ.copy()
    env.update(
        {
            AUTHORITY_ENV: "posix-session-v1",
            "HERMES_DESKTOP_PROCESS_GENERATION": "generation-bootstrap-shim-01",
            "HERMES_DESKTOP_PARENT_PID": str(os.getpid()),
            "HERMES_DESKTOP_PARENT_STARTED_AT_MS": "1700000000000",
            "BOOTSTRAP_INIT_MARKER": str(marker),
            "PYTHONPATH": os.pathsep.join(
                [bootstrap_root.as_posix(), real_package.parent.as_posix()]
            ),
        }
    )
    proc = subprocess.Popen(
        [sys.executable, "-m", "hermes_cli.main"],
        cwd=isolated_cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = proc.communicate(timeout=10)

    assert proc.returncode == 0, stderr
    payload = json.loads(stdout.strip())
    assert payload["backend"] != proc.pid
    initialized_pids = [
        int(value)
        for value in marker.read_text(encoding="utf-8").splitlines()
    ]
    assert initialized_pids == [payload["backend"]], (
        "the retained supervisor must not import the real Hermes package"
    )
