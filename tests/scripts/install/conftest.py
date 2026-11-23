"""A ready pinned-Git store for the install.ps1 stage tests.

install.ps1 keeps its tool store under HERMES_RUNTIME_DIR, else under HERMES_HOME. run_tests.sh
never forwards HERMES_RUNTIME_DIR, so a stage started with a temporary HERMES_HOME found no Git
and downloaded and extracted PortableGit from GitHub, once per home. Tests whose subject is not
that download point HERMES_RUNTIME_DIR here: install.ps1 then finds a ready Git and writes nothing.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[3]
INSTALLER = ROOT / "scripts" / "install.ps1"


def _ready_store_on_path() -> Path | None:
    """The store holding the pinned Git first on PATH (setup-pm stages one on the Windows lanes)."""
    version = json.loads((ROOT / "pm" / "lock.json").read_text(encoding="utf-8"))["packages"]["git"]["version"]
    git = shutil.which("git")
    if not git:
        return None
    entry = Path(git).resolve().parents[1]
    if not entry.name.startswith(f"git-{version}-win32-"):
        return None
    exec_path = subprocess.run([git, "--exec-path"], capture_output=True, text=True, check=False).stdout.strip()
    return entry.parent if exec_path and (Path(exec_path) / "git-remote-https.exe").is_file() else None


@pytest.fixture(scope="session")
def pinned_git_store(tmp_path_factory) -> Path:
    store = _ready_store_on_path()
    if store is not None:
        return store
    store = tmp_path_factory.mktemp("tools")
    powershell = shutil.which("powershell")
    assert powershell, "Windows PowerShell 5.1 is part of every supported Windows"
    script = f". '{INSTALLER}'; if (-not (Get-PinnedGit)) {{ exit 9 }}"
    subprocess.run([powershell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
                   env=dict(os.environ, HERMES_RUNTIME_DIR=str(store)), check=True, capture_output=True, timeout=900)
    return store
