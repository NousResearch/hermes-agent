"""The root ``hermes_bootstrap.py`` repairs a ZIP update killed with the package moved aside.

Every launcher's first Hermes import is ``import hermes_bootstrap``. A ZIP swap killed after it
renamed ``hermes_agent/`` (``hermes_cli/`` in the pre-move layout) to ``*.hermes-update-old`` leaves
no importable package, so the root file must run the moved-aside copy's ``_early_recovery`` (the
code that wrote the journal) before it imports ``hermes_agent``; a pre-move tree runs its live
``hermes_cli/`` copy. The stub repair imports its own
``update_lock`` through the package, as the real one does, so the stand-in packages are covered.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_agent.hermes_cli._startup_fast import project_root_str

_STUB = """\
from {package} import update_lock

def restore_interrupted_pull(root):
    print("restore", root, update_lock.ASIDE, flush=True)
    return True

def relaunch_after_restore():
    print("relaunch", flush=True)
    raise SystemExit(0)
"""


def _install(tmp_path: Path, aside: str, package: str, *, journal: bool) -> Path:
    root = tmp_path / "install"
    recovery = root / aside
    recovery.mkdir(parents=True)
    shutil.copyfile(Path(project_root_str()) / "hermes_bootstrap.py", root / "hermes_bootstrap.py")
    (recovery / "_early_recovery.py").write_text(_STUB.format(package=package), encoding="utf-8")
    (recovery / "update_lock.py").write_text("ASIDE = 'aside'\n", encoding="utf-8")
    if journal:
        (root / ".hermes-update-zip-swap").write_text("{}", encoding="utf-8")
    return root


def _launch(root: Path) -> subprocess.CompletedProcess:
    code = f"import sys; sys.path.insert(0, {str(root)!r}); import hermes_bootstrap"
    return subprocess.run([sys.executable, "-I", "-S", "-c", code], capture_output=True, text=True,
                          timeout=60, check=False)


@pytest.mark.parametrize(("aside", "package"), [
    ("hermes_agent.hermes-update-old/hermes_cli", "hermes_agent.hermes_cli"),
    ("hermes_cli.hermes-update-old", "hermes_cli"),
    ("hermes_cli", "hermes_cli"),  # a pre-move tree: its own hermes_cli/ wrote the journal
])
def test_launch_runs_the_moved_aside_repair_before_importing_the_package(tmp_path, aside, package):
    root = _install(tmp_path, aside, package, journal=True)

    done = _launch(root)

    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines() == [f"restore {root.resolve()} aside", "relaunch"]


def test_launch_without_a_swap_journal_leaves_the_moved_aside_copy_alone(tmp_path):
    root = _install(tmp_path, "hermes_agent.hermes-update-old/hermes_cli", "hermes_agent.hermes_cli",
                    journal=False)

    done = _launch(root)

    assert "restore" not in done.stdout
    assert done.returncode != 0 and "No module named 'hermes_agent'" in done.stderr
