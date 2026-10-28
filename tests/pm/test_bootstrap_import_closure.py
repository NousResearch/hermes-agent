"""The tool-only build stage imports PM without third-party dependencies."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


def test_minimal_bootstrap_closure_reaches_pm_paths_and_locks(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    stage = tmp_path / "stage"
    shutil.copytree(repo / "pm", stage / "pm", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(repo / "hermes_constants.py", stage / "hermes_constants.py")
    (stage / "hermes_cli").mkdir()
    for name in ("__init__.py", "runtime_state.py"):
        shutil.copy2(repo / "hermes_cli" / name, stage / "hermes_cli" / name)
    store = stage / "tools"
    env = dict(os.environ, HERMES_HOME=str(tmp_path / "home"),
               HERMES_RUNTIME_DIR=str(store), PYTHONPATH=str(stage))
    script = """
from pathlib import Path
import pm.paths
from pm.store import Store
from pm.lock import Facts
root = pm.paths.store_root()
with Store(root).install_lock():
    facts = Facts(root / 'facts.json')
    facts.record_state('probe', 'checked', [])
assert facts.path.is_file()
print(root)
"""
    result = subprocess.run([sys.executable, "-S", "-c", script], cwd=stage, env=env,
                            capture_output=True, text=True, timeout=15, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert Path(result.stdout.strip()) == store



def _constants_stage(tmp_path, *, with_acl: bool) -> Path:
    repo = Path(__file__).resolve().parents[2]
    stage = tmp_path / "stage"
    stage.mkdir()
    names = ["hermes_constants.py", "hermes_dir_acl.py"] if with_acl else ["hermes_constants.py"]
    for name in names:
        shutil.copy2(repo / name, stage / name)
    return stage


def _mkdir_in_stage(stage: Path, home: Path) -> subprocess.CompletedProcess:
    script = "import hermes_constants, sys; print(hermes_constants.mkdir_under_hermes_home(sys.argv[1]))"
    return subprocess.run([sys.executable, "-S", "-c", script, str(home / "logs")], cwd=stage,
                          env=dict(os.environ, HERMES_HOME=str(home), PYTHONPATH=str(stage)),
                          capture_output=True, text=True, timeout=15, check=False)


def test_home_mkdir_carries_the_acl_module_in_the_closure(tmp_path):
    """hermes_dir_acl is stdlib-only and ships beside hermes_constants: the two together create a
    home dir, and hermes_constants still imports alone (the import is lazy)."""
    home = tmp_path / "home"
    home.mkdir()
    result = _mkdir_in_stage(_constants_stage(tmp_path, with_acl=True), home)
    assert result.returncode == 0, result.stderr
    assert (home / "logs").is_dir()


def test_home_mkdir_never_silently_skips_the_elevated_grant(tmp_path):
    """Review #140490: an ImportError fallback to plain mkdir made whether an elevated run grants
    the user access depend on which files happened to be present. A missing module is loud."""
    home = tmp_path / "home"
    home.mkdir()
    stage = _constants_stage(tmp_path, with_acl=False)
    alone = subprocess.run([sys.executable, "-S", "-c", "import hermes_constants"], cwd=stage,
                           env=dict(os.environ, PYTHONPATH=str(stage)), capture_output=True, text=True,
                           timeout=15, check=False)
    assert alone.returncode == 0, alone.stderr
    result = _mkdir_in_stage(stage, home)
    assert result.returncode != 0 and "hermes_dir_acl" in result.stderr, result.stdout + result.stderr


def test_managed_python_signing_import_needs_no_pm_or_application(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    script = """
import sys
class NoApplication:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'pm' or fullname == 'utils' or fullname.startswith(('pm.', 'agent.')):
            raise AssertionError('signing imported ' + fullname)
sys.meta_path.insert(0, NoApplication())
from hermes_cli.macos_signing import sign_managed_python
assert callable(sign_managed_python)
"""
    result = subprocess.run([sys.executable, "-S", "-c", script], cwd=repo,
                            env=dict(os.environ, HERMES_HOME=str(tmp_path / "home")),
                            capture_output=True, text=True, timeout=15, check=False)
    assert result.returncode == 0, result.stderr


@pytest.mark.platforms("linux", "macos", "windows")
def test_runtime_staging_streams_uv_output_without_tomllib(tmp_path):
    """Bootstrap runs before PM selects its Python: Docker has 3.10 and
    historical Windows updaters have 3.11, without os.set_blocking for pipes.
    """
    repo = Path(__file__).resolve().parents[2]
    script = """
import os
import sys
if sys.platform == 'win32' and hasattr(os, 'set_blocking'):
    del os.set_blocking
class NoTomllib:
    def find_spec(self, fullname, path=None, target=None):
        if fullname in ('tomllib', 'pm.workspace', 'pm.plugin_declarations'):
            raise AssertionError('runtime staging imported ' + fullname)
sys.meta_path.insert(0, NoTomllib())
import pm.runtime_stage
from pm.environment import _run_streaming
import subprocess
result = _run_streaming([sys.executable, '-c', 'print(\"no solution found\")'],
                        cwd='.', env={}, timeout=30, output=sys.stderr)
assert result.returncode == 0, result
"""
    result = subprocess.run([sys.executable, "-S", "-c", script], cwd=repo,
                            env=dict(os.environ, HERMES_HOME=str(tmp_path / "home")),
                            capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
