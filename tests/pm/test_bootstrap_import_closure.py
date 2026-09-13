"""The tool-only build stage imports PM without third-party dependencies."""

import os
from pathlib import Path
import shutil
import subprocess
import sys


def test_minimal_bootstrap_closure_reaches_pm_paths_and_locks(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    stage = tmp_path / "stage"
    shutil.copytree(repo / "pm", stage / "pm", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(repo / "hermes_constants.py", stage / "hermes_constants.py")
    (stage / "hermes_cli").mkdir()
    for name in ("__init__.py", "runtime_paths.py", "runtime_state.py"):
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
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    assert Path(result.stdout.strip()) == store
