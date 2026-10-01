"""The restart watcher's liveness imports require only the Python stdlib.

Use a real isolated child interpreter: the test runner's loaded dependencies
must not conceal third-party imports in runtime or the frozen updater facade.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("surface", ["runtime", "frozen-updater"])
def test_process_liveness_imports_on_bare_python(tmp_path: Path, surface: str) -> None:
    script = """
import os
import sys

sys.path.insert(0, sys.argv[1])
if sys.argv[2] == 'runtime':
    from runtime.process_identity import pid_exists_stdlib
else:
    from hermes_cli._subprocess_compat import pid_exists_stdlib

assert pid_exists_stdlib(os.getpid())
assert 'utils' not in sys.modules, 'bare process liveness eagerly loaded application dependencies'
print('stdlib liveness import and probe succeeded')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", script, str(ROOT), surface],
        cwd=tmp_path, capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "stdlib liveness import and probe succeeded" in result.stdout
