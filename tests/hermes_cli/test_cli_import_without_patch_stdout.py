"""``cli`` must import even when ``prompt_toolkit.patch_stdout`` is missing (#96075).

The top-level ``from prompt_toolkit.patch_stdout import patch_stdout`` was unguarded,
so a partial/broken prompt_toolkit install (e.g. a stale venv that has the package but
not every submodule) crashed the import chain — including ``tui_gateway.slash_worker``,
which imports ``cli`` in every gateway session. The only use site is a
``with patch_stdout():`` block, so a ``nullcontext`` fallback preserves behaviour.

Modeled on tests/tui_gateway/test_slash_worker_sys_path.py's subprocess pattern.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Import blocker: a sys.meta_path finder that raises ImportError only for
# prompt_toolkit.patch_stdout — the "package present, submodule broken" case.
_BLOCKER = """
import sys

class _BlockPatchStdout:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "prompt_toolkit.patch_stdout":
            raise ImportError("simulated broken prompt_toolkit.patch_stdout")
        return None

sys.meta_path.insert(0, _BlockPatchStdout())
"""


def _import_cli_without_patch_stdout():
    code = _BLOCKER + "import cli\n"
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(PROJECT_ROOT), "PATH": "", "HOME": str(Path.home())},
        cwd=str(PROJECT_ROOT),
        check=False,
    )


def test_cli_imports_without_patch_stdout():
    proc = _import_cli_without_patch_stdout()
    assert proc.returncode == 0, proc.stderr


def test_cli_patch_stdout_falls_back_to_nullcontext():
    code = _BLOCKER + (
        "import cli, contextlib\n"
        "assert cli.patch_stdout is contextlib.nullcontext, cli.patch_stdout\n"
        "assert callable(cli.HermesCLI)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(PROJECT_ROOT), "PATH": "", "HOME": str(Path.home())},
        cwd=str(PROJECT_ROOT),
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
