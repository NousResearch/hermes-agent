"""The worker client must import Hermes, not a ``hermes_cli`` package sitting in the task workspace.

The dispatcher starts the client with the task workspace as its cwd. ``python -m`` puts the cwd
first on ``sys.path``, so a card whose workspace is a hermes-agent checkout (a rebase of a PR
branch, for one) imported that checkout's ``hermes_cli``. An older checkout has no
``kanban_worker_client`` and the worker died on start with
``No module named hermes_cli.kanban_worker_client``.
"""

import subprocess
import sys
from pathlib import Path

from hermes_cli import kanban_db_dispatch as dispatch

REPO_ROOT = Path(__file__).resolve().parents[2]


def _argv():
    return dispatch._worker_argv(None, "worker", None)  # type: ignore[arg-type]  # task is unused


def test_worker_argv_keeps_the_cwd_off_sys_path():
    argv = _argv()
    assert argv[0] == sys.executable
    assert "-P" in argv[1:argv.index("-m")]


def test_worker_client_resolves_to_hermes_even_when_the_cwd_holds_another_hermes_cli(tmp_path):
    shadow = tmp_path / "hermes_cli"
    shadow.mkdir()
    (shadow / "__init__.py").write_text("")  # an old checkout: the package, without the client
    argv = _argv()
    probe = [*argv[:argv.index("-m")], "-c",
             "import importlib.util as u; print(u.find_spec('hermes_cli.kanban_worker_client').origin)"]
    env = {"PYTHONPATH": str(REPO_ROOT), "PATH": "/usr/bin:/bin"}
    out = subprocess.run(probe, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60, check=False)
    assert out.returncode == 0, out.stderr
    assert Path(out.stdout.strip()).resolve() == (REPO_ROOT / "hermes_cli" / "kanban_worker_client.py").resolve()
