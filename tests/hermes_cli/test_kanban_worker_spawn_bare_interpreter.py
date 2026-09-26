"""The kanban worker's spawn argv must import this install's checkout from the
WORKER's cwd, not the dispatcher's.

Regression for #122299. ``_default_spawn`` launches every worker with
``cwd=<per-task workspace>``, and the old argv — a bare
``[sys.executable, "-m", "hermes_cli.main"]`` — only imported because ``python -m``
puts the CURRENT WORKING DIRECTORY on ``sys.path[0]``. From a workspace cwd that
entry is not the checkout, so every worker died with ``ModuleNotFoundError: No
module named 'hermes_cli'`` before running a turn, burned ``failure_limit`` and was
auto-blocked. The argv is now this install's own interpreter-bound entry form
(``hermes_cli._launchers.runtime_command``), whose ``-I`` bootstrap inserts the
checkout on ``sys.path`` in-process.

**Why these tests synthesize a bare interpreter.** The defect only reproduces on an
interpreter that cannot import ``hermes_cli`` unaided — the PM store Python
(``hermes_cli/_launchers.resolve_store_python``; ABI only, no editable install, no
application packages). The pytest interpreter and the repo's legacy ``venv`` both
carry an editable install of this checkout, so ``sys.executable -m hermes_cli.main``
succeeds from ANY cwd and the bug is invisible under them. That is exactly why the
pre-existing ``tests/hermes_cli/test_kanban_db.py::test_resolve_hermes_argv_module_actually_runs``
— which claims to catch this regression — stayed green against the broken code: it
runs the resolver's argv while inheriting the pytest process's cwd, which is the
checkout root.

These tests therefore start a real child from a temp cwd on a synthesized
no-editable-install interpreter, using the REAL ``_worker_argv`` tail (only ``argv[0]``
is substituted, mirroring ``runtime_command(python=...)`` in production). The pre-fix
argv form is spawned alongside as a negative control that must FAIL: if the environment
ever stops being bare, the control fails loudly instead of letting the suite go quietly
green. Assertions are on subprocess BEHAVIOUR — a real import from a foreign cwd —
never on the argv's source text, which would only re-state the implementation.
"""

from __future__ import annotations

import os
import subprocess
import sys
import venv
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The pre-fix argv shape, kept only as a control: it must still fail from a foreign
#: cwd, which is how these tests prove they are running against a bare interpreter.
_BASE_ARGV_TAIL = ["-m", "hermes_cli.main"]

#: How a failed import of this checkout surfaces on stderr (both spellings: the ``-m``
#: finder's and a plain ``import``'s).
_IMPORT_FAILURE_MARKERS = (
    "No module named 'hermes_cli'",
    "Error while finding module specification",
)

_SPAWN_TIMEOUT = 300


@pytest.fixture(scope="module")
def bare_interpreter(tmp_path_factory) -> str:
    """A freshly-created interpreter that cannot import ``hermes_cli`` unaided.

    ``venv --without-pip`` needs no network and installs nothing, so the child is bare
    by construction — the same condition as the PM store Python at spawn time.
    """
    target = tmp_path_factory.mktemp("bare-interpreter") / "env"
    venv.EnvBuilder(with_pip=False, symlinks=False).create(target)
    interpreter = target / ("Scripts/python.exe" if os.name == "nt" else "bin/python3")
    assert interpreter.is_file(), f"could not synthesize a bare interpreter at {interpreter}"
    return str(interpreter)


def _child_env(**extra: str) -> dict:
    """A worker-like env: no ``PYTHONPATH`` (the sanitizer strips Hermes-owned entries)."""
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    env.update(extra)
    return env


def _spawn(argv: list[str], cwd: Path, **env_extra: str):
    """Run *argv* + ``--version`` as a real child whose cwd is NOT the checkout."""
    return subprocess.run(
        [*argv, "--version"],
        cwd=str(cwd),
        env=_child_env(**env_extra),
        capture_output=True,
        text=True,
        timeout=_SPAWN_TIMEOUT,
    )


def _worker_argv_on(interpreter: str) -> list[str]:
    """The real ``_worker_argv`` tail, run by *interpreter*.

    ``-p default`` is a profile that always exists, so the CLI proceeds past argument
    parsing to whatever command it was handed; the contract under test is that the
    child gets that far instead of dying on the import.
    """
    from hermes_cli import kanban_db_dispatch as kbd

    task = SimpleNamespace(
        id="t_probe", skills=None, model_override=None, provider_override=None,
        reasoning_effort=None,
    )
    previous = os.environ.pop("HERMES_BIN", None)
    try:
        argv = kbd._worker_argv(task, "default", None)
    finally:
        if previous is not None:
            os.environ["HERMES_BIN"] = previous
    assert argv[0] != str(interpreter)
    return [interpreter, *argv[1:]]


def test_worker_spawn_imports_the_checkout_from_a_foreign_workspace_cwd(
    bare_interpreter, tmp_path,
):
    """A real worker child started in a task-workspace cwd imports the checkout.

    The base argv form is spawned first as the negative control: on the base revision
    it exits non-zero with ``ModuleNotFoundError: No module named 'hermes_cli'``, which
    is the live failure the dispatcher records as a crashed run.

    The positive assertion is on the ABSENCE of an import failure, not on exit code 0:
    with the import fixed the child reaches the CLI's own argument validation, whose
    exit code is a property of the flags handed to it, not of the spawn path. Any
    rc-0 assertion here would be testing ``-q``'s semantics, not this contract.
    """
    workspace = tmp_path / "task-workspace"
    workspace.mkdir()

    control = _spawn([bare_interpreter, *_BASE_ARGV_TAIL], workspace)
    assert control.returncode != 0, (
        "negative control passed — this interpreter is not bare, so the test would not "
        f"detect the regression. rc={control.returncode} stdout={control.stdout[:200]!r}"
    )
    assert any(marker in control.stderr for marker in _IMPORT_FAILURE_MARKERS), (
        f"negative control did not fail on the import: {control.stderr[:400]!r}"
    )

    result = _spawn(_worker_argv_on(bare_interpreter), workspace)

    for marker in _IMPORT_FAILURE_MARKERS:
        assert marker not in result.stderr, (
            f"worker spawn could not import the checkout from cwd={workspace}: "
            f"rc={result.returncode}\nstderr={result.stderr[-800:]!r}"
        )
    assert result.stderr.strip(), "child produced no stderr at all — it never started"


def test_worker_spawn_survives_pythonsafepath_and_a_stripped_pythonpath(
    bare_interpreter, tmp_path,
):
    """The import must not depend on cwd, ``PYTHONPATH``, or ``PYTHONSAFEPATH``.

    ``PYTHONSAFEPATH=1`` removes the implicit ``-m``/script-dir entry the old form
    relied on, and the child env carries no ``PYTHONPATH`` — both real host variations
    named in #122299, and the multi-profile/routed case where the subprocess-env
    factory strips Hermes-owned PYTHONPATH entries.
    """
    workspace = tmp_path / "stripped-workspace"
    workspace.mkdir()

    result = _spawn(_worker_argv_on(bare_interpreter), workspace, PYTHONSAFEPATH="1")

    for marker in _IMPORT_FAILURE_MARKERS:
        assert marker not in result.stderr, (
            f"worker spawn could not import the checkout under PYTHONSAFEPATH=1 from "
            f"cwd={workspace}: rc={result.returncode}\nstderr={result.stderr[-800:]!r}"
        )


def test_worker_spawn_keeps_the_install_ahead_of_a_path_planted_hermes(tmp_path):
    """#111569 precedence survives: a PATH-planted ``hermes`` never becomes the worker.

    The fix changes the argv *shape*, not the module-beats-PATH rule, so the resolver
    must still return this install's own interpreter rather than the PATH hit, and that
    argv must still import the CLI from a foreign cwd.
    """
    import shutil

    from hermes_cli import kanban_db_dispatch as kbd

    planted = tmp_path / "planted"
    planted.mkdir()
    previous_bin = os.environ.pop("HERMES_BIN", None)
    previous_which, previous_safe = shutil.which, kbd._safe_which_no_cwd
    shutil.which = lambda name, *a, **k: str(planted / "hermes")
    kbd._safe_which_no_cwd = lambda name: str(planted / "hermes")
    try:
        argv = kbd._resolve_hermes_argv()
    finally:
        shutil.which, kbd._safe_which_no_cwd = previous_which, previous_safe
        if previous_bin is not None:
            os.environ["HERMES_BIN"] = previous_bin

    assert argv[0] != str(planted / "hermes")
    assert Path(argv[0]).name.lower().startswith("python")
    assert Path(argv[0]).is_file(), argv[0]
