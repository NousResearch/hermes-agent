"""Regression tests for the kanban worker spawn argv (-P flag, #cwd-shadow).

A card whose workspace is a Hermes source checkout used to kill the worker
before turn 1: ``python -m hermes_cli.kanban_worker_client`` prepends the
process cwd to ``sys.path[0]``, so the checkout's ``hermes_bootstrap.py``
shadowed the pinned live install on PYTHONPATH.  The checkout's bootstrap
resolved ``_root = Path(__file__).parent`` to the checkout, whose install
state has no committed dependency environment (no ``installs/<hash>`` dir),
and ``pm/environments.py::_require_own_dependencies`` raised::

    hermes: no dependency environment is committed for this install;
    run `hermes pm repair`

before the worker reached turn 1.  Fix: add ``-P`` to the spawn argv so the
interpreter never prepends cwd to sys.path, while PYTHONPATH (pointing at
the live install) is still honoured.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_task(**kwargs) -> SimpleNamespace:
    """Minimal Task-like object accepted by _worker_argv."""
    defaults = dict(
        id="t_test0001",
        assignee="test-worker",
        status="running",
        tenant=None,
        workspace_kind="scratch",
        workspace_path=None,
        branch_name=None,
        current_run_id=1,
        claim_lock=None,
        goal_mode=False,
        goal_max_turns=None,
        max_runtime_seconds=None,
        model_override=None,
        provider_override=None,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# 1. Argv shape
# ---------------------------------------------------------------------------

def test_worker_argv_contains_P_flag():
    """_worker_argv must include -P immediately after sys.executable so the
    worker interpreter never prepends its cwd to sys.path[0]."""
    from hermes_cli.kanban_db_dispatch import _worker_argv

    task = _make_task()
    argv = _worker_argv(task, profile_arg="test-worker", hermes_home=None)  # type: ignore[arg-type]

    assert isinstance(argv, list), "argv must be a list"
    assert argv[0] == sys.executable, f"argv[0] must be sys.executable; got {argv[0]!r}"
    assert argv[1] == "-P", (
        f"argv[1] must be '-P' to prevent cwd from shadowing PYTHONPATH; "
        f"got {argv[1]!r}. Full argv: {argv}"
    )
    assert "-m" in argv, "argv must include '-m' module flag"
    assert "hermes_cli.kanban_worker_client" in argv, (
        "argv must launch hermes_cli.kanban_worker_client"
    )


def test_worker_argv_P_precedes_m_flag():
    """-P must come before -m; Python only honours -P when it precedes the
    module flag (the flag must be a top-level interpreter flag, not a module
    argument)."""
    from hermes_cli.kanban_db_dispatch import _worker_argv

    task = _make_task()
    argv = _worker_argv(task, profile_arg="test-worker", hermes_home=None)  # type: ignore[arg-type]

    p_idx = argv.index("-P")
    m_idx = argv.index("-m")
    assert p_idx < m_idx, (
        f"-P (index {p_idx}) must precede -m (index {m_idx}) in argv {argv}"
    )


def test_worker_argv_no_duplicate_P():
    """Ensure -P appears exactly once to keep the argv clean."""
    from hermes_cli.kanban_db_dispatch import _worker_argv

    task = _make_task()
    argv = _worker_argv(task, profile_arg="test-worker", hermes_home=None)  # type: ignore[arg-type]

    assert argv.count("-P") == 1, (
        f"Expected exactly one '-P' in argv; got {argv.count('-P')} in {argv}"
    )


# ---------------------------------------------------------------------------
# 2. Boot probe: cwd=checkout shadow behaviour
#
# We demonstrate the actual sys.path insertion behaviour using subprocess so
# the test is hermetic (it does not depend on the live install's boot state,
# only on the Python interpreter's flag semantics).
#
# Strategy: create a temporary directory that acts as a "checkout" by placing
# a probe module inside it.  With cwd=that-dir and no -P, the probe module
# is importable via cwd insertion.  With -P, it is not (cwd is never added),
# so only modules on PYTHONPATH are found.
# ---------------------------------------------------------------------------

def _run_import_probe(
    *,
    use_p_flag: bool,
    cwd: str,
    extra_pythonpath: str | None = None,
    module_name: str = "cwd_shadow_probe",
) -> subprocess.CompletedProcess:
    """Run ``python [-P] -c "import <module_name>"`` with cwd=cwd.

    Returns the CompletedProcess so callers can check returncode and stderr.
    """
    env = os.environ.copy()
    if extra_pythonpath:
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            extra_pythonpath if not existing else f"{extra_pythonpath}{os.pathsep}{existing}"
        )
    cmd = [sys.executable]
    if use_p_flag:
        cmd.append("-P")
    cmd += ["-c", f"import {module_name}; print('imported ok')"]
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )


def test_cwd_shadow_without_P_flag(tmp_path):
    """NEGATIVE CONTROL: without -P the interpreter inserts cwd into
    sys.path[0], so a module that exists only in cwd is importable even when
    it is absent from PYTHONPATH.

    This is the bad behaviour: if cwd = a Hermes checkout, its
    ``hermes_bootstrap.py`` wins over the PYTHONPATH-pinned live install.
    """
    # Place the probe module only inside the temp dir (the fake "checkout").
    probe = tmp_path / "cwd_shadow_probe.py"
    probe.write_text("# probe found via cwd insertion\n")

    result = _run_import_probe(
        use_p_flag=False,
        cwd=str(tmp_path),
        # PYTHONPATH does NOT include tmp_path; the import can only succeed via cwd.
        extra_pythonpath=None,
    )
    assert result.returncode == 0, (
        f"Without -P, cwd module must be importable (cwd is in sys.path).\n"
        f"stdout: {result.stdout!r}\nstderr: {result.stderr!r}"
    )
    assert "imported ok" in result.stdout


def test_cwd_shadow_blocked_by_P_flag(tmp_path):
    """POSITIVE CONTROL (the fix): with -P the interpreter does NOT insert
    cwd into sys.path, so a module that exists only in cwd is NOT importable
    unless it is also on PYTHONPATH.

    This is the corrected behaviour: a Hermes checkout in cwd cannot shadow
    the PYTHONPATH-pinned live install.
    """
    probe = tmp_path / "cwd_shadow_probe.py"
    probe.write_text("# probe found via cwd insertion — must NOT happen with -P\n")

    result = _run_import_probe(
        use_p_flag=True,
        cwd=str(tmp_path),
        # PYTHONPATH does NOT include tmp_path.
        extra_pythonpath=None,
    )
    assert result.returncode != 0, (
        f"With -P, cwd-only module must NOT be importable (cwd is excluded from sys.path).\n"
        f"stdout: {result.stdout!r}\nstderr: {result.stderr!r}"
    )
    assert "imported ok" not in result.stdout


def test_P_flag_still_honours_PYTHONPATH(tmp_path):
    """With -P, PYTHONPATH is still searched.  A module on PYTHONPATH must
    still be importable from a cwd that does NOT contain it, ensuring the
    live install's tree (pinned on PYTHONPATH by the dispatcher) loads
    correctly regardless of the worker's cwd."""
    # Place the probe on PYTHONPATH only (not in cwd).
    pythonpath_dir = tmp_path / "pinned_install"
    pythonpath_dir.mkdir()
    probe = pythonpath_dir / "cwd_shadow_probe.py"
    probe.write_text("# probe on PYTHONPATH — must import with -P\n")

    # cwd is a different directory that does NOT contain the probe.
    other_cwd = tmp_path / "worker_workspace"
    other_cwd.mkdir()

    result = _run_import_probe(
        use_p_flag=True,
        cwd=str(other_cwd),
        extra_pythonpath=str(pythonpath_dir),
    )
    assert result.returncode == 0, (
        f"With -P, PYTHONPATH-pinned module must still import.\n"
        f"stdout: {result.stdout!r}\nstderr: {result.stderr!r}"
    )
    assert "imported ok" in result.stdout


# ---------------------------------------------------------------------------
# 3. Full spawn-shape integration probe
#
# Verify that _default_spawn builds an argv containing -P even when called
# through the normal dispatch path, without actually launching a child process.
# ---------------------------------------------------------------------------

def test_default_spawn_argv_shape_in_dispatch(monkeypatch, tmp_path):
    """When dispatch_once selects a ready task and calls _default_spawn, the
    captured child argv must include -P.  The spawn function is replaced with
    a spy so no child process is launched."""
    # Isolate to a fresh HERMES_HOME so we don't touch the live board.
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(hermes_home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    # Clear cached module state so the new HERMES_HOME is picked up.
    for mod in list(sys.modules.keys()):
        if mod.startswith("hermes_cli.kanban") or mod == "hermes_cli.kanban_db":
            sys.modules.pop(mod, None)

    import hermes_cli.kanban_db as kb
    import hermes_cli.kanban_db_connect as kbc
    import hermes_cli.kanban_db_dispatch as kbd

    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()

    captured_argv: list[list[str]] = []

    def spy_spawn(task, workspace_path, board=None):
        from hermes_cli.profiles import normalize_profile_name
        profile_arg = normalize_profile_name(task.assignee)
        argv = kbd._worker_argv(task, profile_arg, None)  # type: ignore[arg-type]
        captured_argv.append(argv)
        return 999999  # fake pid

    with kbc.connect() as conn:
        # create_task with no parents creates a 'ready' task (initial_task_state
        # returns 'ready' when parents={} and initial_status='running').
        kb.create_task(conn, title="test-cwd-fix", assignee="default")
        result = kbd.dispatch_once(conn, spawn_fn=spy_spawn)

    assert len(captured_argv) >= 1, (
        f"Expected at least one spawn call; got none. dispatch result: {result}"
    )
    argv = captured_argv[0]
    assert "-P" in argv, (
        f"Worker argv from dispatch path must include -P; got {argv}"
    )
    p_idx = argv.index("-P")
    m_idx = argv.index("-m")
    assert p_idx < m_idx, (
        f"-P (index {p_idx}) must precede -m (index {m_idx}) in dispatch argv {argv}"
    )
