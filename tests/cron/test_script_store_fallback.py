"""cron script resolution must find the OWNING profile's scripts.

A cron job's scripts live beside its cron store (per-profile stores, #4707). Every legitimate
execution context pins the owning store — multiplex tick scope, external worker, routed turns —
so ``_resolve_script_path`` searches the store's scripts dir FIRST and falls back to the
executing process's own ``HERMES_HOME``. Before this fix a profile job executed from a process
whose ambient home differs (an operator's bare probe, the launch home, any pre-override path)
resolved ONLY that process's scripts dir and failed with "Script not found" while the hourly
ticks — which do pin the store — looked healthy. These tests pin the mismatch on purpose:
process home A, owning store home B.
"""

from __future__ import annotations

import importlib
import os

import pytest


@pytest.fixture
def homes(tmp_path, monkeypatch):
    """Process home A (HERMES_HOME) and owning profile home B; cron.jobs re-pinned to A."""
    home_a = tmp_path / "process-home"
    home_b = tmp_path / "profiles" / "coder"
    for home in (home_a, home_b):
        (home / "scripts").mkdir(parents=True)
        (home / "cron").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home_a))
    import hermes_constants
    importlib.reload(hermes_constants)
    import cron.jobs
    importlib.reload(cron.jobs)
    return home_a, home_b


def _pin_store(home_b):
    from cron.jobs import use_cron_store

    return use_cron_store(home_b)


# ---------------------------------------------------------------------------
# cron.scheduler_script._resolve_script_path — the runtime resolver
# ---------------------------------------------------------------------------


def test_relative_script_resolves_in_owning_store(homes):
    _home_a, home_b = homes
    script = home_b / "scripts" / "guard.py"
    script.write_text("print('profile copy')\n", encoding="utf-8")

    from cron.scheduler_script import _resolve_script_path

    with _pin_store(home_b):
        path, error = _resolve_script_path("guard.py")
    assert error is None
    assert path is not None
    assert path.parent == (home_b / "scripts").resolve()


def test_relative_script_without_store_pin_stays_process_home_only(homes):
    """No store pinned (bare probe, root-owned job): the process home is the only candidate —
    a profile-owned script correctly stays unfound, with an error naming what was searched."""
    home_a, home_b = homes
    (home_b / "scripts" / "guard.py").write_text("print('x')\n", encoding="utf-8")

    from cron.scheduler_script import _resolve_script_path

    path, error = _resolve_script_path("guard.py")
    assert path is None
    assert error is not None
    assert "Script not found" in error
    assert str((home_a / "scripts").resolve()) in error


def test_owning_store_takes_precedence_over_process_home(homes):
    """Same name in both homes: the owning profile's copy wins — it is the file that job's
    ticks run (store-first), so a probe must resolve the same file."""
    home_a, home_b = homes
    (home_a / "scripts" / "guard.py").write_text("print('process copy')\n", encoding="utf-8")
    (home_b / "scripts" / "guard.py").write_text("print('profile copy')\n", encoding="utf-8")

    from cron.scheduler_script import _resolve_script_path

    with _pin_store(home_b):
        path, error = _resolve_script_path("guard.py")
    assert error is None
    assert path == (home_b / "scripts" / "guard.py").resolve()


def test_process_home_is_the_fallback_candidate(homes):
    """Script exists only in the executing process's home (launch-home jobs): still found."""
    home_a, home_b = homes
    (home_a / "scripts" / "local.py").write_text("print('x')\n", encoding="utf-8")

    from cron.scheduler_script import _resolve_script_path

    with _pin_store(home_b):
        path, error = _resolve_script_path("local.py")
    assert error is None
    assert path == (home_a / "scripts" / "local.py").resolve()


def test_absolute_script_inside_owning_store_scripts_dir(homes):
    _home_a, home_b = homes
    script = home_b / "scripts" / "abs.py"
    script.write_text("print('x')\n", encoding="utf-8")

    from cron.scheduler_script import _resolve_script_path

    with _pin_store(home_b):
        path, error = _resolve_script_path(str(script))
    assert error is None
    assert path == script.resolve()


def test_absolute_script_outside_every_scripts_dir_is_blocked(homes):
    _home_a, home_b = homes
    outsider = home_b / "evil.py"
    outsider.write_text("print('x')\n", encoding="utf-8")

    from cron.scheduler_script import _resolve_script_path

    with _pin_store(home_b):
        path, error = _resolve_script_path(str(outsider))
    assert path is None
    assert error is not None
    assert "Blocked" in error


def test_relative_traversal_is_blocked_not_missed(homes):
    _home_a, home_b = homes

    from cron.scheduler_script import _resolve_script_path

    with _pin_store(home_b):
        path, error = _resolve_script_path("../evil.py")
    assert path is None
    assert error is not None
    assert "Blocked" in error


def test_run_job_script_executes_the_store_resolved_copy(homes, monkeypatch):
    """End to end: _run_job_script (the monitor/pre-run execution entry) runs the OWNING
    profile's script from a process whose ambient home differs. build_subprocess_env is stubbed
    because the test-suite home-IO guard refuses its real-home manifest probe (see the
    pre-existing spawn-test failures in test_cron_script.py on guarded checkouts)."""
    _home_a, home_b = homes
    (home_b / "scripts" / "marker.py").write_text(
        "print('MARKER profile-copy ran')\n", encoding="utf-8")

    import tools.environments.local as env_local

    monkeypatch.setattr(env_local, "build_subprocess_env",
                        lambda **_kw: dict(os.environ))

    from cron.scheduler_script import _run_job_script

    with _pin_store(home_b):
        ok, output = _run_job_script("marker.py")
    assert ok is True, output
    assert "MARKER profile-copy ran" in output


# ---------------------------------------------------------------------------
# cron.lifecycle_guard / jobs_invariants — door + scanner must agree
# ---------------------------------------------------------------------------


def test_lifecycle_guard_resolves_the_same_file(homes):
    """The guard scans the file that will actually run — store-first, like the runtime."""
    _home_a, home_b = homes
    (home_b / "scripts" / "guarded.py").write_text("echo hi\n", encoding="utf-8")

    from cron.lifecycle_guard import _resolve_script_path

    with _pin_store(home_b):
        path = _resolve_script_path("guarded.py")
    assert path == (home_b / "scripts" / "guarded.py")


def test_guard_owning_store_takes_precedence_over_process_home(homes):
    """Both homes hold a copy: the guard must scan the OWNING store's file — the one the
    runtime actually runs — never a process-home copy left behind by a single-home consumer.
    A home-first candidate order would still reach the store copy via fallback and slip past
    ``test_lifecycle_guard_resolves_the_same_file``, silently re-opening the scanner/runtime
    split, so this pins the ORDER (it is red under a flipped candidate order)."""
    home_a, home_b = homes
    (home_a / "scripts" / "guarded.py").write_text("echo process-copy\n", encoding="utf-8")
    (home_b / "scripts" / "guarded.py").write_text("echo profile-copy\n", encoding="utf-8")

    from cron.lifecycle_guard import _resolve_script_path
    from cron.scheduler_script import _resolve_script_path as _runtime_resolve

    with _pin_store(home_b):
        guard_path = _resolve_script_path("guarded.py")
        runtime_path, runtime_error = _runtime_resolve("guarded.py")
    assert guard_path is not None
    assert guard_path == (home_b / "scripts" / "guarded.py")
    assert runtime_error is None and runtime_path is not None
    assert guard_path.resolve() == runtime_path  # scanner scans the file that runs


def test_guard_process_home_is_the_fallback_candidate(homes):
    """A script that exists only in the executing process's home: the guard must find the file
    the scheduler runs. Resolving the owning store's nonexistent path would make the scanner
    read nothing while the job executes unscanned — the scanner/runtime split this fix closes,
    mirrored in the other direction."""
    home_a, home_b = homes
    (home_a / "scripts" / "local.py").write_text("echo hi\n", encoding="utf-8")

    from cron.lifecycle_guard import _resolve_script_path
    from cron.scheduler_script import _resolve_script_path as _runtime_resolve

    with _pin_store(home_b):
        guard_path = _resolve_script_path("local.py")
        runtime_path, runtime_error = _runtime_resolve("local.py")
    assert runtime_error is None and runtime_path is not None
    assert guard_path is not None
    assert guard_path.resolve() == runtime_path


def test_whitespace_script_existing_in_store_is_a_path_not_a_command(homes):
    """``_script_is_command_line`` must agree with the runtime resolver: a spaced filename that
    exists in the owning store is a creatable path, not a command line; missing everywhere it
    stays a command-line refusal."""
    _home_a, home_b = homes
    (home_b / "scripts" / "my watchdog.sh").write_text("echo hi\n", encoding="utf-8")

    from cron.jobs_invariants import _script_is_command_line

    with _pin_store(home_b):
        assert _script_is_command_line("my watchdog.sh") is False
    assert _script_is_command_line("my watchdog.sh") is True  # unpinned: nowhere to find it
