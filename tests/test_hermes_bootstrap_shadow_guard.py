"""A workspace file must not be able to shadow the stdlib at Hermes boot.

``python -m hermes_cli.main`` — what the Kanban dispatcher launches for every worker,
and what every ``hermes`` console script does — puts the CURRENT WORKING DIRECTORY at
``sys.path[0]``. A Kanban workspace is a scratch directory the worker writes into, so a
file it creates that shares a name with a stdlib module (``inspect.py``, ``json.py``,
``logging.py`` …) wins the import and the process dies while booting, before it can
print a line. Eight consecutive runs on one card died that way and every one of them was
booked as an unrelated failure.

Two layers, because two moments are exposed:

* ``_demote_stdlib_shadowing_paths`` — the guard every entry point gets, making the
  stdlib win a shadowed name while the directory's own modules stay importable. It runs
  as the first statement of ``hermes_bootstrap``; it cannot help with a name the
  interpreter needs *before* user code (``runpy``, and what ``runpy`` imports).
* ``-P`` on the worker argv — keeps the workspace off ``sys.path`` entirely for the
  process the dispatcher spawns, closing that earlier window too. A worker's own
  subprocesses are unaffected: the flag rides the argv, not the environment.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import hermes_bootstrap
from hermes_cli import kanban_db_dispatch as kbd

# ``types`` is imported by ``runpy`` before any user code, so it is exercised in the
# ``-P`` test instead of the guard test.
SHADOWING = ("inspect.py", "json.py", "logging.py")
BEFORE_USER_CODE = ("runpy.py", "types.py", "inspect.py")


def _poison(workspace: Path, names: tuple[str, ...] = SHADOWING) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    for name in names:
        (workspace / name).write_text(
            f'raise SystemExit("workspace {name} was imported")', encoding="utf-8")


def _entry_point_env() -> dict[str, str]:
    """Env for a child that must be poisoned-workspace-faithful, not shell-faithful.

    ``PYTHONSAFEPATH=1`` (equivalently ``-P``) takes the cwd off the child's
    ``sys.path``; the probe below deliberately relies on it being there, and the boot
    child must exercise the guard rather than bypass the whole problem. So the tests own
    that setting instead of inheriting whatever shell they run in — a worker session
    exports ``PYTHONSAFEPATH=1`` box-wide from the local dispatch workaround, and every
    test here would otherwise fail with "No module named probe".
    """
    env = dict(os.environ)
    env.pop("PYTHONSAFEPATH", None)
    root = str(Path(hermes_bootstrap.__file__).resolve().parent)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (root, env.get("PYTHONPATH", "")) if p)
    return env


def _worker_spawn_argv() -> list[str]:
    """The exact argv the dispatcher spawns a worker with, tail included.

    ``_worker_argv`` is the production builder (``_module_hermes_argv`` + profile flags
    + ``chat -q 'work kanban task <id>'``), so a boot test on it covers the spawn shape
    the dispatcher really uses, not just the interpreter prefix.
    """
    task = kbd._kb.Task(
        id="t_probe", title="boot probe", body=None, assignee="probe", status="ready",
        priority=0, created_by="test", created_at=0, started_at=None, completed_at=None,
        workspace_kind="scratch", workspace_path=None, claim_lock=None,
        claim_expires=None, tenant="default")
    return kbd._worker_argv(task, "default", None)


def test_cwd_entry_with_a_stdlib_shadow_is_demoted(monkeypatch, tmp_path):
    workspace = tmp_path / "ws"
    _poison(workspace)
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(sys, "path", ["", str(tmp_path), ""])
    assert hermes_bootstrap._demote_stdlib_shadowing_paths() == ["", ""]
    assert sys.path == [str(tmp_path), "", ""]


def test_clean_cwd_entry_is_left_alone(monkeypatch, tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "helper_script.py").write_text("X = 1", encoding="utf-8")
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(sys, "path", ["", str(tmp_path)])
    assert hermes_bootstrap._demote_stdlib_shadowing_paths() == []
    assert sys.path == ["", str(tmp_path)]


def test_a_bare_directory_does_not_shadow_but_a_package_does(monkeypatch, tmp_path):
    """PEP 420 namespace dirs lose to a real stdlib module further along sys.path."""
    workspace = tmp_path / "ws"
    (workspace / "types").mkdir(parents=True)
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(sys, "path", ["", str(tmp_path)])
    assert hermes_bootstrap._demote_stdlib_shadowing_paths() == []
    (workspace / "types" / "__init__.py").write_text("", encoding="utf-8")
    assert hermes_bootstrap._demote_stdlib_shadowing_paths() == [""]


def test_the_entry_point_boots_from_a_poisoned_workspace(tmp_path):
    """The real worker entry point, launched in a workspace full of shadows, boots.

    ``rc == 0`` alone could pass for the wrong reason, so the probe child also reports
    the demotion it observed and imports a workspace sibling: the stdlib wins the
    shadowed names AND the workspace stays importable.
    """
    workspace = tmp_path / "ws"
    _poison(workspace)
    (workspace / "workspace_sibling.py").write_text("VALUE = 42", encoding="utf-8")
    (workspace / "probe").mkdir()
    (workspace / "probe" / "__main__.py").write_text(
        "import hermes_bootstrap\n"
        "import inspect\n"
        "import workspace_sibling\n"
        "print('INSPECT', inspect.__file__)\n"
        "print('SIBLING', workspace_sibling.VALUE)\n"
        "print('DEMOTED', hermes_bootstrap._DEMOTED_SHADOW_PATHS)\n",
        encoding="utf-8")

    probe = subprocess.run(
        [sys.executable, "-m", "probe"], cwd=workspace, env=_entry_point_env(),
        capture_output=True, text=True, timeout=120, check=False)
    assert probe.returncode == 0, probe.stdout + probe.stderr
    inspect_line = next(ln for ln in probe.stdout.splitlines() if ln.startswith("INSPECT"))
    assert str(workspace) not in inspect_line, probe.stdout
    assert "SIBLING 42" in probe.stdout
    assert f"DEMOTED ['{workspace}']" in probe.stdout

    boot = subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "--help"], cwd=workspace,
        env=_entry_point_env(), capture_output=True, text=True, timeout=300, check=False)
    assert boot.returncode == 0, boot.stderr[-2000:]
    assert "usage: hermes" in boot.stdout


def test_the_worker_launch_survives_shadows_that_precede_user_code(tmp_path):
    """``runpy`` is imported before any user code, so only the argv can save it.

    The dispatcher spawns ``<python> -P -m hermes_cli.main -p … chat -q 'work kanban task
    …'``; a plain ``-m`` over a workspace ``runpy.py`` dies inside ``pymain_run_module``
    ("Could not import runpy module") — no Python-level guard, the shadow guard included,
    ever runs. The argv exercised here is the one the production builder emits, and the
    boot target is the worker's own ``chat`` entry (its usage text is the proof), not just
    the top-level parser. Full-card completion in a poisoned workspace is the live
    dispatcher receipt in the PR body: it needs a model, so it is not a CI test.
    """
    workspace = tmp_path / "ws"
    _poison(workspace, BEFORE_USER_CODE)
    argv = _worker_spawn_argv()
    assert argv[1:3] == ["-P", "-m"], argv
    assert argv[4:6] == ["-p", "default"], argv
    assert argv[-3:] == ["chat", "-q", "work kanban task t_probe"], argv

    boot = subprocess.run(
        [*argv, "--help"], cwd=workspace, env=_entry_point_env(),
        capture_output=True, text=True, timeout=300, check=False)
    assert boot.returncode == 0, boot.stderr[-2000:]
    assert "usage: hermes chat" in boot.stdout


def test_the_import_root_pin_still_matches_the_worker_argv():
    """``-P`` drops the cwd, so the module form MUST keep its PYTHONPATH pin (#122299)."""
    env: dict[str, str] = {}
    kbd._propagate_module_import_root(kbd._module_hermes_argv(), env)
    pinned = env.get("PYTHONPATH", "")
    assert str(Path(kbd.__file__).resolve().parents[1]) in pinned.split(os.pathsep)
