"""Regression test for the kanban dispatcher worker-spawn bootstrap fix (7c37aa8403).

``_module_hermes_argv()`` (``hermes_cli/kanban_db_dispatch.py``) used to build a bare
``[sys.executable, "-m", "hermes_cli.main"]`` for kanban worker spawn. That form depends
entirely on ``hermes_cli`` already being importable in the CHILD -- via an inherited
PYTHONPATH pointing at the repo, a site-installed/editable package, or ``-m``'s own
cwd-based ``sys.path`` insertion. A routed/multiplexed kanban worker gets NONE of these:
``build_subprocess_env(scrub_secrets=True)`` strips Hermes-owned PYTHONPATH entries by
design (``tools/environments/local_pythonpath.py``), and the worker's cwd is the task
workspace, never the repo (``_default_spawn`` spawns with ``cwd=workspace``). The bare
``-m`` form then dies with ``ModuleNotFoundError: No module named 'hermes_cli'`` the
instant it is spawned this way -- exactly the standalone-Python + ``hermes_bootstrap``
deployment the fix commit describes.

The fix routes through ``hermes_cli._launchers.runtime_command``, whose bootstrap does
``sys.path.insert(0, repo_root)`` itself before importing anything, so the child
re-derives the import path independent of both its own PYTHONPATH and its cwd.

This is a real subprocess test, not a mock: it proves the two argv forms actually behave
differently under identical hostile conditions (stripped PYTHONPATH + unrelated cwd),
never a frozen argv literal (root AGENTS.md: behavior contracts over snapshots).
"""

from __future__ import annotations

import os
import subprocess
import sys


def _scrubbed_worker_env(tmp_path, monkeypatch):
    """The real env a routed/multiplexed kanban worker gets, built through the actual
    production sanitizer -- not a hand-rolled dict."""
    from pm.paths import repo_root
    from tools.environments import local as local_env

    root = str(repo_root())
    # Stay off the real hermes home: build_subprocess_env's PATH augmentation resolves
    # the real console-script install dir (_resolve_hermes_bin_dir), which the test
    # suite's HomeIOGuard forbids touching (tests/tools/test_persist_on_release.py
    # carries the same guard). Irrelevant to the PYTHONPATH-stripping contract under
    # test, so it's neutralized rather than patched around per-callsite.
    monkeypatch.setattr(local_env, "_resolve_hermes_bin_dir", lambda: None)
    # Simulate the dispatcher's own ambient shell carrying a Hermes-owned PYTHONPATH
    # entry (what `source ./activate` leaves in a dev shell) -- the exact residue a
    # routed worker must not inherit.
    base_env = dict(os.environ)
    base_env["PYTHONPATH"] = root
    env = local_env.build_subprocess_env(base=base_env, scrub_secrets=True)
    assert root not in (env.get("PYTHONPATH") or "").split(os.pathsep), (
        "test setup invalid: build_subprocess_env did not strip the repo root from "
        "PYTHONPATH -- fix this fixture before trusting the subprocess assertions below"
    )
    # Never let either child touch the developer's real ~/.hermes (update-check cache,
    # sessions, ...).
    home = tmp_path / "child_home"
    home.mkdir()
    env["HERMES_HOME"] = str(home)
    return env, root


def test_worker_argv_survives_pythonpath_stripped_child_env(tmp_path, monkeypatch):
    """The real ``_resolve_hermes_argv()`` chain must import ``hermes_cli`` in a child
    whose PYTHONPATH was stripped by the real sanitizer and whose cwd is the task
    workspace (never the repo). The legacy bare ``-m hermes_cli.main`` form this fix
    replaced cannot, under the identical conditions.
    """
    from hermes_cli.kanban_db_dispatch import _resolve_hermes_argv

    env, _root = _scrubbed_worker_env(tmp_path, monkeypatch)
    workspace = tmp_path / "task_workspace"
    workspace.mkdir()

    # --- RED: the legacy bare "-m hermes_cli.main" argv this fix replaced. ``-S``
    # disables the running interpreter's own site-packages processing, so a dev
    # checkout's editable install of hermes_cli can't quietly paper over the same gap a
    # real standalone-Python + PM-managed dependency install has -- reproducing "no
    # ambient path to hermes_cli except what the argv construction itself provides",
    # exactly what the fix commit's bug report describes.
    legacy_argv = [sys.executable, "-S", "-m", "hermes_cli.main", "--version"]
    legacy = subprocess.run(
        legacy_argv, env=env, cwd=str(workspace),
        capture_output=True, text=True, timeout=60,
    )
    assert legacy.returncode != 0
    assert "ModuleNotFoundError" in legacy.stderr
    assert "hermes_cli" in legacy.stderr

    # --- GREEN: the real production argv this test guards.
    fixed_argv = [*_resolve_hermes_argv(), "--version"]
    fixed = subprocess.run(
        fixed_argv, env=env, cwd=str(workspace),
        capture_output=True, text=True, timeout=60,
    )
    assert fixed.returncode == 0, f"stdout={fixed.stdout!r} stderr={fixed.stderr!r}"
    assert "Install directory" in fixed.stdout
