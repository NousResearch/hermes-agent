"""The managed-worker child must be able to import ``agent`` from the tree it runs in.

The child is spawned as ``python -m agent.managed_worker`` and resolves ``agent`` through
the implicit ``-m`` cwd entry. Two things make that fragile, and both are asserted here:

* the routed-profile child env has the Hermes-owned PYTHONPATH entries stripped on
  purpose (``_strip_hermes_owned_pythonpath``), so the cwd entry is the ONLY route to
  ``agent`` -- and under ``PYTHONSAFEPATH`` even that entry is gone (the same class of
  failure the cron external worker already carries a fix for, #112729 / #122222);
* the cwd must be the tree that OWNS ``agent``. It is pinned onto the child's
  PYTHONPATH, so a cwd pointing at some other tree would pin the wrong tree.

The contract is asserted on the real spawn kwargs (the same Popen-spy shape the cron
worker fix uses), not on the source text.
"""
import asyncio
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


def _spawn_capture(monkeypatch, managed, env):
    """Spawn the managed worker with a captured Popen; return the kwargs it was given."""
    captured = {}

    class _FakeProcess:
        stdin = None
        stdout = None
        stderr = None

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

    def fake_popen(args, **kwargs):
        captured['args'] = args
        captured.update(kwargs)
        return _FakeProcess()

    monkeypatch.setattr(managed, '_worker_env', lambda authority: env)
    monkeypatch.setattr(managed.subprocess, 'Popen', fake_popen)
    authority = SimpleNamespace(profile_id='profile', pending_results={})
    ref = SimpleNamespace(session_id='session')
    row = {'admission_id': 'adm', 'generation': 1}
    asyncio.run(managed._spawn_registered_worker(authority, ref, row))
    return captured


def test_managed_worker_runs_from_the_tree_that_owns_the_agent_package(monkeypatch, tmp_path):
    """The child's cwd is the root the ``agent`` package was imported from.

    ``gateway`` and ``agent`` are not guaranteed to be imported from one tree: an
    installed ``site-packages`` copy of the package next to a source checkout (or the
    reverse) makes them diverge. Deriving the cwd from the module's own ``__file__``
    then points the child at a tree that holds no ``agent`` package, and the child dies
    with ``No module named 'agent'`` before its hello frame -- booked by the owner as
    ``EOFError: managed_worker_pipe_closed``.
    """
    from gateway import session_managed_worker as managed

    import agent
    agent_tree = Path(agent.__file__).resolve().parent.parent

    # This module was imported from a DIFFERENT tree than the ``agent`` package (the
    # editable-install / site-packages-next-to-checkout case). Resolving the cwd from
    # this module's own location would pick that other tree.
    other_root = tmp_path / 'other-tree'
    (other_root / 'gateway').mkdir(parents=True)
    (other_root / 'gateway' / 'session_managed_worker.py').write_text('')
    monkeypatch.setattr(
        managed, '__file__', str(other_root / 'gateway' / 'session_managed_worker.py'))

    captured = _spawn_capture(monkeypatch, managed, {'PATH': os.environ.get('PATH', '')})

    assert Path(captured['cwd']) == agent_tree
    assert (Path(captured['cwd']) / 'agent' / '__init__.py').is_file()


def test_managed_worker_pins_the_tree_on_the_child_pythonpath(monkeypatch):
    """The routed-profile child env has the Hermes tree stripped off PYTHONPATH, so the
    tree goes back on explicitly -- the same pin the cron external worker gets (#112729).
    The pin extends the sanitized env (it must not resurrect entries the sanitizer
    dropped), and the tree comes first."""
    from gateway import session_managed_worker as managed

    import agent
    agent_tree = Path(agent.__file__).resolve().parent.parent

    captured = _spawn_capture(monkeypatch, managed, {'PYTHONPATH': '/user-libs'})

    entries = captured['env']['PYTHONPATH'].split(os.pathsep)
    assert entries[0] == str(agent_tree)
    assert '/user-libs' in entries


def test_managed_worker_child_can_import_agent_with_the_cwd_entry_disabled(monkeypatch):
    """The end-to-end consequence, under the condition that breaks the implicit route:
    ``PYTHONSAFEPATH`` drops the cwd entry from ``sys.path``, and the sanitized child
    env has the tree stripped off PYTHONPATH. With the pin, the child still resolves
    ``agent`` -- from the tree, not from some other checkout on the path."""
    import subprocess
    import sys

    from gateway import session_managed_worker as managed

    import agent
    agent_tree = Path(agent.__file__).resolve().parent.parent

    captured = _spawn_capture(monkeypatch, managed, {'PYTHONPATH': '/user-libs'})

    # The capture patches ``subprocess.Popen`` module-wide; undo it so the real
    # interpreter can be launched for the check below.
    monkeypatch.undo()

    child_env = dict(captured['env'])
    child_env['PYTHONSAFEPATH'] = '1'
    result = subprocess.run(
        [sys.executable, '-c', 'import agent; print(agent.__file__)'],
        cwd=str(captured['cwd']), env=child_env, capture_output=True, text=True, timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()).resolve() == (agent_tree / 'agent' / '__init__.py').resolve()


def test_single_profile_managed_worker_keeps_the_inherited_process_env(monkeypatch):
    """A single-profile gateway's worker inherits the process env byte-for-byte
    (``_worker_env`` returns None). That path is untouched: the child env stays None and
    the cwd is still the tree."""
    from gateway import session_managed_worker as managed

    import agent
    agent_tree = Path(agent.__file__).resolve().parent.parent

    captured = _spawn_capture(monkeypatch, managed, None)

    assert captured['env'] is None
    assert Path(captured['cwd']) == agent_tree
