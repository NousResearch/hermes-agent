"""A linked git worktree borrows the owner checkout's committed dependency generation.

The install key is a hash of the checkout path, so every worktree is a distinct key with
no ``facts.json`` of its own. A worker that imports ``hermes_cli`` from a worktree
therefore resolved to an install with nothing committed, and PM's store Python refuses
such a process outright: "no dependency environment is committed for this install". The
worker died before it did any work -- the defect behind ~64 burned spawns on the
isolation probe (t_86e4721f).

The worktree borrows the owner's committed generation: the same tree, the same
interpreter, the same dependencies. The worktree's own CODE stays first on ``sys.path``
(that is the branch the worker is here to test); only the dependencies are borrowed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import pytest

from pm.environments import (
    committed_venv,
    install_key,
    install_state_dir,
    runtime_facts_path,
    site_packages,
    worktree_owner_root,
)

# Every lane, Windows included: the resolver reads the checkout's ``.git`` file and the
# owner's facts record, neither of which is OS-specific, but the marker keeps the file
# collected by the marked-OS lane collector (scripts/ci/list_os_marked_tests.py).
pytestmark = pytest.mark.platforms("any")


def _checkout(root: Path) -> Path:
    """A main checkout that looks like Hermes (it owns the shared ``.git``)."""
    (root / "hermes_cli").mkdir(parents=True)
    (root / "hermes_cli" / "main.py").write_text("", encoding="utf-8")
    (root / ".git").mkdir()
    return root


def _worktree(root: Path, owner: Path, name: str = "wt") -> Path:
    """A linked worktree of *owner*: ``.git`` is a FILE pointing into the shared gitdir."""
    worktree = root / ".worktrees" / name
    worktree.mkdir(parents=True)
    (worktree / "hermes_cli").mkdir()
    (worktree / "hermes_cli" / "main.py").write_text("", encoding="utf-8")
    gitdir = owner / ".git" / "worktrees" / name
    gitdir.mkdir(parents=True)
    (worktree / ".git").write_text(f"gitdir: {gitdir}\n", encoding="utf-8")
    return worktree


def _commit_generation(owner: Path, tmp_path: Path) -> Path:
    """Give *owner* a committed generation and the facts record that names it."""
    generation = install_state_dir(owner) / "environments" / "gen" / "venv"
    site = site_packages(generation)
    site.mkdir(parents=True)
    (generation / "pyvenv.cfg").write_text("home = test\n", encoding="utf-8")
    (site / "probe_package.py").write_text("version = 'owner'\n", encoding="utf-8")
    runtime_facts_path(owner).parent.mkdir(parents=True, exist_ok=True)
    runtime_facts_path(owner).write_text(json.dumps({
        "schema": 1,
        "packages": {"venv": {"environment": str(generation), "stamp": "verified"}},
    }), encoding="utf-8")
    return generation


def test_a_linked_worktree_resolves_to_its_owning_checkout(tmp_path):
    owner = _checkout(tmp_path / "hermes-agent")
    worktree = _worktree(tmp_path / "hermes-agent", owner)

    assert worktree_owner_root(worktree) == owner.resolve()
    assert worktree_owner_root(owner) is None, "the main checkout owns itself"
    # A checkout that is not a Hermes checkout never borrows anyone.
    other = tmp_path / "other"
    other.mkdir()
    (other / ".git").mkdir()
    assert worktree_owner_root(other) is None


def test_a_worktree_borrows_the_owners_committed_generation(tmp_path):
    owner = _checkout(tmp_path / "hermes-agent")
    worktree = _worktree(tmp_path / "hermes-agent", owner)
    generation = _commit_generation(owner, tmp_path)

    # The worktree has no install state of its own...
    assert install_key(worktree) != install_key(owner)
    assert not runtime_facts_path(worktree).is_file()
    assert not install_state_dir(worktree).is_dir()
    # ...and still resolves to the owner's committed environment.
    assert committed_venv(worktree) == generation.resolve()
    # The owner's own answer is unchanged.
    assert committed_venv(owner) == generation.resolve()


def test_a_worktree_with_its_own_state_prefers_its_own_generation(tmp_path):
    """A worktree that DID commit a generation of its own keeps using it."""
    owner = _checkout(tmp_path / "hermes-agent")
    worktree = _worktree(tmp_path / "hermes-agent", owner)
    _commit_generation(owner, tmp_path)
    own = install_state_dir(worktree) / "environments" / "own" / "venv"
    site = site_packages(own)
    site.mkdir(parents=True)
    (own / "pyvenv.cfg").write_text("home = test\n", encoding="utf-8")
    runtime_facts_path(worktree).parent.mkdir(parents=True, exist_ok=True)
    runtime_facts_path(worktree).write_text(json.dumps({
        "schema": 1, "packages": {"venv": {"environment": str(own), "stamp": "verified"}},
    }), encoding="utf-8")

    assert committed_venv(worktree) == own.resolve()


def test_a_worktree_of_an_owner_with_nothing_committed_borrows_nothing(tmp_path):
    """No committed generation anywhere: the worktree has nothing to borrow, so the
    process keeps its own packages (today's behaviour for an install with no state)."""
    owner = _checkout(tmp_path / "hermes-agent")
    worktree = _worktree(tmp_path / "hermes-agent", owner)

    assert committed_venv(worktree) is None


def test_the_borrowed_generation_is_what_the_worker_actually_imports(tmp_path, monkeypatch):
    """The end-to-end shape: a worker whose cwd is the worktree imports the worktree's
    CODE and the owner's DEPENDENCIES."""
    from pm import environments

    owner = _checkout(tmp_path / "hermes-agent")
    worktree = _worktree(tmp_path / "hermes-agent", owner)
    generation = _commit_generation(owner, tmp_path)
    site = site_packages(generation)
    (site / "probe_package.py").write_text("version = 'owner'\n", encoding="utf-8")

    monkeypatch.setattr(sys, "prefix", str(generation))
    monkeypatch.setattr(sys, "base_prefix", str(generation))
    selected = environments.activate_dependencies(worktree)

    assert selected == owner.resolve()
    import probe_package
    assert probe_package.version == "owner"
    # The worktree's own checkout stays first on sys.path -- that is the branch under test.
    assert Path(sys.path[0]).resolve() == worktree.resolve()


def test_a_broken_owner_record_does_not_fail_the_worktree(tmp_path):
    """The owner's own record is broken: the worktree surfaces nothing of its own, and
    the owner's launch reports the real error."""
    owner = _checkout(tmp_path / "hermes-agent")
    worktree = _worktree(tmp_path / "hermes-agent", owner)
    runtime_facts_path(owner).parent.mkdir(parents=True, exist_ok=True)
    runtime_facts_path(owner).write_text("{not json", encoding="utf-8")

    assert committed_venv(worktree) is None
    with pytest.raises(RuntimeError):
        committed_venv(owner)


def test_the_owner_record_still_rejects_an_environment_outside_its_install(tmp_path):
    """The containment check is not weakened for the owner: a record naming a foreign
    tree is an error, not a selection."""
    owner = _checkout(tmp_path / "hermes-agent")
    worktree = _worktree(tmp_path / "hermes-agent", owner)
    foreign = tmp_path / "foreign" / "venv"
    (foreign).mkdir(parents=True)
    (foreign / "pyvenv.cfg").write_text("home = test\n", encoding="utf-8")
    runtime_facts_path(owner).parent.mkdir(parents=True, exist_ok=True)
    runtime_facts_path(owner).write_text(json.dumps({
        "schema": 1, "packages": {"venv": {"environment": str(foreign)}},
    }), encoding="utf-8")

    with pytest.raises(RuntimeError):
        committed_venv(owner)
    # The worktree borrows nothing from a broken owner.
    assert committed_venv(worktree) is None


def test_the_resolution_needs_no_subprocess_and_no_git_on_path(tmp_path, monkeypatch):
    """Boot-time cost matters: the owner is read off the filesystem, never by running
    ``git`` (a spawn at boot can fail under the worker's mount namespace)."""
    owner = _checkout(tmp_path / "hermes-agent")
    worktree = _worktree(tmp_path / "hermes-agent", owner)

    import subprocess

    def refuse(*args, **kwargs):  # pragma: no cover - the guard is the assertion
        raise AssertionError("the resolver spawned a subprocess")

    monkeypatch.setattr(subprocess, "run", refuse)
    monkeypatch.setattr(subprocess, "Popen", refuse)
    assert worktree_owner_root(worktree) == owner.resolve()


@pytest.mark.skipif(os.name == "nt", reason="POSIX path layout")
def test_a_bare_dot_git_file_is_not_a_worktree(tmp_path):
    """A checkout whose ``.git`` is a file but not a gitdir pointer never borrows."""
    root = tmp_path / "fake"
    (root / "hermes_cli").mkdir(parents=True)
    (root / "hermes_cli" / "main.py").write_text("", encoding="utf-8")
    (root / ".git").write_text("gitdir: /nonexistent/elsewhere\n", encoding="utf-8")
    assert worktree_owner_root(root) is None
