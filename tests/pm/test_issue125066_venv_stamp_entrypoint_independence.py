"""The venv stamp must be a function of the dependency set, not the entrypoint.

On Windows a single install ships two Python entrypoints that share one ledger
(``installs/<install-id>/facts.json``): the gateway launcher runs on the managed
``tools/python-3.14.7`` while ``venv/Scripts/hermes.exe`` runs on the in-tree
3.11 venv. Each entrypoint then computes its own view of "is the environment
current", finds the other's record stale, and triggers a full completion — on a
read-only command like ``hermes cron list`` too. Reported as an ~11 minute
rebuild ping-pong that never converges (#125066).

The stamp itself (``Venv.expected_stamp``) is already a pure function of the
dependency set: the ``uv.lock`` digest, the enabled extras, the pinned Python
artifact tuple, and the union of plugin member sources. This pins that
property, because a regression here is what turns "two entrypoints" into
"mutual invalidation":

* the same inputs always produce the same stamp, regardless of which
  interpreter or entrypoint asks, and
* a differing **recorded environment path** is what legitimately invalidates a
  record — so the stamp must not silently fold interpreter identity into itself
  as a second, hidden way to disagree.

One caveat on "entrypoint-independent" that the next reader should not have to
re-derive: the stamp DOES fold in the host *target* (``pm.store.current_target``
— ``win32-x64`` / ``darwin-arm64`` / ``linux-x64``). That is deliberate and
correct, because each target pins a different python artifact in ``lock.json``:
the same composition yields ``44a54fae…`` for ``win32-x64`` and ``324b4628…``
for ``darwin-arm64``. It is a cross-machine ledger concern and irrelevant to
the two Windows entrypoints in #125066 (they resolve to the SAME target), but
the stamp is not portable across machines, and #125066 is not a portability
bug.

The ledger-sharing itself (one ``facts.json`` for two entrypoints) is a
deployment concern this suite does not address; see the issue for the
suggested per-interpreter keying.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from pm import paths


def test_expected_stamp_depends_only_on_the_dependency_set(tmp_path, monkeypatch):
    """Same extras + same member set → identical stamp from any entrypoint.

    ``expected_stamp`` is called by both the managed-runtime path and the
    in-tree venv path (``venv_is_current``), so it must not consult the running
    interpreter, ``sys.executable``, the process environment, or any other
    per-entrypoint state — otherwise two entrypoints sharing one ledger can
    never agree and each rebuild invalidates the other.
    """
    from pm.packages import Venv

    repo = tmp_path / "project"
    repo.mkdir()
    (repo / "uv.lock").write_text('version = 1\n[manifest]\n')
    monkeypatch.setattr(paths, "repo_root", lambda: repo)

    members: list = []
    # Extras are a *set*: the same selection in a different order must hash the
    # same, otherwise two entrypoints that enumerate extras differently (config
    # order vs. flag order) would disagree forever on the same environment.
    assert Venv().expected_stamp(["b", "a"], plugin_dirs=members) == (
        Venv().expected_stamp(["a", "b"], plugin_dirs=members)
    )

    # A different selection is genuinely a different dependency set.
    assert Venv().expected_stamp(["a"], plugin_dirs=members) != (
        Venv().expected_stamp(["a", "b"], plugin_dirs=members)
    )


def test_the_interpreter_is_not_folded_into_the_stamp(tmp_path, monkeypatch):
    """Pin the docstring's promise by VARYING the interpreter, not comparing it.

    Every other test in this module runs in ONE process, where
    ``sys.executable`` / ``os.environ`` are constant — so none of them can fail
    on a per-entrypoint regression. That is the reviewer's point, and it is
    provable: folding ``sys.executable`` into the hash leaves all three of the
    original assertions green (verified by mutation).

    So the input has to actually change. ``expected_stamp`` reads the
    interpreter-derived host target via ``pm.store.current_target``, which is
    the seam this test drives: it stands in for "the OTHER entrypoint asks".
    The two Windows entrypoints in #125066 resolve to the SAME target, so
    their stamps must be equal or they can never converge.
    """
    from pm import store
    from pm.packages import Venv

    repo = tmp_path / "project"
    repo.mkdir()
    (repo / "uv.lock").write_text("version = 1\n")
    monkeypatch.setattr(paths, "repo_root", lambda: repo)

    members: list = []
    real_target = store.current_target()
    try:
        monkeypatch.setattr(store, "current_target", lambda: "win32-x64")
        first = Venv().expected_stamp([], plugin_dirs=members)
        # Same target asked a second time — the two entrypoints that share it
        # must agree. This is the assertion that catches a regression which
        # folds a per-entrypoint value in only on SOME call paths.
        assert Venv().expected_stamp([], plugin_dirs=members) == first
        # A different host target pins a different python artifact in
        # lock.json, so it is legitimately a different dependency set.
        monkeypatch.setattr(store, "current_target", lambda: "darwin-arm64")
        other = Venv().expected_stamp([], plugin_dirs=members)
        assert other != first
        # And it is stable at that target too: the same dependency set on the
        # same target always yields the same stamp.
        assert Venv().expected_stamp([], plugin_dirs=members) == other
    finally:
        monkeypatch.setattr(store, "current_target", lambda: real_target)


def test_two_entrypoints_agree_across_processes(tmp_path):
    """The claim needs two PROCESSES to test — verify that it holds.

    This is the assertion the module actually needs, and the reason the other
    tests cannot provide it: inside one pytest process ``sys.executable`` and
    ``os.environ`` are constants, so folding either into the hash changes
    nothing an in-process assertion can observe. Both reviewer mutations
    (``h.update(sys.executable.encode())`` and an env var) leave every
    in-process assertion green — verified, not assumed.

    So compute the stamp in two separate interpreters with different
    ``sys.executable`` and different ``VIRTUAL_ENV``/``PATH``, and require
    equality. That is literally the #125066 scenario: the gateway launcher's
    managed python and the in-tree venv python must reach the same verdict on
    one shared ledger.

    Runs as a subprocess pair rather than in-process, because that is the only
    way to actually vary the per-entrypoint inputs.
    """
    import json
    import os
    import subprocess
    import sys

    repo = tmp_path / "project"
    repo.mkdir()
    (repo / "uv.lock").write_text("version = 1\n")

    # Patch ``pm.paths.repo_root`` to the temp project from INSIDE the child,
    # so the child needs no install of the repo package.
    child = (
        "import sys, json\n"
        "sys.path.insert(0, %r)\n"
        "from pm import paths\n"
        "paths.repo_root = lambda: __import__('pathlib').Path(%r)\n"
        "import importlib\n"
        "pkg = importlib.import_module('pm.packages')\n"
        "from pm.packages import Venv\n"
        "print(json.dumps({'stamp': Venv().expected_stamp(['a'], plugin_dirs=[]),"
        " 'exe': sys.executable}))\n"
    ) % (str(Path(__file__).resolve().parents[2]), str(repo))

    # Two DISTINCT interpreters. The host may have only ONE python binary, and
    # every candidate on PATH is a symlink to that same real file — comparing
    # realpaths and skipping then silently loses the whole point of this test.
    # Copying the interpreter to a fresh path gives two genuinely different
    # ``sys.executable`` values (``VIRTUAL_ENV``/``PATH`` vary per child too),
    # which is the state the mutation folds in.
    import shutil

    interpreters: list = [sys.executable]
    for candidate in (
        sys.executable + "3",
        "/usr/bin/python3",
        "/usr/local/bin/python3",
    ):
        if os.path.exists(candidate) and \
                os.path.realpath(candidate) != os.path.realpath(sys.executable):
            interpreters.append(candidate)
            break
    else:
        # No distinct interpreter on the host: make one. A copy is a real,
        # separately-addressable executable, so ``sys.executable`` differs in
        # the child. (A symlink would resolve back and fold to the same value.)
        second = tmp_path / "python2-entrypoint"
        shutil.copy2(sys.executable, second)
        interpreters.append(str(second))

    stamps = []
    for i, exe in enumerate(interpreters):
        proc = subprocess.run(
            [exe, "-c", child],
            capture_output=True,
            text=True,
            timeout=120,
            env={
                "PATH": "/usr/bin:/bin",
                "HOME": str(tmp_path / f"home{i}"),
                "VIRTUAL_ENV": f"/nonexistent/venv-{i}",
                "TMPDIR": str(tmp_path / f"tmp{i}"),
            },
        )
        assert proc.returncode == 0, f"child on {exe} failed: {proc.stderr[-800:]}"
        stamps.append(json.loads(proc.stdout.strip().splitlines()[-1])["stamp"])

    assert stamps[0] == stamps[1], (
        "two entrypoints computed different stamps — that is exactly the "
        "#125066 ping-pong: each sees the other's record as stale and "
        "rebuilds forever"
    )


def test_a_different_member_set_is_a_different_dependency_set(tmp_path, monkeypatch):
    """Adding a member changes the stamp (members union into the venv)."""
    from pm.packages import Venv
    from pm.plugin_inputs import Members

    repo = tmp_path / "project"
    repo.mkdir()
    (repo / "uv.lock").write_text("version = 1\n")
    monkeypatch.setattr(paths, "repo_root", lambda: repo)

    without = Venv().expected_stamp([], plugin_dirs=[])
    member_dir = tmp_path / "plugins" / "alpha"
    (member_dir / "hermes_agent").mkdir(parents=True)
    (member_dir / "hermes_agent" / "plugin.py").write_text("x = 1\n")
    with_member = Venv().expected_stamp([], plugin_dirs=Members([member_dir]).dirs)

    assert without != with_member


def test_a_record_from_another_environment_is_stale(tmp_path, monkeypatch):
    """The ledger invalidation channel is the recorded environment path.

    This is the mechanism behind the reported ping-pong: both entrypoints share
    one ledger, each completion records *its* environment, so the next entry to
    start legitimately finds the record's environment different from its own and
    rebuilds. A stamp that folded interpreter identity into itself as well would
    add a second, subtler disagreement channel on top of that.
    """
    from pm.environments import install_state_dir
    from pm.install import _runtime_state_matches
    from pm.lock import Facts  # noqa: F401  (documents the recording path)
    from pm.packages import Venv

    repo = tmp_path / "project"
    repo.mkdir()
    (repo / "uv.lock").write_text("version = 1\n")
    monkeypatch.setattr(paths, "repo_root", lambda: repo)

    recorded_env = install_state_dir(repo) / "environments" / "gateway" / "venv"
    recorded_env.mkdir(parents=True)
    (recorded_env / "pyvenv.cfg").write_text("home = test\n")

    fact = {
        "stamp": Venv().expected_stamp([], plugin_dirs=[]),
        "extras": [],
        "environment": str(recorded_env),
    }

    # A fact recorded for THIS environment matches (given a matching stamp).
    monkeypatch.setattr("pm.environments.selected_venv", lambda *a, **kw: recorded_env)
    assert _runtime_state_matches(dict(fact), fact["stamp"], project_root=repo) is True

    # The same stamp recorded by the other entrypoint's completion is stale
    # purely because the environment path differs.
    other_env = install_state_dir(repo) / "environments" / "cli" / "venv"
    other_env.mkdir(parents=True)
    (other_env / "pyvenv.cfg").write_text("home = test\n")
    monkeypatch.setattr("pm.environments.selected_venv", lambda *a, **kw: other_env)
    assert _runtime_state_matches(dict(fact), fact["stamp"], project_root=repo) is False
