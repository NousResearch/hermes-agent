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

The ledger-sharing itself (one ``facts.json`` for two entrypoints) is a
deployment concern this suite does not address; see the issue for the
suggested per-interpreter keying.
"""
from __future__ import annotations

import importlib

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
