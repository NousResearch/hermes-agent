"""`pm gc` must key its collectors on the owning checkout, not the executing tree.

A generation workspace executing the CLI hashes to a second install key nobody
writes under (#125537): the collectors' missing-root early-returns then render
the run as a successful zero, and a hard error is the only honest answer once
the state dir itself cannot be found.
"""
import json
import sys
import time
from pathlib import Path

from pm.environments import install_key, install_state_dir, installs_root, owning_install_root
from pm.runtime import collect_runtime_generations


def _select(repo, state, name):
    environment = state / "environments" / name / "venv"
    (state / "facts.json").write_text(
        json.dumps({"packages": {"venv": {"environment": str(environment)}}}), encoding="utf-8")
    return environment


def test_owning_install_root_follows_the_project_stamp(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    repo = tmp_path / "repo"
    repo.mkdir()
    state = install_state_dir(repo)
    workspace = state / "environments" / "gen-active" / "workspace"
    workspace.mkdir(parents=True)
    (state / "inputs").mkdir(parents=True)
    (state / "inputs" / ".project-root").write_text(str(repo), encoding="utf-8")

    # The workspace is a plain path here, not the executing tree, so its own
    # key derives a state dir that does not exist -- the silent-zero trap.
    assert not install_state_dir(workspace).exists()
    assert owning_install_root(workspace) == repo

    # Any tree that is not an installs/<key>/environments/<gen>/workspace is literal.
    assert owning_install_root(repo) == repo
    lookalike = tmp_path / "not-installs" / "environments" / "gen" / "workspace"
    lookalike.mkdir(parents=True)
    assert owning_install_root(lookalike) == lookalike
    # A generation workspace without the stamp never invents a root (different key).
    bare = installs_root() / install_key(tmp_path / "elsewhere") / "environments" / "gen-stale" / "workspace"
    bare.mkdir(parents=True)
    assert owning_install_root(bare) == bare


def test_gc_from_a_generation_workspace_prunes_the_owning_install(tmp_path, monkeypatch, capsys):
    from pm import cli

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    repo = tmp_path / "repo"
    repo.mkdir()
    state = install_state_dir(repo)
    for name, old in (("gen-active", False), ("gen-stale", True)):
        venv = state / "environments" / name / "venv"
        venv.mkdir(parents=True)
        (venv / "pyvenv.cfg").write_text("version = 3.11")
        marker = venv.parent / ".lease-managed"
        marker.touch()
        if old:
            how_old = time.time() - 2 * 86400
            import os
            os.utime(marker, (how_old, how_old))
    _select(repo, state, "gen-active")

    # The CLI executes from the active generation's workspace: its key is wrong.
    workspace = state / "environments" / "gen-active" / "workspace"
    workspace.mkdir(parents=True)
    (state / "inputs").mkdir(parents=True)
    (state / "inputs" / ".project-root").write_text(str(repo), encoding="utf-8")
    monkeypatch.setattr("pm.paths.repo_root", lambda: workspace)
    monkeypatch.setattr("pm.paths.writable_store_root", lambda: tmp_path / "tools")

    assert cli.cmd_gc(None) == 0
    out = capsys.readouterr().out
    assert "removed 1 dependency generations" in out
    assert not (state / "environments" / "gen-stale").exists()
    assert (state / "environments" / "gen-active").is_dir()


def test_gc_errors_when_the_install_state_cannot_be_found(tmp_path, monkeypatch, capsys):
    from pm import cli

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    # A tree that is neither a generation workspace nor an installed checkout:
    # the old code printed "removed 0 dependency generations" with exit 0.
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    monkeypatch.setattr("pm.paths.repo_root", lambda: checkout)
    monkeypatch.setattr("pm.paths.writable_store_root", lambda: tmp_path / "tools")

    assert cli.cmd_gc(None) == 1
    assert "no dependency generations" in capsys.readouterr().err
