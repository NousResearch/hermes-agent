"""`hermes pm lock` with no arguments relocks uv.lock through PM and nothing else."""
from __future__ import annotations

from argparse import Namespace

import pm
from pm import cli
from pm.package import InstallError

PYPROJECT = """\
[project]
name = "demo"
version = "0"

[project.optional-dependencies]
alpha = ["six"]
beta = ["idna"]
all = ["demo[alpha]"]
"""
LOCK = """\
version = 1

[[package]]
name = "demo"
version = "0"
source = { editable = "." }

[package.metadata]
provides-extras = ["alpha", "all"]
"""


def test_relock_writes_nothing_when_current_and_names_new_opt_in_extras(tmp_path, monkeypatch, capsys):
    (tmp_path / "pyproject.toml").write_text(PYPROJECT, encoding="utf-8")
    (tmp_path / "uv.lock").write_text(LOCK, encoding="utf-8")
    monkeypatch.setattr(cli, "repo_root", lambda: tmp_path)
    stale = []
    relocks = []

    def check(source, **kw):
        assert (source, kw.get("explicit")) == (tmp_path, True)
        # A stale lock is this command's expected case: no red failure report.
        assert kw.get("quiet") is True
        if stale:
            raise InstallError("venv", "The lockfile at `uv.lock` needs to be updated")

    monkeypatch.setattr(pm, "check_project_lock", check)
    monkeypatch.setattr(pm, "lock_project", lambda source, **kw: relocks.append((source, kw)))
    monkeypatch.setattr(pm, "sync_venv", lambda *a, **kw: relocks.append("sync"))

    before = (tmp_path / "uv.lock").read_bytes()
    assert cli.cmd_lock(Namespace(name=None, version=None)) == 0
    assert relocks == []
    assert (tmp_path / "uv.lock").read_bytes() == before
    assert "already current" in capsys.readouterr().out

    stale.append(True)
    assert cli.cmd_lock(Namespace(name=None, version=None)) == 0
    # Relocking never syncs an environment; activation owns that.
    assert relocks == [(tmp_path, {"explicit": True})]
    out = capsys.readouterr().out
    assert "out of date" in out and "✗" not in out
    assert "activate" in out
    # beta is new and outside [all]; alpha was already locked and is in [all].
    assert "all,beta" in out and "alpha" not in out
