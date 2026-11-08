"""`hermes pm sync` refreshes the committed workspace snapshot without re-resolving (#122425)."""
from __future__ import annotations

import argparse

from pm import cli
from pm.package import InstallError


def test_pm_sync_wires_to_cmd_sync(monkeypatch, capsys):
    seen = {}

    def fake_sync(root):
        seen["root"] = root
        return root / "workspace"

    monkeypatch.setattr("pm.workspace.sync_sources", fake_sync)
    monkeypatch.setattr("pm.runtime.is_runtime", lambda: True)
    assert cli.main(["sync"]) == 0
    from pm.paths import repo_root

    assert seen["root"] == repo_root()
    assert "workspace snapshot refreshed" in capsys.readouterr().out


def test_cmd_sync_failure_names_the_remedy(monkeypatch, capsys):
    def boom(_root):
        raise InstallError("venv", "no dependency environment is committed for this install",
                           "run `hermes pm install`")

    monkeypatch.setattr("pm.workspace.sync_sources", boom)
    assert cli.cmd_sync(argparse.Namespace()) == 1
    assert "hermes pm install" in capsys.readouterr().out


def test_pm_sync_from_managed_venv_uses_owning_checkout(tmp_path, monkeypatch):
    """The managed venv's frozen workspace is not the source to refresh (#122425)."""
    import sys

    from pm.environments import install_key

    checkout = tmp_path / "checkout"
    (checkout / "hermes_cli").mkdir(parents=True)
    (checkout / "hermes_cli" / "main.py").write_text("", encoding="utf-8")
    state = tmp_path / "installs" / install_key(checkout)
    (state / "inputs").mkdir(parents=True)
    (state / "inputs" / ".project-root").write_text(str(checkout), encoding="utf-8")
    generation = state / "environments" / "gen"
    venv, workspace = generation / "venv", generation / "workspace"
    venv.mkdir(parents=True)
    workspace.mkdir()

    # The PM command re-execs through a separate runtime, not the app venv.
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "pm-runtime"))
    monkeypatch.setattr(sys, "base_prefix", str(tmp_path / "system"))
    monkeypatch.setattr(cli, "repo_root", lambda: workspace)
    monkeypatch.setenv("PYTHONPATH", "")
    seen = []

    def fake_sync(root):
        seen.append(root)
        return workspace

    monkeypatch.setattr("pm.workspace.sync_sources", fake_sync)
    assert cli.cmd_sync(argparse.Namespace()) == 0
    assert seen == [checkout]
