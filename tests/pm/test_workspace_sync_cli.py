"""`hermes pm sync` refreshes the committed workspace snapshot without re-resolving (#122425)."""
from __future__ import annotations

import argparse

import pm.cli as cli
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
