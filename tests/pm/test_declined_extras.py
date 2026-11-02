"""`hermes pm install --without EXTRA` drops a recorded extra, and nothing that carries a
selection forward (recorded extras, the main-era venv scan, a lazy first use) re-adds it
until the user asks for it again with `--extra EXTRA`."""

from __future__ import annotations

import argparse
import importlib
from types import SimpleNamespace

import pytest

import pm.cli
from pm import extras, install
from pm.package import InstallError
from pm.paths import repo_root


@pytest.fixture()
def synced(monkeypatch):
    calls = []
    monkeypatch.setattr(pm.cli, "_install_names", lambda names, target=None, *, verify=True: 0)
    monkeypatch.setattr(importlib.import_module("pm.install"), "activate", lambda **kwargs: [])
    monkeypatch.setattr(importlib.import_module("pm.install"), "sync_venv",
                        lambda extras=None, **kwargs: calls.append(list(extras or [])))
    return calls


def _next_selection(recorded, requested=None):
    package = SimpleNamespace(project_root=repo_root, expected_stamp=lambda enabled, **_: "stamp")
    return install._target_selection(package, {"extras": recorded}, extras=requested, inputs={},
                                     repair=False, shipped=None, frozen=None)[0]


def test_without_drops_a_recorded_extra_from_every_later_selection(synced, tmp_path):
    assert pm.cli.cmd_install(argparse.Namespace(names=None, tools_only=False, without=["matrix"])) == 0

    # The update's sync carries recorded extras forward; the dropped one stays out, others stay.
    assert _next_selection(["all", "fal", "matrix"]) == ["all", "fal"]
    # A data root without facts re-scans the main-era venv, which is never deleted.
    site = tmp_path / "venv" / "lib" / "python3.11" / "site-packages"
    for anchor in (*extras._anchors("matrix"), "fal_client"):
        (site / anchor).mkdir(parents=True)
    assert extras.legacy_selection(tmp_path) == ["all", "fal", "matrix"]
    extras.record_declined_extras(add=["matrix"], project_root=tmp_path)
    assert extras.legacy_selection(tmp_path) == ["all", "fal"]
    # A lazy first use refuses instead of silently reinstalling it.
    package = SimpleNamespace(project_root=repo_root)
    with pytest.raises(InstallError, match="pm install --extra matrix"):
        install._refuse_declined(package, ["matrix"], explicit=False)


def test_extra_opts_a_dropped_extra_back_in():
    extras.record_declined_extras(add=["matrix"])
    assert _next_selection(["all"], requested=["matrix"]) == ["all", "matrix"]
    install._refuse_declined(SimpleNamespace(project_root=repo_root), ["matrix"], explicit=True)


def test_without_refuses_an_extra_the_default_sync_selects_anyway(synced, capsys):
    assert pm.cli.cmd_install(argparse.Namespace(names=None, tools_only=False, without=["web"])) == 1
    assert "extras outside [all]" in capsys.readouterr().out
    assert synced == []
