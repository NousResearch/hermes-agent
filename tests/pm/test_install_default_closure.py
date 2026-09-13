"""The default `pm install` closure must stage the interpreter boot requires.

A fresh source install emits boot launchers (scripts/install.sh stage_path,
hermes_cli/_launchers.py) that exec the pm STORE interpreter. The `python`
package is marked optional (dev installs use their own venv; sealed bundles
adopt a shipped one), so the old default closure — every non-optional
lockfile package — skipped it and left `hermes` unbootable (audit C05).

Behavioral: drive cmd_install with the real lockfile/registry and stub
installers, asserting on the requested package names, not on script text.
"""

from __future__ import annotations

import argparse
import importlib

import pytest

import pm.cli
from pm.lock import Lockfile
from pm.paths import repo_root
from pm.registry import get_package


@pytest.fixture()
def install_spy(monkeypatch):
    calls = {"names": None, "sync_extras": None}

    def fake_install_names(names, target=None):
        calls["names"] = list(names)
        return 0

    def fake_sync_venv(extras=None, **kwargs):
        calls["sync_extras"] = list(extras or [])
        return None

    monkeypatch.setattr(pm.cli, "_install_names", fake_install_names)
    monkeypatch.setattr(importlib.import_module("pm.ensure"), "sync_venv", fake_sync_venv)
    return calls


def test_default_closure_includes_the_boot_interpreter(install_spy) -> None:
    assert pm.cli.cmd_install(argparse.Namespace(names=None)) == 0

    lockfile = Lockfile(repo_root() / "pm" / "lock.json")
    names = install_spy["names"]
    # The launchers' interpreter must be requested...
    assert "python" in names
    # ...through the existing authorities: it is pinned in pm/lock.json and
    # defined in the package registry — no parallel hand-written list here.
    assert "python" in lockfile.names()
    get_package("python")
    # ...and the rest of the root closure is unchanged (non-optional packages).
    expected = {
        n for n in lockfile.names() if not get_package(n).optional
    } | {"python"}
    assert set(names) == expected


def test_default_closure_still_syncs_the_venv_with_all_extras(install_spy) -> None:
    assert pm.cli.cmd_install(argparse.Namespace(names=None)) == 0
    assert install_spy["sync_extras"] == ["all"]


def test_explicit_names_pass_through_untouched(install_spy) -> None:
    assert pm.cli.cmd_install(argparse.Namespace(names=["npm", "ripgrep"])) == 0
    assert install_spy["names"] == ["npm", "ripgrep"]
    assert install_spy["sync_extras"] is None
