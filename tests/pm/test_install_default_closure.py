"""The default `pm install` closure must stage the interpreter boot requires.

A fresh source install emits boot launchers (the source completion's
publish_launchers, hermes_cli/_launchers.py) that exec the pm STORE
interpreter. The `python`
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
    monkeypatch.setattr(importlib.import_module("pm.install"), "sync_venv", fake_sync_venv)
    return calls


def test_default_closure_includes_the_boot_interpreter(install_spy):
    assert pm.cli.cmd_install(argparse.Namespace(names=None)) == 0
    assert "python" in install_spy["names"]
    assert all(not pm.cli.get_package(name).internal for name in install_spy["names"])
    assert install_spy["sync_extras"] == ["all"]



@pytest.mark.parametrize("names", [["npm", "ripgrep"], ["dmgbuild"]])
def test_explicit_names_pass_through_untouched(install_spy, names) -> None:
    assert pm.cli.cmd_install(argparse.Namespace(names=names)) == 0
    assert install_spy["names"] == names
    assert install_spy["sync_extras"] is None
