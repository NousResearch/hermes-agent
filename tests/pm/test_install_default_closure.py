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
    calls = {"names": None, "verified": None, "sync_extras": None, "activated": [], "fail": set()}

    def fake_install_names(names, target=None, *, verify=True):
        # Accumulate: optional defaults install in their own call after the
        # required closure.
        calls["names"] = [*(calls["names"] or []), *names]
        calls["verified"] = verify
        return len(calls["fail"] & set(names))

    def fake_sync_venv(extras=None, **kwargs):
        calls["sync_extras"] = list(extras or [])
        return None

    def fake_activate(**kwargs):
        calls["activated"].append(kwargs)
        return []

    monkeypatch.setattr(pm.cli, "_install_names", fake_install_names)
    monkeypatch.setattr(importlib.import_module("pm.install"), "sync_venv", fake_sync_venv)
    monkeypatch.setattr(importlib.import_module("pm.install"), "activate", fake_activate)
    return calls


def test_default_closure_includes_the_boot_interpreter(install_spy):
    assert pm.cli.cmd_install(argparse.Namespace(names=None, tools_only=False)) == 0
    assert "python" in install_spy["names"]
    assert "venv" not in install_spy["names"]
    assert all(not pm.cli.get_package(name).internal for name in install_spy["names"])
    assert install_spy["sync_extras"] == ["all"]
    assert install_spy["activated"] == [{"allow_incomplete": True}]


@pytest.mark.parametrize("names", [["npm", "ripgrep"], ["dmgbuild"]])
def test_explicit_names_pass_through_untouched(install_spy, names) -> None:
    assert pm.cli.cmd_install(argparse.Namespace(names=names, tools_only=False)) == 0
    assert install_spy["names"] == names
    assert install_spy["sync_extras"] is None
    assert install_spy["activated"] == []


def test_tools_only_publishes_tools_and_stops_before_the_venv(install_spy):
    assert pm.cli.cmd_install(argparse.Namespace(names=None, extra=[], target=None, tools_only=True)) == 0
    assert "python" in install_spy["names"]
    assert "venv" not in install_spy["names"]
    assert install_spy["sync_extras"] is None
    assert install_spy["activated"] == [{"allow_incomplete": True}]


def test_trust_recorded_skips_the_byte_check_and_still_syncs(install_spy):
    assert pm.cli.cmd_install(argparse.Namespace(
        names=None, extra=[], target=None, tools_only=False, trust_recorded=True)) == 0
    assert "python" in install_spy["names"]
    assert install_spy["verified"] is False
    assert install_spy["sync_extras"] == ["all"]


@pytest.mark.parametrize("kwargs, message", [
    ({"names": ["ripgrep"], "extra": [], "target": None, "tools_only": False, "trust_recorded": True},
     "--trust-recorded"),
    ({"names": None, "extra": ["anthropic"], "target": None, "tools_only": False, "trust_recorded": True},
     "--trust-recorded"),
    ({"names": None, "extra": [], "target": "linux-x64", "tools_only": False, "trust_recorded": True},
     "--target"),
])
def test_trust_recorded_refuses_a_narrowed_install(install_spy, capsys, kwargs, message):
    assert pm.cli.cmd_install(argparse.Namespace(**kwargs)) == 1
    assert message in capsys.readouterr().out
    assert install_spy["names"] is None
    assert install_spy["sync_extras"] is None


def test_a_missing_tool_blocks_the_venv_sync(install_spy, monkeypatch, capsys):
    monkeypatch.setattr(
        importlib.import_module("pm.install"), "activate",
        lambda **kwargs: ["git: not installed or outdated"],
    )
    assert pm.cli.cmd_install(argparse.Namespace(names=None, tools_only=False)) == 1
    assert install_spy["sync_extras"] is None
    assert "git: not installed or outdated" in capsys.readouterr().out


def _bare_install(install_spy, **kwargs) -> list[str]:
    install_spy["names"] = None
    assert pm.cli.cmd_install(argparse.Namespace(names=None, tools_only=False, **kwargs)) == 0
    return install_spy["names"]


def test_browser_tools_are_a_default_the_user_can_decline_and_restore(install_spy):
    assert "agent-browser" in _bare_install(install_spy)
    assert "agent-browser" not in _bare_install(install_spy, without=["agent-browser"])
    # The opt-out is recorded: a later bare install (the updater's shape) keeps it.
    assert "agent-browser" not in _bare_install(install_spy)
    install_spy["names"] = None
    assert pm.cli.cmd_install(argparse.Namespace(names=["agent-browser"], tools_only=False)) == 0
    assert "agent-browser" in _bare_install(install_spy)


def test_a_failed_default_download_does_not_fail_the_install(install_spy, capsys):
    install_spy["fail"] = {"agent-browser"}
    assert "agent-browser" in _bare_install(install_spy)
    assert install_spy["sync_extras"] == ["all"]
    assert "hermes pm install agent-browser" in capsys.readouterr().out


def test_termux_has_no_pm_browser_default():
    from pm.defaults import default_packages

    names = pm.cli._lockfile().names()
    assert "agent-browser" in default_packages(names, target="linux-x64", declined_names=frozenset())
    assert default_packages(names, target="linux-arm64-bionic", declined_names=frozenset()) == []


def test_without_refuses_a_required_package(install_spy, capsys):
    assert pm.cli.cmd_install(argparse.Namespace(names=None, tools_only=False, without=["git"])) == 1
    assert "--without accepts only" in capsys.readouterr().out
    assert install_spy["names"] is None
