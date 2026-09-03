"""WeRoll fleet: `hermes update` on the parked branch `fleet` is refused unless gated."""
import subprocess
from types import SimpleNamespace

import pytest

from hermes_cli import update_cmd


def _fake_git(branch):
    def run(cmd, **kw):
        return SimpleNamespace(stdout=branch + "\n", returncode=0)
    return run


def test_refuses_on_fleet(monkeypatch):
    monkeypatch.delenv("HERMES_FLEET_GATED_UPDATE", raising=False)
    monkeypatch.setattr(subprocess, "run", _fake_git("fleet"))
    with pytest.raises(SystemExit) as e:
        update_cmd._fleet_update_gate(SimpleNamespace())
    assert e.value.code == 1


def test_allows_when_gated(monkeypatch):
    monkeypatch.setenv("HERMES_FLEET_GATED_UPDATE", "1")
    monkeypatch.setattr(subprocess, "run", _fake_git("fleet"))
    update_cmd._fleet_update_gate(SimpleNamespace())  # no exit


def test_allows_off_fleet(monkeypatch):
    monkeypatch.delenv("HERMES_FLEET_GATED_UPDATE", raising=False)
    monkeypatch.setattr(subprocess, "run", _fake_git("main"))
    update_cmd._fleet_update_gate(SimpleNamespace())  # no exit
