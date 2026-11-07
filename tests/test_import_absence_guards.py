"""Optional-import guards: a missing target or parent package is absence; a failure inside it is not.

The guards compare ``ModuleNotFoundError.name`` with the module they import. A missing package above
the target (``gateway`` for ``gateway.relay.egress``) means the target is absent too; a missing
dependency the target itself imports means a broken install, which must surface.
"""
from __future__ import annotations

import sys

import pytest

from gateway.relay.egress import _is_missing_gateway_relay


@pytest.mark.parametrize("name", ["gateway", "gateway.relay", "gateway.relay.egress"])
def test_relay_guard_reads_a_missing_target_or_parent_as_absence(name):
    assert _is_missing_gateway_relay(ModuleNotFoundError(name=name)) is True


@pytest.mark.parametrize("exc", [
    ModuleNotFoundError(name="gateway.relay.egress.dependency"),
    ModuleNotFoundError(name="gateway.relay.egress_dependency"),
    ModuleNotFoundError(name="requests"),
    ModuleNotFoundError(name=None),
    ImportError(name="gateway.relay.egress"),
])
def test_relay_guard_reads_anything_else_as_a_fault(exc):
    assert _is_missing_gateway_relay(exc) is False


class _Missing:
    """Make importing ``target`` fail as if module ``name`` were not installed."""

    def __init__(self, target: str, name: str):
        self.target, self.name = target, name

    def find_spec(self, fullname, path=None, target=None):
        if fullname == self.target:
            raise ModuleNotFoundError(f"No module named {self.name!r}", name=self.name)


@pytest.mark.parametrize("missing,allowed", [("hermes_cli", True), ("hermes_cli.config", True), ("yaml", False)])
def test_lazy_install_policy_without_the_config_package(missing, allowed, monkeypatch):
    from pm.install import lazy_installs_allowed

    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
    monkeypatch.delitem(sys.modules, "hermes_cli.config", raising=False)
    monkeypatch.setattr(sys, "meta_path", [_Missing("hermes_cli.config", missing), *sys.meta_path])
    assert lazy_installs_allowed() is allowed
