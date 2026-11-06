"""``ImportError.name`` guards keep telling absence from breakage after their module is renamed.

These guards fall back only when the module they import is missing and re-raise any other
failure. A move rewrites ``import gateway.relay.egress`` and the matching string to
``hermes_agent.gateway.relay.egress``; when ``hermes_agent`` itself is missing, Python reports
``exc.name == "hermes_agent"``, a parent of the target. A guard that lists names by hand misses
that parent and turns absence into a crash, so every guard treats the target and each of its
parent packages as absence.
"""
from __future__ import annotations

import ast
import importlib
import inspect
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PREFIX = "hermes_relocated"

ENTRY_FILES = (
    "acp_adapter/entry.py", "agent/managed_worker.py", "batch_runner.py", "cli.py", "gateway/run.py",
    "hermes_cli/kanban_worker_client.py", "hermes_cli/main.py", "run_agent.py",
)


def _relocate(source: str, top: str) -> str:
    return re.sub(rf"(?<![\w.]){top}(?=[.\"'\s])", f"{PREFIX}.{top}", source)


def _entry_guard(path: str) -> str:
    source = (REPO_ROOT / path).read_text(encoding="utf-8")
    for node in ast.parse(source).body:
        if isinstance(node, ast.Try) and any(isinstance(s, ast.Import) and s.names[0].name == "hermes_bootstrap" for s in node.body):
            return ast.get_source_segment(source, node)
    raise AssertionError(f"{path}: no hermes_bootstrap guard")


def _relocated_function(module_name: str, function: str, top: str, **overrides):
    module = importlib.import_module(module_name)
    namespace = {**vars(module), **overrides}
    exec(compile(_relocate(inspect.getsource(getattr(module, function)), top), module.__file__, "exec"), namespace)
    return namespace[function]


def _backend_for():
    import hermes_cli.config_backend as backend
    relocated = _relocated_function("hermes_cli.config_backend", "_backend_for", "plugins",
                                    _BACKENDS_PACKAGE=f"{PREFIX}.{backend._BACKENDS_PACKAGE}")
    with pytest.raises(backend.ConfigBackendUnavailable):
        relocated("remote")


def _is_missing_gateway_relay():
    relocated = _relocated_function("gateway.relay.egress", "_is_missing_gateway_relay", "gateway")
    assert relocated(ModuleNotFoundError(name=PREFIX)) is True
    assert relocated(ModuleNotFoundError(name=f"{PREFIX}.gateway.relay.egress_dependency")) is False


def _authorize_relay_target():
    relocated = _relocated_function("tools.send_message_tool", "_authorize_relay_target", "gateway")
    assert relocated("telegram", "123") is None


FUNCTION_GUARDS = {
    "pm.store._native_machine": lambda: isinstance(
        _relocated_function("pm.store", "_native_machine", "hermes_platform")(), str),
    "pm.install.lazy_installs_allowed": lambda: _relocated_function(
        "pm.install", "lazy_installs_allowed", "hermes_cli")() is True,
    "hermes_cli.config._worker_config_snapshot": lambda: _relocated_function(
        "hermes_cli.config", "_worker_config_snapshot", "agent")() is None,
    "hermes_cli.__getattr__": lambda: isinstance(
        _relocated_function("hermes_cli", "__getattr__", "pm")("__version__"), str),
    "hermes_cli.config_backend._backend_for": lambda: _backend_for() is None,
    "gateway.relay.egress._is_missing_gateway_relay": lambda: _is_missing_gateway_relay() is None,
    "tools.send_message_tool._authorize_relay_target": lambda: _authorize_relay_target() is None,
}


@pytest.mark.parametrize("path", ENTRY_FILES)
def test_entry_bootstrap_guard_treats_a_missing_parent_package_as_absence(path):
    exec(compile(_relocate(_entry_guard(path), "hermes_bootstrap"), path, "exec"), {"__name__": "relocated_entry"})
    assert f"{PREFIX}.hermes_bootstrap" not in sys.modules


@pytest.mark.parametrize("site", sorted(FUNCTION_GUARDS))
def test_function_guard_treats_a_missing_parent_package_as_absence(site, monkeypatch):
    monkeypatch.delenv("HERMES_DISABLE_LAZY_INSTALLS", raising=False)
    assert FUNCTION_GUARDS[site]()


class _BrokenTarget:
    """Importing ``target`` fails on a dependency it needs: a broken install, not an absent module."""

    def __init__(self, target: str):
        self.target = target

    def find_spec(self, fullname, path=None, target=None):
        if fullname == self.target:
            raise ModuleNotFoundError("No module named 'missing_dependency'", name="missing_dependency")


@pytest.mark.parametrize("path", ENTRY_FILES)
def test_entry_bootstrap_guard_reraises_when_the_bootstrap_cannot_load(path, monkeypatch):
    monkeypatch.delitem(sys.modules, "hermes_bootstrap", raising=False)
    monkeypatch.setattr(sys, "meta_path", [_BrokenTarget("hermes_bootstrap"), *sys.meta_path])
    with pytest.raises(ModuleNotFoundError) as raised:
        exec(compile(_entry_guard(path), path, "exec"), {"__name__": "broken_entry"})
    assert raised.value.name == "missing_dependency"
