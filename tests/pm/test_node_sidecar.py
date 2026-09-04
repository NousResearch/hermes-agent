"""Tests: install_node_sidecar — the npm ci executor for plugin package.json
sidecars (plugin-deps plan §B item 2, wired). Hermetic: runner + binary
injected, lazy-gate patched."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import pm.workspace as ws


@pytest.fixture
def lazy_on(monkeypatch):
    import sys

    if "pm.ensure" not in sys.modules:
        import importlib

        importlib.import_module("pm.ensure")
    ensure_mod = sys.modules["pm.ensure"]
    monkeypatch.setattr(ensure_mod, "lazy_installs_allowed", lambda: True)


def _plug(tmp_path: Path, with_lock: bool = False) -> Path:
    plug = tmp_path / "node-plug"
    plug.mkdir()
    (plug / "package.json").write_text('{"name": "node-plug"}\n', encoding="utf-8")
    if with_lock:
        (plug / "package-lock.json").write_text("{}\n", encoding="utf-8")
    return plug


def test_no_package_json_is_a_noop(tmp_path, lazy_on):
    plug = tmp_path / "plain"
    plug.mkdir()
    assert ws.install_node_sidecar(plug, npm_bin="npm") is None


def test_ci_when_lockfile_present(tmp_path, lazy_on):
    plug = _plug(tmp_path, with_lock=True)
    calls = []

    def runner(cmd, **k):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    assert ws.install_node_sidecar(plug, npm_bin="npm", runner=runner) is None
    assert calls == [["npm", "ci", "--no-audit", "--no-fund"]]


def test_install_without_lockfile(tmp_path, lazy_on):
    plug = _plug(tmp_path)  # no package-lock.json
    calls = []

    def runner(cmd, **k):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    assert ws.install_node_sidecar(plug, npm_bin="npm", runner=runner) is None
    assert calls == [["npm", "install", "--no-audit", "--no-fund"]]


def test_lazy_off_refuses(tmp_path, monkeypatch):
    import sys

    if "pm.ensure" not in sys.modules:
        import importlib

        importlib.import_module("pm.ensure")
    ensure_mod = sys.modules["pm.ensure"]
    monkeypatch.setattr(ensure_mod, "lazy_installs_allowed", lambda: False)
    plug = _plug(tmp_path)
    reason = ws.install_node_sidecar(plug, npm_bin="npm")
    assert reason and "disabled" in reason


def test_npm_failure_returns_reason_not_raise(tmp_path, lazy_on):
    plug = _plug(tmp_path)

    def runner(cmd, **k):
        return SimpleNamespace(returncode=1, stdout="", stderr="ERESOLVE unable to resolve dependency tree")

    reason = ws.install_node_sidecar(plug, npm_bin="npm", runner=runner)
    assert "exited 1" in reason and "ERESOLVE" in reason


def test_runner_explosion_is_a_reason(tmp_path, lazy_on):
    plug = _plug(tmp_path)

    def runner(cmd, **k):
        raise OSError("spawn denied")

    reason = ws.install_node_sidecar(plug, npm_bin="npm", runner=runner)
    assert "failed to run" in reason
