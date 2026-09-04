"""pm.extras: anchor availability, ensure_import, ensure_and_bind, and the
spec→extra install shim. Network-free — sync_venv is always stubbed (via the
pm.ensure module object; the pm package re-exports the ensure() FUNCTION,
which shadows the submodule attribute for string-path monkeypatching)."""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest

import pm
import pm.extras as extras

ensure_mod = importlib.import_module("pm.ensure")


# ---- per-extra platform gates ([tool.hermes.extras-platforms]) ----


def test_extra_supported_ungated_extra_is_true():
    extras._PLATFORM_GATES = None
    assert extras.extra_supported("no-such-gate-for-this-one") is True


def test_extra_supported_gate_excludes_platform(monkeypatch):
    extras._PLATFORM_GATES = {"gated-extra": "sys_platform == 'linux'"}
    try:
        # On this Windows host the linux gate must read False — unless the
        # anchors happen to be installed (installed-override beats table).
        if sys.platform == "linux":
            pytest.skip("host is linux — the linux gate is inclusive here")
        assert extras.extra_supported("gated-extra") is False
    finally:
        extras._PLATFORM_GATES = None


def test_extra_supported_installed_override_beats_gate(monkeypatch):
    extras._PLATFORM_GATES = {"gated-extra": "sys_platform == 'linux'"}
    try:
        # anchors importable → supported even if the gate would exclude
        monkeypatch.setitem(sys.modules, "gated_extra", SimpleNamespace())
        assert extras.extra_supported("gated-extra") is True
    finally:
        extras._PLATFORM_GATES = None


def test_ensure_import_raises_on_gated_off_extra(monkeypatch, synced):
    extras._PLATFORM_GATES = {"gated-extra": "sys_platform == 'linux'"}
    try:
        if sys.platform == "linux":
            pytest.skip("host is linux — gate is inclusive here")
        monkeypatch.setattr(extras, "available", lambda e: False)
        with pytest.raises(pm.InstallError) as exc:
            extras.ensure_import("gated-extra")
        assert "not supported on this platform" in str(exc.value)
        assert synced == []  # never reached the venv sync
    finally:
        extras._PLATFORM_GATES = None


@pytest.fixture
def synced(monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(ensure_mod, "sync_venv", lambda x=None: calls.append(list(x or [])))
    return calls


def test_available_known_anchor_present():
    assert extras.available("web") is True  # fastapi ships in this venv


def test_available_missing_module():
    assert extras.available("no-such-extra-anywhere") is False


def test_available_counts_sys_modules_fakes(monkeypatch):
    monkeypatch.setitem(sys.modules, "hindsight", SimpleNamespace())
    assert extras.available("hindsight") is True


def test_available_unknown_extra_uses_underscore_guess(monkeypatch):
    monkeypatch.setitem(sys.modules, "some_new_thing", SimpleNamespace())
    assert extras.available("some-new-thing") is True


def test_ensure_import_noop_when_available(monkeypatch, synced):
    monkeypatch.setitem(sys.modules, "fal_client", SimpleNamespace())
    extras.ensure_import("fal")
    assert synced == []


def test_ensure_import_syncs_when_missing(monkeypatch, synced):
    monkeypatch.setattr(extras, "available", lambda e: False)
    extras.ensure_import("fal")
    assert synced == [["fal"]]


def test_ensure_import_propagates_install_error(monkeypatch):
    def boom(x=None):
        raise pm.InstallError("venv", "lazy installs are disabled")

    monkeypatch.setattr(ensure_mod, "sync_venv", boom)
    monkeypatch.setattr(extras, "available", lambda e: False)
    with pytest.raises(pm.InstallError):
        extras.ensure_import("fal")


def test_ensure_and_bind_binds_on_success(monkeypatch, synced):
    monkeypatch.setattr(extras, "available", lambda e: True)
    target: dict = {}
    ok = extras.ensure_and_bind("fal", lambda: {"NAME": 42}, target)
    assert ok is True and target["NAME"] == 42


def test_ensure_and_bind_false_on_install_failure(monkeypatch):
    def boom(x=None):
        raise pm.InstallError("venv", "nope")

    monkeypatch.setattr(ensure_mod, "sync_venv", boom)
    monkeypatch.setattr(extras, "available", lambda e: False)
    target: dict = {}
    assert extras.ensure_and_bind("fal", lambda: {"X": 1}, target) is False
    assert target == {}


def test_ensure_and_bind_false_on_import_failure(monkeypatch, synced):
    monkeypatch.setattr(extras, "available", lambda e: True)

    def importer():
        raise ImportError("still broken")

    assert extras.ensure_and_bind("fal", importer, {}) is False


def test_package_reexports():
    assert pm.available is extras.available
    assert pm.ensure_import is extras.ensure_import


def test_every_anchor_extra_exists_in_pyproject():
    """Contract: ANCHORS maps real pyproject extras (no orphaned names)."""
    import tomllib
    from pathlib import Path

    py = tomllib.loads(
        (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text(encoding="utf-8")
    )
    declared = set(py["project"]["optional-dependencies"])
    orphans = set(extras.ANCHORS) - declared
    assert not orphans, f"ANCHORS names extras pyproject does not declare: {sorted(orphans)}"
