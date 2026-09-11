"""pm.extras: anchor availability, ensure_import, ensure_and_bind, and the
spec→extra install shim. Network-free — sync_venv is stubbed at the client
seam used by extras; the engine and worker have separate transaction tests."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

import pm
import pm.client as client
import pm.extras as extras


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


def test_sync_refuses_python_gated_extra_before_touching_environment(monkeypatch, tmp_path):
    import importlib
    from pathlib import Path
    from packaging.markers import default_environment

    engine = importlib.import_module("pm.ensure")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    version = default_environment()["python_full_version"]
    monkeypatch.setattr(extras, "_PLATFORM_GATES", {
        "unavailable-engine": f"python_full_version < '{version}'",
    })
    # Installed caller anchors cannot make a new managed graph compatible.
    monkeypatch.setitem(sys.modules, "unavailable_engine", SimpleNamespace())
    assert extras.extra_supported("unavailable-engine")
    assert not extras.extra_supported("unavailable-engine", importable=lambda _: False)

    def refuse_environment_access(*args, **kwargs):
        pytest.fail("unsupported request reached dependency environment machinery")

    monkeypatch.setattr(engine, "get_package", refuse_environment_access)
    with pytest.raises(pm.InstallError, match="not supported by this Python/platform"):
        engine.sync_venv(["unavailable-engine"], explicit=True)


def test_declared_extra_gates_match_dependency_selection():
    import tomllib
    from pathlib import Path
    from packaging.markers import default_environment
    from packaging.requirements import Requirement

    root = Path(__file__).resolve().parents[2]
    metadata = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    optional = metadata["project"]["optional-dependencies"]
    targets = [
        ("linux", "Linux", "x86_64"), ("linux", "Linux", "aarch64"),
        ("darwin", "Darwin", "x86_64"), ("darwin", "Darwin", "arm64"),
        ("win32", "Windows", "AMD64"), ("win32", "Windows", "ARM64"),
    ]
    for system, platform_system, machine in targets:
        for python in ("3.12", "3.13", "3.14"):
            environment = {**default_environment(), "sys_platform": system,
                           "platform_system": platform_system, "platform_machine": machine,
                           "python_version": python, "python_full_version": python + ".0"}
            for extra in metadata["tool"]["hermes"]["extras-platforms"]:
                selected = any(req.marker is None or req.marker.evaluate(environment)
                               for req in map(Requirement, optional[extra]))
                assert extras.extra_supported(extra, environment=environment,
                                              importable=lambda _: False) == selected, (
                    extra, system, machine, python,
                )


@pytest.fixture
def synced(monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(client, "sync_venv", lambda x=None: calls.append(list(x or [])))
    return calls


def test_available_known_anchor_present():
    assert extras.available("web") is True  # fastapi ships in this venv


def test_available_missing_module():
    assert extras.available("no-such-extra-anywhere") is False


@pytest.mark.parametrize(("extra", "module"), [
    ("hindsight", "hindsight_client"),
    ("teams", "microsoft_teams.apps"),
])
def test_available_counts_sys_modules_fakes(monkeypatch, extra, module):
    monkeypatch.setitem(sys.modules, module, SimpleNamespace())
    assert extras.available(extra) is True


def test_google_readiness_requires_its_oauth_imports(monkeypatch):
    present = {"googleapiclient", "google.auth", "google_auth_httplib2"}
    monkeypatch.setattr(extras, "_importable", lambda name: name in present)
    assert not extras.available("google")
    present.add("google_auth_oauthlib.flow")
    assert extras.available("google")


def test_available_unknown_extra_uses_underscore_guess(monkeypatch):
    monkeypatch.setitem(sys.modules, "some_new_thing", SimpleNamespace())
    assert extras.available("some-new-thing") is True


def test_ensure_import_noop_when_available(monkeypatch, synced):
    monkeypatch.setitem(sys.modules, "fal_client", SimpleNamespace())
    extras.ensure_import("fal")
    assert synced == []


def test_ensure_import_syncs_when_missing(monkeypatch, synced, tmp_path):
    from pathlib import Path

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(extras, "available", lambda e: False)
    extras.ensure_import("fal")
    assert synced == [["fal"]]


def test_ensure_import_propagates_install_error(monkeypatch):
    def boom(x=None):
        raise pm.InstallError("venv", "lazy installs are disabled")

    monkeypatch.setattr(client, "sync_venv", boom)
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

    monkeypatch.setattr(client, "sync_venv", boom)
    monkeypatch.setattr(extras, "available", lambda e: False)
    target: dict = {}
    assert extras.ensure_and_bind("fal", lambda: {"X": 1}, target) is False
    assert target == {}


def test_ensure_and_bind_false_on_import_failure(monkeypatch, synced):
    monkeypatch.setattr(extras, "available", lambda e: True)

    def importer():
        raise ImportError("still broken")

    assert extras.ensure_and_bind("fal", importer, {}) is False


def test_package_exports_preserve_availability_and_noop_install(monkeypatch, synced):
    monkeypatch.setitem(sys.modules, "some_new_thing", SimpleNamespace())
    assert pm.available("some-new-thing") == extras.available("some-new-thing") is True
    assert pm.available("no-such-extra-anywhere") == extras.available("no-such-extra-anywhere") is False
    pm.ensure_import("some-new-thing")
    assert synced == []


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
