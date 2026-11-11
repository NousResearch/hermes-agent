"""Windows dependency setup must keep npm and its Node outside a sealed package."""

import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import hermes_constants
import pm
import pm.install
from pm import paths
from pm.lock import Facts, Lockfile
from pm.package import InstallError, Runner
from pm.packages import Nodejs, Npm
from pm.store import current_target, tree_digest

pytestmark = pytest.mark.platforms("windows")


@pytest.fixture
def packaged_tools(tmp_path, monkeypatch):
    shipped, writable = tmp_path / "payload/tools", tmp_path / "home/tools"
    shipped.mkdir(parents=True)
    (shipped.parent / "manifest.json").write_text("{}", encoding="utf-8")
    lock = Lockfile(tmp_path / "lock.json")
    artifact, target = "1" * 64, current_target()
    for package in (Nodejs(), Npm()):
        # Synthetic executable bytes exercise copying and resolution without being executed.
        monkeypatch.setattr(package, "verify", lambda entry, target: "")
        monkeypatch.setitem(
            importlib.import_module("pm.registry")._packages, package.name, package
        )
        lock.set_pin(
            package.name,
            "test",
            {target: {"url": "https://test.invalid/tool", "sha256": artifact}},
        )
        entry = shipped / package.store_entry("test", target)
        binary = package.binary(entry, target)
        assert binary is not None
        binary.parent.mkdir(parents=True)
        binary.write_bytes(b"synthetic " + package.name.encode())
        Facts(shipped / "facts.json").record(
            package.name,
            "test",
            entry.name,
            package.env(entry, target),
            shipped,
            target=target,
            artifacts=[artifact],
            digest=tree_digest(entry),
        )
    lock.save()
    monkeypatch.setattr(paths, "lockfile_path", lambda: lock.path)
    monkeypatch.setattr(paths, "store_root", lambda: shipped)
    monkeypatch.setattr(paths, "writable_store_root", lambda: writable)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(pm.install, "lazy_installs_allowed", lambda: True)
    monkeypatch.setattr(
        pm, "ensure", lambda name, **kwargs: Runner(name, pm.env_for(name))
    )
    return SimpleNamespace(shipped=shipped, writable=writable)


def test_read_only_context_does_not_copy_packaged_tools(packaged_tools):
    env = hermes_constants.with_hermes_node_path({"PATH": "ambient"})
    assert str(packaged_tools.shipped) in env["PATH"]
    assert not packaged_tools.writable.exists()


def test_subprocess_context_copies_both_pins_and_preserves_caller_env(packaged_tools):
    original = {"PATH": "ambient", "PRESERVE_THIS": "value"}
    env = hermes_constants.with_hermes_node_path(original, for_subprocess=True)
    directories = env["PATH"].split(os.pathsep)
    assert directories == [
        str(packaged_tools.writable / f"{name}-test-{current_target()}")
        for name in ("npm", "node")
    ] + ["ambient"]
    for name in ("npm", "node"):
        copied = Facts(packaged_tools.writable / "facts.json").get(name)
        shipped = Facts(packaged_tools.shipped / "facts.json").get(name)
        assert copied is not None and shipped is not None
        assert copied["digest"] == shipped["digest"]
        assert pm.installed_package(name).path.is_relative_to(packaged_tools.shipped)
    assert env["PRESERVE_THIS"] == "value"
    assert original == {"PATH": "ambient", "PRESERVE_THIS": "value"}


def test_copy_obeys_lazy_policy_and_explicit_setup_then_reuses_copies(
    packaged_tools, monkeypatch
):
    monkeypatch.setattr(pm.install, "lazy_installs_allowed", lambda: False)
    with pytest.raises(InstallError, match="lazy installs are disabled"):
        hermes_constants.with_hermes_node_path(for_subprocess=True)
    assert not packaged_tools.writable.exists()
    env = hermes_constants.with_hermes_node_path(for_subprocess=True, explicit=True)
    assert str(packaged_tools.shipped) not in env["PATH"]
    assert hermes_constants.with_hermes_node_path(for_subprocess=True) == env


@pytest.fixture
def setup_path(request, tmp_path, monkeypatch):
    bridge = tmp_path / "bridge"
    bridge.mkdir()
    (bridge / "package.json").write_text(
        json.dumps({"name": "synthetic-sidecar", "version": "1.0.0"}), encoding="utf-8"
    )
    name = request.param
    if name == "whatsapp_gateway":
        from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

        adapter = Mock(name="whatsapp", _set_fatal_error=Mock())
        return lambda: WhatsAppAdapter._ensure_bridge_deps(adapter, bridge)
    if name == "whatsapp_cli":
        from hermes_cli.main_platform_setup import _whatsapp_install_bridge

        return lambda: _whatsapp_install_bridge(bridge)
    if name == "whatsapp_web":
        from hermes_cli.web_routers.messaging import (
            _ensure_whatsapp_bridge_dependencies,
        )

        return lambda: _ensure_whatsapp_bridge_dependencies(bridge)
    if name.startswith("photon_"):
        module = importlib.import_module(
            "plugins.platforms.photon." + ("cli" if name == "photon_cli" else "adapter")
        )
        monkeypatch.setattr(module, "_sidecar_dir", lambda: bridge)
        return (
            module._install_sidecar
            if name == "photon_cli"
            else module._reinstall_sidecar_deps
        )
    from pm.workspace import install_node_sidecar

    return lambda: install_node_sidecar(bridge, explicit=True)


SETUP_PATHS = [
    "whatsapp_gateway",
    "whatsapp_cli",
    "whatsapp_web",
    "photon_gateway",
    "photon_cli",
    "plugin_sidecar",
]


@pytest.mark.parametrize("setup_path", SETUP_PATHS, indirect=True)
def test_setup_entry_points_use_writable_paired_context(
    packaged_tools, setup_path, monkeypatch
):
    calls = []

    def installed(command, **kwargs):
        expected_npm = Npm().binary(
            packaged_tools.writable / f"npm-test-{current_target()}", current_target()
        )
        assert command[0] == str(expected_npm)
        directories = kwargs["env"]["PATH"].split(os.pathsep)
        assert directories[:2] == [
            str(packaged_tools.writable / f"{name}-test-{current_target()}")
            for name in ("npm", "node")
        ]
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("subprocess.run", installed)
    setup_path()
    assert calls


@pytest.mark.parametrize("setup_path", SETUP_PATHS, indirect=True)
def test_setup_reports_copy_failure_before_launching_npm(
    packaged_tools, setup_path, monkeypatch, capsys, caplog
):
    from fastapi import HTTPException
    from pm import user_facing

    def refused(*args, **kwargs):
        raise InstallError("node", "synthetic copy refusal")

    monkeypatch.setattr(user_facing, "writable_copy", refused)
    run = Mock(side_effect=AssertionError("must not launch npm after copy refusal"))
    monkeypatch.setattr("subprocess.run", run)
    try:
        result = setup_path()
    except HTTPException as exc:
        result = exc.detail
    captured = capsys.readouterr()
    assert (
        "synthetic copy refusal"
        in str(result) + captured.out + captured.err + caplog.text
    )
    run.assert_not_called()
