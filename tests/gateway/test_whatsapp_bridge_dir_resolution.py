"""Tests for resolve_whatsapp_bridge_dir() — read-only install tree handling.

Regression coverage for #49561: in the Docker image the install tree
(/opt/hermes/scripts/whatsapp-bridge) is read-only, so `npm install` fails
with EACCES. The resolver must detect the read-only install dir and mirror the
bridge source into a writable HERMES_HOME location instead.
"""
import importlib
from pathlib import Path
from unittest.mock import patch

import pytest

from gateway.platforms import whatsapp_common


def _seed_install_tree(install_bridge: Path) -> None:
    """Create a minimal fake bridge source tree."""
    install_bridge.mkdir(parents=True, exist_ok=True)
    (install_bridge / "bridge.js").write_text("// bridge\n")
    (install_bridge / "package.json").write_text('{"name": "whatsapp-bridge"}\n')


def test_readonly_install_mirrors_to_hermes_home(tmp_path, monkeypatch):
    """A read-only install tree is mirrored into a writable HERMES_HOME."""
    install_root = tmp_path / "install"
    install_bridge = install_root / "scripts" / "whatsapp-bridge"
    _seed_install_tree(install_bridge)

    hermes_home = tmp_path / "hermes_home"
    hermes_home.mkdir()

    monkeypatch.setattr(
        whatsapp_common, "__file__",
        str(install_root / "gateway" / "platforms" / "whatsapp_common.py"),
    )
    monkeypatch.setattr(
        "hermes_constants.get_hermes_home", lambda: hermes_home
    )

    # Simulate a read-only install tree. chmod(0o555) is unreliable under
    # root (CI/Docker bypass permission bits), so force the write probe to
    # fail by raising on the .write_test touch for the install dir only.
    _real_touch = Path.touch

    def _fake_touch(self, *a, **kw):
        if self.name == ".write_test" and install_bridge in self.parents:
            raise PermissionError("read-only install tree")
        return _real_touch(self, *a, **kw)

    monkeypatch.setattr(Path, "touch", _fake_touch)

    resolved = whatsapp_common.resolve_whatsapp_bridge_dir()

    expected = hermes_home / "scripts" / "whatsapp-bridge"
    assert resolved == expected
    # Source was mirrored, not symlinked.
    assert (expected / "bridge.js").read_text() == "// bridge\n"
    assert (expected / "package.json").exists()


def test_readonly_upgrade_refreshes_source_and_preserves_runtime_state(
    tmp_path, monkeypatch
):
    """An existing writable mirror is upgraded before it can be selected."""
    install_root = tmp_path / "install"
    install_bridge = install_root / "scripts" / "whatsapp-bridge"
    _seed_install_tree(install_bridge)
    (install_bridge / "bridge.js").write_text("// authenticated IPC bridge\n")
    (install_bridge / "bridge_helpers.js").write_text("// IPC helpers\n")
    (install_bridge / "bridge-launcher.js").write_text("// auth launcher\n")
    # Runtime state and secrets are never part of the shipped-source contract.
    (install_bridge / ".bridge-token").write_text("must-not-be-mirrored")

    hermes_home = tmp_path / "hermes_home"
    mirror = hermes_home / "scripts" / "whatsapp-bridge"
    node_modules = mirror / "node_modules"
    node_modules.mkdir(parents=True)
    (mirror / "bridge.js").write_text("// old unauthenticated TCP bridge\n")
    (mirror / "package.json").write_text('{"name": "old-bridge"}\n')
    dependency_sentinel = node_modules / "preserved-package.js"
    dependency_sentinel.write_text("// installed dependency\n")
    runtime_state = mirror / "session-runtime.json"
    runtime_state.write_text('{"preserve": true}\n')

    monkeypatch.setattr(
        whatsapp_common,
        "__file__",
        str(install_root / "gateway" / "platforms" / "whatsapp_common.py"),
    )
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: hermes_home)
    _force_readonly_install(monkeypatch, install_bridge)

    resolved = whatsapp_common.resolve_whatsapp_bridge_dir()

    assert resolved == mirror
    assert (resolved / "bridge.js").read_text() == "// authenticated IPC bridge\n"
    assert (resolved / "bridge_helpers.js").read_text() == "// IPC helpers\n"
    assert (resolved / "bridge-launcher.js").read_text() == "// auth launcher\n"
    assert dependency_sentinel.read_text() == "// installed dependency\n"
    assert runtime_state.read_text() == '{"preserve": true}\n'
    assert not (resolved / ".bridge-token").exists()
    assert (resolved / ".hermes-source-manifest.json").is_file()

    # The default adapter path is resolved only after the refresh completes,
    # so startup cannot select the stale TCP implementation.
    from gateway.config import PlatformConfig
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter

    monkeypatch.setattr(WhatsAppAdapter, "_DEFAULT_BRIDGE_DIR", None)
    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={}))
    assert Path(adapter._bridge_script).read_text() == "// authenticated IPC bridge\n"


def test_failed_readonly_refresh_never_returns_partial_mirror(tmp_path, monkeypatch):
    """A failed atomic source update falls back instead of selecting stale code."""
    install_root = tmp_path / "install"
    install_bridge = install_root / "scripts" / "whatsapp-bridge"
    _seed_install_tree(install_bridge)
    (install_bridge / "bridge.js").write_text("// authenticated IPC bridge\n")

    hermes_home = tmp_path / "hermes_home"
    mirror = hermes_home / "scripts" / "whatsapp-bridge"
    mirror.mkdir(parents=True)
    (mirror / "bridge.js").write_text("// old unauthenticated TCP bridge\n")

    monkeypatch.setattr(
        whatsapp_common,
        "__file__",
        str(install_root / "gateway" / "platforms" / "whatsapp_common.py"),
    )
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: hermes_home)
    _force_readonly_install(monkeypatch, install_bridge)

    real_replace = whatsapp_common.os.replace

    def fail_manifest_commit(source, destination):
        if Path(destination).name == ".hermes-source-manifest.json":
            raise OSError("disk full")
        return real_replace(source, destination)

    with patch(
        "gateway.platforms.whatsapp_common.os.replace",
        side_effect=fail_manifest_commit,
    ):
        resolved = whatsapp_common.resolve_whatsapp_bridge_dir()

    assert resolved == install_bridge
    assert (mirror / "bridge.js").read_text() == "// authenticated IPC bridge\n"
    assert not (mirror / ".hermes-source-manifest.json").exists()


def test_readonly_refresh_refuses_symlink_mirror(tmp_path, monkeypatch):
    """Refresh never follows an existing mirror symlink to overwrite its target."""
    install_root = tmp_path / "install"
    install_bridge = install_root / "scripts" / "whatsapp-bridge"
    _seed_install_tree(install_bridge)
    hermes_home = tmp_path / "hermes_home"
    target = tmp_path / "operator-files"
    target.mkdir()
    mirror = hermes_home / "scripts" / "whatsapp-bridge"
    mirror.parent.mkdir(parents=True)
    mirror.symlink_to(target, target_is_directory=True)

    monkeypatch.setattr(
        whatsapp_common,
        "__file__",
        str(install_root / "gateway" / "platforms" / "whatsapp_common.py"),
    )
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: hermes_home)
    _force_readonly_install(monkeypatch, install_bridge)

    assert whatsapp_common.resolve_whatsapp_bridge_dir() == install_bridge
    assert list(target.iterdir()) == []


def _force_readonly_install(monkeypatch, install_bridge: Path) -> None:
    real_touch = Path.touch

    def fake_touch(self, *args, **kwargs):
        if self.name == ".write_test" and install_bridge in self.parents:
            raise PermissionError("read-only install tree")
        return real_touch(self, *args, **kwargs)

    monkeypatch.setattr(Path, "touch", fake_touch)
