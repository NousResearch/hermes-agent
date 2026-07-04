"""Tests for resolve_whatsapp_bridge_dir() — read-only install tree handling.

Regression coverage for #49561: in the Docker image the install tree
(/opt/hermes/scripts/whatsapp-bridge) is read-only, so `npm install` fails
with EACCES. The resolver must detect the read-only install dir and mirror the
bridge source into a writable HERMES_HOME location instead.
"""
import importlib
from pathlib import Path

import pytest

from gateway.platforms import whatsapp_common


def _seed_install_tree(install_bridge: Path) -> None:
    """Create a minimal fake bridge source tree."""
    install_bridge.mkdir(parents=True, exist_ok=True)
    (install_bridge / "bridge.js").write_text("// bridge\n")
    (install_bridge / "package.json").write_text('{"name": "whatsapp-bridge"}\n')


def test_writable_install_returns_install_dir(tmp_path, monkeypatch):
    """When the install tree is writable, the resolver returns it unchanged."""
    install_root = tmp_path / "install"
    install_bridge = install_root / "scripts" / "whatsapp-bridge"
    _seed_install_tree(install_bridge)

    hermes_home = tmp_path / "hermes_home"
    hermes_home.mkdir()

    # Point the resolver's two anchors at our temp dirs.
    monkeypatch.setattr(
        whatsapp_common, "__file__",
        str(install_root / "gateway" / "platforms" / "whatsapp_common.py"),
    )
    monkeypatch.setattr(
        "hermes_constants.get_hermes_home", lambda: hermes_home
    )

    resolved = whatsapp_common.resolve_whatsapp_bridge_dir()
    assert resolved == install_bridge
    # Nothing mirrored into HERMES_HOME.
    assert not (hermes_home / "scripts" / "whatsapp-bridge").exists()


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


def test_readonly_install_reuses_existing_mirror(tmp_path, monkeypatch):
    """If the HERMES_HOME mirror already exists, return it without re-copying."""
    install_root = tmp_path / "install"
    install_bridge = install_root / "scripts" / "whatsapp-bridge"
    _seed_install_tree(install_bridge)

    hermes_home = tmp_path / "hermes_home"
    mirror = hermes_home / "scripts" / "whatsapp-bridge"
    mirror.mkdir(parents=True)
    # A sentinel file proves the resolver returned the EXISTING mirror
    # rather than wiping/recopying it.
    (mirror / "node_modules").mkdir()
    (mirror / "node_modules" / "sentinel").write_text("keep me\n")

    monkeypatch.setattr(
        whatsapp_common, "__file__",
        str(install_root / "gateway" / "platforms" / "whatsapp_common.py"),
    )
    monkeypatch.setattr(
        "hermes_constants.get_hermes_home", lambda: hermes_home
    )

    _real_touch = Path.touch

    def _fake_touch(self, *a, **kw):
        if self.name == ".write_test" and install_bridge in self.parents:
            raise PermissionError("read-only install tree")
        return _real_touch(self, *a, **kw)

    monkeypatch.setattr(Path, "touch", _fake_touch)

    resolved = whatsapp_common.resolve_whatsapp_bridge_dir()

    assert resolved == mirror
    # Existing node_modules left intact (no destructive re-copy).
    assert (mirror / "node_modules" / "sentinel").read_text() == "keep me\n"
