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
    (install_bridge / "bridge_auth.js").write_text("// auth helper\n")
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
    assert (expected / "bridge_auth.js").read_text() == "// auth helper\n"
    assert (expected / "package.json").exists()


def test_readonly_install_refreshes_existing_mirror_without_deleting_dependencies(
    tmp_path, monkeypatch
):
    install_root = tmp_path / "install"
    install_bridge = install_root / "scripts" / "whatsapp-bridge"
    _seed_install_tree(install_bridge)

    hermes_home = tmp_path / "hermes_home"
    existing = hermes_home / "scripts" / "whatsapp-bridge"
    existing.mkdir(parents=True)
    (existing / "bridge.js").write_text("// stale bridge\n")
    (existing / "removed-helper.js").write_text("// removed upstream\n")
    removed_dir = existing / "removed-source-dir"
    removed_dir.mkdir()
    (removed_dir / "old.js").write_text("// removed upstream\n")
    (existing / "file-to-dir").write_text("// old file\n")
    source_dir = install_bridge / "file-to-dir"
    source_dir.mkdir()
    (source_dir / "current.js").write_text("// current directory\n")
    mirror_dir = existing / "dir-to-file"
    mirror_dir.mkdir()
    (mirror_dir / "old.js").write_text("// old directory\n")
    (install_bridge / "dir-to-file").write_text("// current file\n")
    node_modules = existing / "node_modules"
    node_modules.mkdir()
    dependency_marker = node_modules / "keep-me"
    dependency_marker.write_text("installed")

    monkeypatch.setattr(
        whatsapp_common, "__file__",
        str(install_root / "gateway" / "platforms" / "whatsapp_common.py"),
    )
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: hermes_home)

    real_touch = Path.touch

    def fake_touch(self, *args, **kwargs):
        if self.name == ".write_test" and install_bridge in self.parents:
            raise PermissionError("read-only install tree")
        return real_touch(self, *args, **kwargs)

    monkeypatch.setattr(Path, "touch", fake_touch)

    resolved = whatsapp_common.resolve_whatsapp_bridge_dir()

    assert resolved == existing
    assert (existing / "bridge.js").read_text() == "// bridge\n"
    assert (existing / "bridge_auth.js").read_text() == "// auth helper\n"
    assert not (existing / "removed-helper.js").exists()
    assert not removed_dir.exists()
    assert (existing / "file-to-dir" / "current.js").read_text() == (
        "// current directory\n"
    )
    assert (existing / "dir-to-file").read_text() == "// current file\n"
    assert dependency_marker.read_text() == "installed"


