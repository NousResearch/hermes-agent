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


@pytest.fixture
def readonly_install(tmp_path, monkeypatch):
    """A read-only install tree plus a writable HERMES_HOME; yields (install_bridge, mirror)."""
    install_root = tmp_path / "install"
    install_bridge = install_root / "scripts" / "whatsapp-bridge"
    _seed_install_tree(install_bridge)
    hermes_home = tmp_path / "hermes_home"
    hermes_home.mkdir()
    monkeypatch.setattr(whatsapp_common, "__file__", str(install_root / "gateway" / "platforms" / "whatsapp_common.py"))
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: hermes_home)
    _real_touch = Path.touch

    def _fake_touch(self, *a, **kw):
        if self.name == ".write_test" and install_bridge in self.parents:
            raise PermissionError("read-only install tree")
        return _real_touch(self, *a, **kw)

    monkeypatch.setattr(Path, "touch", _fake_touch)
    return install_bridge, hermes_home / "scripts" / "whatsapp-bridge"


def test_upgraded_install_tree_reaches_existing_mirror(readonly_install):
    """An image upgrade must update the mirror the bridge actually runs from."""
    install_bridge, mirror = readonly_install
    assert whatsapp_common.resolve_whatsapp_bridge_dir() == mirror
    (install_bridge / "bridge.js").write_text("// bridge v2\n")
    (install_bridge / "new_helper.js").write_text("// new\n")

    assert whatsapp_common.resolve_whatsapp_bridge_dir() == mirror
    assert (mirror / "bridge.js").read_text() == "// bridge v2\n"
    assert (mirror / "new_helper.js").read_text() == "// new\n"
    assert not list(mirror.glob("*.local-*"))  # an upstream change is not a local edit


def test_node_modules_left_alone(readonly_install):
    install_bridge, mirror = readonly_install
    whatsapp_common.resolve_whatsapp_bridge_dir()
    (mirror / "node_modules" / "pkg").mkdir(parents=True)
    (mirror / "node_modules" / "pkg" / "index.js").write_text("installed\n")
    (install_bridge / "bridge.js").write_text("// bridge v2\n")

    whatsapp_common.resolve_whatsapp_bridge_dir()
    assert (mirror / "node_modules" / "pkg" / "index.js").read_text() == "installed\n"


def test_locally_edited_mirror_file_is_kept(readonly_install):
    install_bridge, mirror = readonly_install
    whatsapp_common.resolve_whatsapp_bridge_dir()
    (mirror / "bridge.js").write_text("// hand-patched\n")
    (install_bridge / "bridge.js").write_text("// bridge v2\n")

    whatsapp_common.resolve_whatsapp_bridge_dir()
    assert (mirror / "bridge.js").read_text() == "// bridge v2\n"
    kept = list(mirror.glob("bridge.js.local-*"))
    assert len(kept) == 1 and kept[0].read_text() == "// hand-patched\n"


def test_legacy_mirror_without_manifest_keeps_differing_files(readonly_install):
    install_bridge, mirror = readonly_install
    mirror.mkdir(parents=True)
    (mirror / "bridge.js").write_text("// first-install copy\n")
    (mirror / "package.json").write_text('{"name": "whatsapp-bridge"}\n')

    assert whatsapp_common.resolve_whatsapp_bridge_dir() == mirror
    assert (mirror / "bridge.js").read_text() == "// bridge\n"
    assert len(list(mirror.glob("bridge.js.local-*"))) == 1
    assert not list(mirror.glob("package.json.local-*"))  # identical files are not touched


def test_refresh_failure_falls_back_to_existing_mirror(readonly_install, monkeypatch):
    install_bridge, mirror = readonly_install
    whatsapp_common.resolve_whatsapp_bridge_dir()

    def _boom(*a, **kw):
        raise OSError("disk full")

    monkeypatch.setattr(whatsapp_common, "_refresh_bridge_mirror", _boom)
    assert whatsapp_common.resolve_whatsapp_bridge_dir() == mirror


