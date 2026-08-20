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


def test_packaged_bridge_env_is_used_by_shared_resolver(tmp_path, monkeypatch):
    """Nix/Homebrew wrapper paths feed every WhatsApp caller centrally."""
    packaged_bridge = tmp_path / "store" / "whatsapp-bridge"
    _seed_install_tree(packaged_bridge)
    hermes_home = tmp_path / "hermes_home"
    hermes_home.mkdir()

    monkeypatch.setenv("HERMES_WHATSAPP_BRIDGE_DIR", str(packaged_bridge))
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: hermes_home)

    resolved = whatsapp_common.resolve_whatsapp_bridge_dir()

    assert resolved == packaged_bridge
    assert not (hermes_home / "scripts" / "whatsapp-bridge").exists()


def test_readonly_packaged_bridge_env_mirrors_to_hermes_home(tmp_path, monkeypatch):
    packaged_bridge = tmp_path / "nix-store" / "whatsapp-bridge"
    _seed_install_tree(packaged_bridge)
    hermes_home = tmp_path / "hermes_home"
    hermes_home.mkdir()

    monkeypatch.setenv("HERMES_WHATSAPP_BRIDGE_DIR", str(packaged_bridge))
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: hermes_home)
    real_touch = Path.touch

    def fake_touch(self, *args, **kwargs):
        if self.name == ".write_test" and packaged_bridge in self.parents:
            raise PermissionError("read-only Nix store")
        return real_touch(self, *args, **kwargs)

    monkeypatch.setattr(Path, "touch", fake_touch)

    resolved = whatsapp_common.resolve_whatsapp_bridge_dir()

    expected = hermes_home / "scripts" / "whatsapp-bridge"
    assert resolved == expected
    assert (expected / "bridge.js").read_text() == "// bridge\n"


def test_readonly_packaged_source_mirror_is_writable(tmp_path, monkeypatch, request):
    """A read-only packaged source mirrors into a WRITABLE tree.

    Regression for #15336: Nix store directories are ``0555`` and files
    ``0444``. ``shutil.copytree`` preserves those bits, so an unpatched mirror
    is itself read-only and the bridge's ``npm install`` (which must create
    ``node_modules`` inside the mirror) fails with EACCES. The resolver must
    restore owner write on the mirror.
    """
    import os
    import stat as _stat

    packaged_bridge = tmp_path / "nix-store" / "whatsapp-bridge"
    _seed_install_tree(packaged_bridge)
    (packaged_bridge / "lib").mkdir()
    (packaged_bridge / "lib" / "allowlist.js").write_text("// allowlist\n")

    # Make the packaged source read-only like the Nix store (deepest first).
    for entry in sorted(packaged_bridge.rglob("*"), reverse=True):
        entry.chmod(0o444 if entry.is_file() else 0o555)
    packaged_bridge.chmod(0o555)
    # Restore write bits so pytest can clean tmp_path regardless of pytest ver.
    request.addfinalizer(
        lambda: [
            p.chmod(0o755) for p in [packaged_bridge, *packaged_bridge.rglob("*")]
        ]
    )

    hermes_home = tmp_path / "hermes_home"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_WHATSAPP_BRIDGE_DIR", str(packaged_bridge))
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: hermes_home)

    # Force the read-only branch deterministically even when tests run as root
    # (root bypasses the 0555 write probe in Docker/CI).
    real_touch = Path.touch

    def fake_touch(self, *args, **kwargs):
        if self.name == ".write_test" and packaged_bridge in self.parents:
            raise PermissionError("read-only Nix store")
        return real_touch(self, *args, **kwargs)

    monkeypatch.setattr(Path, "touch", fake_touch)

    resolved = whatsapp_common.resolve_whatsapp_bridge_dir()

    expected = hermes_home / "scripts" / "whatsapp-bridge"
    assert resolved == expected
    assert (expected / "bridge.js").read_text() == "// bridge\n"
    # The mirror and its contents must be owner-writable so npm can install.
    assert os.stat(expected).st_mode & _stat.S_IWUSR, "mirror dir not writable"
    assert os.stat(expected / "lib").st_mode & _stat.S_IWUSR, "nested dir not writable"
    assert os.stat(expected / "bridge.js").st_mode & _stat.S_IWUSR, "mirror file not writable"


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


def test_readonly_existing_mirror_is_healed_on_reuse(tmp_path, monkeypatch, request):
    """A stale read-only mirror is healed in place on reuse, deps preserved.

    Regression for #15336: a mirror created before this permission fix (or by an
    interrupted copy) is itself read-only (dirs ``0555`` / files ``0444``). The
    resolver's reuse path must restore owner write so ``npm install`` can run,
    without re-copying over an already-installed ``node_modules``.
    """
    import os
    import stat as _stat

    packaged_bridge = tmp_path / "nix-store" / "whatsapp-bridge"
    _seed_install_tree(packaged_bridge)

    hermes_home = tmp_path / "hermes_home"
    mirror = hermes_home / "scripts" / "whatsapp-bridge"
    _seed_install_tree(mirror)
    # Installed deps that must survive (no destructive re-copy).
    (mirror / "node_modules").mkdir()
    (mirror / "node_modules" / "sentinel").write_text("keep me\n")

    # Make the mirror read-only like a pre-fix Nix mirror; node_modules stays
    # writable (npm created it).
    (mirror / "bridge.js").chmod(0o444)
    (mirror / "package.json").chmod(0o444)
    mirror.chmod(0o555)
    request.addfinalizer(
        lambda: [p.chmod(0o755) for p in [mirror, *mirror.rglob("*")]]
    )

    monkeypatch.setenv("HERMES_WHATSAPP_BRIDGE_DIR", str(packaged_bridge))
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: hermes_home)

    # Force the read-only (install) branch deterministically even under root.
    real_touch = Path.touch

    def fake_touch(self, *args, **kwargs):
        if self.name == ".write_test" and packaged_bridge in self.parents:
            raise PermissionError("read-only Nix store")
        return real_touch(self, *args, **kwargs)

    monkeypatch.setattr(Path, "touch", fake_touch)

    resolved = whatsapp_common.resolve_whatsapp_bridge_dir()

    assert resolved == mirror
    # Deps preserved (reuse, not re-copy).
    assert (mirror / "node_modules" / "sentinel").read_text() == "keep me\n"
    # Mirror healed to writable so npm can run.
    assert os.stat(mirror).st_mode & _stat.S_IWUSR, "mirror dir not healed"
    assert os.stat(mirror / "bridge.js").st_mode & _stat.S_IWUSR, "mirror file not healed"


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
