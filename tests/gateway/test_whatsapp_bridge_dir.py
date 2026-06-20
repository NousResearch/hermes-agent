"""Tests for writable WhatsApp bridge directory resolution (issue #49561).

In Docker the install tree (e.g. ``/opt/hermes``) is read-only for the runtime
user, so ``npm install`` inside ``<install>/scripts/whatsapp-bridge`` fails with
``EACCES``. ``resolve_whatsapp_bridge_dir()`` must fall back to a writable
``HERMES_HOME`` copy in that case, while leaving writable installs untouched.
"""

import os
from pathlib import Path

import pytest

from gateway.platforms import whatsapp_common as wc


def _make_source(tmp: Path) -> Path:
    src = tmp / "install" / "scripts" / "whatsapp-bridge"
    src.mkdir(parents=True)
    (src / "bridge.js").write_text("// bridge v1\n")
    (src / "package.json").write_text('{"name":"bridge","version":"1"}\n')
    (src / "package-lock.json").write_text("{}\n")
    (src / "allowlist.js").write_text("// allow\n")
    return src


def test_resolve_uses_install_dir_when_writable(tmp_path, monkeypatch):
    src = _make_source(tmp_path)
    monkeypatch.setattr(wc, "bridge_source_dir", lambda: src)

    # A real writable source dir: resolve returns it unchanged, no copy made.
    result = wc.resolve_whatsapp_bridge_dir()

    assert result == src


def test_resolve_falls_back_to_hermes_home_when_readonly(tmp_path, monkeypatch):
    src = _make_source(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(wc, "bridge_source_dir", lambda: src)
    monkeypatch.setattr(wc, "_dir_is_writable", lambda path: path != src)
    monkeypatch.setenv("HERMES_HOME", str(home))

    result = wc.resolve_whatsapp_bridge_dir()

    expected = home / "scripts" / "whatsapp-bridge"
    assert result == expected
    # Source files are mirrored so connect()/pairing find bridge.js + deps.
    assert (expected / "bridge.js").read_text() == "// bridge v1\n"
    assert (expected / "package.json").exists()
    assert (expected / "package-lock.json").exists()
    assert (expected / "allowlist.js").exists()


def test_sync_preserves_existing_node_modules(tmp_path):
    src = _make_source(tmp_path)
    target = tmp_path / "home" / "scripts" / "whatsapp-bridge"
    (target / "node_modules" / "baileys").mkdir(parents=True)
    (target / "node_modules" / "baileys" / "index.js").write_text("module\n")

    wc._sync_bridge_source(src, target)

    # node_modules is left intact; source files are copied in alongside it.
    assert (target / "node_modules" / "baileys" / "index.js").read_text() == "module\n"
    assert (target / "bridge.js").read_text() == "// bridge v1\n"


def test_sync_refreshes_changed_source(tmp_path):
    src = _make_source(tmp_path)
    target = tmp_path / "home" / "scripts" / "whatsapp-bridge"
    target.mkdir(parents=True)
    (target / "bridge.js").write_text("// bridge OLD\n")

    wc._sync_bridge_source(src, target)

    assert (target / "bridge.js").read_text() == "// bridge v1\n"


def test_dir_is_writable(tmp_path):
    writable = tmp_path / "w"
    writable.mkdir()
    assert wc._dir_is_writable(writable) is True

    # Non-existent leaf under a writable parent is treated as writable
    # (npm would create the leaf dir).
    assert wc._dir_is_writable(writable / "scripts" / "whatsapp-bridge") is True


@pytest.mark.skipif(os.geteuid() == 0 if hasattr(os, "geteuid") else True,
                    reason="root bypasses POSIX write permission bits")
def test_dir_is_writable_false_on_readonly(tmp_path):
    ro = tmp_path / "ro"
    ro.mkdir()
    os.chmod(ro, 0o500)
    try:
        assert wc._dir_is_writable(ro) is False
    finally:
        os.chmod(ro, 0o700)
