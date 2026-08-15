"""Plugin directory scans must skip dunder dirs and survive OSError (#86996)."""

from __future__ import annotations

from pathlib import Path

from hermes_cli.plugins import PluginManager


def test_scan_skips_dunder_dirs_and_still_loads_real_plugin(tmp_path: Path):
    root = tmp_path / "plugins"
    demo = root / "demo"
    demo.mkdir(parents=True)
    (demo / "plugin.yaml").write_text("name: demo\nversion: 0.1.0\ndescription: x\n")
    (demo / "__pycache__").mkdir()
    pycache = root / "__pycache__"
    pycache.mkdir()
    pycache.chmod(0o000)
    try:
        found = PluginManager()._scan_directory(root, "user")
    finally:
        pycache.chmod(0o700)

    assert [manifest.name for manifest in found] == ["demo"]


def test_scan_survives_iterdir_oserror(tmp_path: Path, monkeypatch):
    root = tmp_path / "plugins"
    root.mkdir()

    def _boom(_self):
        raise PermissionError("unreadable plugin root")

    monkeypatch.setattr(Path, "iterdir", _boom, raising=False)
    found = PluginManager()._scan_directory(root, "user")
    assert found == []


def test_scan_skips_child_whose_is_dir_raises(tmp_path: Path, monkeypatch):
    root = tmp_path / "plugins"
    demo = root / "demo"
    demo.mkdir(parents=True)
    (demo / "plugin.yaml").write_text("name: demo\nversion: 0.1.0\ndescription: x\n")
    spooky = root / "spooky"
    spooky.mkdir()

    original_is_dir = Path.is_dir

    def _is_dir(self):
        if self.name == "spooky":
            raise PermissionError("stat failed")
        return original_is_dir(self)

    monkeypatch.setattr(Path, "is_dir", _is_dir)
    found = PluginManager()._scan_directory(root, "user")
    assert [manifest.name for manifest in found] == ["demo"]
