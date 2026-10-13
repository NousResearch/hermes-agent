"""Plugin directory scans must skip dunder and dot-prefixed dirs and survive OSError (#86996, #137974)."""

from __future__ import annotations

from pathlib import Path

from hermes_cli.plugins import PluginManager


def test_scan_skips_dunder_dirs_and_still_loads_real_plugin(tmp_path: Path):
    root = tmp_path / "plugins"
    demo = root / "demo"
    demo.mkdir(parents=True)
    (demo / "plugin.yaml").write_text("name: demo\nversion: 0.1.0\ndescription: x\n")
    (demo / "__pycache__").mkdir()
    (root / "__pycache__").mkdir()
    (root / "__MACOSX__").mkdir()

    found = PluginManager()._scan_directory(root, "user")

    assert [manifest.name for manifest in found] == ["demo"]


def test_scan_skips_leftover_update_staging_dir(tmp_path: Path):
    root = tmp_path / "plugins"
    demo = root / "demo"
    demo.mkdir(parents=True)
    (demo / "plugin.yaml").write_text("name: demo\nversion: 0.1.0\ndescription: x\n")
    # A killed `plugins update` leaves the `.update-<rand>/` staging tree
    # behind; its `plugin/` payload holds a full copy of the new revision.
    staging = root / ".update-tmp1234567" / "plugin"
    staging.mkdir(parents=True)
    (staging / "plugin.yaml").write_text("name: demo\nversion: 0.4.0\ndescription: staged\n")
    (root / ".git").mkdir()

    found = PluginManager()._scan_directory(root, "user")

    assert [manifest.name for manifest in found] == ["demo"]


def test_scan_still_finds_hermes_plugin_manifest_dir(tmp_path: Path):
    # `.hermes-plugin/` is the one dot dir that IS a manifest location
    # (multi-harness repo convention); the dot-dir skip must not hide it.
    root = tmp_path / "plugins"
    sp = root / "superpowers"
    sp.mkdir(parents=True)
    (sp / ".hermes-plugin").mkdir()
    (sp / ".hermes-plugin" / "plugin.yaml").write_text(
        "name: superpowers\nversion: 6.3.0\ndescription: multi-harness\n"
    )

    found = PluginManager()._scan_directory(root, "user")

    assert [manifest.name for manifest in found] == ["superpowers"]


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
