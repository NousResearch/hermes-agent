"""Plugin install must not probe or repair permissions outside its staged tree."""

from pathlib import Path

import pytest

from hermes_cli import plugins_cmd as pc


@pytest.mark.parametrize("target_kind", ["file", "directory", "missing"])
def test_install_rejects_external_links_before_probing(tmp_path, monkeypatch, target_kind):
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()
    outside = tmp_path / "outside"
    if target_kind == "directory":
        outside.mkdir()
    elif target_kind == "file":
        outside.write_text("untouched", encoding="utf-8")
    original_mode = outside.stat().st_mode if outside.exists() else None

    def clone(staging, *_args):
        staging.mkdir()
        (staging / "plugin.yaml").write_text("name: bounded\n", encoding="utf-8")
        (staging / "escape").symlink_to(outside, target_is_directory=target_kind == "directory")
        return "a" * 40

    real_probe = pc._probe_readable

    def probe_inside_only(path):
        assert path.resolve().is_relative_to(plugins_dir), "probed an external link target"
        real_probe(path)

    monkeypatch.setattr(pc, "_plugins_dir", lambda: plugins_dir)
    monkeypatch.setattr(pc, "_clone_plugin_repo", clone)
    monkeypatch.setattr(pc, "_probe_readable", probe_inside_only)

    with pytest.raises(pc.PluginOperationError, match="escapes the plugin tree"):
        pc._install_plugin_core("https://github.com/example/bounded", force=False)

    assert list(plugins_dir.iterdir()) == []
    if original_mode is not None:
        assert outside.stat().st_mode == original_mode
    if target_kind == "file":
        assert outside.read_text(encoding="utf-8") == "untouched"


def test_install_preserves_readable_internal_links(tmp_path, monkeypatch):
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()

    def clone(staging, *_args):
        staging.mkdir()
        (staging / "plugin.yaml").write_text("name: bounded\nmanifest_version: 1\n", encoding="utf-8")
        (staging / "assets").mkdir()
        (staging / "assets" / "note.txt").write_text("safe", encoding="utf-8")
        (staging / "note.txt").symlink_to(Path("assets") / "note.txt")
        (staging / "assets-link").symlink_to("assets", target_is_directory=True)
        return "a" * 40

    monkeypatch.setattr(pc, "_plugins_dir", lambda: plugins_dir)
    monkeypatch.setattr(pc, "_clone_plugin_repo", clone)

    target, manifest, name = pc._install_plugin_core("https://github.com/example/bounded", force=False)

    assert name == manifest["name"] == "bounded"
    assert (target / "note.txt").read_text(encoding="utf-8") == "safe"
    assert (target / "assets-link" / "note.txt").read_text(encoding="utf-8") == "safe"
