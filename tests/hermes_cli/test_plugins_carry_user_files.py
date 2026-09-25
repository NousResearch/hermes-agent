"""Regression tests for issue #122006: force-replace plugin updates must keep user files."""

import shutil

from hermes_cli.plugins_cmd_catalog import _carry_unshipped_files, _stash_local_files, _tree_files


def test_stash_local_files_copies_wholly_ignored_directory_tree(tmp_path):
    target = tmp_path / "plugin"
    (target / "data" / "sub").mkdir(parents=True)
    (target / "data" / "index.db").write_text("keep me")
    (target / "data" / "sub" / "nested.bin").write_text("keep me too")
    stash = tmp_path / "stash"
    _stash_local_files(target, ["data/"], stash)
    assert (stash / "data" / "index.db").read_text() == "keep me"
    assert (stash / "data" / "sub" / "nested.bin").read_text() == "keep me too"


def test_carry_unshipped_files_keeps_user_config_and_data(tmp_path):
    target = tmp_path / "plugin"
    (target / "data").mkdir(parents=True)
    (target / "config.yaml").write_text("threshold: 42\n")
    (target / "user-data.db").write_text("keep me")
    (target / "data" / "index.db").write_text("keep me too")
    (target / "plugin.py").write_text("SHIPPED = 1\n")
    (target / "__pycache__").mkdir()
    (target / "__pycache__" / "plugin.cpython-314.pyc").write_text("junk")
    stash = tmp_path / "stash"
    carried = _carry_unshipped_files(
        target, {"plugin.py", "config.example.yaml"}, stash)
    assert carried == ["config.yaml", "data/index.db", "user-data.db"]
    assert (stash / "config.yaml").read_text() == "threshold: 42\n"
    assert not (stash / "plugin.py").exists()
    assert not (stash / "__pycache__").exists()


def test_tree_files_skips_cache_and_sidecar_parts(tmp_path):
    tree = tmp_path / "t"
    (tree / "__pycache__").mkdir(parents=True)
    (tree / "__pycache__" / "x.pyc").write_text("junk")
    (tree / "a.pyc").write_text("junk")
    (tree / "sub").mkdir()
    (tree / "sub" / "keep.txt").write_text("keep")
    assert _tree_files(tree) == {"sub/keep.txt"}


def test_reclone_update_carries_user_files_the_new_tree_does_not_ship(tmp_path, monkeypatch):
    from hermes_cli import plugins_cmd, plugins_cmd_update

    target = tmp_path / "plugins" / "myplug"
    (target / "data").mkdir(parents=True)
    (target / "config.yaml").write_text("api_endpoint: https://my-real-endpoint\n")
    (target / "user-data.db").write_text("keep me")
    (target / "data" / "index.db").write_text("keep me too")
    (target / "plugin.py").write_text("OLD = 1\n")

    def fake_install(source, force=False):
        assert force is True
        shutil.rmtree(target)
        target.mkdir(parents=True)
        (target / "plugin.py").write_text("NEW = 2\n")
        (target / "config.example.yaml").write_text("api_endpoint: http://example.invalid\n")
        return target, {"name": "myplug"}, "myplug"

    monkeypatch.setattr(plugins_cmd, "_install_plugin_core", fake_install)
    monkeypatch.setattr(
        plugins_cmd, "_read_install_metadata",
        lambda: {"myplug": {"revision": "b" * 40}})

    out = plugins_cmd_update._reclone_plugin_update("https://example.com/repo", "a" * 40, target)
    assert "Re-installed" in out
    # user files carried across the swap
    assert (target / "config.yaml").read_text() == "api_endpoint: https://my-real-endpoint\n"
    assert (target / "user-data.db").read_text() == "keep me"
    assert (target / "data" / "index.db").read_text() == "keep me too"
    # shipped files come from the new tree
    assert (target / "plugin.py").read_text() == "NEW = 2\n"
