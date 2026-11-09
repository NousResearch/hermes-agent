"""A published plugin takes ``plugins/``'s permissions before its publication commits.

Recovery keeps a committed publication, so a permission fix that ran after the commit could be
skipped by a caller killed in between, leaving an elevated install Administrators-only (the
``tempfile`` staging DACL). The DACL reset itself is proved on Windows by
``test_worker_publication.py::test_published_plugin_takes_the_plugins_dir_permissions_not_the_staging_folders``
and ``test_filesystem_inherit_acl.py``.
"""
from pathlib import Path


def test_the_published_root_takes_its_parents_permissions_before_the_commit(tmp_path, monkeypatch):
    from pm import publication
    from pm.publication import StagedPlugin

    home = tmp_path / "home"
    staged, target = tmp_path / "staging" / "plugin", home / "plugins" / "acl-probe"
    (staged / "pkg").mkdir(parents=True)
    (staged / "plugin.yaml").write_text("name: acl-probe\n", encoding="utf-8")
    (staged / "pkg" / "__init__.py").write_text("def register(ctx):\n    pass\n", encoding="utf-8")
    target.parent.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    project = tmp_path / "project"
    project.mkdir()
    plugin = StagedPlugin({
        "staged": str(staged), "target": str(target), "target_digest": None,
        "old_metadata": {}, "new_metadata": {"acl-probe": {"revision": "new"}},
    })
    calls = []
    monkeypatch.setattr(publication, "inherit_parent_acl",
                        lambda path: calls.append((Path(path), (path / "plugin.yaml").is_file(),
                                                   plugin.metadata.exists())) or True)

    plugin.publish(project)

    # Once, on the published root, after the move and before the install metadata exists.
    assert calls == [(target, True, False)]
    assert "acl-probe" in plugin.metadata.read_text(encoding="utf-8")
