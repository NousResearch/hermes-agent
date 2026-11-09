"""A plugin an elevated (UAC) process installs or updates stays usable by the same user unelevated.

Publication moves a tree staged under a ``tempfile`` directory into ``plugins/``. Python 3.13+ gives
that directory a protected DACL whose only user-reachable entry is OWNER RIGHTS, and an elevated
process's default owner is Administrators, so the user's normal token got WinError 5 on the plugin
and discovery skipped it with only a log line. Elevation is the host fact the code reads; only the
native ACL writer is replaced, to record what an elevated grant touched (the effect itself is
proved on Windows by ``test_hermes_dir_acl_windows.py``).
"""
import shutil
from pathlib import Path

import pytest

import hermes_dir_acl


@pytest.fixture
def staged_plugin(tmp_path, monkeypatch):
    home = tmp_path / "home"
    staged, target = tmp_path / "staging" / "plugin", home / "plugins" / "acl-probe"
    (staged / "pkg").mkdir(parents=True)
    (staged / "plugin.yaml").write_text("name: acl-probe\n", encoding="utf-8")
    (staged / "pkg" / "__init__.py").write_text("def register(ctx):\n    pass\n", encoding="utf-8")
    target.parent.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("pm.client.sync_venv", lambda **kwargs: shutil.move(kwargs["plugins"].data["staged"],
                                                                             kwargs["plugins"].data["target"]))
    writes = []
    monkeypatch.setattr(hermes_dir_acl, "_grant_with_pywin32",
                        lambda path, *_a, **_k: writes.append(Path(path).relative_to(target).as_posix()))
    return staged, target, writes


def test_an_elevated_publication_grants_the_user_the_whole_published_tree(staged_plugin, monkeypatch):
    from hermes_cli.plugins_transaction import publish_plugin

    staged, target, writes = staged_plugin
    monkeypatch.setattr(hermes_dir_acl, "process_is_elevated", lambda: True)

    publish_plugin(staged, target, {}, {"acl-probe": {}})

    assert (target / "plugin.yaml").is_file()
    assert sorted(writes) == [".", "pkg", "pkg/__init__.py", "plugin.yaml"]


def test_an_unelevated_publication_touches_no_acl(staged_plugin, monkeypatch):
    from hermes_cli.plugins_transaction import publish_plugin

    staged, target, writes = staged_plugin
    monkeypatch.setattr(hermes_dir_acl, "process_is_elevated", lambda: False)

    publish_plugin(staged, target, {}, {"acl-probe": {}})

    assert (target / "plugin.yaml").is_file() and writes == []

def test_worker_grants_acl_before_cli_returns(staged_plugin, monkeypatch, tmp_path):
    """A killed caller cannot strand committed plugin code before it runs post-sync cleanup."""
    from pm.publication import StagedPlugin

    staged, target, writes = staged_plugin
    monkeypatch.setattr(hermes_dir_acl, "process_is_elevated", lambda: True)
    project = tmp_path / "project"
    project.mkdir()

    # Exercise the real PM publication seam without ever resuming publish_plugin().
    # Recovery retains this new tree if the worker committed it before the caller died.
    StagedPlugin({
        "staged": str(staged), "target": str(target), "target_digest": None,
        "old_metadata": {}, "new_metadata": {"acl-probe": {"revision": "new"}},
    }).publish(project)

    assert (target / "plugin.yaml").is_file()
    assert sorted(writes) == [".", "pkg", "pkg/__init__.py", "plugin.yaml"]
