"""A plugin an elevated (UAC) process installs or updates stays usable by the same user unelevated.

Publication moves a tree staged under a ``tempfile`` directory into ``plugins/``. Python 3.13+ gives
that directory a protected DACL whose only user-reachable entry is OWNER RIGHTS, and an elevated
process's default owner is Administrators, so the user's normal token got WinError 5 on the plugin
and discovery skipped it with only a log line. Elevation is the host fact the code reads. ``sync_venv``
runs only the real PM publication move (``StagedPlugin.publish``), without a dependency build, and
the native ACL writer is replaced to record what an elevated grant touched (the effect itself is
proved on Windows by ``test_hermes_dir_acl_windows.py``).
"""
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
    from pm.publication import StagedPlugin

    project = tmp_path / "pm-project"
    project.mkdir()
    monkeypatch.setattr("pm.client.sync_venv",
                        lambda **kwargs: StagedPlugin(dict(kwargs["plugins"].data)).publish(project))
    writes = []
    # True like the real writer when it wrote: that write does not propagate, so children are owed.
    monkeypatch.setattr(hermes_dir_acl, "_grant_with_pywin32",
                        lambda path, *_a, **_k: writes.append(Path(path).relative_to(target).as_posix()) or True)
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


def test_without_pywin32_one_inheritable_grant_covers_the_published_tree(staged_plugin, monkeypatch):
    """PM's worker runtime has no pywin32, so publication takes the icacls fallback. Its inheritable
    ACE on the plugin root propagates to the tree, so a per-entry walk (``whoami`` + ``icacls`` per
    file, under the install lock) is one process launch per entry too many."""
    from hermes_cli.plugins_transaction import publish_plugin

    staged, target, _ = staged_plugin
    monkeypatch.setattr(hermes_dir_acl, "process_is_elevated", lambda: True)

    def no_pywin32(*_a, **_k):
        raise ImportError("win32security")

    icacls = []
    monkeypatch.setattr(hermes_dir_acl, "_grant_with_pywin32", no_pywin32)
    monkeypatch.setattr(hermes_dir_acl, "_grant_with_icacls",
                        lambda path, is_dir, *_a, **_k: icacls.append((Path(path).relative_to(target).as_posix(), is_dir)))

    publish_plugin(staged, target, {}, {"acl-probe": {}})

    assert (target / "plugin.yaml").is_file() and icacls == [(".", True)]
