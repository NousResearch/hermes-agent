"""Shared link parents remain protected without repeated ancestor expansion."""
from pathlib import Path

from hermes_dir_acl import _link_ancestor_dirs


def test_shared_links_protect_all_and_only_parents_within_root(tmp_path):
    root = tmp_path / "home"
    linked = {
        root,
        root / "cache" / "packages" / "first",
        root / "cache" / "packages" / "second",
        root / "cache" / "other" / "third",
        root / "profiles" / "bai" / "link",
        tmp_path / "outside" / "link",
    }
    assert _link_ancestor_dirs(root, linked) == {
        root, root / "cache", root / "cache" / "packages",
        root / "cache" / "other", root / "profiles", root / "profiles" / "bai",
    }


def test_many_links_with_shared_ancestors_keep_the_same_boundary():
    root = Path.cwd() / "unused-test-root"
    branch = root.joinpath(*(f"level-{n}" for n in range(30)))
    linked = {branch / f"link-{n}" for n in range(10000)}
    expected = {branch, *list(branch.parents)[:30]}
    assert _link_ancestor_dirs(root, linked) == expected
