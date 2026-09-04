"""Uninstall and profile cloning respect the install/profile bucket split.

Runtime artifacts belong to an install; config, sessions and skills belong
to a profile. Uninstall removes the former (in either mode — they are not
data); profile clone/export never copies them.
"""

import pytest

from hermes_cli.uninstall import remove_legacy_runtime_trees


class TestRemoveLegacyRuntimeTrees:
    def test_removes_a_pre_split_node_tree(self, tmp_path):
        home = tmp_path / "home"
        (home / "node" / "bin").mkdir(parents=True)
        (home / "node" / "bin" / "node").write_text("#!/bin/sh\n", encoding="utf-8")

        removed = remove_legacy_runtime_trees(home)

        assert (home / "node") not in [p for p in home.iterdir()]
        assert removed == [home / "node"]

    def test_removes_only_the_uv_binary_not_the_whole_bin_dir(self, tmp_path):
        """A user's own scripts live in bin/ — deleting the directory would
        take them with it."""
        home = tmp_path / "home"
        (home / "bin").mkdir(parents=True)
        (home / "bin" / "uv").write_text("#!/bin/sh\n", encoding="utf-8")
        (home / "bin" / "my-script").write_text("#!/bin/sh\n", encoding="utf-8")

        removed = remove_legacy_runtime_trees(home)

        assert removed == [home / "bin" / "uv"]
        assert (home / "bin").is_dir()
        assert (home / "bin" / "my-script").is_file()

    def test_never_touches_profile_state(self, tmp_path):
        home = tmp_path / "home"
        home.mkdir()
        for name in ("config.yaml", "auth.json", "SOUL.md"):
            (home / name).write_text("keep me", encoding="utf-8")
        for name in ("sessions", "skills", "memories", "profiles"):
            (home / name).mkdir()

        remove_legacy_runtime_trees(home)

        for name in ("config.yaml", "auth.json", "SOUL.md"):
            assert (home / name).is_file(), name
        for name in ("sessions", "skills", "memories", "profiles"):
            assert (home / name).is_dir(), name

    def test_no_runtime_trees_is_a_quiet_no_op(self, tmp_path):
        home = tmp_path / "home"
        home.mkdir()
        assert remove_legacy_runtime_trees(home) == []


class TestProfileCopyExclusions:
    @pytest.mark.parametrize(
        "excluded", [".hermes-runtime", "node", "hermes-agent", "profiles"]
    )
    def test_clone_all_excludes_install_artifacts(self, excluded):
        from hermes_cli.profiles import _CLONE_ALL_DEFAULT_EXCLUDE_ROOT

        assert excluded in _CLONE_ALL_DEFAULT_EXCLUDE_ROOT

    @pytest.mark.parametrize("excluded", [".hermes-runtime", "node"])
    def test_export_excludes_managed_runtimes(self, excluded):
        from hermes_cli.profiles import _DEFAULT_EXPORT_EXCLUDE_ROOT

        assert excluded in _DEFAULT_EXPORT_EXCLUDE_ROOT

    @pytest.mark.parametrize("excluded", [".hermes-runtime", "node"])
    def test_distribution_excludes_managed_runtimes(self, excluded):
        from hermes_cli.profile_distribution import USER_OWNED_EXCLUDE

        assert excluded in USER_OWNED_EXCLUDE

    def test_profile_state_is_still_copied(self):
        """The exclusions must not swallow actual profile data."""
        from hermes_cli.profiles import _CLONE_ALL_DEFAULT_EXCLUDE_ROOT

        for kept in ("config.yaml", "skills", "memories", "SOUL.md"):
            assert kept not in _CLONE_ALL_DEFAULT_EXCLUDE_ROOT
