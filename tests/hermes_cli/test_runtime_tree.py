"""Tests for hermes_cli/runtime_tree.py — tree classification and channel."""

import json
from pathlib import Path

from hermes_cli.runtime_tree import (
    CHANNEL_MAIN,
    CHANNEL_STABLE,
    STEWARD_UPDATE_MESSAGES,
    GitCheckout,
    Sealed,
    is_managed_install_root,
    resolve_update_channel,
    runtime_tree,
    steward_update_message,
)


class TestRuntimeTree:
    def test_a_tree_with_git_is_a_checkout(self, tmp_path):
        (tmp_path / ".git").mkdir()
        tree = runtime_tree(tmp_path)
        assert isinstance(tree, GitCheckout)
        assert tree.root == tmp_path

    def test_a_worktree_gitfile_also_counts(self, tmp_path):
        # Linked worktrees and submodules have a .git FILE, not a directory.
        (tmp_path / ".git").write_text("gitdir: /somewhere/else\n")
        assert isinstance(runtime_tree(tmp_path), GitCheckout)

    def test_a_gitless_tree_is_sealed_with_the_stamped_steward(self, tmp_path):
        (tmp_path / ".hermes_build_info.json").write_text(
            json.dumps({"commit": "a" * 40, "distribution": "desktop-app"})
        )
        tree = runtime_tree(tmp_path)
        assert isinstance(tree, Sealed)
        assert tree.steward == "desktop-app"

    def test_a_gitless_tree_without_a_stamp_is_sealed_unknown(self, tmp_path):
        tree = runtime_tree(tmp_path)
        assert isinstance(tree, Sealed)
        assert tree.steward == "unknown"

    def test_a_corrupt_stamp_degrades_to_unknown(self, tmp_path):
        (tmp_path / ".hermes_build_info.json").write_text("{not json")
        tree = runtime_tree(tmp_path)
        assert isinstance(tree, Sealed)
        assert tree.steward == "unknown"


class TestStewardMessages:
    def test_every_known_steward_names_its_mechanism(self):
        assert "--eject" in steward_update_message("desktop-app")
        assert "docker pull" in steward_update_message("docker")
        assert "flake" in steward_update_message("nix")

    def test_an_unknown_steward_gets_the_fallback_with_its_name(self):
        message = steward_update_message("pacman")
        assert "pacman" in message
        assert "cannot update" in message

    def test_every_table_entry_is_a_refusal(self):
        for steward, message in STEWARD_UPDATE_MESSAGES.items():
            assert message.startswith("\u2717"), steward


class TestManagedInstallRoot:
    def test_hermes_home_checkout_is_managed(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        root = tmp_path / ".hermes" / "hermes-agent"
        root.mkdir(parents=True)
        assert is_managed_install_root(root) is True

    def test_a_dev_tree_is_not_managed(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        dev = tmp_path / "src" / "hermes-agent"
        dev.mkdir(parents=True)
        assert is_managed_install_root(dev) is False

    def test_the_fhs_root_layout_is_managed(self):
        # /usr/local/lib/hermes-agent need not exist for the answer; the
        # comparison is by path. If it does not resolve, False is safe.
        result = is_managed_install_root(Path("/usr/local/lib/hermes-agent"))
        assert result in (True, False)  # never raises
        # On machines without the dir, resolve() still succeeds (no symlinks
        # involved), so the comparison holds.
        assert result is True


class TestResolveUpdateChannel:
    def test_stable_from_config(self):
        assert resolve_update_channel({"update": {"channel": "stable"}}) == CHANNEL_STABLE

    def test_main_is_the_default(self):
        assert resolve_update_channel(None) == CHANNEL_MAIN
        assert resolve_update_channel({}) == CHANNEL_MAIN
        assert resolve_update_channel({"update": {}}) == CHANNEL_MAIN

    def test_auto_and_unknown_mean_main(self):
        assert resolve_update_channel({"update": {"channel": "auto"}}) == CHANNEL_MAIN
        assert resolve_update_channel({"update": {"channel": "nightly"}}) == CHANNEL_MAIN

    def test_case_and_whitespace_are_forgiven(self):
        assert resolve_update_channel({"update": {"channel": " Stable "}}) == CHANNEL_STABLE
