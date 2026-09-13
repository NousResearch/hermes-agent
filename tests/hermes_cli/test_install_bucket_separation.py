"""Uninstall and profile cloning respect the install/profile bucket split.

Runtime artifacts belong to an install; config, sessions and skills belong
to a profile. Uninstall removes the former (in either mode — they are not
data); profile clone/export never copies them.
"""

from pathlib import Path
import tarfile

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
    @pytest.mark.parametrize("operation", ["clone", "export", "distribution"])
    def test_copies_profile_payload_without_install_artifacts(self, tmp_path, monkeypatch, operation):
        from hermes_cli import profiles, profile_distribution

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        home = tmp_path / ".hermes"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setattr(profiles, "_maybe_register_gateway_service", lambda name: None)
        kept = {"config.yaml": "model: {}\n", "SOUL.md": "profile identity\n",
                "skills/demo/SKILL.md": "demo instructions\n"}
        if operation != "distribution":
            kept["memories/MEMORY.md"] = "profile memory\n"
        excluded = (".hermes-runtime", "node", "hermes-agent", "profiles")
        for rel, content in kept.items():
            path = home / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        for name in excluded:
            path = home / name / "must-not-copy"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("install state", encoding="utf-8")

        if operation == "clone":
            target = profiles.create_profile("clone", clone_from="default", clone_all=True, no_alias=True)
        elif operation == "distribution":
            profile_distribution.write_manifest(home, profile_distribution.DistributionManifest(name="copy", version="1.0.0"))
            profile_distribution.install_distribution(str(home), name="copy", create_alias=False)
            target = profiles.get_profile_dir("copy")
        else:
            archive = profiles.export_profile("default", str(tmp_path / "profile.tar.gz"))
            with tarfile.open(archive) as bundle:
                for rel, content in kept.items():
                    payload = bundle.extractfile(f"default/{rel}")
                    assert payload is not None, rel
                    assert payload.read().decode() == content
                roots = {name.split("/")[1] for name in bundle.getnames() if "/" in name}
                assert not roots.intersection(excluded)
            return

        for rel, content in kept.items():
            assert (target / rel).read_text(encoding="utf-8") == content
        for name in excluded:
            assert not (target / name).exists(), name
