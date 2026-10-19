"""Uninstall and profile cloning respect the install/profile bucket split.

Runtime artifacts belong to an install; config, sessions and skills belong
to a profile. Uninstall removes the former (in either mode — they are not
data); profile clone/export never copies them.
"""

from pathlib import Path
import tarfile

import pytest

from hermes_cli.uninstall import remove_legacy_runtime_trees


def test_legacy_cleanup_removes_only_runtime_bytes_and_is_idempotent(tmp_path):
    runtime = ('node/bin/node', 'bin/uv', 'bin/uv.exe')
    user = ('bin/my-script', 'config.yaml', 'auth.json', 'SOUL.md',
            'sessions/session', 'skills/demo/SKILL.md', 'memories/MEMORY.md', 'profiles/other/config.yaml')
    for name in (*runtime, *user):
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(name, encoding='utf-8')
    removed = remove_legacy_runtime_trees(tmp_path)
    assert set(removed) == {tmp_path / 'node', tmp_path / 'bin/uv', tmp_path / 'bin/uv.exe'}
    assert all(not (tmp_path / name).exists() for name in runtime)
    assert all((tmp_path / name).read_text(encoding='utf-8') == name for name in user)
    assert remove_legacy_runtime_trees(tmp_path) == []


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
