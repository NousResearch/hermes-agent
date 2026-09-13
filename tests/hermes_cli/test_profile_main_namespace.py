"""New profile targets must not reuse the default session namespace."""

import tarfile
from pathlib import Path

import pytest

from hermes_cli import profiles
from hermes_cli.profile_distribution import (
    DistributionManifest,
    install_distribution,
    update_distribution,
    write_manifest,
)
from hermes_constants import mark_named_profile_deleted, named_profile_is_deleted


@pytest.fixture
def profile_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Lifecycle tests use real profile files, never host services or running backends.
    for name in (
        "_cleanup_gateway_service", "_maybe_register_gateway_service", "_maybe_unregister_gateway_service",
        "_stop_profile_backends", "_notify_multiplexer",
    ):
        monkeypatch.setattr(profiles, name, lambda *args, **kwargs: None)
    return home


@pytest.mark.parametrize("operation", ["create", "import", "rename", "distribution", "distribution_tombstone"])
@pytest.mark.parametrize("name", ["main", "MAIN", " main "])
def test_new_profile_targets_cannot_claim_default_namespace(profile_home, tmp_path, operation, name):
    source = profile_home / "profiles" / "source"
    source.mkdir(parents=True)
    config = "model:\n  provider: custom\n  default: local-model\n"
    (source / "config.yaml").write_text(config)
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(source, arcname="source")
    write_manifest(source, DistributionManifest(name="source"))
    target = profile_home / "profiles" / "main"
    tombstoned = operation == "distribution_tombstone"
    if tombstoned:
        target.mkdir()
        mark_named_profile_deleted(target)
    actions = {
        "create": lambda: profiles.create_profile(name, no_skills=True),
        "import": lambda: profiles.import_profile(str(archive), name=name),
        "rename": lambda: profiles.rename_profile("source", name),
        "distribution": lambda: install_distribution(str(source), name=name, create_alias=True),
        "distribution_tombstone": lambda: install_distribution(
            str(source), name=name, force=True, create_alias=True,
        ),
    }

    with pytest.raises(ValueError, match="main"):
        actions[operation]()

    assert target.exists() == tombstoned
    if tombstoned:
        assert named_profile_is_deleted(target)
        assert not list(target.iterdir())
    assert (source / "config.yaml").read_text() == config
    assert not profiles.find_alias_for_profile("main")


@pytest.mark.parametrize("finish", ["rename", "delete"])
def test_legacy_main_profile_remains_manageable(profile_home, tmp_path, finish):
    legacy = profile_home / "profiles" / "main"
    legacy.mkdir(parents=True)
    config = "model:\n  provider: custom\n  default: local-model\n"
    (legacy / "config.yaml").write_text(config)
    if finish == "rename":
        profiles.create_wrapper_script("main")
    profiles.set_active_profile("main")

    assert profiles.resolve_profile_env("main") == str(legacy)
    assert any(profile.name == "main" for profile in profiles.list_profiles())
    clone = profiles.create_profile("copy", clone_from="main", no_alias=True)
    assert (clone / "config.yaml").is_file()

    source = tmp_path / "distribution"
    source.mkdir()
    (source / "SOUL.md").write_text("Updated distribution content")
    manifest = DistributionManifest(name="main", source=str(source))
    write_manifest(source, manifest)
    write_manifest(legacy, manifest)
    updated = update_distribution("main")
    assert updated.target_dir == legacy
    assert (legacy / "SOUL.md").read_text() == (source / "SOUL.md").read_text()
    assert (legacy / "config.yaml").read_text() == config

    if finish == "rename":
        renamed = profiles.rename_profile("main", "renamed")
        assert profiles.resolve_profile_env("renamed") == str(renamed)
        assert (renamed / "config.yaml").read_text() == config
        assert profiles.find_alias_for_profile("main") is None
    else:
        profiles.delete_profile("main", yes=True)

    assert not legacy.exists()
    assert (clone / "config.yaml").is_file()
