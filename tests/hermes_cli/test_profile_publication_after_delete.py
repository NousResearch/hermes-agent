"""A profile published under a previously deleted name is a live profile.

``profile delete`` leaves a tombstone at ``profiles/.deleted/<name>``. Import, distribution
install and rename each publish a directory under a name; a tombstone left for that name made
the new profile unlisted, unresolvable through ``-p``, and refused by ``create`` as existing.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli import profiles
from hermes_cli.profile_distribution import DistributionManifest, install_distribution, write_manifest


@pytest.fixture()
def profile_env(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return tmp_path


def _create_and_delete(name: str) -> None:
    profiles.create_profile(name, no_alias=True)
    with patch("hermes_cli.profiles._cleanup_gateway_service"), \
         patch("hermes_cli.profiles._live_default_multiplexer", return_value=False):
        profiles.delete_profile(name, yes=True)


def _publish_by_import(tmp_path: Path, name: str) -> None:
    profiles.create_profile("source", no_alias=True)
    archive = profiles.export_profile("source", str(tmp_path / "source.tar.gz"))
    profiles.import_profile(str(archive), name=name)


def _publish_by_install(tmp_path: Path, name: str) -> None:
    source = tmp_path / "distribution"
    source.mkdir()
    (source / "SOUL.md").write_text("Distribution.\n")
    (source / "config.yaml").write_text("model:\n  model: m\n")
    write_manifest(source, DistributionManifest(name=name, version="0.1.0"))
    install_distribution(str(source), name=name)


def _publish_by_rename(tmp_path: Path, name: str) -> None:
    profiles.create_profile("source", no_alias=True)
    with patch("hermes_cli.profiles._cleanup_gateway_service"):
        profiles.rename_profile("source", name)


@pytest.mark.parametrize("publish", [_publish_by_import, _publish_by_install, _publish_by_rename],
                         ids=["import", "install", "rename"])
def test_profile_published_under_a_deleted_name_is_live(profile_env, publish):
    _create_and_delete("reused")

    publish(profile_env, "reused")

    assert "reused" in [p.name for p in profiles.list_profiles()]
    assert profiles.resolve_profile_env("reused") == str(profiles.get_profile_dir("reused"))
