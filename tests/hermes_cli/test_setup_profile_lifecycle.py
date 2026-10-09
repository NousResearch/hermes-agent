"""The setup marker survives metadata updates, but never crosses profile publication as a copy."""

from pathlib import Path
import tarfile

import pytest

from hermes_cli import profile_lifecycle, profiles, setup_profile
from hermes_constants import named_profile_is_deleted
from hermes_cli.profile_incarnation import read_profile_incarnation


@pytest.fixture
def profile_root(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    root.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(profiles, "_notify_multiplexer", lambda _name: None)
    monkeypatch.setattr(profiles, "_maybe_register_gateway_service", lambda _name: None)
    return root


def _is_setup_profile(home: Path) -> bool:
    return (home / profiles.SETUP_PROFILE_MARKER).is_file()


def test_setup_marker_survives_metadata_and_refuses_retirement(profile_root):
    setup = setup_profile.ensure_setup_profile()
    ordinary = profiles.create_profile("ordinary", no_alias=True, no_skills=True)
    guide = setup.path
    incarnation = read_profile_incarnation(guide)
    marker = (guide / profiles.SETUP_PROFILE_MARKER).read_bytes()
    profiles.write_profile_meta(guide, display_name="My guide", previous_names=["earlier-guide"])

    # Discovery uses the marker, not the profile name, and preserves user edits.
    before = (guide / "profile.yaml").read_bytes()
    assert setup_profile.ensure_setup_profile() == setup_profile.SetupProfile(setup.name, guide, False)
    assert (guide / "profile.yaml").read_bytes() == before
    assert (guide / profiles.SETUP_PROFILE_MARKER).read_bytes() == marker
    meta = profiles.read_profile_meta(guide)
    assert meta["description"] == setup_profile.SETUP_PROFILE_DESCRIPTION
    assert meta["display_name"] == "My guide"
    assert meta["previous_names"] == ["earlier-guide"]
    assert read_profile_incarnation(guide) == incarnation

    assert not _is_setup_profile(ordinary)

    profile_lifecycle.mark_profile_deleting(guide, incarnation)
    with pytest.raises(FileNotFoundError, match="being deleted"):
        profiles.write_profile_meta(guide, description="must not land")
    assert (guide / "profile.yaml").read_bytes() == before
    assert profiles.profile_exists(setup.name) is False


@pytest.mark.parametrize("copy_kind", ["clone", "import"])
def test_setup_copies_drop_marker_before_fresh_generation_publication(profile_root, monkeypatch, copy_kind):
    setup = setup_profile.ensure_setup_profile()
    assert setup.created is True
    source_incarnation = read_profile_incarnation(setup.path)
    assert source_incarnation is not None
    source_marker = (setup.path / profiles.SETUP_PROFILE_MARKER).read_bytes()
    target = profiles.get_profile_dir("copied-guide")
    published = []
    real_publish = profile_lifecycle.publish_profile_generation

    def publish(home, incarnation):
        assert Path(home) == target
        assert named_profile_is_deleted(home)
        assert not _is_setup_profile(Path(home))
        assert incarnation is not None and incarnation != source_incarnation
        assert read_profile_incarnation(home) == incarnation
        published.append(incarnation)
        real_publish(home, incarnation)

    monkeypatch.setattr(profile_lifecycle, "publish_profile_generation", publish)
    if copy_kind == "clone":
        copied = profiles.create_profile("copied-guide", clone_from=setup.name, clone_all=True, no_alias=True)
    else:
        # Include the original incarnation too: an imported archive must not carry its authority.
        archive = profile_root / "setup.tar.gz"
        with tarfile.open(archive, "w:gz") as bundle:
            bundle.add(setup.path, arcname=setup.name)
        copied = profiles.import_profile(str(archive), name="copied-guide")

    assert copied == target
    assert published == [read_profile_incarnation(target)]
    assert profiles.profile_exists("copied-guide") is True
    assert not _is_setup_profile(target)
    assert (setup.path / profiles.SETUP_PROFILE_MARKER).read_bytes() == source_marker
    assert read_profile_incarnation(setup.path) == source_incarnation
    assert setup_profile.find_setup_profile() == (setup.name, setup.path)
