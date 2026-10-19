"""Profile metadata refuses a retiring home, and a copy (clone, import) never carries its source's generation."""

from pathlib import Path
import tarfile

import pytest

from hermes_cli import profile_lifecycle, profiles
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


def test_profile_metadata_survives_updates_and_refuses_retirement(profile_root):
    home = profiles.create_profile("guide", no_alias=True, no_skills=True)
    incarnation = read_profile_incarnation(home)
    profiles.write_profile_meta(home, display_name="My guide", previous_names=["earlier-guide"])

    before = (home / "profile.yaml").read_bytes()
    meta = profiles.read_profile_meta(home)
    assert meta["display_name"] == "My guide"
    assert meta["previous_names"] == ["earlier-guide"]
    assert read_profile_incarnation(home) == incarnation

    profile_lifecycle.mark_profile_deleting(home, incarnation)
    with pytest.raises(FileNotFoundError, match="being deleted"):
        profiles.write_profile_meta(home, description="must not land")
    assert (home / "profile.yaml").read_bytes() == before
    assert profiles.profile_exists("guide") is False


@pytest.mark.parametrize("copy_kind", ["clone", "import"])
def test_profile_copies_publish_a_fresh_generation_behind_the_tombstone(profile_root, monkeypatch, copy_kind):
    source = profiles.create_profile("guide", no_alias=True, no_skills=True)
    source_incarnation = read_profile_incarnation(source)
    assert source_incarnation is not None
    target = profiles.get_profile_dir("copied-guide")
    published = []
    real_publish = profile_lifecycle.publish_profile_generation

    def publish(home, incarnation):
        assert Path(home) == target
        assert named_profile_is_deleted(home)
        assert incarnation is not None and incarnation != source_incarnation
        assert read_profile_incarnation(home) == incarnation
        published.append(incarnation)
        real_publish(home, incarnation)

    monkeypatch.setattr(profile_lifecycle, "publish_profile_generation", publish)
    if copy_kind == "clone":
        copied = profiles.create_profile("copied-guide", clone_from="guide", clone_all=True, no_alias=True)
    else:
        # Include the original incarnation too: an imported archive must not carry its authority.
        archive = profile_root / "guide.tar.gz"
        with tarfile.open(archive, "w:gz") as bundle:
            bundle.add(source, arcname="guide")
        copied = profiles.import_profile(str(archive), name="copied-guide")

    assert copied == target
    assert published == [read_profile_incarnation(target)]
    assert profiles.profile_exists("copied-guide") is True
    assert read_profile_incarnation(source) == source_incarnation
