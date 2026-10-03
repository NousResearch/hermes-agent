"""Setup roles survive metadata updates, but never cross profile publication as a copy."""

from pathlib import Path
import tarfile

import pytest

from hermes_cli import profile_lifecycle, profiles, setup_profile
from hermes_cli.profile_incarnation import read_profile_incarnation
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from toolsets import profile_role_toolsets


@pytest.fixture
def profile_root(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    root.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(profiles, "_notify_multiplexer", lambda _name: None)
    monkeypatch.setattr(profiles, "_maybe_register_gateway_service", lambda _name: None)
    return root


def test_setup_role_metadata_is_scoped_and_refuses_retirement(profile_root):
    guide = profiles.create_profile("guide", no_alias=True, no_skills=True)
    ordinary = profiles.create_profile("ordinary", no_alias=True, no_skills=True)
    incarnation = read_profile_incarnation(guide)
    with profile_lifecycle.profile_lifecycle_lease(guide):
        profiles.write_profile_meta(guide, role=profiles.SETUP_ROLE, description="guide notes")
    profiles.write_profile_meta(guide, display_name="My guide", previous_names=["earlier-guide"])

    # Discovery uses the role, not the old hermes-setup slug, and preserves user edits.
    before = (guide / "profile.yaml").read_bytes()
    assert setup_profile.ensure_setup_profile() == setup_profile.SetupProfile("guide", guide, False)
    assert (guide / "profile.yaml").read_bytes() == before
    meta = profiles.read_profile_meta(guide)
    assert meta["role"] == profiles.SETUP_ROLE
    assert meta["description"] == "guide notes"
    assert meta["display_name"] == "My guide"
    assert meta["previous_names"] == ["earlier-guide"]
    assert read_profile_incarnation(guide) == incarnation

    for home, has_setup in ((guide, True), (ordinary, False), (guide, True)):
        token = set_hermes_home_override(str(home))
        try:
            granted, denied = profile_role_toolsets()
            assert ("setup" in granted) is has_setup
            assert ("setup" in denied) is not has_setup
        finally:
            reset_hermes_home_override(token)

    with pytest.raises(ValueError, match="unknown profile role"):
        profiles.write_profile_meta(guide, role="not-a-role")
    assert (guide / "profile.yaml").read_bytes() == before
    profile_lifecycle.mark_profile_deleting(guide, incarnation)
    with pytest.raises(FileNotFoundError, match="being deleted"):
        profiles.write_profile_meta(guide, role=profiles.SETUP_ROLE, description="must not land")
    assert (guide / "profile.yaml").read_bytes() == before
    assert profiles.profile_exists("guide") is False


@pytest.mark.parametrize("copy_kind", ["clone", "import"])
def test_setup_copies_drop_role_before_fresh_generation_publication(profile_root, monkeypatch, copy_kind):
    setup = setup_profile.ensure_setup_profile()
    assert setup.created is True
    source_incarnation = read_profile_incarnation(setup.path)
    assert source_incarnation is not None
    source_meta = (setup.path / "profile.yaml").read_bytes()
    target = profiles.get_profile_dir("copied-guide")
    published = []
    real_publish = profile_lifecycle.publish_profile_generation

    def publish(home, incarnation):
        assert Path(home) == target
        assert profile_lifecycle.profile_home_is_tombstoned(home)
        assert profiles.read_profile_meta(Path(home))["role"] is None
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
    assert profiles.read_profile_meta(target)["role"] is None
    assert (setup.path / "profile.yaml").read_bytes() == source_meta
    assert read_profile_incarnation(setup.path) == source_incarnation
    assert setup_profile.find_setup_profile() == (setup.name, setup.path)