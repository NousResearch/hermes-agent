"""Profile-owned, compare-and-swap bindings for disabled descriptors."""

from pathlib import Path

import pytest

from hermes_cli import profiles


@pytest.fixture()
def profile_env(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return tmp_path


FIRST_REF = "specialist-descriptor:" + "a" * 64
SECOND_REF = "specialist-descriptor:" + "b" * 64


def test_specialist_descriptor_binding_is_exact_cas_and_reversible(profile_env):
    profiles.create_profile("explorer", no_alias=True)
    profile_dir = profiles.get_profile_dir("explorer")
    profiles.write_profile_meta(profile_dir, description="keep this metadata")

    assert (
        profiles.set_specialist_descriptor_ref("explorer", FIRST_REF) == FIRST_REF
    )
    assert profiles.get_specialist_descriptor_ref("explorer") == FIRST_REF
    assert profiles.read_profile_meta(profile_dir)["description"] == "keep this metadata"

    with pytest.raises(ValueError, match="SPECIALIST_DESCRIPTOR_BINDING_CONFLICT"):
        profiles.set_specialist_descriptor_ref("explorer", SECOND_REF)
    assert profiles.get_specialist_descriptor_ref("explorer") == FIRST_REF

    with pytest.raises(ValueError, match="SPECIALIST_DESCRIPTOR_BINDING_CONFLICT"):
        profiles.clear_specialist_descriptor_ref("explorer", SECOND_REF)
    assert profiles.get_specialist_descriptor_ref("explorer") == FIRST_REF

    profiles.clear_specialist_descriptor_ref("explorer", FIRST_REF)
    assert profiles.get_specialist_descriptor_ref("explorer") is None
    assert profiles.read_profile_meta(profile_dir)["description"] == "keep this metadata"


@pytest.mark.parametrize(
    "reference",
    ["", "contract:" + "a" * 64, "specialist-descriptor:not-a-sha256"],
)
def test_specialist_descriptor_binding_rejects_non_descriptor_reference(
    profile_env, reference
):
    profiles.create_profile("explorer", no_alias=True)

    with pytest.raises(ValueError, match="INVALID_SPECIALIST_DESCRIPTOR_REF"):
        profiles.set_specialist_descriptor_ref("explorer", reference)