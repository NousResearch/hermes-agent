"""A deleted profile is not a snapshot sibling (#144748).

`_sibling_profile_homes` filtered on directory shape only, so a profile the operator had just
deleted (tombstone under `profiles/.deleted/`, husk dir possibly still present) was handed to
`create_quick_snapshot`, which mkdir'd `.backup.lock` and `state-snapshots/` inside that husk.
"""

from hermes_cli import backup
from hermes_constants import mark_named_profile_deleted


def _install(monkeypatch, tmp_path):
    profiles = tmp_path / "hermes" / "profiles"
    profiles.mkdir(parents=True)
    monkeypatch.setattr(backup, "_sibling_roots", None, raising=False)
    from hermes_cli import profiles as profiles_mod

    monkeypatch.setattr(profiles_mod, "_get_profiles_root", lambda: profiles)
    monkeypatch.setattr(profiles_mod, "_get_default_hermes_home", lambda: tmp_path / "hermes")
    return profiles


def test_a_tombstoned_profile_is_not_a_sibling(monkeypatch, tmp_path):
    profiles = _install(monkeypatch, tmp_path)
    live, gone = profiles / "live1", profiles / "gone1"
    live.mkdir()
    gone.mkdir()
    mark_named_profile_deleted(gone)

    names = [name for name, _home in backup._sibling_profile_homes(tmp_path / "invoking")]

    assert "live1" in names
    assert "gone1" not in names


def test_clearing_the_tombstone_brings_the_sibling_back(monkeypatch, tmp_path):
    profiles = _install(monkeypatch, tmp_path)
    gone = profiles / "gone2"
    gone.mkdir()
    mark_named_profile_deleted(gone)
    from hermes_constants import clear_named_profile_deleted

    clear_named_profile_deleted(gone)

    names = [name for name, _home in backup._sibling_profile_homes(tmp_path / "invoking")]

    assert "gone2" in names
