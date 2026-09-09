"""Deletion retries must recognize only known pre-incarnation tombstones."""

from pathlib import Path

import pytest

from hermes_cli import profile_lifecycle, profiles
from hermes_cli.profile_incarnation import (
    PROFILE_INCARNATION_FILENAME,
    ensure_profile_incarnation,
)


@pytest.fixture
def legacy_profile(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    profile = home / "profiles" / "worker"
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Never inspect/stop services or processes outside this disposable fixture.
    monkeypatch.setattr(profiles, "_check_gateway_running", lambda *_: False)
    monkeypatch.setattr(profiles, "_get_wrapper_dir", lambda: tmp_path / "bin")
    for name in ("_cleanup_gateway_service", "_maybe_unregister_gateway_service", "_stop_profile_backends"):
        monkeypatch.setattr(profiles, name, lambda *_: None)
    monkeypatch.setattr(profile_lifecycle, "external_profile_file_holders", lambda *_: [])
    return profile


@pytest.mark.parametrize("contents", ["", "deleted\n"])
def test_legacy_delete_retry_never_backfills_through_fence(legacy_profile, contents, monkeypatch):
    profile = legacy_profile
    marker = profile_lifecycle.mark_profile_deleting(profile)
    marker.write_text(contents, encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        ensure_profile_incarnation(profile)

    real_remove = profiles._rmtree_with_retry

    def remove_behind_fence(path, *args):
        assert profile_lifecycle.profile_home_is_tombstoned(path)
        assert not (path / PROFILE_INCARNATION_FILENAME).exists()
        return real_remove(path, *args)

    monkeypatch.setattr(profiles, "_rmtree_with_retry", remove_behind_fence)
    assert profiles.delete_profile("worker", yes=True) == profile
    assert not profile.exists()
    assert marker.exists()


@pytest.mark.parametrize("contents", ["", "deleted\n", "a" * 32 + "\n"])
@pytest.mark.parametrize("holder", ["local", "external"])
def test_failed_retry_preserves_existing_fence(legacy_profile, monkeypatch, contents, holder):
    profile = legacy_profile
    marker = profile_lifecycle.mark_profile_deleting(profile)
    marker.write_text(contents, encoding="utf-8")
    monkeypatch.setattr(
        profile_lifecycle, "wait_for_profile_state_db_release", lambda *_: holder != "local",
    )
    monkeypatch.setattr(
        profile_lifecycle, "wait_for_external_profile_file_release", lambda *_: [12345],
    )
    with pytest.raises(RuntimeError, match="still in use"):
        profiles.delete_profile("worker", yes=True)
    assert profile.exists()
    assert marker.exists(), "failed retry cleared the pre-existing deletion fence"
    assert marker.read_text(encoding="utf-8") == contents
    with pytest.raises(FileNotFoundError):
        ensure_profile_incarnation(profile)
    assert not (profile / PROFILE_INCARNATION_FILENAME).exists()


@pytest.mark.parametrize("contents", ["not-a-generation\n", "deleted-but-malformed\n", "a" * 31 + "\n"])
def test_malformed_token_tombstone_is_not_a_legacy_bypass(legacy_profile, contents):
    marker = profile_lifecycle.mark_profile_deleting(legacy_profile)
    marker.write_text(contents, encoding="utf-8")
    with pytest.raises(RuntimeError, match="Invalid profile incarnation marker"):
        profiles.delete_profile("worker", yes=True)
    assert legacy_profile.exists()
    assert marker.read_text(encoding="utf-8") == contents
    assert not (legacy_profile / PROFILE_INCARNATION_FILENAME).exists()
