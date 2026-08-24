"""Deleted named profiles must stay gone until explicitly recreated.

A live serve/logging process can mkdir ``profiles/<name>/logs`` after
``hermes profile delete`` removes the tree. That empty shell then
reappears in ``hermes profile list`` and Desktop Bot Mode. These tests
lock the tombstone + no-mkdir contract without depending on Desktop.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli.config import ensure_hermes_home
from hermes_cli.profiles import (
    backfill_profile_envs,
    create_profile,
    delete_profile,
    list_profiles,
    profile_exists,
    profiles_to_serve,
    resolve_profile_env,
    set_active_profile,
)
from hermes_constants import named_profile_home
from hermes_logging import setup_logging


@pytest.fixture()
def profile_env(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    default_home = tmp_path / ".hermes"
    default_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    return tmp_path


def _named_homes(tmp_path: Path) -> list[str]:
    return [info.name for info in list_profiles() if not info.is_default]


def _delete(name: str) -> None:
    with patch("hermes_cli.profiles._cleanup_gateway_service"), patch(
        "hermes_cli.profiles._stop_profile_backends"
    ):
        delete_profile(name, yes=True)


class TestDeletedProfileTombstone:
    def test_delete_then_logging_setup_does_not_recreate_home(self, profile_env, monkeypatch):
        profile_dir = create_profile("worker", no_alias=True, no_skills=True)
        with patch("hermes_cli.profiles._cleanup_gateway_service"), patch(
            "hermes_cli.profiles._stop_profile_backends"
        ):
            delete_profile("worker", yes=True)

        assert not profile_dir.exists()
        assert "worker" not in _named_homes(profile_env)

        monkeypatch.setenv("HERMES_HOME", str(profile_dir))
        with pytest.raises(FileNotFoundError, match="Named profile home does not exist"):
            setup_logging(hermes_home=profile_dir, force=True)

        assert not profile_dir.exists()
        monkeypatch.setenv("HERMES_HOME", str(profile_env / ".hermes"))
        assert "worker" not in _named_homes(profile_env)

    def test_empty_shell_after_delete_is_not_listed_or_served(self, profile_env):
        profile_dir = create_profile("worker", no_alias=True, no_skills=True)
        with patch("hermes_cli.profiles._cleanup_gateway_service"), patch(
            "hermes_cli.profiles._stop_profile_backends"
        ):
            delete_profile("worker", yes=True)

        # Simulate a stale mkdir that only rebuilds the directory itself.
        profile_dir.mkdir(parents=True)
        (profile_dir / "state.db").write_bytes(b"")

        assert "worker" not in _named_homes(profile_env)
        served = [name for name, _ in profiles_to_serve(True)]
        assert "worker" not in served

    def test_tombstoned_home_is_not_bootstrapped(self, profile_env, monkeypatch):
        profile_dir = create_profile("worker", no_alias=True, no_skills=True)
        with patch("hermes_cli.profiles._cleanup_gateway_service"), patch(
            "hermes_cli.profiles._stop_profile_backends"
        ):
            delete_profile("worker", yes=True)
        profile_dir.mkdir(parents=True)

        monkeypatch.setenv("HERMES_HOME", str(profile_dir))
        with pytest.raises(FileNotFoundError, match="Named profile home does not exist"):
            ensure_hermes_home()
        assert not (profile_dir / "sessions").exists()

    def test_create_after_delete_clears_tombstone(self, profile_env):
        create_profile("worker", no_alias=True, no_skills=True)
        with patch("hermes_cli.profiles._cleanup_gateway_service"), patch(
            "hermes_cli.profiles._stop_profile_backends"
        ):
            delete_profile("worker", yes=True)

        recreated = create_profile("worker", no_alias=True, no_skills=True)
        assert recreated.is_dir()
        assert "worker" in _named_homes(profile_env)

    def test_profile_exists_is_false_for_tombstoned_shell(self, profile_env):
        profile_dir = create_profile("worker", no_alias=True, no_skills=True)
        assert profile_exists("worker") is True
        _delete("worker")
        profile_dir.mkdir(parents=True)

        assert profile_exists("worker") is False
        with pytest.raises(FileNotFoundError, match="does not exist"):
            set_active_profile("worker")
        with pytest.raises(FileNotFoundError, match="does not exist"):
            resolve_profile_env("worker")

    def test_backfill_skips_tombstoned_directory(self, profile_env):
        profile_dir = create_profile("worker", no_alias=True, no_skills=True)
        _delete("worker")
        profile_dir.mkdir(parents=True)
        (profile_env / ".hermes" / ".env").write_text(
            "OPENROUTER_API_KEY=root-key\n", encoding="utf-8"
        )

        backfilled = backfill_profile_envs(quiet=True)

        assert "worker" not in backfilled
        assert not (profile_dir / ".env").exists()

    def test_create_after_delete_replaces_empty_shell(self, profile_env):
        profile_dir = create_profile("worker", no_alias=True, no_skills=True)
        _delete("worker")
        profile_dir.mkdir(parents=True)
        (profile_dir / "logs").mkdir()

        recreated = create_profile("worker", no_alias=True, no_skills=True)
        assert recreated.is_dir()
        assert "worker" in _named_homes(profile_env)

    @pytest.mark.parametrize("leftover", ["config.yaml", ".env"])
    def test_create_after_delete_refuses_when_identity_files_remain(
        self, profile_env, leftover
    ):
        profile_dir = create_profile("worker", no_alias=True, no_skills=True)
        _delete("worker")
        profile_dir.mkdir(parents=True)
        leftover_path = profile_dir / leftover
        leftover_path.write_text("keep-me\n", encoding="utf-8")

        with pytest.raises(FileExistsError, match="already exists"):
            create_profile("worker", no_alias=True, no_skills=True)

        assert leftover_path.read_text(encoding="utf-8") == "keep-me\n"
        assert leftover_path.exists()


class TestNamedProfileHome:
    def test_logs_under_named_profile_resolve_to_profile_home(self, tmp_path):
        # tmp_path acts as a real Hermes home (Docker/custom layout): it
        # carries a home marker file, so profiles/ under it is canonical.
        (tmp_path / "config.yaml").write_text("{}\n", encoding="utf-8")
        worker = tmp_path / "profiles" / "worker"
        assert named_profile_home(worker / "logs") == worker
        assert named_profile_home(worker) == worker

    def test_dot_hermes_layout_resolves_without_markers(self, tmp_path):
        worker = tmp_path / ".hermes" / "profiles" / "worker"
        assert named_profile_home(worker / "logs") == worker
        assert named_profile_home(worker) == worker

    def test_default_home_with_profiles_in_path_is_not_named(self, tmp_path):
        default_home = tmp_path / "foo" / "profiles" / "notahome" / ".hermes"
        assert named_profile_home(default_home) is None
        assert named_profile_home(default_home / "logs") is None

    def test_unrelated_profiles_dir_is_not_named(self, tmp_path):
        # Review point 1 regression: a custom home like
        # /srv/profiles/buildcache must NOT be treated as a named profile —
        # its parent is not a Hermes home, so logging must keep mkdir-ing.
        custom_home = tmp_path / "srv" / "profiles" / "buildcache"
        assert named_profile_home(custom_home) is None
        assert named_profile_home(custom_home / "logs") is None

    def test_unrelated_profiles_dir_still_mkdirs(self, tmp_path):
        from hermes_constants import mkdir_under_hermes_home

        custom_home = tmp_path / "srv" / "profiles" / "buildcache"
        log_dir = mkdir_under_hermes_home(custom_home / "logs")
        assert log_dir.is_dir()

    def test_setup_logging_under_unrelated_profiles_dir_succeeds(self, tmp_path):
        # End-to-end shape of the point-1 regression: setup_logging on a
        # non-profile custom home whose path contains a 'profiles' segment
        # must create the log dir instead of raising FileNotFoundError.
        custom_home = tmp_path / "srv" / "profiles" / "buildcache"
        custom_home.mkdir(parents=True)
        log_dir = setup_logging(hermes_home=custom_home, force=True)
        assert log_dir == custom_home / "logs"
        assert log_dir.is_dir()

    def test_tombstone_dir_marks_profiles_root(self, tmp_path):
        # A profiles/.deleted directory is only ever created by
        # `hermes profile delete` — its presence alone anchors recognition,
        # so tombstones are honored even when root markers are missing.
        profiles_dir = tmp_path / "opt" / "profiles"
        (profiles_dir / ".deleted").mkdir(parents=True)
        worker = profiles_dir / "worker"
        assert named_profile_home(worker / "logs") == worker
