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
    create_profile,
    delete_profile,
    list_profiles,
    profiles_to_serve,
)
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
