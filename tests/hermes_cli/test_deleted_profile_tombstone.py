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

from cron.scheduler_delivery import BOT_CHAT_PLATFORM, cron_delivery_targets
from hermes_cli.config import ensure_hermes_home
from gateway.profile_serving import profiles_to_serve
from profiles.current import set_active_profile
from profiles.paths import resolve_profile_env
from profiles.registry import list_profile_names, profile_exists
from hermes_cli.profiles import (
    backfill_profile_envs,
    create_profile,
    delete_profile,
    list_profiles,
)
from hermes_constants import (
    named_profile_home,
    reset_hermes_home_override,
    set_hermes_home_override,
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

    def test_late_reasoning_caps_save_does_not_recreate_deleted_home(self, profile_env):
        """A daemon retaining a deleted profile context must not recreate its cache tree."""
        from hermes_cli import models_reasoning_caps

        profile_dir = create_profile("worker", no_alias=True, no_skills=True)
        _delete("worker")

        token = set_hermes_home_override(profile_dir)
        try:
            models_reasoning_caps._save_reasoning_caps_disk(
                "https://example.test/v1/models",
                {"example/model": {"supports_reasoning": True}},
            )
        finally:
            reset_hermes_home_override(token)

        assert not profile_dir.exists()

    def test_atomic_cache_write_allows_live_profile_home(self, profile_env):
        from hermes_cli.models import _write_json_cache

        profile_dir = create_profile("worker", no_alias=True, no_skills=True)
        cache_path = profile_dir / "cache" / "models.json"

        _write_json_cache(cache_path, {"cached": True})

        assert cache_path.read_text(encoding="utf-8") == '{\n  "cached": true\n}'

    def test_atomic_cache_write_allows_unrelated_profiles_path(self, tmp_path):
        from hermes_cli.models import _write_json_cache

        cache_path = tmp_path / "custom" / "profiles" / "cache" / "models.json"

        _write_json_cache(cache_path, {"cached": True})

        assert cache_path.read_text(encoding="utf-8") == '{\n  "cached": true\n}'

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
        # The name-only hot path (cron Bot Chat targets, kanban profile hints)
        # honors the same tombstone: a recreated shell must not resurface as a
        # `bot-chat:worker` delivery target.
        assert "worker" not in list_profile_names()
        assert f"{BOT_CHAT_PLATFORM}:worker" not in [t["id"] for t in cron_delivery_targets()]

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

    def test_delete_releases_this_process_session_db(self, profile_env):
        import hermes_state_registry as registry

        profile_dir = create_profile("worker", no_alias=True, no_skills=True)
        held = registry.acquire(profile_dir / "state.db")
        assert held._conn is not None
        try:
            _delete("worker")
            assert held._conn is None
            recreated = create_profile("worker", no_alias=True, no_skills=True)
            again = registry.acquire(recreated / "state.db")
            assert again._conn is not None
            registry.release(again)
        finally:
            registry.close_all()

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

    def test_marker_less_shell_is_not_a_profile(self, profile_env):
        """A ``profiles/<name>`` dir with no identity file (a pre-tombstone ghost shell left by a
        cron ticker, or a stray infrastructure dir) is not listed, served, or seeded with the
        default install's ``.env`` — that seeding is what legitimised ghosts on ``hermes update``
        (#95188 path D, #94823, #99392). ``profile create`` may take the name back."""
        default_env = profile_env / ".hermes" / ".env"
        default_env.write_text("OPENAI_API_KEY=sk-real\n", encoding="utf-8")
        shell = profile_env / ".hermes" / "profiles" / "ghost"
        (shell / "cron").mkdir(parents=True)
        (shell / "cron" / "ticker_heartbeat").write_text("1\n", encoding="utf-8")
        legacy = profile_env / ".hermes" / "profiles" / "legacy"
        legacy.mkdir()
        (legacy / "state.db").write_bytes(b"")

        assert backfill_profile_envs(quiet=True) == ["legacy"]
        assert not (shell / ".env").exists()
        assert _named_homes(profile_env) == ["legacy"]
        assert [name for name, _ in profiles_to_serve(True)] == ["default", "legacy"]
        # ``hermes --profile ghost serve`` (a stale Desktop boot target) must not start a backend
        # in the shell — its ensure_hermes_home() would rebuild the full profile tree.
        assert not profile_exists("ghost")
        with pytest.raises(FileNotFoundError):
            resolve_profile_env("ghost")
        assert Path(resolve_profile_env("legacy")) == legacy

        # A live marker-less dir may still hold user files: ``profile create`` must fail closed,
        # naming the stray dir, and leave every byte in place (never rmtree a non-tombstoned dir).
        (shell / "skills" / "my-skill").mkdir(parents=True)
        (shell / "skills" / "my-skill" / "SKILL.md").write_text("# mine\n", encoding="utf-8")
        with pytest.raises(FileExistsError, match=str(shell)):
            create_profile("ghost", no_alias=True, no_skills=True)
        assert (shell / "skills" / "my-skill" / "SKILL.md").exists()
        assert "ghost" not in _named_homes(profile_env)

    def test_dangling_symlink_marker_is_still_identity(self, profile_env):
        """A profile whose only marker is a dangling symlinked ``config.yaml`` (clone/migration
        leftover) stays resolvable: ``is_file()`` follows links and would make it invisible."""
        legacy = profile_env / ".hermes" / "profiles" / "legacy"
        legacy.mkdir(parents=True)
        (legacy / "config.yaml").symlink_to(profile_env / "gone" / "config.yaml")
        assert profile_exists("legacy")
        assert Path(resolve_profile_env("legacy")) == legacy

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
