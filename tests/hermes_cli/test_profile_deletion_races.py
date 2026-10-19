"""A deleted, renamed or replaced profile stays gone: no writer recreates its home.

Credential, config and profile-lifecycle writers that raced a delete (or ran for a home whose
generation was replaced) used to resurrect the directory or publish a partial one. Moved out of
``tests/tui_gateway/test_profiles_list_delete_race.py``, which keeps the gateway's own cases.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import config, profile_lifecycle, profiles
from hermes_constants import named_profile_is_deleted, reset_hermes_home_override, set_hermes_home_override
from hermes_state import SessionDB


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    monkeypatch.setattr(profiles, "_get_wrapper_dir", lambda: fake_home / ".local" / "bin")
    return hermes_home


def test_provider_auth_write_cannot_resurrect_deleted_profile(home: Path) -> None:
    from hermes_cli import auth

    profile_dir = home / "profiles" / "worker"
    profile_dir.mkdir(parents=True)
    SessionDB(db_path=profile_dir / "state.db").close()
    profiles.delete_profile("worker", yes=True)
    token = set_hermes_home_override(profile_dir)
    try:
        with pytest.raises(FileNotFoundError, match="missing or being deleted"):
            auth._update_config_for_provider("openrouter", "https://example.invalid/api")
    finally:
        reset_hermes_home_override(token)

    assert not profile_dir.exists()
    assert named_profile_is_deleted(profile_dir) is True


def test_direct_auth_store_save_cannot_resurrect_deleted_profile(home: Path) -> None:
    from hermes_cli import auth

    profile_dir = home / "profiles" / "worker"
    profile_dir.mkdir(parents=True)
    profiles.delete_profile("worker", yes=True)
    token = set_hermes_home_override(profile_dir)
    try:
        with pytest.raises(FileNotFoundError, match="missing or being deleted"):
            auth._save_auth_store({"providers": {}})
    finally:
        reset_hermes_home_override(token)

    assert not profile_dir.exists()
    assert named_profile_is_deleted(profile_dir) is True


def test_credential_mirror_rewrite_cannot_recreate_profile_deleted_before_commit(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli import credential_lifecycle
    import utils

    profile_dir = home / "profiles" / "worker"
    profile_dir.mkdir(parents=True)
    (profile_dir / "config.yaml").write_text(
        "model:\n  api_key: old-secret\n",
        encoding="utf-8",
    )
    token = set_hermes_home_override(profile_dir)
    real_atomic_write = utils.atomic_roundtrip_yaml_save

    def delete_before_commit(path, data, **kwargs):
        profile_lifecycle.mark_profile_deleting(profile_dir)
        shutil.rmtree(profile_dir)
        return real_atomic_write(path, data, **kwargs)

    monkeypatch.setattr(utils, "atomic_roundtrip_yaml_save", delete_before_commit)
    try:
        with pytest.raises(FileNotFoundError):
            credential_lifecycle._scrub_config_yaml_mirrors(
                "old-secret",
                "new-secret",
            )
    finally:
        reset_hermes_home_override(token)

    assert not profile_dir.exists()
    assert named_profile_is_deleted(profile_dir) is True


def test_delete_retry_recovers_incarnation_after_partial_rmtree(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_dir = profiles.create_profile("worker", no_alias=True, no_skills=True)
    incarnation = profile_dir.joinpath(".profile-incarnation").read_text(
        encoding="utf-8"
    ).strip()
    real_rmtree = profiles._rmtree_with_retry
    attempts = 0

    def partial_then_remove(path: Path, onerror) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            path.joinpath(".profile-incarnation").unlink()
            raise OSError("injected partial removal")
        real_rmtree(path, onerror)

    monkeypatch.setattr(profiles, "_rmtree_with_retry", partial_then_remove)

    with pytest.raises(RuntimeError, match="Could not remove profile directory"):
        profiles.delete_profile("worker", yes=True)

    tombstone = profile_lifecycle.profile_deletion_marker(profile_dir)
    assert tombstone.read_text(encoding="utf-8").strip() == incarnation
    assert not profile_dir.joinpath(".profile-incarnation").exists()

    profiles.delete_profile("worker", yes=True)

    assert attempts == 2
    assert not profile_dir.exists()


@pytest.mark.parametrize("prior_tombstone", [False, True])
def test_failed_profile_initialization_publishes_no_partial_home(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
    prior_tombstone: bool,
) -> None:
    profile_dir = home / "profiles" / "worker"
    if prior_tombstone:
        profile_lifecycle.mark_profile_deleting(profile_dir)

    def fail_meta(*_args, **_kwargs):
        raise RuntimeError("simulated metadata failure")

    monkeypatch.setattr(profiles, "write_profile_meta", fail_meta)

    with pytest.raises(RuntimeError, match="metadata failure"):
        profiles.create_profile(
            "worker",
            no_alias=True,
            no_skills=True,
            description="must be atomic",
        )

    assert not profile_dir.exists()
    assert named_profile_is_deleted(profile_dir) is prior_tombstone
    staging_root = home / "profiles" / ".profile-creating"
    assert not staging_root.exists() or list(staging_root.iterdir()) == []


def test_named_profile_config_never_mkdir_after_validation_race(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_dir = home / "profiles" / "worker"
    profile_dir.mkdir(parents=True)
    checked = False

    def remove_after_validation(_profile_home: Path | str, **_kwargs) -> bool:
        nonlocal checked
        if not checked:
            checked = True
            shutil.rmtree(profile_dir)
        return False

    monkeypatch.setattr(
        "hermes_constants.named_profile_home_is_unavailable",
        remove_after_validation,
    )
    token = set_hermes_home_override(profile_dir)
    config._HERMES_HOME_ENSURED.pop(str(profile_dir), None)
    try:
        with pytest.raises(FileNotFoundError, match="disappeared during initialization"):
            config.ensure_hermes_home()
    finally:
        reset_hermes_home_override(token)

    assert not profile_dir.exists()


@pytest.mark.parametrize("with_marker", (False, True), ids=("legacy", "marked"))
def test_config_memo_never_crosses_profile_directory_generation(home: Path, with_marker: bool) -> None:
    from hermes_cli.profile_incarnation import write_fresh_profile_incarnation

    profile_dir = home / "profiles" / "worker"
    profile_dir.mkdir(parents=True)
    if with_marker:
        write_fresh_profile_incarnation(profile_dir)
    token = set_hermes_home_override(profile_dir)
    try:
        config.ensure_hermes_home()
        assert (profile_dir / "cron").is_dir()
        shutil.rmtree(profile_dir)
        profile_dir.mkdir()
        if with_marker:
            write_fresh_profile_incarnation(profile_dir)

        config.ensure_hermes_home()

        assert all((profile_dir / subdir).is_dir() for subdir in config._HERMES_HOME_SUBDIRS)
        assert (profile_dir / "SOUL.md").is_file()
    finally:
        config._HERMES_HOME_ENSURED.pop(str(profile_dir), None)
        reset_hermes_home_override(token)


@pytest.mark.platforms("any")
def test_profile_mutation_lock_is_cross_process(home: Path) -> None:
    profile_dir = home / "profiles" / "locked"
    script = (
        "from hermes_cli import profile_lifecycle, profiles; "
        "profile_lifecycle._PROFILE_LIFECYCLE_LOCK_TIMEOUT_SECONDS=0.1; "
        "\ntry:\n profiles.create_profile('locked', no_alias=True, no_skills=True)\n"
        "except TimeoutError:\n raise SystemExit(0)\nraise SystemExit(1)"
    )

    with profile_lifecycle.profile_lifecycle_lease(profile_dir):
        result = subprocess.run(
            [sys.executable, "-c", script],
            env={**os.environ, "HERMES_HOME": str(home)},
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

    assert result.returncode == 0, result.stderr
    assert not profile_dir.exists()
    assert profiles.create_profile("locked", no_alias=True, no_skills=True) == profile_dir


def test_interactive_delete_confirmation_does_not_hold_lifecycle_lock(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_dir = home / "profiles" / "worker"
    profile_dir.mkdir(parents=True)
    probes: list[bool] = []

    def lease_is_free() -> None:
        # Nonblocking on another thread: a lease held across the prompt refuses it.
        try:
            with profile_lifecycle.profile_lifecycle_lease(profile_dir, timeout=0):
                probes.append(True)
        except TimeoutError:
            probes.append(False)

    def cancel_after_probe(_prompt: str) -> str:
        probe = threading.Thread(target=lease_is_free)
        probe.start()
        probe.join(10)
        return "cancel"

    monkeypatch.setattr("builtins.input", cancel_after_probe)

    assert profiles.delete_profile("worker", yes=False) == profile_dir
    assert probes == [True], "the confirmation prompt held the profile lifecycle lease"
    assert profile_dir.is_dir()


@pytest.mark.parametrize("with_marker", (False, True), ids=("legacy", "marked"))
def test_delete_confirmation_never_deletes_replacement_generation(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
    with_marker: bool,
) -> None:
    from hermes_cli.profile_incarnation import write_fresh_profile_incarnation

    profile_dir = home / "profiles" / "worker"
    profile_dir.mkdir(parents=True)
    (profile_dir / "old-generation").write_text("old", encoding="utf-8")
    if with_marker:
        write_fresh_profile_incarnation(profile_dir)
    original_stat = Path.stat
    first_stat = profile_dir.stat()

    def reused_stat(path, *args, **kwargs):
        value = original_stat(path, *args, **kwargs)
        if path != profile_dir:
            return value
        fields = {name: getattr(value, name) for name in dir(value) if name.startswith("st_")}
        fields.update(
            st_dev=first_stat.st_dev, st_ino=first_stat.st_ino, st_ctime_ns=first_stat.st_ctime_ns,
        )
        return SimpleNamespace(**fields)

    def replace_then_confirm(_prompt: str) -> str:
        shutil.rmtree(profile_dir)
        profile_dir.mkdir()
        (profile_dir / "replacement-generation").write_text("new", encoding="utf-8")
        if with_marker:
            write_fresh_profile_incarnation(profile_dir)
        return "worker"

    monkeypatch.setattr(Path, "stat", reused_stat)
    monkeypatch.setattr("builtins.input", replace_then_confirm)

    with pytest.raises(RuntimeError, match="changed while deletion was being confirmed"):
        profiles.delete_profile("worker", yes=False)

    assert (profile_dir / "replacement-generation").read_text(encoding="utf-8") == "new"


def test_failed_rename_preserves_prior_destination_tombstone(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_dir = home / "profiles" / "worker"
    new_dir = home / "profiles" / "research"
    old_dir.mkdir(parents=True)
    profile_lifecycle.mark_profile_deleting(new_dir)

    original_rename = Path.rename

    def fail_target_rename(path: Path, target: Path) -> Path:
        if path == old_dir and target == new_dir:
            raise OSError("simulated rename failure")
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_target_rename)

    with pytest.raises(OSError, match="rename failure"):
        profiles.rename_profile("worker", "research")

    assert old_dir.is_dir()
    assert not new_dir.exists()
    assert named_profile_is_deleted(new_dir) is True


def test_concurrent_profile_use_cannot_restore_retired_name(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_dir = home / "profiles" / "worker"
    old_dir.mkdir(parents=True)
    (old_dir / "config.yaml").write_text("{}\n", encoding="utf-8")
    profiles.set_active_profile("worker")
    finishing = threading.Event()
    release = threading.Event()
    rename_errors: list[Exception] = []
    use_errors: list[Exception] = []

    def block_finish(*_args) -> None:
        finishing.set()
        release.wait(timeout=10)

    monkeypatch.setattr(profiles, "_finish_profile_rename", block_finish)

    def rename() -> None:
        try:
            profiles.rename_profile("worker", "research")
        except Exception as exc:
            rename_errors.append(exc)

    def reuse_old_name() -> None:
        try:
            profiles.set_active_profile("worker")
        except Exception as exc:
            use_errors.append(exc)

    rename_thread = threading.Thread(target=rename)
    rename_thread.start()
    assert finishing.wait(timeout=10)
    use_thread = threading.Thread(target=reuse_old_name)
    use_thread.start()
    release.set()
    rename_thread.join(timeout=3)
    use_thread.join(timeout=3)

    assert not rename_errors
    assert len(use_errors) == 1
    assert isinstance(use_errors[0], FileNotFoundError)
    assert profiles.get_active_profile() == "research"


def test_failed_fresh_import_hides_partial_destination(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_dir = home / "profiles" / "worker"

    def fake_extract(_archive: Path, staging_root: Path) -> None:
        extracted = staging_root / "worker"
        extracted.mkdir()
        (extracted / "config.yaml").write_text("model: test\n", encoding="utf-8")

    def partial_move(_source: str, destination: str) -> None:
        partial = Path(destination)
        partial.mkdir()
        (partial / "config.yaml").write_text("partial", encoding="utf-8")
        raise OSError("simulated cross-filesystem copy failure")

    monkeypatch.setattr(profiles, "safe_extract_targz", fake_extract)
    monkeypatch.setattr(shutil, "move", partial_move)

    with pytest.raises(OSError, match="copy failure"):
        profiles._import_profile_into_home(
            home / "unused.tar.gz",
            "worker",
            "worker",
            profile_dir,
        )

    assert profile_dir.is_dir()
    assert named_profile_is_deleted(profile_dir) is True
    assert profiles.profile_exists("worker") is False
