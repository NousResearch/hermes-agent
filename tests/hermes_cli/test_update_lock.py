"""Focused tests for the repository-scoped legacy update lock."""

from pathlib import Path

import pytest


def test_shared_lock_identity_normalizes_symlinked_repository(tmp_path: Path) -> None:
    from hermes_cli.update_lock import shared_update_lock_identity, shared_update_lock_path

    repo = tmp_path / "repo"
    repo.mkdir()
    alias = tmp_path / "repo-alias"
    alias.symlink_to(repo, target_is_directory=True)

    assert shared_update_lock_identity(alias) == shared_update_lock_identity(repo)
    assert shared_update_lock_path(alias) == shared_update_lock_path(repo)
    assert repo not in shared_update_lock_path(repo).parents


def test_shared_lock_rejects_missing_and_non_directory_identities(tmp_path: Path) -> None:
    from hermes_cli.update_lock import UpdateLockError, shared_update_lock_identity

    regular_file = tmp_path / "not-a-repository"
    regular_file.write_text("x")

    with pytest.raises(UpdateLockError):
        shared_update_lock_identity(tmp_path / "missing")
    with pytest.raises(UpdateLockError):
        shared_update_lock_identity(regular_file)


@pytest.mark.parametrize("timeout", [-1.0, float("nan"), float("inf"), 301.0])
def test_shared_lock_rejects_unbounded_timeouts(tmp_path: Path, timeout: float) -> None:
    from hermes_cli.update_lock import UpdateLockError, acquire_shared_update_lock

    repo = tmp_path / "repo"
    repo.mkdir()
    with pytest.raises(UpdateLockError):
        with acquire_shared_update_lock(repo, timeout_seconds=timeout):
            pytest.fail("invalid timeout must fail before acquisition")


def test_shared_lock_is_reentrant_for_same_repository(tmp_path: Path, monkeypatch) -> None:
    from hermes_cli import update_lock
    from hermes_cli.update_lock import acquire_shared_update_lock

    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(update_lock, "_LOCK_ROOT", tmp_path / "locks")

    with acquire_shared_update_lock(repo, timeout_seconds=0.0):
        with acquire_shared_update_lock(repo.resolve(), timeout_seconds=0.0):
            pass


def test_lock_namespace_swap_after_validation_fails_closed(
    tmp_path: Path, monkeypatch
) -> None:
    from hermes_cli import update_lock
    from hermes_cli.update_lock import UpdateLockError, acquire_shared_update_lock

    repo = tmp_path / "repo"
    repo.mkdir()
    lock_root = tmp_path / "locks"
    attacker_root = tmp_path / "attacker"
    attacker_root.mkdir(mode=0o700)
    monkeypatch.setattr(update_lock, "_LOCK_ROOT", lock_root)

    real_validate = update_lock._validate_private_directory
    validation_count = 0

    def validate_then_swap(path: Path):
        nonlocal validation_count
        validated = real_validate(path)
        validation_count += 1
        if validation_count == 2:
            path.rename(tmp_path / "parked-locks")
            attacker_root.rename(path)
        return validated

    monkeypatch.setattr(update_lock, "_validate_private_directory", validate_then_swap)

    with pytest.raises(UpdateLockError):
        with acquire_shared_update_lock(repo, timeout_seconds=0.0):
            pytest.fail("swapped lock namespace must never be acquired")
    assert list(lock_root.iterdir()) == []
