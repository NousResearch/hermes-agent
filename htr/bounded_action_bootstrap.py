"""Root, Successor, and case coordination bootstrap for Task 28 Phase 28A."""

from __future__ import annotations

import fcntl
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from htr import paths
from htr.bounded_action_control_paths import (
    CONTROL_DIR_MODE,
    fsync_dir_fd,
    mkdirat,
    open_dir_no_follow,
    openat_dir_no_follow,
    validate_dir_mode_0700,
    validate_new_task28_ownership,
    validate_preexisting_control_dir,
)
from htr.state import BoundedActionDurabilityError, BoundedActionValidationError

_O_CREAT = os.O_CREAT
_O_EXCL = os.O_EXCL
_O_DIRECTORY = os.O_DIRECTORY
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)


@dataclass
class PublicationBootstrap:
    """Root publication coordination handles after bootstrap (lock order: root → successor → case)."""

    runs_root_fd: int
    control_fd: int
    bounded_actions_fd: int
    publication_coord_fd: int
    successor_coord_fd: int | None = None
    case_fd: int | None = None

    def release(self) -> None:
        release_publication_locks(self)


# Back-compat alias for earlier draft name.
PublicationLocks = PublicationBootstrap


def _open_or_create_control(base_dir: Path | None) -> tuple[int, int, bool]:
    runs_root = paths.runs_root(base_dir)
    runs_fd = open_dir_no_follow(runs_root, context="runs_root")
    control_path = paths.control_root(base_dir)
    created_control = False
    if control_path.exists():
        control_fd = open_dir_no_follow(control_path, context=".control")
        validate_preexisting_control_dir(control_fd, context=".control")
    else:
        try:
            os.mkdir(control_path, CONTROL_DIR_MODE)
            created_control = True
        except FileExistsError:
            pass
        control_fd = open_dir_no_follow(control_path, context=".control")
        validate_preexisting_control_dir(control_fd, context=".control")
        if created_control:
            validate_new_task28_ownership(control_fd, context=".control")
    if created_control:
        try:
            fsync_dir_fd(runs_fd)
        except OSError as exc:
            raise BoundedActionDurabilityError(
                f"control bootstrap fsync failed: {exc}",
                proposal_id="",
                record_name="bootstrap",
                durability_stage="control_fsync",
                record_may_have_committed=False,
            ) from exc
    return runs_fd, control_fd, created_control


def _open_or_create_bounded_actions(control_fd: int, base_dir: Path | None) -> tuple[int, bool]:
    name = paths.BOUNDED_ACTIONS_DIR_NAME
    created = mkdirat(control_fd, name, CONTROL_DIR_MODE, context="bounded_actions")
    ba_fd = openat_dir_no_follow(control_fd, name, context="bounded_actions")
    validate_dir_mode_0700(ba_fd, context="bounded_actions")
    if created:
        validate_new_task28_ownership(ba_fd, context="bounded_actions")
        try:
            fsync_dir_fd(ba_fd)
            fsync_dir_fd(control_fd)
        except OSError as exc:
            raise BoundedActionDurabilityError(
                f"bounded_actions bootstrap fsync failed: {exc}",
                proposal_id="",
                record_name="bootstrap",
                durability_stage="bounded_actions_fsync",
                record_may_have_committed=False,
            ) from exc
    return ba_fd, created


def _open_or_create_publication_coord(bounded_actions_fd: int) -> tuple[int, bool]:
    name = paths.PUBLICATION_COORD_DIR_NAME
    created = mkdirat(bounded_actions_fd, name, CONTROL_DIR_MODE, context="_publication_coord")
    coord_fd = openat_dir_no_follow(bounded_actions_fd, name, context="_publication_coord")
    validate_dir_mode_0700(coord_fd, context="_publication_coord")
    if created:
        validate_new_task28_ownership(coord_fd, context="_publication_coord")
        fsync_dir_fd(coord_fd)
        fsync_dir_fd(bounded_actions_fd)
    return coord_fd, created


def bootstrap_publication_tree(base_dir: Path | None) -> PublicationBootstrap:
    runs_fd, control_fd, _ = _open_or_create_control(base_dir)
    try:
        ba_fd, _ = _open_or_create_bounded_actions(control_fd, base_dir)
    except Exception:
        os.close(control_fd)
        os.close(runs_fd)
        raise
    try:
        pub_fd, created = _open_or_create_publication_coord(ba_fd)
    except Exception:
        os.close(ba_fd)
        os.close(control_fd)
        os.close(runs_fd)
        raise
    if created:
        try:
            fsync_dir_fd(pub_fd)
            fsync_dir_fd(ba_fd)
            fsync_dir_fd(control_fd)
            fsync_dir_fd(runs_fd)
        except OSError as exc:
            raise BoundedActionDurabilityError(
                f"publication bootstrap fsync failed: {exc}",
                proposal_id="",
                record_name="bootstrap",
                durability_stage="parent_fsync",
                record_may_have_committed=False,
            ) from exc
    return PublicationBootstrap(
        runs_root_fd=runs_fd,
        control_fd=control_fd,
        bounded_actions_fd=ba_fd,
        publication_coord_fd=pub_fd,
    )


@contextmanager
def root_publication_lock(locks: PublicationBootstrap) -> Iterator[int]:
    fcntl.flock(locks.publication_coord_fd, fcntl.LOCK_EX)
    try:
        yield locks.publication_coord_fd
    finally:
        fcntl.flock(locks.publication_coord_fd, fcntl.LOCK_UN)


@contextmanager
def successor_coord_lock(locks: PublicationBootstrap, successor_run_id: str, base_dir: Path | None) -> Iterator[int]:
    name = paths.SUCCESSOR_COORD_DIR_NAME
    succ_root_created = mkdirat(locks.bounded_actions_fd, name, CONTROL_DIR_MODE, context="_successor_coord")
    succ_root_fd = openat_dir_no_follow(locks.bounded_actions_fd, name, context="_successor_coord")
    validate_dir_mode_0700(succ_root_fd, context="_successor_coord")
    if succ_root_created:
        validate_new_task28_ownership(succ_root_fd, context="_successor_coord")
        fsync_dir_fd(succ_root_fd)
        fsync_dir_fd(locks.bounded_actions_fd)
    created = mkdirat(succ_root_fd, successor_run_id, CONTROL_DIR_MODE, context=successor_run_id)
    succ_fd = openat_dir_no_follow(succ_root_fd, successor_run_id, context=successor_run_id)
    validate_dir_mode_0700(succ_fd, context=successor_run_id)
    if created:
        validate_new_task28_ownership(succ_fd, context=successor_run_id)
        fsync_dir_fd(succ_fd)
        fsync_dir_fd(succ_root_fd)
    os.close(succ_root_fd)
    fcntl.flock(succ_fd, fcntl.LOCK_EX)
    try:
        yield succ_fd
    finally:
        fcntl.flock(succ_fd, fcntl.LOCK_UN)
        os.close(succ_fd)


@contextmanager
def case_lock(locks: PublicationBootstrap, proposal_id: str, *, create: bool) -> Iterator[int]:
    if create:
        created = mkdirat(locks.bounded_actions_fd, proposal_id, CONTROL_DIR_MODE, context=proposal_id)
        case_fd = openat_dir_no_follow(locks.bounded_actions_fd, proposal_id, context=proposal_id)
        validate_dir_mode_0700(case_fd, context=proposal_id, require_ownership=created)
        if created:
            validate_new_task28_ownership(case_fd, context=proposal_id)
            fsync_dir_fd(case_fd)
            fsync_dir_fd(locks.bounded_actions_fd)
    else:
        case_fd = openat_dir_no_follow(locks.bounded_actions_fd, proposal_id, context=proposal_id)
        validate_dir_mode_0700(case_fd, context=proposal_id)
    fcntl.flock(case_fd, fcntl.LOCK_EX)
    try:
        yield case_fd
    finally:
        fcntl.flock(case_fd, fcntl.LOCK_UN)
        os.close(case_fd)


def release_publication_locks(locks: PublicationBootstrap) -> None:
    for fd in (
        locks.case_fd,
        locks.successor_coord_fd,
        locks.publication_coord_fd,
        locks.bounded_actions_fd,
        locks.control_fd,
        locks.runs_root_fd,
    ):
        if fd is not None:
            os.close(fd)
