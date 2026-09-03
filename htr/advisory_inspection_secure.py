"""Task 29 — secure read-only filesystem primitives (R5-02)."""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from htr import paths
from htr.advisory_inspection_constants import (
    MAX_BYTES_PER_ARTIFACT,
    MAX_CONTROL_RECORD_FILE_BYTES,
    MAX_DIRECT_DIRECTORY_ENTRIES_OBSERVED,
    MAX_RAW_READ_BYTES,
)
from htr.advisory_inspection_decoder import ControlJsonDecodeResult, decode_control_json
from htr.bounded_action_control_paths import (
    fstat_identity,
    is_directory_mode,
    is_regular_file_mode,
    open_dir_no_follow,
    openat_dir_no_follow,
    openat_file_no_follow,
    stat_entry_identity,
    stat_entry_mode,
)

_O_RDONLY = os.O_RDONLY
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
FILE_OPEN_FLAGS = _O_RDONLY | _O_NOFOLLOW | _O_CLOEXEC


@dataclass
class SecureReadResult:
    ok: bool
    raw_bytes: bytes = b""
    raw_digest: str = ""
    filesystem_status: str = "filesystem_not_attempted"
    decode: ControlJsonDecodeResult | None = None
    pre_open_size: int | None = None
    file_type_status: str = "file_type_not_inspected"
    hardlink_status: str = "hardlink_not_inspected"
    budget_exceeded: bool = False


@dataclass
class RunsRootContext:
    runs_root_fd: int
    runs_root_path: Path


@dataclass
class WalkContext:
    fds: list[int] = field(default_factory=list)
    current_fd: int = -1

    def close_all(self) -> None:
        for fd in reversed(self.fds):
            try:
                os.close(fd)
            except OSError:
                pass
        self.fds.clear()


def raw_sha256_digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def semantic_sha256_digest(data: bytes) -> str:
    body = data
    if body.endswith(b"\n") and not body.endswith(b"\n\n"):
        body = body[:-1]
    return "sha256:" + hashlib.sha256(body).hexdigest()


def validate_runs_root_s0() -> tuple[RunsRootContext | None, str]:
    """S0 platform / configured root validation."""
    if _O_NOFOLLOW == 0:
        return None, "platform_nofollow_unavailable"

    runs_root_path = paths.default_runs_root()
    try:
        st = os.lstat(str(runs_root_path))
    except FileNotFoundError:
        return None, "runs_root_absent"
    except OSError:
        return None, "runs_root_absent"

    if stat.S_ISLNK(st.st_mode):
        return None, "runs_root_symlink_blocked"
    if not stat.S_ISDIR(st.st_mode):
        return None, "runs_root_wrong_type"

    try:
        runs_root_fd = open_dir_no_follow(runs_root_path, context="runs_root")
    except Exception:
        return None, "runs_root_absent"

    return RunsRootContext(runs_root_fd=runs_root_fd, runs_root_path=runs_root_path), "filesystem_observed"


def open_intermediate_dir(
    parent_fd: int,
    name: str,
    *,
    context: str,
) -> tuple[int | None, str | None]:
    """S1 intermediate directory open with identity compare."""
    try:
        st_mode = stat_entry_mode(parent_fd, name)
    except FileNotFoundError:
        return None, "target_disappeared_before_open"
    except OSError:
        return None, "target_disappeared_before_open"

    if stat.S_ISLNK(st_mode):
        return None, "filesystem_path_component_not_directory"
    if not is_directory_mode(st_mode):
        return None, "filesystem_path_component_not_directory"

    try:
        pre_identity = stat_entry_identity(parent_fd, name)
    except OSError:
        return None, "target_disappeared_before_open"

    try:
        child_fd = openat_dir_no_follow(parent_fd, name, context=context)
    except OSError:
        return None, "target_disappeared_before_open"

    try:
        post_identity = fstat_identity(child_fd)
    except OSError:
        os.close(child_fd)
        return None, "opened_fd_identity_mismatch"

    if pre_identity != post_identity:
        os.close(child_fd)
        return None, "opened_fd_identity_mismatch"

    return child_fd, None


def walk_attempt_path(
    runs_root_ctx: RunsRootContext,
    run_id: str,
    task_id: str,
    attempt_id: str,
) -> tuple[WalkContext | None, str]:
    """Walk run_id/tasks/task_id/attempts/attempt_id under runs root."""
    walk = WalkContext(current_fd=runs_root_ctx.runs_root_fd)
    walk.fds.append(runs_root_ctx.runs_root_fd)

    for name in (run_id, "tasks", task_id, "attempts", attempt_id):
        child_fd, err = open_intermediate_dir(walk.current_fd, name, context=f"walk/{name}")
        if err is not None:
            walk.close_all()
            return None, err
        walk.fds.append(child_fd)
        walk.current_fd = child_fd

    return walk, "filesystem_observed"


def walk_run_path(runs_root_ctx: RunsRootContext, run_id: str) -> tuple[WalkContext | None, str]:
    walk = WalkContext(current_fd=runs_root_ctx.runs_root_fd)
    walk.fds.append(runs_root_ctx.runs_root_fd)

    child_fd, err = open_intermediate_dir(walk.current_fd, run_id, context=f"walk/{run_id}")
    if err is not None:
        walk.close_all()
        return None, err
    walk.fds.append(child_fd)
    walk.current_fd = child_fd
    return walk, "filesystem_observed"


def read_regular_control_file(
    parent_fd: int,
    filename: str,
    *,
    decode_kind: str,
    context: str,
) -> SecureReadResult:
    """S2 regular control file read with five-way identity compare."""
    result = SecureReadResult(ok=False)

    try:
        st_mode = stat_entry_mode(parent_fd, filename)
    except FileNotFoundError:
        result.filesystem_status = "target_disappeared_before_open"
        return result
    except OSError:
        result.filesystem_status = "target_disappeared_before_open"
        return result

    if stat.S_ISLNK(st_mode):
        result.filesystem_status = "filesystem_observed"
        return result
    if not is_regular_file_mode(st_mode):
        result.filesystem_status = "filesystem_observed"
        return result

    try:
        pre_st = os.stat(filename, dir_fd=parent_fd, follow_symlinks=False)
    except OSError:
        result.filesystem_status = "target_disappeared_before_open"
        return result

    pre_identity = (pre_st.st_dev, pre_st.st_ino)
    pre_size = pre_st.st_size
    result.pre_open_size = pre_size

    if pre_st.st_nlink != 1:
        result.hardlink_status = "manifest_hardlink_blocked" if "manifest" in filename else "hardlink_not_inspected"
        result.filesystem_status = "filesystem_observed"
        return result

    if pre_size > MAX_CONTROL_RECORD_FILE_BYTES:
        result.budget_exceeded = True
        result.filesystem_status = "filesystem_observed"
        return result

    try:
        parent_pre_identity = fstat_identity(parent_fd)
    except OSError:
        result.filesystem_status = "filesystem_not_attempted"
        return result

    try:
        file_fd = openat_file_no_follow(parent_fd, filename, FILE_OPEN_FLAGS, context=context)
    except OSError:
        result.filesystem_status = "target_disappeared_before_open"
        return result

    try:
        fd_identity = fstat_identity(file_fd)
        if fd_identity != pre_identity:
            result.filesystem_status = "opened_fd_identity_mismatch"
            return result

        chunks: list[bytes] = []
        total = 0
        while True:
            part = os.read(file_fd, 65536)
            if not part:
                break
            total += len(part)
            if total > MAX_RAW_READ_BYTES:
                result.budget_exceeded = True
                result.filesystem_status = "filesystem_observed"
                return result
            chunks.append(part)

        raw = b"".join(chunks)

        post_fd_identity = fstat_identity(file_fd)
        if post_fd_identity != fd_identity:
            result.filesystem_status = "opened_fd_identity_mismatch"
            return result

        if len(raw) != pre_size:
            result.filesystem_status = "file_size_changed_during_read"
            return result

        try:
            post_entry = os.stat(filename, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            result.filesystem_status = "target_disappeared_after_open"
            return result
        except OSError:
            result.filesystem_status = "target_disappeared_after_open"
            return result

        post_entry_identity = (post_entry.st_dev, post_entry.st_ino)
        if post_entry_identity != fd_identity:
            result.filesystem_status = "target_name_replaced_while_fd_remained_open"
            return result

        try:
            parent_post_identity = fstat_identity(parent_fd)
        except OSError:
            result.filesystem_status = "parent_directory_component_replaced"
            return result

        if parent_pre_identity != parent_post_identity:
            result.filesystem_status = "parent_directory_component_replaced"
            return result

        result.raw_bytes = raw
        result.raw_digest = raw_sha256_digest(raw)
        result.filesystem_status = "filesystem_observed"
        result.file_type_status = "file_regular"
        result.hardlink_status = "hardlink_count_one"
        result.decode = decode_control_json(raw, kind=decode_kind)  # type: ignore[arg-type]
        result.ok = True
        return result
    finally:
        os.close(file_fd)


def os_close_runs_root(ctx: RunsRootContext) -> None:
    try:
        os.close(ctx.runs_root_fd)
    except OSError:
        pass


@dataclass
class HashArtifactResult:
    ok: bool = False
    filesystem_status: str = "filesystem_not_attempted"
    file_type_status: str = "file_type_not_inspected"
    hardlink_status: str = "hardlink_not_inspected"
    identity_status: str = "identity_not_applicable"
    size_status: str = "size_not_inspected"
    digest_status: str = "digest_not_inspected"
    stability_status: str = "stability_not_applicable"
    budget_exceeded: bool = False
    computed_digest: str | None = None
    observed_size: int | None = None


def open_nested_path(parent_fd: int, components: tuple[str, ...], *, context: str) -> tuple[int | None, str]:
    """Open a nested relative path under *parent_fd* (directories + terminal file fd)."""
    if not components:
        return None, "filesystem_path_component_not_directory"

    walk_fd = parent_fd
    opened: list[int] = []

    for index, name in enumerate(components):
        is_last = index == len(components) - 1
        try:
            st_mode = stat_entry_mode(walk_fd, name)
        except FileNotFoundError:
            for fd in reversed(opened):
                os.close(fd)
            return None, "target_disappeared_before_open"
        except OSError:
            for fd in reversed(opened):
                os.close(fd)
            return None, "target_disappeared_before_open"

        if stat.S_ISLNK(st_mode):
            for fd in reversed(opened):
                os.close(fd)
            return None, "path_symlink_blocked" if is_last else "filesystem_path_component_not_directory"

        if is_last:
            if not is_regular_file_mode(st_mode):
                for fd in reversed(opened):
                    os.close(fd)
                if stat.S_ISDIR(st_mode):
                    return None, "filesystem_observed"
                if stat.S_ISFIFO(st_mode):
                    return None, "filesystem_observed"
                if stat.S_ISSOCK(st_mode):
                    return None, "filesystem_observed"
                return None, "filesystem_observed"
            try:
                pre_identity = stat_entry_identity(walk_fd, name)
            except OSError:
                for fd in reversed(opened):
                    os.close(fd)
                return None, "target_disappeared_before_open"
            try:
                file_fd = openat_file_no_follow(walk_fd, name, FILE_OPEN_FLAGS, context=context)
            except OSError:
                for fd in reversed(opened):
                    os.close(fd)
                return None, "target_disappeared_before_open"
            try:
                if fstat_identity(file_fd) != pre_identity:
                    os.close(file_fd)
                    for fd in reversed(opened):
                        os.close(fd)
                    return None, "opened_fd_identity_mismatch"
            except OSError:
                os.close(file_fd)
                for fd in reversed(opened):
                    os.close(fd)
                return None, "opened_fd_identity_mismatch"
            for fd in reversed(opened):
                os.close(fd)
            return file_fd, "filesystem_observed"

        if not is_directory_mode(st_mode):
            for fd in reversed(opened):
                os.close(fd)
            return None, "filesystem_path_component_not_directory"

        child_fd, err = open_intermediate_dir(walk_fd, name, context=f"{context}/{name}")
        if err is not None or child_fd is None:
            for fd in reversed(opened):
                os.close(fd)
            return None, err or "target_disappeared_before_open"
        if walk_fd != parent_fd:
            opened.append(walk_fd)
        walk_fd = child_fd

    return None, "filesystem_path_component_not_directory"


def hash_artifact_file(parent_fd: int, rel_components: tuple[str, ...]) -> HashArtifactResult:
    """L1 artifact hash with five-way identity compare (R5-02 S3)."""
    result = HashArtifactResult()
    if not rel_components:
        result.filesystem_status = "filesystem_path_component_not_directory"
        return result

    file_fd, open_status = open_nested_path(parent_fd, rel_components, context="artifact_l1")
    if file_fd is None:
        result.filesystem_status = "filesystem_observed" if open_status == "path_symlink_blocked" else open_status
        if open_status == "path_symlink_blocked":
            result.file_type_status = "file_symlink"
        return result

    filename = rel_components[-1]
    dir_fd = parent_fd
    if len(rel_components) > 1:
        dir_fd = None
        walk_fd = parent_fd
        opened: list[int] = []
        for name in rel_components[:-1]:
            child_fd, err = open_intermediate_dir(walk_fd, name, context=f"artifact_l1/{name}")
            if err is not None or child_fd is None:
                for fd in reversed(opened):
                    os.close(fd)
                os.close(file_fd)
                result.filesystem_status = err or "target_disappeared_before_open"
                return result
            if walk_fd != parent_fd:
                opened.append(walk_fd)
            walk_fd = child_fd
        dir_fd = walk_fd

    try:
        try:
            pre_st = os.stat(filename, dir_fd=dir_fd, follow_symlinks=False)
        except OSError:
            result.filesystem_status = "target_disappeared_before_open"
            return result

        pre_identity = (pre_st.st_dev, pre_st.st_ino)
        pre_size = pre_st.st_size
        result.observed_size = pre_size

        if stat.S_ISLNK(pre_st.st_mode):
            result.file_type_status = "file_symlink"
            result.filesystem_status = "filesystem_observed"
            return result
        if stat.S_ISDIR(pre_st.st_mode):
            result.file_type_status = "file_directory"
            result.filesystem_status = "filesystem_observed"
            return result
        if stat.S_ISFIFO(pre_st.st_mode):
            result.file_type_status = "file_fifo"
            result.filesystem_status = "filesystem_observed"
            return result
        if stat.S_ISSOCK(pre_st.st_mode):
            result.file_type_status = "file_socket"
            result.filesystem_status = "filesystem_observed"
            return result
        if not stat.S_ISREG(pre_st.st_mode):
            result.file_type_status = "file_type_unknown"
            result.filesystem_status = "filesystem_observed"
            return result

        if pre_st.st_nlink != 1:
            result.hardlink_status = "artifact_hardlink_blocked"
            result.file_type_status = "file_regular"
            result.filesystem_status = "filesystem_observed"
            return result

        if pre_size > MAX_BYTES_PER_ARTIFACT:
            result.budget_exceeded = True
            result.file_type_status = "file_regular"
            result.hardlink_status = "hardlink_count_one"
            result.filesystem_status = "filesystem_observed"
            result.size_status = "size_not_inspected"
            return result

        try:
            parent_pre_identity = fstat_identity(dir_fd)
        except OSError:
            result.filesystem_status = "filesystem_not_attempted"
            return result

        if fstat_identity(file_fd) != pre_identity:
            result.filesystem_status = "opened_fd_identity_mismatch"
            result.identity_status = "identity_fd_mismatch"
            return result

        hasher = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(file_fd, 65536)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_BYTES_PER_ARTIFACT:
                result.budget_exceeded = True
                result.filesystem_status = "filesystem_observed"
                result.file_type_status = "file_regular"
                result.hardlink_status = "hardlink_count_one"
                return result
            hasher.update(chunk)

        post_fd_identity = fstat_identity(file_fd)
        if post_fd_identity != pre_identity:
            result.filesystem_status = "opened_fd_identity_mismatch"
            result.identity_status = "identity_fd_mismatch"
            result.stability_status = "stability_race_detected"
            return result

        if total != pre_size:
            result.filesystem_status = "file_size_changed_during_read"
            result.stability_status = "stability_race_detected"
            result.digest_status = "digest_indeterminate"
            return result

        try:
            post_entry = os.stat(filename, dir_fd=dir_fd, follow_symlinks=False)
        except OSError:
            result.filesystem_status = "target_disappeared_after_open"
            result.stability_status = "stability_race_detected"
            result.digest_status = "digest_indeterminate"
            return result

        if (post_entry.st_dev, post_entry.st_ino) != pre_identity:
            result.filesystem_status = "target_name_replaced_while_fd_remained_open"
            result.stability_status = "stability_race_detected"
            result.digest_status = "digest_indeterminate"
            return result

        try:
            parent_post_identity = fstat_identity(dir_fd)
        except OSError:
            result.filesystem_status = "parent_directory_component_replaced"
            result.stability_status = "stability_race_detected"
            result.digest_status = "digest_indeterminate"
            return result

        if parent_pre_identity != parent_post_identity:
            result.filesystem_status = "parent_directory_component_replaced"
            result.stability_status = "stability_race_detected"
            result.digest_status = "digest_indeterminate"
            return result

        result.computed_digest = "sha256:" + hasher.hexdigest()
        result.filesystem_status = "filesystem_observed"
        result.file_type_status = "file_regular"
        result.hardlink_status = "hardlink_count_one"
        result.size_status = "size_undeclared"
        result.digest_status = "digest_undeclared"
        result.stability_status = "stability_identities_held"
        result.ok = True
        return result
    finally:
        os.close(file_fd)
        if dir_fd is not None and dir_fd != parent_fd:
            os.close(dir_fd)


def list_attempt_dirents(attempt_fd: int) -> list[str]:
    """Sorted direct directory entry names under an attempt fd."""
    from htr.bounded_action_control_paths import list_dir_names_sorted

    return list_dir_names_sorted(attempt_fd)


def open_artifacts_dir(attempt_fd: int) -> tuple[int | None, str | None]:
    """Open the ``artifacts/`` directory under an attempt fd."""
    return open_intermediate_dir(attempt_fd, "artifacts", context="attempt/artifacts")


def scan_unreferenced_artifacts(
    attempt_fd: int,
    referenced_names: set[str],
) -> tuple[list[tuple[str, list[str]]], bool]:
    """Non-recursive metadata scan of ``artifacts/`` (hashed=false)."""
    findings: list[tuple[str, list[str]]] = []
    dir_exceeded = False

    artifacts_fd, err = open_artifacts_dir(attempt_fd)
    if err is not None or artifacts_fd is None:
        return findings, dir_exceeded

    try:
        from htr.bounded_action_control_paths import list_dir_names_sorted

        names = list_dir_names_sorted(artifacts_fd)
        if len(names) > MAX_DIRECT_DIRECTORY_ENTRIES_OBSERVED:
            dir_exceeded = True
            names = names[:MAX_DIRECT_DIRECTORY_ENTRIES_OBSERVED]

        for name in names:
            if name in referenced_names:
                continue
            item_findings: list[str] = []
            try:
                st_mode = stat_entry_mode(artifacts_fd, name)
            except OSError:
                continue
            if stat.S_ISLNK(st_mode):
                item_findings.append("path_symlink_blocked")
            findings.append((name, item_findings))
    finally:
        os.close(artifacts_fd)

    return findings, dir_exceeded


def classify_regular_file_presence(parent_fd: int, filename: str) -> tuple[str, str, str]:
    """Classify absent/symlink/wrong_type/hardlink without full read."""
    try:
        st = os.stat(filename, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return "absent", "filesystem_not_attempted", "file_type_not_inspected"
    except OSError:
        return "absent", "filesystem_not_attempted", "file_type_not_inspected"

    if stat.S_ISLNK(st.st_mode):
        return "symlink", "filesystem_observed", "file_symlink"
    if not stat.S_ISREG(st.st_mode):
        return "wrong_type", "filesystem_observed", "file_directory" if stat.S_ISDIR(st.st_mode) else "file_type_unknown"
    if st.st_nlink != 1:
        return "hardlink", "filesystem_observed", "file_regular"
    if st.st_size > MAX_CONTROL_RECORD_FILE_BYTES:
        return "size_budget", "filesystem_observed", "file_regular"
    return "regular", "filesystem_observed", "file_regular"
