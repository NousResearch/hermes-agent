#!/usr/bin/env python3
"""Owner-provenance seal for the standalone owner-gate stage-zero kit.

This module is copied to a root-only transient path and verified with the
exact locally authored SHA-256 *before* Python executes it remotely.  It then
validates and publishes the complete stdlib-only stage-zero closure outside
the untrusted incoming package.  No incoming payload module is imported.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, BinaryIO, Callable, Mapping, NoReturn, Sequence


SCHEMA = "muncho-owner-gate-outer-stage0-kit.v1"
RECEIPT_SCHEMA = "muncho-owner-gate-outer-stage0-seal.v1"
INCOMING_BASE = Path("/var/tmp/muncho-owner-gate-stage0-incoming")
RELEASE_BASE = Path("/opt/muncho-owner-gate-stage0/releases")
RECEIPT_BASE = Path("/var/lib/muncho-owner-gate-stage0/receipts")
TRANSPORT_RECEIPT_BASE = Path(
    "/var/lib/muncho-owner-gate-stage0/transport-receipts"
)
BUNDLE_INCOMING_BASE = Path("/var/tmp/muncho-owner-gate-incoming")
TRUSTED_RUNNER = "bin/muncho-owner-gate-trusted-stage0"
FIXED_FILES: Mapping[str, int] = {
    "scripts/__init__.py": 0o444,
    "scripts/canary/__init__.py": 0o444,
    "scripts/canary/owner_gate_outer_stage0.py": 0o444,
    "scripts/canary/owner_gate_stage0.py": 0o444,
    "scripts/canary/trusted_signer_stage0.py": 0o444,
    "bin/muncho-owner-gate-trusted-stage0": 0o555,
}
SOURCE_FILES: Mapping[str, str] = {
    **{path: path for path in FIXED_FILES if not path.startswith("bin/")},
    "bin/muncho-owner-gate-trusted-stage0": (
        "ops/muncho/owner-gate/bin/muncho-owner-gate-trusted-stage0"
    ),
}
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_FILE_BYTES = 128 * 1024 * 1024
MAX_TREE_FILES = 20_000
MAX_TREE_BYTES = 4 * 1024 * 1024 * 1024
TREE_STREAM_SCHEMA = "muncho-owner-gate-exact-tree-stream.v1"
TREE_RECEIPT_SCHEMA = "muncho-owner-gate-exact-tree-receipt.v1"
TREE_STREAM_MAGIC = b"MUNCHO_OWNER_GATE_TREE_STREAM_V1\n"

_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class OwnerGateOuterStage0Error(RuntimeError):
    """Stable, secret-free outer stage-zero failure."""


def _error(code: str, exc: BaseException | None = None) -> NoReturn:
    del exc
    raise OwnerGateOuterStage0Error(code) from None


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii", errors="strict")
    except (TypeError, ValueError, UnicodeError) as exc:
        _error("owner_gate_outer_stage0_json_invalid", exc)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _safe_relative(value: Any) -> Path:
    if not isinstance(value, str) or not value:
        _error("owner_gate_outer_stage0_path_invalid")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        _error("owner_gate_outer_stage0_path_invalid")
    return path


def _read_regular(
    path: Path,
    *,
    maximum: int,
    expected_uid: int | None,
    expected_gid: int | None,
    allowed_modes: frozenset[int],
    allow_empty: bool = False,
    allowed_nlinks: frozenset[int] = frozenset({1}),
) -> tuple[bytes, os.stat_result]:
    descriptor: int | None = None
    try:
        before = path.lstat()
        if stat.S_ISLNK(before.st_mode):
            _error("owner_gate_outer_stage0_file_invalid")
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
        if (
            stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(opened.st_mode)
            or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
            or opened.st_nlink not in allowed_nlinks
            or (expected_uid is not None and opened.st_uid != expected_uid)
            or (expected_gid is not None and opened.st_gid != expected_gid)
            or stat.S_IMODE(opened.st_mode) not in allowed_modes
            or (opened.st_size < 1 and not allow_empty)
            or opened.st_size > maximum
        ):
            _error("owner_gate_outer_stage0_file_invalid")
        raw = bytearray()
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                _error("owner_gate_outer_stage0_file_invalid")
            raw.extend(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        if (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            _error("owner_gate_outer_stage0_file_changed")
        return bytes(raw), after
    except OwnerGateOuterStage0Error:
        raise
    except OSError as exc:
        _error("owner_gate_outer_stage0_file_unavailable", exc)
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        os.fsync(descriptor)
    except OSError as exc:
        _error("owner_gate_outer_stage0_sync_failed", exc)
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _ensure_directory(
    path: Path,
    *,
    uid: int,
    gid: int,
    mode: int,
) -> None:
    if not path.is_absolute() or ".." in path.parts:
        _error("owner_gate_outer_stage0_directory_invalid")
    try:
        if not os.path.lexists(path):
            parent = path.parent.lstat()
            if stat.S_ISLNK(parent.st_mode) or not stat.S_ISDIR(parent.st_mode):
                _error("owner_gate_outer_stage0_directory_invalid")
            path.mkdir(mode=mode)
            os.chown(path, uid, gid)
            os.chmod(path, mode)
            _fsync_directory(path.parent)
        state = path.lstat()
    except OwnerGateOuterStage0Error:
        raise
    except OSError as exc:
        _error("owner_gate_outer_stage0_directory_invalid", exc)
    if (
        stat.S_ISLNK(state.st_mode)
        or not stat.S_ISDIR(state.st_mode)
        or state.st_uid != uid
        or state.st_gid != gid
        or stat.S_IMODE(state.st_mode) != mode
    ):
        _error("owner_gate_outer_stage0_directory_invalid")


def _ensure_base_tree(
    base: Path,
    *,
    uid: int,
    gid: int,
    leaf_mode: int,
) -> None:
    missing: list[Path] = []
    current = base
    while not os.path.lexists(current):
        missing.append(current)
        current = current.parent
    current_state = current.lstat()
    if stat.S_ISLNK(current_state.st_mode) or not stat.S_ISDIR(
        current_state.st_mode
    ):
        _error("owner_gate_outer_stage0_directory_invalid")
    for path in reversed(missing):
        mode = leaf_mode if path == base else 0o755
        _ensure_directory(path, uid=uid, gid=gid, mode=mode)
    _ensure_directory(base, uid=uid, gid=gid, mode=leaf_mode)


def _publish_exact_bytes(
    destination: Path,
    payload: bytes,
    *,
    uid: int,
    gid: int,
    mode: int,
    after_open: Callable[[], None] | None = None,
    after_write_chunk: Callable[[], None] | None = None,
    after_fsync: Callable[[], None] | None = None,
    after_link: Callable[[], None] | None = None,
) -> None:
    if not payload or len(payload) > MAX_FILE_BYTES:
        _error("owner_gate_outer_stage0_publish_invalid")
    staging = destination.parent / f".{destination.name}.outer-staged"
    descriptor: int | None = None
    try:
        if os.path.lexists(destination):
            final_raw, final_state = _read_regular(
                destination,
                maximum=len(payload),
                expected_uid=uid,
                expected_gid=gid,
                allowed_modes=frozenset({mode}),
                allowed_nlinks=frozenset({1, 2}),
            )
            if final_raw != payload:
                _error("owner_gate_outer_stage0_publish_conflict")
            if os.path.lexists(staging):
                staged_raw, staged_state = _read_regular(
                    staging,
                    maximum=len(payload),
                    expected_uid=uid,
                    expected_gid=gid,
                    allowed_modes=frozenset({mode}),
                    allow_empty=True,
                    allowed_nlinks=frozenset({1, 2}),
                )
                same = (staged_state.st_dev, staged_state.st_ino) == (
                    final_state.st_dev,
                    final_state.st_ino,
                )
                if (
                    staged_state.st_nlink == 2
                    and not same
                    or staged_state.st_nlink == 1
                    and not payload.startswith(staged_raw)
                ):
                    _error("owner_gate_outer_stage0_publish_conflict")
                staging.unlink()
                _fsync_directory(destination.parent)
            final_raw, _ = _read_regular(
                destination,
                maximum=len(payload),
                expected_uid=uid,
                expected_gid=gid,
                allowed_modes=frozenset({mode}),
            )
            if final_raw != payload:
                _error("owner_gate_outer_stage0_publish_conflict")
            return
        if os.path.lexists(staging):
            try:
                staged_raw, staged_state = _read_regular(
                    staging,
                    maximum=len(payload),
                    expected_uid=uid,
                    expected_gid=gid,
                    allowed_modes=frozenset({mode}),
                    allow_empty=True,
                )
            except OwnerGateOuterStage0Error:
                state = staging.lstat()
                if (
                    stat.S_ISLNK(state.st_mode)
                    or not stat.S_ISREG(state.st_mode)
                    or state.st_nlink != 1
                    or state.st_size != 0
                    or state.st_uid not in {0, uid}
                    or state.st_gid not in {0, gid}
                    or stat.S_IMODE(state.st_mode) & ~mode
                ):
                    raise
                staging.unlink()
                _fsync_directory(destination.parent)
                staged_raw = None
                staged_state = state
            if staged_raw == payload:
                recovered = os.open(
                    staging,
                    os.O_RDONLY
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                )
                try:
                    os.fsync(recovered)
                finally:
                    os.close(recovered)
                _fsync_directory(destination.parent)
            elif staged_raw is None:
                pass
            elif (
                staged_state.st_nlink == 1
                and len(staged_raw) < len(payload)
                and payload.startswith(staged_raw)
            ):
                staging.unlink()
                _fsync_directory(destination.parent)
            else:
                _error("owner_gate_outer_stage0_publish_conflict")
        if not os.path.lexists(staging):
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(staging, flags, mode)
            if after_open is not None:
                after_open()
            os.fchmod(descriptor, mode)
            os.fchown(descriptor, uid, gid)
            view = memoryview(payload)
            while view:
                chunk = view[: min(len(view), 1024 * 1024)]
                written = os.write(descriptor, chunk)
                if written <= 0:
                    raise OSError
                view = view[written:]
                if after_write_chunk is not None:
                    after_write_chunk()
            os.fsync(descriptor)
            if after_fsync is not None:
                after_fsync()
            os.close(descriptor)
            descriptor = None
            _fsync_directory(destination.parent)
        try:
            os.link(staging, destination, follow_symlinks=False)
        except FileExistsError:
            pass
        if after_link is not None:
            after_link()
        _fsync_directory(destination.parent)
        final_raw, final_state = _read_regular(
            destination,
            maximum=len(payload),
            expected_uid=uid,
            expected_gid=gid,
            allowed_modes=frozenset({mode}),
            allowed_nlinks=frozenset({1, 2}),
        )
        staged_raw, staged_state = _read_regular(
            staging,
            maximum=len(payload),
            expected_uid=uid,
            expected_gid=gid,
            allowed_modes=frozenset({mode}),
            allowed_nlinks=frozenset({1, 2}),
        )
        if (
            final_raw != payload
            or staged_raw != payload
            or staged_state.st_nlink == 2
            and (staged_state.st_dev, staged_state.st_ino)
            != (final_state.st_dev, final_state.st_ino)
        ):
            _error("owner_gate_outer_stage0_publish_conflict")
        staging.unlink()
        _fsync_directory(destination.parent)
    except OwnerGateOuterStage0Error:
        raise
    except OSError as exc:
        _error("owner_gate_outer_stage0_publish_failed", exc)
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _git_output(source_root: Path, arguments: Sequence[str]) -> str:
    try:
        completed = subprocess.run(
            ("/usr/bin/git", "-C", str(source_root), *arguments),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
        )
    except (OSError, subprocess.SubprocessError) as exc:
        _error("owner_gate_outer_stage0_git_invalid", exc)
    if completed.returncode != 0:
        _error("owner_gate_outer_stage0_git_invalid")
    return completed.stdout.decode("ascii", errors="strict").strip()


def verify_local_provenance(
    source_root: Path,
    *,
    release_revision: str,
) -> str:
    if (
        not source_root.is_absolute()
        or _REVISION.fullmatch(release_revision or "") is None
        or _git_output(source_root, ("rev-parse", "HEAD")) != release_revision
        or _git_output(
            source_root,
            ("status", "--porcelain=v1", "--untracked-files=all"),
        )
    ):
        _error("owner_gate_outer_stage0_git_invalid")
    tree = _git_output(source_root, ("rev-parse", "HEAD^{tree}"))
    if _REVISION.fullmatch(tree) is None:
        _error("owner_gate_outer_stage0_git_invalid")
    return tree


def build_manifest(
    source_root: Path,
    *,
    release_revision: str,
    source_tree_oid: str,
) -> Mapping[str, Any]:
    if (
        not source_root.is_absolute()
        or _REVISION.fullmatch(release_revision or "") is None
        or _REVISION.fullmatch(source_tree_oid or "") is None
    ):
        _error("owner_gate_outer_stage0_manifest_invalid")
    files: list[Mapping[str, Any]] = []
    for relative, mode in FIXED_FILES.items():
        path = source_root / _safe_relative(SOURCE_FILES[relative])
        raw, _ = _read_regular(
            path,
            maximum=MAX_FILE_BYTES,
            expected_uid=None,
            expected_gid=None,
            allowed_modes=frozenset({0o400, 0o440, 0o444, 0o500, 0o550, 0o555, 0o644, 0o755}),
        )
        files.append({
            "path": relative,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size": len(raw),
            "mode": f"{mode:04o}",
        })
    unsigned = {
        "schema": SCHEMA,
        "source_release_revision": release_revision,
        "source_tree_oid": source_tree_oid,
        "trusted_runner": TRUSTED_RUNNER,
        "files": files,
        "incoming_payload_code_executed": False,
        "network_fetch_required": False,
        "generic_shell_runtime_added": False,
    }
    return {**unsigned, "kit_manifest_sha256": sha256_json(unsigned)}


def validate_manifest(value: Any) -> Mapping[str, Any]:
    fields = {
        "schema",
        "source_release_revision",
        "source_tree_oid",
        "trusted_runner",
        "files",
        "incoming_payload_code_executed",
        "network_fetch_required",
        "generic_shell_runtime_added",
        "kit_manifest_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        _error("owner_gate_outer_stage0_manifest_invalid")
    files = value.get("files")
    if not isinstance(files, list) or len(files) != len(FIXED_FILES):
        _error("owner_gate_outer_stage0_manifest_invalid")
    expected_paths = list(FIXED_FILES)
    for expected_path, item in zip(expected_paths, files, strict=True):
        if (
            not isinstance(item, Mapping)
            or set(item) != {"path", "sha256", "size", "mode"}
            or item.get("path") != expected_path
            or item.get("mode") != f"{FIXED_FILES[expected_path]:04o}"
            or _SHA256.fullmatch(str(item.get("sha256", ""))) is None
            or type(item.get("size")) is not int
            or item["size"] < 1
            or item["size"] > MAX_FILE_BYTES
        ):
            _error("owner_gate_outer_stage0_manifest_invalid")
    unsigned = {
        key: item for key, item in value.items() if key != "kit_manifest_sha256"
    }
    if (
        value.get("schema") != SCHEMA
        or _REVISION.fullmatch(str(value.get("source_release_revision", "")))
        is None
        or _REVISION.fullmatch(str(value.get("source_tree_oid", ""))) is None
        or value.get("trusted_runner") != TRUSTED_RUNNER
        or value.get("incoming_payload_code_executed") is not False
        or value.get("network_fetch_required") is not False
        or value.get("generic_shell_runtime_added") is not False
        or value.get("kit_manifest_sha256") != sha256_json(unsigned)
    ):
        _error("owner_gate_outer_stage0_manifest_invalid")
    return dict(value)


def _tree_source_inventory(root: Path) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    if not root.is_absolute():
        _error("owner_gate_outer_stage0_tree_source_invalid")
    try:
        root_state = root.lstat()
    except OSError as exc:
        _error("owner_gate_outer_stage0_tree_source_invalid", exc)
    if stat.S_ISLNK(root_state.st_mode) or not stat.S_ISDIR(root_state.st_mode):
        _error("owner_gate_outer_stage0_tree_source_invalid")
    directories: list[Mapping[str, Any]] = []
    files: list[Mapping[str, Any]] = []
    allowed_source_modes = frozenset({
        0o400,
        0o440,
        0o444,
        0o500,
        0o550,
        0o555,
        0o600,
        0o640,
        0o644,
        0o700,
        0o750,
        0o755,
    })
    for path in sorted(root.rglob("*"), key=lambda item: str(item.relative_to(root))):
        relative = str(path.relative_to(root))
        _safe_relative(relative)
        try:
            state = path.lstat()
        except OSError as exc:
            _error("owner_gate_outer_stage0_tree_source_invalid", exc)
        if stat.S_ISLNK(state.st_mode):
            _error("owner_gate_outer_stage0_tree_source_invalid")
        if stat.S_ISDIR(state.st_mode):
            directories.append({"path": relative, "mode": "0555"})
            continue
        if not stat.S_ISREG(state.st_mode):
            _error("owner_gate_outer_stage0_tree_source_invalid")
        raw, opened = _read_regular(
            path,
            maximum=MAX_FILE_BYTES,
            expected_uid=None,
            expected_gid=None,
            allowed_modes=allowed_source_modes,
        )
        mode = "0555" if stat.S_IMODE(opened.st_mode) & 0o111 else "0444"
        files.append({
            "path": relative,
            "mode": mode,
            "size": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        })
    if not files or len(files) > MAX_TREE_FILES:
        _error("owner_gate_outer_stage0_tree_source_invalid")
    if sum(item["size"] for item in files) > MAX_TREE_BYTES:
        _error("owner_gate_outer_stage0_tree_source_invalid")
    return directories, files


def build_tree_stream_manifest(
    root: Path,
    *,
    purpose: str,
    release_id: str,
) -> Mapping[str, Any]:
    directories, files = _tree_source_inventory(root)
    projection = {"directories": directories, "files": files}
    unsigned = {
        "schema": TREE_STREAM_SCHEMA,
        "purpose": purpose,
        "release_id": release_id,
        "directories": directories,
        "files": files,
        "source_tree_projection_sha256": sha256_json(projection),
        "symlinks_allowed": False,
        "special_files_allowed": False,
        "extra_paths_allowed": False,
    }
    return validate_tree_stream_manifest({
        **unsigned,
        "transport_manifest_sha256": sha256_json(unsigned),
    })


def validate_tree_stream_manifest(
    value: Any,
    *,
    expected_purpose: str | None = None,
    expected_release_id: str | None = None,
) -> Mapping[str, Any]:
    fields = {
        "schema",
        "purpose",
        "release_id",
        "directories",
        "files",
        "source_tree_projection_sha256",
        "symlinks_allowed",
        "special_files_allowed",
        "extra_paths_allowed",
        "transport_manifest_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        _error("owner_gate_outer_stage0_tree_manifest_invalid")
    purpose = value.get("purpose")
    release_id = value.get("release_id")
    identifier_valid = (
        purpose == "outer-stage0-kit"
        and _SHA256.fullmatch(str(release_id or "")) is not None
        or purpose == "owner-gate-bundle"
        and _REVISION.fullmatch(str(release_id or "")) is not None
    )
    directories = value.get("directories")
    files = value.get("files")
    if (
        not identifier_valid
        or expected_purpose is not None
        and purpose != expected_purpose
        or expected_release_id is not None
        and release_id != expected_release_id
        or not isinstance(directories, list)
        or not isinstance(files, list)
        or not files
        or len(files) > MAX_TREE_FILES
        or len(directories) > MAX_TREE_FILES
    ):
        _error("owner_gate_outer_stage0_tree_manifest_invalid")
    directory_paths: list[str] = []
    for item in directories:
        if (
            not isinstance(item, Mapping)
            or set(item) != {"path", "mode"}
            or item.get("mode") != "0555"
        ):
            _error("owner_gate_outer_stage0_tree_manifest_invalid")
        directory_paths.append(str(_safe_relative(item.get("path"))))
    file_paths: list[str] = []
    total = 0
    for item in files:
        if (
            not isinstance(item, Mapping)
            or set(item) != {"path", "mode", "size", "sha256"}
            or item.get("mode") not in {"0444", "0555"}
            or type(item.get("size")) is not int
            or item["size"] < 1
            or item["size"] > MAX_FILE_BYTES
            or _SHA256.fullmatch(str(item.get("sha256", ""))) is None
        ):
            _error("owner_gate_outer_stage0_tree_manifest_invalid")
        file_paths.append(str(_safe_relative(item.get("path"))))
        total += item["size"]
    directory_set = set(directory_paths)
    file_set = set(file_paths)
    required_directories = {
        str(parent)
        for relative in (*directory_paths, *file_paths)
        for parent in _safe_relative(relative).parents
        if str(parent) != "."
    }
    projection = {"directories": directories, "files": files}
    unsigned = {
        key: item
        for key, item in value.items()
        if key != "transport_manifest_sha256"
    }
    if (
        value.get("schema") != TREE_STREAM_SCHEMA
        or directory_paths != sorted(directory_paths)
        or file_paths != sorted(file_paths)
        or len(directory_set) != len(directory_paths)
        or len(file_set) != len(file_paths)
        or directory_set & file_set
        or not required_directories.issubset(directory_set)
        or total > MAX_TREE_BYTES
        or value.get("source_tree_projection_sha256") != sha256_json(projection)
        or value.get("symlinks_allowed") is not False
        or value.get("special_files_allowed") is not False
        or value.get("extra_paths_allowed") is not False
        or value.get("transport_manifest_sha256") != sha256_json(unsigned)
    ):
        _error("owner_gate_outer_stage0_tree_manifest_invalid")
    return dict(value)


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view[: min(len(view), 1024 * 1024)])
        if written <= 0:
            raise OSError
        view = view[written:]


def write_tree_stream(
    root: Path,
    output: Path,
    *,
    purpose: str,
    release_id: str,
) -> Mapping[str, Any]:
    manifest = build_tree_stream_manifest(
        root,
        purpose=purpose,
        release_id=release_id,
    )
    manifest_raw = canonical_json_bytes(manifest)
    if (
        not output.is_absolute()
        or os.path.lexists(output)
        or not output.parent.is_dir()
        or output.parent.is_symlink()
    ):
        _error("owner_gate_outer_stage0_stream_output_invalid")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            output,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o400,
        )
        os.fchmod(descriptor, 0o400)
        _write_all(descriptor, TREE_STREAM_MAGIC)
        _write_all(descriptor, len(manifest_raw).to_bytes(8, "big"))
        _write_all(descriptor, manifest_raw)
        by_path = {item["path"]: item for item in manifest["files"]}
        for relative, item in by_path.items():
            raw, _ = _read_regular(
                root / _safe_relative(relative),
                maximum=item["size"],
                expected_uid=None,
                expected_gid=None,
                allowed_modes=frozenset({
                    0o400,
                    0o440,
                    0o444,
                    0o500,
                    0o550,
                    0o555,
                    0o600,
                    0o640,
                    0o644,
                    0o700,
                    0o750,
                    0o755,
                }),
            )
            if (
                len(raw) != item["size"]
                or hashlib.sha256(raw).hexdigest() != item["sha256"]
            ):
                _error("owner_gate_outer_stage0_tree_source_changed")
            _write_all(descriptor, raw)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        _fsync_directory(output.parent)
        state = output.stat()
    except OwnerGateOuterStage0Error:
        if descriptor is not None:
            os.close(descriptor)
            descriptor = None
        output.unlink(missing_ok=True)
        _fsync_directory(output.parent)
        raise
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
            descriptor = None
        output.unlink(missing_ok=True)
        _fsync_directory(output.parent)
        _error("owner_gate_outer_stage0_stream_output_failed", exc)
    return {
        "schema": "muncho-owner-gate-exact-tree-stream-build.v1",
        "purpose": purpose,
        "release_id": release_id,
        "stream_path": str(output),
        "stream_size": state.st_size,
        "stream_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "source_tree_projection_sha256": manifest[
            "source_tree_projection_sha256"
        ],
        "file_count": len(manifest["files"]),
        "directory_count": len(manifest["directories"]),
    }


def _read_stream_exact(stream: BinaryIO, size: int) -> bytes:
    if size < 0 or size > MAX_FILE_BYTES:
        _error("owner_gate_outer_stage0_stream_invalid")
    raw = bytearray()
    remaining = size
    while remaining:
        chunk = stream.read(min(remaining, 1024 * 1024))
        if not isinstance(chunk, bytes) or not chunk:
            _error("owner_gate_outer_stage0_stream_invalid")
        raw.extend(chunk)
        remaining -= len(chunk)
    return bytes(raw)


def _verify_exact_tree(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    uid: int,
    gid: int,
) -> Mapping[str, Any]:
    root_state = root.lstat()
    if (
        stat.S_ISLNK(root_state.st_mode)
        or not stat.S_ISDIR(root_state.st_mode)
        or root_state.st_uid != uid
        or root_state.st_gid != gid
        or stat.S_IMODE(root_state.st_mode) != 0o555
    ):
        _error("owner_gate_outer_stage0_received_tree_invalid")
    expected_directories = {
        item["path"]: item for item in manifest["directories"]
    }
    expected_files = {item["path"]: item for item in manifest["files"]}
    observed_directories: set[str] = set()
    observed_files: set[str] = set()
    projection: list[Mapping[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda item: str(item.relative_to(root))):
        relative = str(path.relative_to(root))
        state = path.lstat()
        if stat.S_ISLNK(state.st_mode) or state.st_uid != uid or state.st_gid != gid:
            _error("owner_gate_outer_stage0_received_tree_invalid")
        if stat.S_ISDIR(state.st_mode):
            if (
                relative not in expected_directories
                or stat.S_IMODE(state.st_mode) != 0o555
            ):
                _error("owner_gate_outer_stage0_received_tree_invalid")
            observed_directories.add(relative)
            projection.append({"path": relative, "mode": "0555", "type": "directory"})
            continue
        item = expected_files.get(relative)
        if item is None:
            _error("owner_gate_outer_stage0_received_tree_invalid")
        raw, _ = _read_regular(
            path,
            maximum=item["size"],
            expected_uid=uid,
            expected_gid=gid,
            allowed_modes=frozenset({int(item["mode"], 8)}),
        )
        if (
            len(raw) != item["size"]
            or hashlib.sha256(raw).hexdigest() != item["sha256"]
        ):
            _error("owner_gate_outer_stage0_received_tree_invalid")
        observed_files.add(relative)
        projection.append({
            "path": relative,
            "mode": item["mode"],
            "type": "file",
            "size": item["size"],
            "sha256": item["sha256"],
        })
    if (
        observed_directories != set(expected_directories)
        or observed_files != set(expected_files)
    ):
        _error("owner_gate_outer_stage0_received_tree_invalid")
    return {
        "path": str(root),
        "uid": uid,
        "gid": gid,
        "mode": "0555",
        "projection_sha256": sha256_json(projection),
        "projection_count": len(projection),
    }


def receive_exact_tree_stream(
    stream: BinaryIO,
    *,
    purpose: str,
    release_id: str,
    expected_stream_manifest_sha256: str,
    expected_self_sha256: str,
    kit_base: Path = INCOMING_BASE,
    bundle_base: Path = BUNDLE_INCOMING_BASE,
    receipt_base: Path = TRANSPORT_RECEIPT_BASE,
    owner_uid: int = 0,
    owner_gid: int = 0,
    require_root: bool = True,
    sealer_path: Path | None = None,
) -> Mapping[str, Any]:
    if require_root and os.geteuid() != 0:  # windows-footgun: ok — Debian root boundary
        _error("owner_gate_outer_stage0_root_required")
    if (
        _SHA256.fullmatch(expected_stream_manifest_sha256 or "") is None
        or _SHA256.fullmatch(expected_self_sha256 or "") is None
    ):
        _error("owner_gate_outer_stage0_stream_authority_invalid")
    actual_sealer_path = sealer_path or Path(__file__)
    if not actual_sealer_path.is_absolute():
        _error("owner_gate_outer_stage0_self_invalid")
    self_raw, _ = _read_regular(
        actual_sealer_path,
        maximum=MAX_FILE_BYTES,
        expected_uid=owner_uid,
        expected_gid=owner_gid,
        allowed_modes=frozenset({0o400}),
    )
    if hashlib.sha256(self_raw).hexdigest() != expected_self_sha256:
        _error("owner_gate_outer_stage0_self_invalid")
    if _read_stream_exact(stream, len(TREE_STREAM_MAGIC)) != TREE_STREAM_MAGIC:
        _error("owner_gate_outer_stage0_stream_invalid")
    manifest_size = int.from_bytes(_read_stream_exact(stream, 8), "big")
    if manifest_size < 1 or manifest_size > MAX_MANIFEST_BYTES:
        _error("owner_gate_outer_stage0_stream_invalid")
    manifest_raw = _read_stream_exact(stream, manifest_size)
    if (
        hashlib.sha256(manifest_raw).hexdigest()
        != expected_stream_manifest_sha256
    ):
        _error("owner_gate_outer_stage0_stream_authority_invalid")
    try:
        decoded = json.loads(manifest_raw.decode("utf-8", errors="strict"))
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        _error("owner_gate_outer_stage0_tree_manifest_invalid", exc)
    manifest = validate_tree_stream_manifest(
        decoded,
        expected_purpose=purpose,
        expected_release_id=release_id,
    )
    if canonical_json_bytes(manifest) != manifest_raw:
        _error("owner_gate_outer_stage0_tree_manifest_invalid")
    destination_base = (
        kit_base if purpose == "outer-stage0-kit" else bundle_base
    )
    _ensure_base_tree(
        destination_base,
        uid=owner_uid,
        gid=owner_gid,
        leaf_mode=0o700,
    )
    _ensure_base_tree(
        receipt_base.parent,
        uid=owner_uid,
        gid=owner_gid,
        leaf_mode=0o700,
    )
    _ensure_directory(receipt_base, uid=owner_uid, gid=owner_gid, mode=0o700)
    final = destination_base / release_id
    staging = destination_base / (
        f".{release_id}.{expected_stream_manifest_sha256}.receiving"
    )
    final_exists = os.path.lexists(final)
    if not final_exists:
        if not os.path.lexists(staging):
            staging.mkdir(mode=0o700)
            os.chown(staging, owner_uid, owner_gid)
            _fsync_directory(destination_base)
        staging_state = staging.lstat()
        if (
            stat.S_ISLNK(staging_state.st_mode)
            or not stat.S_ISDIR(staging_state.st_mode)
            or staging_state.st_uid != owner_uid
            or staging_state.st_gid != owner_gid
            or stat.S_IMODE(staging_state.st_mode) not in {0o700, 0o555}
        ):
            _error("owner_gate_outer_stage0_received_tree_invalid")
        if stat.S_IMODE(staging_state.st_mode) == 0o555:
            staging.chmod(0o700)
        for item in manifest["directories"]:
            directory = staging / _safe_relative(item["path"])
            if not os.path.lexists(directory):
                directory.mkdir(mode=0o700)
                os.chown(directory, owner_uid, owner_gid)
            state = directory.lstat()
            if (
                stat.S_ISLNK(state.st_mode)
                or not stat.S_ISDIR(state.st_mode)
                or state.st_uid != owner_uid
                or state.st_gid != owner_gid
                or stat.S_IMODE(state.st_mode) not in {0o700, 0o555}
            ):
                _error("owner_gate_outer_stage0_received_tree_invalid")
            if stat.S_IMODE(state.st_mode) == 0o555:
                directory.chmod(0o700)
    for item in manifest["files"]:
        raw = _read_stream_exact(stream, item["size"])
        if hashlib.sha256(raw).hexdigest() != item["sha256"]:
            _error("owner_gate_outer_stage0_stream_invalid")
        if not final_exists:
            _publish_exact_bytes(
                staging / _safe_relative(item["path"]),
                raw,
                uid=owner_uid,
                gid=owner_gid,
                mode=int(item["mode"], 8),
            )
    trailing = stream.read(1)
    if not isinstance(trailing, bytes) or trailing:
        _error("owner_gate_outer_stage0_stream_invalid")
    if final_exists:
        tree_evidence = _verify_exact_tree(
            final,
            manifest,
            uid=owner_uid,
            gid=owner_gid,
        )
    else:
        for directory in sorted(
            (item for item in staging.rglob("*") if item.is_dir()),
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            directory.chmod(0o555)
        staging.chmod(0o555)
        _fsync_tree(staging, uid=owner_uid, gid=owner_gid)
        _verify_exact_tree(staging, manifest, uid=owner_uid, gid=owner_gid)
        _rename_noreplace(staging, final)
        _fsync_directory(destination_base)
        tree_evidence = _verify_exact_tree(
            final,
            manifest,
            uid=owner_uid,
            gid=owner_gid,
        )
    unsigned = {
        "schema": TREE_RECEIPT_SCHEMA,
        "purpose": purpose,
        "release_id": release_id,
        "stream_manifest_sha256": expected_stream_manifest_sha256,
        "transport_manifest_sha256": manifest["transport_manifest_sha256"],
        "source_tree_projection_sha256": manifest[
            "source_tree_projection_sha256"
        ],
        "receiver_self_sha256": expected_self_sha256,
        "received_tree": tree_evidence,
        "input_code_executed": False,
        "input_code_imported": False,
        "symlinks_received": False,
        "special_files_received": False,
        "extra_paths_received": False,
    }
    receipt = {**unsigned, "receipt_sha256": sha256_json(unsigned)}
    receipt_name = f"{purpose}-{release_id}-{expected_stream_manifest_sha256}.json"
    _publish_exact_bytes(
        receipt_base / receipt_name,
        canonical_json_bytes(receipt),
        uid=owner_uid,
        gid=owner_gid,
        mode=0o400,
    )
    return receipt


def materialize_kit(
    source_root: Path,
    output: Path,
    *,
    release_revision: str,
    source_tree_oid: str | None = None,
) -> Mapping[str, Any]:
    tree = source_tree_oid or verify_local_provenance(
        source_root, release_revision=release_revision
    )
    manifest = build_manifest(
        source_root,
        release_revision=release_revision,
        source_tree_oid=tree,
    )
    if (
        not output.is_absolute()
        or ".." in output.parts
        or os.path.lexists(output)
        or not output.parent.is_dir()
    ):
        _error("owner_gate_outer_stage0_output_invalid")
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    if os.path.lexists(temporary):
        _error("owner_gate_outer_stage0_output_invalid")
    temporary.mkdir(mode=0o700)
    try:
        by_path = {item["path"]: item for item in manifest["files"]}
        for relative, item in by_path.items():
            source = source_root / _safe_relative(SOURCE_FILES[relative])
            raw, _ = _read_regular(
                source,
                maximum=item["size"],
                expected_uid=None,
                expected_gid=None,
                allowed_modes=frozenset({0o400, 0o440, 0o444, 0o500, 0o550, 0o555, 0o644, 0o755}),
            )
            if (
                len(raw) != item["size"]
                or hashlib.sha256(raw).hexdigest() != item["sha256"]
            ):
                _error("owner_gate_outer_stage0_source_changed")
            destination = temporary / _safe_relative(relative)
            destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            descriptor = os.open(
                destination,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                int(item["mode"], 8),
            )
            try:
                view = memoryview(raw)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise OSError
                    view = view[written:]
                os.fchmod(descriptor, int(item["mode"], 8))
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        manifest_raw = canonical_json_bytes(manifest)
        manifest_path = temporary / "outer-stage0-manifest.json"
        descriptor = os.open(
            manifest_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o444,
        )
        try:
            view = memoryview(manifest_raw)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError
                view = view[written:]
            os.fchmod(descriptor, 0o444)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        for directory in sorted(
            (item for item in temporary.rglob("*") if item.is_dir()),
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            directory.chmod(0o555)
        temporary.chmod(0o555)
        os.rename(temporary, output)
        _fsync_directory(output.parent)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def _rename_noreplace(source: Path, destination: Path) -> None:
    if os.path.lexists(destination):
        _error("owner_gate_outer_stage0_release_conflict")
    if sys.platform != "linux":
        os.rename(source, destination)
        return
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        _error("owner_gate_outer_stage0_noreplace_unavailable")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    if renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1):
        failure = ctypes.get_errno()
        if failure == errno.EEXIST:
            _error("owner_gate_outer_stage0_release_conflict")
        _error(
            "owner_gate_outer_stage0_publish_failed",
            OSError(failure, os.strerror(failure)),
        )


def _fsync_tree(root: Path, *, uid: int, gid: int) -> None:
    paths = sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True)
    for path in paths:
        state = path.lstat()
        if stat.S_ISLNK(state.st_mode) or state.st_uid != uid or state.st_gid != gid:
            _error("owner_gate_outer_stage0_release_invalid")
        if stat.S_ISREG(state.st_mode):
            descriptor = os.open(
                path,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        elif not stat.S_ISDIR(state.st_mode):
            _error("owner_gate_outer_stage0_release_invalid")
    directories = [
        path for path in paths if stat.S_ISDIR(path.lstat().st_mode)
    ]
    for directory in (*directories, root):
        _fsync_directory(directory)


def _verify_release(
    release: Path,
    manifest: Mapping[str, Any],
    *,
    uid: int,
    gid: int,
) -> Mapping[str, Any]:
    state = release.lstat()
    if (
        stat.S_ISLNK(state.st_mode)
        or not stat.S_ISDIR(state.st_mode)
        or state.st_uid != uid
        or state.st_gid != gid
        or stat.S_IMODE(state.st_mode) != 0o555
    ):
        _error("owner_gate_outer_stage0_release_invalid")
    expected_files = {
        **{item["path"]: item for item in manifest["files"]},
        "outer-stage0-manifest.json": {
            "sha256": hashlib.sha256(canonical_json_bytes(manifest)).hexdigest(),
            "size": len(canonical_json_bytes(manifest)),
            "mode": "0444",
        },
    }
    expected_directories = {
        str(parent)
        for relative in expected_files
        for parent in _safe_relative(relative).parents
        if str(parent) != "."
    }
    observed_paths: set[str] = set()
    observed_directories: set[str] = set()
    projection: list[Mapping[str, Any]] = []
    for path in sorted(release.rglob("*"), key=lambda item: str(item.relative_to(release))):
        relative = str(path.relative_to(release))
        item_state = path.lstat()
        if (
            stat.S_ISLNK(item_state.st_mode)
            or item_state.st_uid != uid
            or item_state.st_gid != gid
        ):
            _error("owner_gate_outer_stage0_release_invalid")
        if stat.S_ISDIR(item_state.st_mode):
            if (
                relative not in expected_directories
                or stat.S_IMODE(item_state.st_mode) != 0o555
            ):
                _error("owner_gate_outer_stage0_release_invalid")
            observed_directories.add(relative)
            projection.append({"path": relative, "type": "directory", "mode": "0555"})
            continue
        expected = expected_files.get(relative)
        if expected is None or not stat.S_ISREG(item_state.st_mode):
            _error("owner_gate_outer_stage0_release_invalid")
        expected_mode = expected.get("mode")
        expected_size = expected.get("size")
        expected_sha256 = expected.get("sha256")
        if (
            not isinstance(expected_mode, str)
            or type(expected_size) is not int
            or not isinstance(expected_sha256, str)
        ):
            _error("owner_gate_outer_stage0_release_invalid")
        raw, _ = _read_regular(
            path,
            maximum=MAX_FILE_BYTES,
            expected_uid=uid,
            expected_gid=gid,
            allowed_modes=frozenset({int(expected_mode, 8)}),
        )
        if (
            len(raw) != expected_size
            or hashlib.sha256(raw).hexdigest() != expected_sha256
        ):
            _error("owner_gate_outer_stage0_release_invalid")
        observed_paths.add(relative)
        projection.append({
            "path": relative,
            "type": "file",
            "mode": expected_mode,
            "size": len(raw),
            "sha256": expected_sha256,
        })
    if (
        observed_paths != set(expected_files)
        or observed_directories != expected_directories
    ):
        _error("owner_gate_outer_stage0_release_invalid")
    return {
        "path": str(release),
        "uid": uid,
        "gid": gid,
        "mode": "0555",
        "projection_sha256": sha256_json(projection),
        "projection_count": len(projection),
    }


def seal_incoming_kit(
    incoming: Path,
    *,
    expected_manifest_sha256: str,
    incoming_base: Path = INCOMING_BASE,
    release_base: Path = RELEASE_BASE,
    receipt_base: Path = RECEIPT_BASE,
    owner_uid: int = 0,
    owner_gid: int = 0,
    require_root: bool = True,
    sealer_path: Path | None = None,
) -> Mapping[str, Any]:
    if require_root and os.geteuid() != 0:  # windows-footgun: ok — Debian root boundary
        _error("owner_gate_outer_stage0_root_required")
    if (
        _SHA256.fullmatch(expected_manifest_sha256 or "") is None
        or incoming.parent != incoming_base
        or incoming.name != expected_manifest_sha256
        or not incoming.is_absolute()
        or not release_base.is_absolute()
        or not receipt_base.is_absolute()
    ):
        _error("owner_gate_outer_stage0_incoming_invalid")
    incoming_base_state = incoming_base.lstat()
    incoming_state = incoming.lstat()
    if (
        stat.S_ISLNK(incoming_base_state.st_mode)
        or not stat.S_ISDIR(incoming_base_state.st_mode)
        or incoming_base_state.st_uid != owner_uid
        or incoming_base_state.st_gid != owner_gid
        or stat.S_IMODE(incoming_base_state.st_mode) not in {0o700, 0o755}
        or stat.S_ISLNK(incoming_state.st_mode)
        or not stat.S_ISDIR(incoming_state.st_mode)
        or incoming_state.st_uid != owner_uid
        or incoming_state.st_gid != owner_gid
        or stat.S_IMODE(incoming_state.st_mode) != 0o555
    ):
        _error("owner_gate_outer_stage0_incoming_invalid")
    manifest_raw, _ = _read_regular(
        incoming / "outer-stage0-manifest.json",
        maximum=MAX_MANIFEST_BYTES,
        expected_uid=owner_uid,
        expected_gid=owner_gid,
        allowed_modes=frozenset({0o400, 0o440, 0o444}),
    )
    if hashlib.sha256(manifest_raw).hexdigest() != expected_manifest_sha256:
        _error("owner_gate_outer_stage0_manifest_digest_invalid")
    try:
        decoded = json.loads(manifest_raw.decode("utf-8", errors="strict"))
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        _error("owner_gate_outer_stage0_manifest_invalid", exc)
    manifest = validate_manifest(decoded)
    if canonical_json_bytes(manifest) != manifest_raw:
        _error("owner_gate_outer_stage0_manifest_invalid")
    by_path = {item["path"]: item for item in manifest["files"]}
    try:
        _verify_release(incoming, manifest, uid=owner_uid, gid=owner_gid)
    except OwnerGateOuterStage0Error as exc:
        _error("owner_gate_outer_stage0_incoming_invalid", exc)
    sealer_item = by_path["scripts/canary/owner_gate_outer_stage0.py"]
    actual_sealer_path = sealer_path or Path(__file__)
    if not actual_sealer_path.is_absolute():
        _error("owner_gate_outer_stage0_self_invalid")
    self_raw, _ = _read_regular(
        actual_sealer_path,
        maximum=MAX_FILE_BYTES,
        expected_uid=owner_uid,
        expected_gid=owner_gid,
        allowed_modes=frozenset({0o400}),
    )
    if (
        len(self_raw) != sealer_item["size"]
        or hashlib.sha256(self_raw).hexdigest() != sealer_item["sha256"]
    ):
        _error("owner_gate_outer_stage0_self_invalid")
    _ensure_base_tree(
        release_base.parent,
        uid=owner_uid,
        gid=owner_gid,
        leaf_mode=0o755,
    )
    _ensure_directory(release_base, uid=owner_uid, gid=owner_gid, mode=0o755)
    _ensure_base_tree(
        receipt_base.parent,
        uid=owner_uid,
        gid=owner_gid,
        leaf_mode=0o700,
    )
    _ensure_directory(receipt_base, uid=owner_uid, gid=owner_gid, mode=0o700)
    final = release_base / expected_manifest_sha256
    staging = release_base / f".{expected_manifest_sha256}.sealing"
    if os.path.lexists(final):
        release_evidence = _verify_release(
            final, manifest, uid=owner_uid, gid=owner_gid
        )
    else:
        if not os.path.lexists(staging):
            staging.mkdir(mode=0o700)
            os.chown(staging, owner_uid, owner_gid)
            _fsync_directory(release_base)
        state = staging.lstat()
        if (
            stat.S_ISLNK(state.st_mode)
            or not stat.S_ISDIR(state.st_mode)
            or state.st_uid != owner_uid
            or state.st_gid != owner_gid
            or stat.S_IMODE(state.st_mode) not in {0o700, 0o555}
        ):
            _error("owner_gate_outer_stage0_release_invalid")
        if stat.S_IMODE(state.st_mode) == 0o555:
            staging.chmod(0o700)
        for relative, item in by_path.items():
            try:
                source_raw, _ = _read_regular(
                    incoming / _safe_relative(relative),
                    maximum=item["size"],
                    expected_uid=None,
                    expected_gid=None,
                    allowed_modes=frozenset({int(item["mode"], 8)}),
                )
            except OwnerGateOuterStage0Error as exc:
                _error("owner_gate_outer_stage0_incoming_invalid", exc)
            if (
                len(source_raw) != item["size"]
                or hashlib.sha256(source_raw).hexdigest() != item["sha256"]
            ):
                _error("owner_gate_outer_stage0_incoming_invalid")
            destination = staging / _safe_relative(relative)
            destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            for parent in destination.parents:
                if parent == staging.parent:
                    break
                if parent.is_relative_to(staging):
                    os.chown(parent, owner_uid, owner_gid)
                    os.chmod(parent, 0o700)
            _publish_exact_bytes(
                destination,
                source_raw,
                uid=owner_uid,
                gid=owner_gid,
                mode=int(item["mode"], 8),
            )
        _publish_exact_bytes(
            staging / "outer-stage0-manifest.json",
            manifest_raw,
            uid=owner_uid,
            gid=owner_gid,
            mode=0o444,
        )
        for directory in sorted(
            (item for item in staging.rglob("*") if item.is_dir()),
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            directory.chmod(0o555)
        staging.chmod(0o555)
        _fsync_tree(staging, uid=owner_uid, gid=owner_gid)
        _verify_release(staging, manifest, uid=owner_uid, gid=owner_gid)
        _rename_noreplace(staging, final)
        _fsync_directory(release_base)
        release_evidence = _verify_release(
            final, manifest, uid=owner_uid, gid=owner_gid
        )
    unsigned = {
        "schema": RECEIPT_SCHEMA,
        "kit_manifest_sha256": expected_manifest_sha256,
        "kit_self_hash": manifest["kit_manifest_sha256"],
        "source_release_revision": manifest["source_release_revision"],
        "source_tree_oid": manifest["source_tree_oid"],
        "outer_sealer_sha256": hashlib.sha256(self_raw).hexdigest(),
        "trusted_runner": str(final / TRUSTED_RUNNER),
        "release": release_evidence,
        "incoming_payload_code_executed": False,
        "incoming_payload_imported": False,
        "network_fetch_performed": False,
        "generic_shell_runtime_added": False,
    }
    receipt = {**unsigned, "receipt_sha256": sha256_json(unsigned)}
    _publish_exact_bytes(
        receipt_base / f"{expected_manifest_sha256}.json",
        canonical_json_bytes(receipt),
        uid=owner_uid,
        gid=owner_gid,
        mode=0o400,
    )
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    build = subparsers.add_parser("build", allow_abbrev=False)
    build.add_argument("--source-root", type=Path, required=True)
    build.add_argument("--release-revision", required=True)
    build.add_argument("--output", type=Path, required=True)
    stream_build = subparsers.add_parser("stream-build", allow_abbrev=False)
    stream_build.add_argument("--source-root", type=Path, required=True)
    stream_build.add_argument(
        "--purpose",
        choices=("outer-stage0-kit", "owner-gate-bundle"),
        required=True,
    )
    stream_build.add_argument("--release-id", required=True)
    stream_build.add_argument("--output", type=Path, required=True)
    stream_receive = subparsers.add_parser(
        "stream-receive",
        allow_abbrev=False,
    )
    stream_receive.add_argument(
        "--purpose",
        choices=("outer-stage0-kit", "owner-gate-bundle"),
        required=True,
    )
    stream_receive.add_argument("--release-id", required=True)
    stream_receive.add_argument(
        "--expected-stream-manifest-sha256",
        required=True,
    )
    stream_receive.add_argument("--expected-self-sha256", required=True)
    seal = subparsers.add_parser("seal", allow_abbrev=False)
    seal.add_argument("--incoming", type=Path, required=True)
    seal.add_argument("--expected-manifest-sha256", required=True)
    arguments = parser.parse_args(argv)
    if arguments.operation == "build":
        result = materialize_kit(
            arguments.source_root,
            arguments.output,
            release_revision=arguments.release_revision,
        )
    elif arguments.operation == "stream-build":
        result = write_tree_stream(
            arguments.source_root,
            arguments.output,
            purpose=arguments.purpose,
            release_id=arguments.release_id,
        )
    elif arguments.operation == "stream-receive":
        result = receive_exact_tree_stream(
            sys.stdin.buffer,
            purpose=arguments.purpose,
            release_id=arguments.release_id,
            expected_stream_manifest_sha256=(
                arguments.expected_stream_manifest_sha256
            ),
            expected_self_sha256=arguments.expected_self_sha256,
        )
    else:
        result = seal_incoming_kit(
            arguments.incoming,
            expected_manifest_sha256=arguments.expected_manifest_sha256,
        )
    print(canonical_json_bytes(result).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FIXED_FILES",
    "BUNDLE_INCOMING_BASE",
    "INCOMING_BASE",
    "OwnerGateOuterStage0Error",
    "RECEIPT_BASE",
    "RELEASE_BASE",
    "SCHEMA",
    "SOURCE_FILES",
    "TRUSTED_RUNNER",
    "TRANSPORT_RECEIPT_BASE",
    "TREE_RECEIPT_SCHEMA",
    "TREE_STREAM_MAGIC",
    "TREE_STREAM_SCHEMA",
    "build_manifest",
    "build_tree_stream_manifest",
    "canonical_json_bytes",
    "materialize_kit",
    "receive_exact_tree_stream",
    "seal_incoming_kit",
    "validate_manifest",
    "validate_tree_stream_manifest",
    "verify_local_provenance",
    "write_tree_stream",
]
