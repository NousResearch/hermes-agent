"""Safe ``tar.gz`` primitives shared by the profile and kanban transfer paths."""

from __future__ import annotations

import os
import shutil
import tarfile
import tempfile
from contextlib import suppress
from pathlib import Path, PurePosixPath, PureWindowsPath


def normalize_archive_parts(member_name: str) -> list[str]:
    """Return safe path parts for an archive member, or raise ``ValueError``.

    Rejects absolute paths (POSIX and Windows, including drive letters), empty names, and any ``..``
    component. Backslashes are folded to ``/`` first so a Windows-authored archive can't smuggle a
    separator past the POSIX parse.
    """
    normalized_name = member_name.replace("\\", "/")
    posix_path = PurePosixPath(normalized_name)
    windows_path = PureWindowsPath(member_name)
    parts = [part for part in posix_path.parts if part not in {"", "."}]
    if (not normalized_name or posix_path.is_absolute() or windows_path.is_absolute()
            or windows_path.drive or not parts or ".." in parts):
        raise ValueError(f"Unsafe archive member path: {member_name}")
    return parts


def make_targz(base: str, root_dir: str, base_dir: str) -> str:
    """Create ``<base>.tar.gz`` of ``root_dir/base_dir`` in GNU tar format.

    Writes to a sibling temp file and renames onto the archive path only after it is fully written.
    """
    archive_path = f"{base}.tar.gz"
    dest_dir = os.path.dirname(archive_path) or "."
    fd, tmp_path = tempfile.mkstemp(dir=dest_dir, prefix=".archive_", suffix=".tar.gz.tmp")
    try:
        with os.fdopen(fd, "wb") as f, \
                tarfile.open(fileobj=f, mode="w:gz", format=tarfile.GNU_FORMAT) as tf:
            tf.add(str(Path(root_dir) / base_dir), arcname=base_dir)
        os.replace(tmp_path, archive_path)
    except BaseException:
        with suppress(OSError):
            os.unlink(tmp_path)
        raise
    return archive_path


def safe_extract_targz(archive: Path, destination: Path) -> None:
    """Extract ``archive`` into ``destination`` without path escapes or links.

    Only directories and regular files are extracted; symlinks, hardlinks, and device nodes raise
    rather than being silently skipped, so a tampered archive fails the import instead of landing a
    partial tree.
    """
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf.getmembers():
            target = destination.joinpath(*normalize_archive_parts(member.name))
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ValueError(f"Unsupported archive member type: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            extracted = tf.extractfile(member)
            if extracted is None:
                raise ValueError(f"Cannot read archive member: {member.name}")
            with extracted, open(target, "wb") as dst:
                shutil.copyfileobj(extracted, dst)
            with suppress(OSError):
                os.chmod(target, member.mode & 0o777)


def archive_root_dirs(archive: Path) -> set[str]:
    """Return the archive's top-level directory names.

    Transfer archives carry exactly one root directory, which names the thing being imported.
    Inspecting before extraction lets the caller resolve the target name (and refuse a malformed
    archive) without first mutating a live tree.
    """
    with tarfile.open(archive, "r:gz") as tf:
        return {parts[0] for member in tf.getmembers()
                for parts in [normalize_archive_parts(member.name)]
                if len(parts) > 1 or member.isdir()}


def copy_regular_files(src: Path, dst: Path) -> int:
    """Copy the regular files under ``src`` into ``dst``, skipping symlinks; return the count.

    Used on the *export* side so a symlink planted in an attachments or logs tree can't pull an
    arbitrary file into the archive. A missing ``src`` copies nothing.
    """
    if not src.is_dir():
        return 0
    copied = 0
    for entry in sorted(src.rglob("*")):
        if entry.is_symlink() or not entry.is_file():
            continue
        target = dst / entry.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(entry, target)
        copied += 1
    return copied
