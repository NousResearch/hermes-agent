"""Safe ``tar.gz`` primitives shared by the profile and kanban transfer paths."""

from __future__ import annotations

import os
import posixpath
import shutil
import tarfile
import tempfile
import uuid
from contextlib import suppress
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Optional


def _fsync_parent(path: str) -> None:
    """Make a just-renamed file's directory entry durable, and that directory's own entry (best effort).

    The grandparent is synced too because an export may have just created the directory.
    Stdlib-only on purpose: this module loads on the ``python -S`` early-recovery path
    (profile override), where ``utils`` and its ruamel import are unavailable.
    """
    if os.name == "nt":
        return
    parent = os.path.dirname(os.path.abspath(path))
    for directory in (parent, os.path.dirname(parent)):
        with suppress(OSError):
            fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(fd)
            finally:
                os.close(fd)


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


def make_targz(base: str, root_dir: str, base_dir: str, *, link_source: Optional[Path] = None) -> str:
    """Create ``<base>.tar.gz`` of ``root_dir/base_dir`` in GNU tar format.

    Writes to a sibling temp file and renames onto the archive path only after it is fully written
    and fsynced, so a crash after the rename cannot leave a zero-tailed archive in its place.
    ``root_dir/base_dir`` may be a symlink-preserving copy of ``link_source``: an absolute link into
    ``link_source`` (installers link ``lsp/bin/<server>`` that way) is then archived as the matching
    relative link, since :func:`safe_extract_targz` refuses absolute links.
    """
    archive_path = f"{base}.tar.gz"
    dest_dir = os.path.dirname(archive_path) or "."
    source = os.path.realpath(link_source) if link_source is not None else None

    def _relative_links(info: tarfile.TarInfo) -> tarfile.TarInfo:
        if source is not None and info.issym() and os.path.isabs(info.linkname):
            real = os.path.realpath(info.linkname)
            if real == source or real.startswith(source + os.sep):
                inside = posixpath.join(base_dir, *Path(os.path.relpath(real, source)).parts)
                info.linkname = posixpath.relpath(inside, posixpath.dirname(info.name))
        return info

    fd, tmp_path = tempfile.mkstemp(dir=dest_dir, prefix=".archive_", suffix=".tar.gz.tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            with tarfile.open(fileobj=f, mode="w:gz", format=tarfile.GNU_FORMAT) as tf:
                tf.add(str(Path(root_dir) / base_dir), arcname=base_dir, filter=_relative_links)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, archive_path)
        _fsync_parent(archive_path)
    except BaseException:
        with suppress(OSError):
            os.unlink(tmp_path)
        raise
    return archive_path


def _create_archive_symlinks(links: list[tuple[tarfile.TarInfo, Path, Path]]) -> None:
    """Create the archive's symlinks after every file is written, then refuse any that escapes.

    Each entry is ``(member, path, root)``, ``root`` being the member's top-level directory. Links
    come last so no file or directory is ever written through one. Containment is judged on the
    real resolved path once all links exist, because a target such as ``y/../..`` only escapes
    through another link (``y``), and that link may come later in the archive. That final check
    runs with ``root`` moved to a random name: the caller publishes it under another name, so a
    link that leaves ``root`` and walks back in by its archive name (``../<root>/x``) would then
    point at another profile.
    """
    def _inside(root: Path, path: Path) -> bool:
        real_root, resolved = os.path.realpath(root), os.path.realpath(path)
        return resolved == real_root or resolved.startswith(real_root + os.sep)

    made: dict[Path, bool] = {}
    for member, target, root in links:
        # Every parent must be a real directory, so no link is created through another link
        # (in either archive order: a later link over an implied parent collides below).
        parents = target.relative_to(root).parents
        if PurePosixPath(member.linkname.replace("\\", "/")).is_absolute() \
                or PureWindowsPath(member.linkname).drive \
                or any((root / parent).is_symlink() for parent in parents):
            raise ValueError(f"Archive symlink escapes the destination: {member.name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():  # must not replace a member already extracted
            raise ValueError(f"Archive symlink collides with another member: {member.name}")
        made[target] = _symlink(member, target)
    # Windows fixes a link's file/dir type at creation, and a link through another link created
    # later saw no directory yet: recreate mismatched ones until every type matches its target.
    for _ in links:
        wrong = [(m, t) for m, t, _r in links if made[t] != os.path.isdir(t)]
        if not wrong:
            break
        for member, target in wrong:
            target.unlink()
            made[target] = _symlink(member, target)
    for root in {root for _m, _t, root in links}:
        moved = root.with_name(f".{uuid.uuid4().hex}")
        root.rename(moved)
        try:
            for member, target, _r in [entry for entry in links if entry[2] == root]:
                if not _inside(moved, moved / target.relative_to(root)):
                    raise ValueError(f"Archive symlink escapes the destination: {member.name}")
        finally:
            moved.rename(root)


def _symlink(member: tarfile.TarInfo, target: Path) -> bool:
    """Create ``target`` -> ``member.linkname`` (either separator style); return the dir flag used."""
    link_text = member.linkname.replace("\\", "/").replace("/", os.sep)
    is_dir = os.path.isdir(target.parent / link_text)
    try:
        os.symlink(link_text, target, target_is_directory=is_dir)
    except OSError as exc:  # Windows without the symlink privilege (Developer Mode off)
        raise ValueError(f"Cannot create the archive symlink {member.name} on this system: {exc}") from exc
    return is_dir


def safe_extract_targz(archive: Path, destination: Path, *, allow_symlinks: bool = False) -> None:
    """Extract ``archive`` into ``destination`` without path escapes.

    Directories and regular files are extracted as-is. With ``allow_symlinks`` a symlink is
    recreated when it resolves inside its top-level archive directory (see
    :func:`_create_archive_symlinks`); otherwise it raises, as hardlinks and device nodes always
    do, so a tampered archive fails the import instead of landing a partial tree. ``destination``
    must be a fresh staging directory the caller discards on error.
    """
    links: list[tuple[tarfile.TarInfo, Path, Path]] = []
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf.getmembers():
            parts = normalize_archive_parts(member.name)
            target = destination.joinpath(*parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if member.issym() and allow_symlinks:
                links.append((member, target, destination / parts[0]))
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
    _create_archive_symlinks(links)


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
