"""Safe ``tar.gz`` primitives shared by the profile and kanban transfer paths."""

from __future__ import annotations

import os
import shutil
import tarfile
import tempfile
from contextlib import suppress
from pathlib import Path, PurePosixPath, PureWindowsPath


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


def make_targz(base: str, root_dir: str, base_dir: str) -> str:
    """Create ``<base>.tar.gz`` of ``root_dir/base_dir`` in GNU tar format.

    Writes to a sibling temp file and renames onto the archive path only after it is fully written
    and fsynced, so a crash after the rename cannot leave a zero-tailed archive in its place.
    """
    archive_path = f"{base}.tar.gz"
    dest_dir = os.path.dirname(archive_path) or "."
    fd, tmp_path = tempfile.mkstemp(dir=dest_dir, prefix=".archive_", suffix=".tar.gz.tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            with tarfile.open(fileobj=f, mode="w:gz", format=tarfile.GNU_FORMAT) as tf:
                tf.add(str(Path(root_dir) / base_dir), arcname=base_dir)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, archive_path)
        _fsync_parent(archive_path)
    except BaseException:
        with suppress(OSError):
            os.unlink(tmp_path)
        raise
    return archive_path


def relativize_internal_links(staged: Path, source: Path) -> None:
    """Rewrite each absolute symlink in ``staged`` (a copy of ``source``) that points inside
    ``source`` as a relative link, so the archive imports anywhere.

    Installers link by absolute path (``lsp/bin/<server>`` -> ``<profile>/lsp/node_modules/...``)
    and :func:`safe_extract_targz` refuses absolute links. Links leaving ``source`` are left as-is.
    """
    root = os.path.realpath(source)
    for link in [p for p in staged.rglob("*") if p.is_symlink()]:
        target = os.readlink(link)
        real = os.path.realpath(target)
        if os.path.isabs(target) and real.startswith(root + os.sep):
            inside = staged / os.path.relpath(real, root)
            link.unlink()
            os.symlink(os.path.relpath(inside, link.parent), link, target_is_directory=os.path.isdir(real))


def _create_archive_symlinks(links: list[tuple[tarfile.TarInfo, Path]], destination: Path) -> None:
    """Create the archive's symlinks after every file is written, then refuse any that escapes.

    Links come last so no file or directory is ever written through one. Containment is judged on
    the real resolved path once all links exist, because a target such as ``y/../..`` only escapes
    through another link (``y``), and that link may come later in the archive. The bound is the
    member's own top-level directory, not ``destination``: the caller moves that directory into
    place, so a link to a sibling of it would point outside the imported tree afterwards.
    """
    def _inside(member: tarfile.TarInfo, path: Path) -> bool:
        root = os.path.realpath(destination.joinpath(normalize_archive_parts(member.name)[0]))
        resolved = os.path.realpath(path)
        return resolved == root or resolved.startswith(root + os.sep)

    for member, target in links:
        # The parent check stops a link (or its parent dirs) being created through an earlier link.
        if PurePosixPath(member.linkname.replace("\\", "/")).is_absolute() \
                or PureWindowsPath(member.linkname).drive or not _inside(member, target.parent):
            raise ValueError(f"Archive symlink escapes the destination: {member.name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        try:  # Windows needs the directory flag up front; every file and dir already exists
            os.symlink(member.linkname, target, target_is_directory=(target.parent / member.linkname).is_dir())
        except FileExistsError as exc:  # a link member must not replace a member already extracted
            raise ValueError(f"Archive symlink collides with another member: {member.name}") from exc
        except OSError as exc:  # Windows without the symlink privilege (Developer Mode off)
            raise ValueError(f"Cannot create the archive symlink {member.name} on this system: {exc}") from exc
    for member, target in links:
        if not _inside(member, target):
            raise ValueError(f"Archive symlink escapes the destination: {member.name}")


def safe_extract_targz(archive: Path, destination: Path, *, allow_symlinks: bool = False) -> None:
    """Extract ``archive`` into ``destination`` without path escapes.

    Directories and regular files are extracted as-is. With ``allow_symlinks`` a symlink is
    recreated when it resolves inside its top-level archive directory (see
    :func:`_create_archive_symlinks`); otherwise it raises, as hardlinks and device nodes always
    do, so a tampered archive fails the import instead of landing a partial tree. ``destination``
    must be a fresh staging directory the caller discards on error.
    """
    links: list[tuple[tarfile.TarInfo, Path]] = []
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf.getmembers():
            target = destination.joinpath(*normalize_archive_parts(member.name))
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if member.issym() and allow_symlinks:
                links.append((member, target))
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
    _create_archive_symlinks(links, destination)


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
