"""Entries that never leave a profile, and the child mount points among them.

Split out of :mod:`hermes_cli.profiles` (that facade is past the file-size cap and may not
grow). ``_non_exportable_entries`` is the single filter every profile copy path funnels
through — ``--clone-all``, the ``plugins/`` and ``skills/`` clones, and ``export_profile`` —
so skipping child mount points here fixes the whole class at once.

Why mount points need special handling (#156026): a host directory bind-mounted into
``HERMES_HOME`` (Docker ``-v /host/xchg:/opt/data/xchg``) is an ordinary ``S_ISDIR`` to
``shutil.copytree``, so a clone recursed into the host's tree — slow, and recursive when the
mount sits under the copy destination (``profiles/.<name>.staging-<pid>``). ``os.path.ismount``
does not catch it either: it compares only ``st_dev`` of the path and its parent, and a
same-device bind mount has the parent's ``st_dev``. Linux ``/proc/self/mountinfo`` lists every
mount point regardless of device, so on Linux we consult it; elsewhere ``os.path.ismount`` is
the fallback (the procfs read fails and yields no mount points).
"""

from __future__ import annotations

import os
import re
import stat

_MOUNTINFO_PATH = "/proc/self/mountinfo"


def _read_mount_points(mountinfo_path: str = _MOUNTINFO_PATH) -> frozenset[str]:
    """Absolute mount points listed by Linux ``mountinfo``.

    Empty when the table is unreadable or absent (every non-Linux platform), which makes
    :func:`is_child_mount_point` fall back to ``os.path.ismount``. ``mountinfo_path`` is
    injectable so the parser is testable without a real mount table.
    """
    try:
        with open(mountinfo_path, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.read().splitlines()
    except OSError:
        return frozenset()
    points: set[str] = set()
    for line in lines:
        # proc(5): <id> <parent> <maj:min> <root> <mount point> <opts> [optional...] - <fstype> ...
        fields = line.partition(" - ")[0].split()
        if len(fields) < 5:
            continue
        point = fields[4]
        if "\\" in point:  # octal escapes (\040 = space, \011 = tab)
            point = re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), point)
        points.add(point)
    return frozenset(points)


def is_child_mount_point(path: str, mount_points: frozenset[str]) -> bool:
    """True when *path* is itself a mount point, given a :func:`_read_mount_points` snapshot.

    The realpath is compared because ``copytree`` hands us the walked directory, which can
    differ from the kernel's resolved mount target by symlinks in the prefix. A non-empty
    ``mount_points`` is authoritative on Linux (the table lists every mount, cross-device
    included); an empty snapshot defers to ``os.path.ismount`` for other platforms.
    """
    if mount_points:
        try:
            return os.path.realpath(path) in mount_points or path in mount_points
        except (OSError, ValueError):
            return False
    return os.path.ismount(path)


def _non_exportable_entries(
    directory: str, contents: list, mount_points: frozenset[str] | None = None
) -> set:
    """Entries under *directory* that must never be copied out of a profile: bytecode caches,
    ``*.sock``/``*.tmp`` names, child mount points, and anything that is not a regular file,
    directory, or symlink. :func:`shutil.copytree` cannot copy special files, so a single live
    Unix socket without a ``.sock`` name (or a FIFO, or a device node) would abort the whole
    export or clone with ``[Errno 6] No such device or address``. A bind-mounted host directory
    is skipped so a clone never recurses into host state outside the profile (#156026).
    Symlinks survive — copytree recreates them.

    ``mount_points`` is an optional :func:`_read_mount_points` snapshot: a full-tree copy
    (``--clone-all``) reads the table once and passes it to every per-directory call, while a
    caller that uses this directly as ``copytree``'s ignore leaves it ``None`` to read fresh."""
    ignored: set = set()
    snapshot = mount_points
    for entry in contents:
        if entry == "__pycache__" or entry.endswith((".sock", ".tmp", ".pyc", ".pyo")):
            ignored.add(entry)
            continue
        entry_path = os.path.join(directory, entry)
        try:
            mode = os.lstat(entry_path).st_mode
        except OSError:
            ignored.add(entry)  # vanished mid-walk — copytree would fail on it anyway
            continue
        if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode) or stat.S_ISLNK(mode)):
            ignored.add(entry)
            continue
        if stat.S_ISDIR(mode) and not stat.S_ISLNK(mode):
            if snapshot is None:
                snapshot = _read_mount_points()  # read once for this call
            if is_child_mount_point(entry_path, snapshot):
                ignored.add(entry)
    return ignored
