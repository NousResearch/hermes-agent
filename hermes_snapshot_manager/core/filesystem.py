from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path, PurePosixPath
import fnmatch
import os
import shutil
import signal
import stat
import subprocess
import threading


ProgressCallback = Callable[[int, int, str], None]
ScanProgressCallback = Callable[[int, str], None]  # (scanned_count, current_relative_path) -> None
SkipCallback = Callable[[str, str], None]  # (relative_path, error_message) -> None
# (relative_path, reason) -> None; reason is one of:
# "fifo", "socket", "device", "symlink", "mount_point:<path>",
# "fs_type:<type>", "cross_device", "lstat_failed"
TraversalSkipCallback = Callable[[str, str], None]


def _relpath(path: Path, root: Path) -> str:
    """Return path relative to root, or the path string if that fails."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        # Not relative to root (e.g., different drive on Windows, or absolute path
        # that doesn't share the root prefix) — return as absolute posix string.
        return path.as_posix()

# Default timeout for copying a single file (seconds)
DEFAULT_COPY_TIMEOUT_SECONDS = 30

# Default max retries per file
DEFAULT_MAX_RETRIES = 2

# Max number of consecutive skipped (stuck) files before we treat the snapshot as degraded.
# Prevents accumulating unbounded zombie processes / pathological I/O.
DEFAULT_MAX_CONSECUTIVE_SKIPPED = 20

# Signals we treat as "stuck" when sent to the process group.
_STUCK_SIGNALS = (signal.SIGKILL, signal.SIGTERM)

# Filesystem types that are safe to traverse.  procfs/sysfs/devpts are virtual
# and either contain no real files or will block when accessed.
# cgroup, cgroup2, devtmpfs, autofs, snap, overlay are also excluded by default.
# Network-backed types are excluded by default because they can hang on
# unresponsive servers.
_DEFAULT_SKIP_FS_TYPES: frozenset[str] = frozenset({
    "proc", "sysfs", "devpts", "devtmpfs", "cgroup", "cgroup2",
    "autofs", "pstore", "securityfs", "debugfs", "hugetlbfs",
    "mqueue", "fusectl", "configfs", "binfmt_misc",
    # Network / FUSE types
    "nfs", "nfs4", "cifs", "smb", "smb3", "sshfs", "fuse.sshfs",
    "fuse", "fuseblk", "overlay", "aufs", "btrfs",
    # Container runtimes
    "container", "containerd", "crio", "runtime",
    # WSL / Plan 9 shared-folder protocol
    "9p",
})

# Mount points that are almost always virtual / system mounts and should be
# excluded from traversal even if their fstype is not known.
_DEFAULT_SKIP_MOUNT_POINTS: tuple[str, ...] = (
    "/proc", "/sys", "/dev", "/dev/pts", "/dev/shm",
    "/run", "/run/lock",
    # /tmp is intentionally NOT included here because it may contain user files
    # and is often on the same filesystem as the source.  If you want to skip
    # /tmp, add it to skip_mount_points in settings or set skip_tmp=True in
    # iter_included_files().  We also skip /mnt and /media as they commonly
    # contain external mounts.
    "/mnt", "/media", "/snap",
)


def _is_real_regular_file(entry: os.DirEntry) -> bool:
    """Return True only if entry is a real regular file (not symlink, FIFO, socket, device).

    Uses lstat internally so it never blocks on FIFO read or socket connect.
    This is the primary defense: we skip non-regular files before they reach
    the copy stage, where cp would block indefinitely on FIFOs/sockets.
    """
    try:
        st = entry.stat(follow_symlinks=False)
    except OSError:
        return False
    return stat.S_ISREG(st.st_mode)


def _is_skipable_by_stat(entry_path: str | Path) -> tuple[bool, str]:
    """Check if a path is a special file that would block during traversal.

    Returns (should_skip, reason) where reason is empty string if not skipable.
    Uses lstat so it never blocks on FIFO read or socket connect.
    """
    try:
        st = os.lstat(entry_path)
    except OSError:
        return True, "lstat_failed"

    mode = st.st_mode

    # Non-regular files that would cause cp to block forever
    if stat.S_ISFIFO(mode):
        return True, "fifo"
    if stat.S_ISSOCK(mode):
        return True, "socket"

    # Device files — cp will try to open and read them (returning ENODEV or
    # blocking on a raw character device)
    if stat.S_ISBLK(mode) or stat.S_ISCHR(mode):
        return True, "device"

    # Broken symlink — would fail with ENOENT during copy
    if stat.S_ISLNK(mode):
        return True, "symlink"

    return False, ""


def _safe_scandir_recurse(
    source: Path,
    skip_mount_points: tuple[str, ...] | None = None,
    skip_fs_types: frozenset[str] | None = None,
    cross_device: bool = False,
    stop_event: threading.Event | None = None,
    _device_id: int | None = None,
    skip_symlinks: bool = True,
    traversal_skip_callback: TraversalSkipCallback | None = None,
    _root: Path | None = None,
) -> Iterator[Path]:
    """Recursively walk source using os.scandir with mount-boundary and
    special-file guards.

    This replaces rglob for snapshot-traversal because:
    - rglob + is_file() can block on FIFOs/sockets (opens the file descriptor).
    - rglob follows symlink directories by default in Python 3.11.
    - os.scandir + lstat never opens file content, so never blocks on a FIFO.

    Args:
        source: Root directory to traverse.
        skip_mount_points: Mount-point prefixes to skip (absolute paths).
        skip_fs_types: Filesystem type names to skip (from /proc/mounts).
        cross_device: If False, stop traversal when a directory's device ID
            differs from _device_id (prevents traversing into bind mounts).
        stop_event: Optional threading.Event; if set, abort traversal early.
        _device_id: Device ID of the source root (used internally for
            cross-device boundary detection).  Set automatically on first call.
        skip_symlinks: If True (default), skip all symlinks (including working
            symlinks to regular files).  This prevents snapshotting symlinks
            that point outside the source tree.  If False, symlinks to regular
            files are included and copied as symlinks via `cp -p`.
        traversal_skip_callback: If provided, called with (relative_path, reason)
            whenever a file or directory is skipped during traversal (mount point,
            filesystem type, cross-device boundary, or special file).  The path is
            relative to the original source root.  Reasons are one of:
            "fifo", "socket", "device", "symlink", "mount_point:<path>",
            "fs_type:<type>", "cross_device", "lstat_failed".
        _root: Internal; set to the original source root on first call so
            relative paths can be computed for the skip callback.
    """
    if _root is None:
        _root = source

    if skip_mount_points is None:
        skip_mount_points = _DEFAULT_SKIP_MOUNT_POINTS
    if skip_fs_types is None:
        skip_fs_types = _DEFAULT_SKIP_FS_TYPES

    if _device_id is None:
        try:
            _device_id = os.stat(source).st_dev
        except OSError:
            return

    # Check if this path is under a skipable mount point
    try:
        real_path = os.path.realpath(source)
    except OSError:
        if traversal_skip_callback is not None:
            traversal_skip_callback(_relpath(source, _root), "lstat_failed")
        return
    for mp in skip_mount_points:
        if real_path == mp or real_path.startswith(mp + "/"):
            if traversal_skip_callback is not None:
                traversal_skip_callback(_relpath(source, _root), f"mount_point:{mp}")
            return

    # Check filesystem type
    try:
        fs_type = _get_filesystem_type(source)
        if fs_type and fs_type in skip_fs_types:
            if traversal_skip_callback is not None:
                traversal_skip_callback(_relpath(source, _root), f"fs_type:{fs_type}")
            return
    except OSError:
        pass

    try:
        entries = list(os.scandir(source))
    except OSError:
        # Directory disappeared or permission denied — skip silently
        if traversal_skip_callback is not None:
            traversal_skip_callback(_relpath(source, _root), "lstat_failed")
        return
    except PermissionError:
        if traversal_skip_callback is not None:
            traversal_skip_callback(_relpath(source, _root), "lstat_failed")
        return

    for entry in entries:
        if stop_event and stop_event.is_set():
            return

        try:
            is_dir = entry.is_dir(follow_symlinks=False)
        except OSError:
            continue

        if is_dir:
            # Check cross-device boundary
            if not cross_device:
                try:
                    child_dev = os.stat(entry.path).st_dev
                except OSError:
                    if traversal_skip_callback is not None:
                        traversal_skip_callback(_relpath(Path(entry.path), _root), "lstat_failed")
                    continue
                if child_dev != _device_id:
                    # Crossed a mount boundary — skip the mount entirely
                    if traversal_skip_callback is not None:
                        traversal_skip_callback(_relpath(Path(entry.path), _root), "cross_device")
                    continue

            # Recurse into subdirectory
            yield from _safe_scandir_recurse(
                Path(entry.path),
                skip_mount_points=skip_mount_points,
                skip_fs_types=skip_fs_types,
                cross_device=cross_device,
                stop_event=stop_event,
                _device_id=_device_id,
                skip_symlinks=skip_symlinks,
                traversal_skip_callback=traversal_skip_callback,
                _root=_root,
            )
        else:
            # It's a file (or symlink-to-file). Use lstat to check file type
            # safely without blocking.
            should_skip, reason = _is_skipable_by_stat(entry.path)
            if should_skip:
                # Symlinks are only skipped if skip_symlinks is True
                if reason == "symlink" and not skip_symlinks:
                    # Include working symlinks when skip_symlinks=False
                    yield Path(entry.path)
                else:
                    if traversal_skip_callback is not None:
                        traversal_skip_callback(_relpath(Path(entry.path), _root), reason)
                continue
            yield Path(entry.path)


def _get_filesystem_type(path: Path) -> str | None:
    """Return the filesystem type of the mount containing path (e.g. 'ext4', 'nfs4').

    Returns None if the type cannot be determined.
    """
    try:
        # Use /proc/mounts for reliable cross-platform type info
        with open("/proc/mounts", "r") as f:
            mounts = f.read()
    except (OSError, IOError):
        return None

    # Find the most-specific matching mount entry
    best_match = None
    best_len = -1
    for line in mounts.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        mount_point = parts[1]
        fs_type = parts[2]
        # mount_point may contain octal escapes for spaces
        import urllib.parse
        try:
            decoded_mp = urllib.parse.unquote(mount_point)
        except Exception:
            decoded_mp = mount_point

        if path == Path(decoded_mp) or str(path).startswith(decoded_mp + "/"):
            if len(decoded_mp) > best_len:
                best_match = fs_type
                best_len = len(decoded_mp)

    return best_match


def iter_included_files(
    source: Path,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
    skip_stuck_patterns: list[str] | None = None,
    skip_mount_points: tuple[str, ...] | None = None,
    skip_fs_types: frozenset[str] | None = None,
    cross_device: bool = False,
    skip_symlinks: bool = True,
    traversal_skip_callback: TraversalSkipCallback | None = None,
    scan_progress_callback: ScanProgressCallback | None = None,
) -> list[Path]:
    """Iterate over regular files under source, skipping special files and mount boundaries.

    This is the safe replacement for source.rglob('*') + is_file() filtering.

    Special-file defense (via lstat):
      - FIFOs  → skipped (cp would block forever reading from them)
      - Sockets → skipped (cp would block forever)
      - Block/char devices → skipped (cp would return ENODEV or block)
      - Broken symlinks → skipped (cp would fail with ENOENT)

    Mount-boundary defense (when cross_device=False):
      - If a subdirectory's device ID differs from the source root's device ID,
        it is treated as a bind mount and its contents are NOT traversed.
      - This prevents infinite/huge traversals when /mnt/foo is a bind mount of /home.

    Args:
        source: Root directory to traverse.
        include_patterns: Glob patterns for files to include.
        exclude_patterns: Glob patterns for files to exclude.
        skip_stuck_patterns: Additional glob patterns; matching paths are skipped
            immediately without traversal (e.g. '**/__pycache__').
        skip_mount_points: Additional mount-point prefixes to skip.
        skip_fs_types: Additional filesystem-type names to skip.
        cross_device: If True, cross filesystem/device boundaries during traversal.
            Default False (safer: stops at mount boundaries).
        traversal_skip_callback: If provided, called with (relative_path, reason)
            for every item skipped during traversal.  Also called for items
            skipped by skip_stuck_patterns or include/exclude filters.
    """
    combined_skip_mounts = (
        tuple(set(_DEFAULT_SKIP_MOUNT_POINTS) | set(skip_mount_points or ()))
    )
    combined_skip_fs = (
        _DEFAULT_SKIP_FS_TYPES | (skip_fs_types or frozenset())
    )

    files: list[Path] = []
    scanned_count = 0
    for item in _safe_scandir_recurse(
        source,
        skip_mount_points=combined_skip_mounts,
        skip_fs_types=combined_skip_fs,
        cross_device=cross_device,
        skip_symlinks=skip_symlinks,
        traversal_skip_callback=traversal_skip_callback,
    ):
        scanned_count += 1
        relative_str = item.relative_to(source).as_posix()
        if scan_progress_callback is not None and (scanned_count == 1 or scanned_count % 250 == 0):
            scan_progress_callback(scanned_count, relative_str)
        # Additional pattern-based skip (after lstat check)
        if skip_stuck_patterns and _matches_any(relative_str, skip_stuck_patterns):
            if traversal_skip_callback is not None:
                traversal_skip_callback(relative_str, "skipped_by_pattern")
            continue
        if not should_include(relative_str, include_patterns=include_patterns, exclude_patterns=exclude_patterns):
            if traversal_skip_callback is not None:
                traversal_skip_callback(relative_str, "excluded_by_filter")
            continue
        files.append(item)
    return files


def _matches_any(relative_path: str, patterns: list[str]) -> bool:
    if not patterns:
        return False
    if "**/*" in patterns:
        return True
    path = PurePosixPath(relative_path)
    candidates = {relative_path, path.as_posix()}
    if relative_path:
        parts = relative_path.split("/")
        candidates.add(parts[-1])
        for i in range(1, len(parts) + 1):
            candidates.add("/".join(parts[:i]))
    for pattern in patterns:
        pattern_variants = {pattern}
        if pattern.startswith("**/"):
            pattern_variants.add(pattern[3:])
        for candidate in candidates:
            if any(fnmatch.fnmatchcase(candidate, variant) for variant in pattern_variants):
                return True
    return False


def should_include(relative_path: str, include_patterns: list[str] | None = None, exclude_patterns: list[str] | None = None) -> bool:
    include_patterns = include_patterns or ["**/*"]
    exclude_patterns = exclude_patterns or []
    if exclude_patterns and _matches_any(relative_path, exclude_patterns):
        return False
    return _matches_any(relative_path, include_patterns) or relative_path == ""


def copy_tree(
    source: Path,
    destination: Path,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
    progress_callback: ProgressCallback | None = None,
    scan_progress_callback: ScanProgressCallback | None = None,
    skip_callback: SkipCallback | None = None,
    copy_timeout: float = DEFAULT_COPY_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    skip_stuck_patterns: list[str] | None = None,
    max_consecutive_skipped: int = DEFAULT_MAX_CONSECUTIVE_SKIPPED,
    skip_mount_points: tuple[str, ...] | None = None,
    skip_fs_types: frozenset[str] | None = None,
    cross_device: bool = False,
    skip_symlinks: bool = True,
    traversal_skip_callback: TraversalSkipCallback | None = None,
    traversal_skipped_collector: list[tuple[str, str]] | None = None,
) -> tuple[list[Path], list[tuple[str, str]], bool]:
    """Copy a source tree to destination, skipping files that fail.

    Returns:
        A tuple of (included_files, skipped_files, is_degraded) where:
        - included_files: list of Path objects that were successfully copied.
        - skipped_files: list of (relative_path, error_message) tuples.
        - is_degraded: True if any files were skipped.

    The copy operation is fail-safe: individual file failures are logged via
    skip_callback (if provided) and do not abort the overall copy operation.
    A metadata-only placeholder is created for any skipped files so the
    manifest still references them.
    """
    destination.mkdir(parents=True, exist_ok=True)

    # Collector wraps traversal_skip_callback to also accumulate a list
    _traversal_skipped: list[tuple[str, str]] = []
    _cb = traversal_skip_callback

    def _wrapped_cb(path: str, reason: str) -> None:
        _traversal_skipped.append((path, reason))
        if _cb is not None:
            _cb(path, reason)

    included_files = iter_included_files(
        source,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
        skip_stuck_patterns=skip_stuck_patterns,
        skip_mount_points=skip_mount_points,
        skip_fs_types=skip_fs_types,
        cross_device=cross_device,
        skip_symlinks=skip_symlinks,
        traversal_skip_callback=_wrapped_cb,
        scan_progress_callback=scan_progress_callback,
    )

    # Merge traversal skips into the collector for the caller
    if traversal_skipped_collector is not None:
        traversal_skipped_collector.extend(_traversal_skipped)

    # Collect files to copy (exclude dirs — those are created as we go)
    files_to_copy = [f for f in included_files]

    # Internal progress counter so copy_files_with_progress can use it
    # without us having to compute it again.
    copied_count = [0]  # mutable container so closure can mutate

    def _progress_wrapper(completed: int, total: int, relative_str: str) -> None:
        copied_count[0] = completed
        if progress_callback is not None:
            progress_callback(completed, total, relative_str)

    skipped_files, is_degraded = copy_files_with_progress(
        source,
        destination,
        files_to_copy,
        progress_callback=_progress_wrapper,
        skip_callback=skip_callback,
        copy_timeout=copy_timeout,
        max_retries=max_retries,
        skip_stuck_patterns=skip_stuck_patterns,
        max_consecutive_skipped=max_consecutive_skipped,
    )

    # Re-collect included files from the destination as a Path list.
    # Any file that was skipped still has a placeholder (via _metadata_only_copy).
    result_files = [f for f in destination.rglob("*") if f.is_file()]
    return result_files, skipped_files, is_degraded


def _copy_file_with_timeout(source: Path, dest: Path, timeout: float) -> None:
    """Copy a single file with a timeout mechanism.

    Uses a subprocess spawned with os.setsid so we can kill the entire
    process group if the copy hangs (e.g., on locked files, network
    drives, or stale NFS handles).  This is far more reliable than
    threading-based interrupt because kill(2) on a subprocess can break
    blocking C-level syscalls that no Python-level join() can unwind.
    """
    # Use cp(1) which is available on all Unix systems and handles
    # metadata-copy (permissions + mtime) via -p flag.
    # --interval=1 makes rsync-check less aggressive; not used here but
    # retained for API parity with any future rsync path.
    cmd = ["cp", "-p", str(source), str(dest)]

    try:
        proc = subprocess.run(
            cmd,
            timeout=timeout,
            # Start a new session so killpg targets the whole process group.
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except subprocess.TimeoutExpired:
        raise TimeoutError(f"Copy timed out after {timeout}s: {source}")
    except OSError as exc:
        raise OSError(f"Failed to spawn copy process for {source}: {exc}")

    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise OSError(f"cp exited with {proc.returncode}: {stderr}")


def _metadata_only_copy(source: Path, dest: Path) -> bool:
    """Try to preserve at least mode+mtime when a full copy failed.

    Returns True if metadata was applied, False if even that failed.
    This is a last-ditch fallback so a stuck file still gets a
    placeholder in the archive (same mode/mtime as source) rather than
    silently disappearing from the manifest.

    Uses a subprocess with a hard 5-second timeout.
    """
    try:
        stat_info = source.stat()
    except OSError:
        return False

    # Touch dest to create a zero-length placeholder
    try:
        dest.write_bytes(b"")
    except OSError:
        return False

    # Set mtime (and mode if we have permission)
    try:
        os.utime(dest, (stat_info.st_atime, stat_info.st_mtime))
    except OSError:
        pass

    try:
        os.chmod(dest, stat_info.st_mode & 0o777)
    except OSError:
        pass

    return True


def copy_files_with_progress(
    source_root: Path,
    destination_root: Path,
    files: list[Path],
    progress_callback: ProgressCallback | None = None,
    skip_callback: SkipCallback | None = None,
    copy_timeout: float = DEFAULT_COPY_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    skip_stuck_patterns: list[str] | None = None,
    max_consecutive_skipped: int = DEFAULT_MAX_CONSECUTIVE_SKIPPED,
) -> tuple[list[tuple[str, str]], bool]:
    """Copy a list of files with progress reporting and fail-safe skipping.

    Args:
        source_root: Root of the source tree.
        destination_root: Root of the destination tree.
        files: List of file paths to copy (relative to source_root).
        progress_callback: Called with (index, total, relative_path).
        skip_callback: Called with (relative_path, error_message) when a
            file is skipped.
        copy_timeout: Max seconds to wait for a single file copy.
        max_retries: Number of retry attempts after timeout/error.
        skip_stuck_patterns: Glob patterns. Matching files are skipped
            immediately without retry (useful for network mounts, sockets).
        max_consecutive_skipped: Abort copy if this many files are skipped
            consecutively (prevents pathological I/O storms).

    Returns:
        A two-tuple (skipped_files, is_degraded) where:
        - skipped_files: List of (relative_path, error_message) tuples.
        - is_degraded: True if any files were skipped.
    """
    total_files = len(files)
    skipped_files: list[tuple[str, str]] = []
    is_degraded = False
    consecutive_skipped = 0

    for index, item in enumerate(files, start=1):
        relative = item.relative_to(source_root)
        relative_str = relative.as_posix()
        target = destination_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)

        # Fast-path: known-stuck files are skipped immediately without retry.
        if skip_stuck_patterns and _matches_any(relative_str, skip_stuck_patterns):
            skipped_files.append((relative_str, "skipped_by_pattern"))
            is_degraded = True
            consecutive_skipped += 1
            if skip_callback is not None:
                skip_callback(relative_str, "skipped_by_pattern")
            if consecutive_skipped >= max_consecutive_skipped:
                skipped_files.append(
                    (relative_str, f"max_consecutive_skipped ({max_consecutive_skipped}) reached — aborting")
                )
                break
            continue

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                _copy_file_with_timeout(item, target, timeout=copy_timeout)
                last_error = None
                consecutive_skipped = 0
                break
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                if attempt < max_retries:
                    threading.Event().wait(0.1)

        if last_error is not None:
            # Last-ditch metadata fallback so the manifest still references
            # a file (placeholder with correct mode/mtime) rather than skipping
            # it entirely from the archive.
            _metadata_only_copy(item, target)
            skipped_files.append((relative_str, last_error))
            is_degraded = True
            consecutive_skipped += 1
            if skip_callback is not None:
                skip_callback(relative_str, last_error)
            if consecutive_skipped >= max_consecutive_skipped:
                skipped_files.append(
                    (relative_str, f"max_consecutive_skipped ({max_consecutive_skipped}) reached — aborting")
                )
                break
            continue

        if progress_callback is not None:
            progress_callback(index, total_files, relative_str)

    return skipped_files, is_degraded


def replace_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination, dirs_exist_ok=True)
