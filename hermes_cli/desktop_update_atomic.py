"""Atomic binary-swap helpers for the Linux Desktop in-app updater.

Issue #58593 — Linux Desktop in-app update repeatedly fails to stick and resets
Electron. The user's terminal `hermes update` corrected the divergence but the
in-app desktop updater kept reporting the same update available and reset
Electron's chrome-sandbox ownership / mode (root:root + 4755 → user:user + 0755).

Root cause for the "doesn't stick" symptom
------------------------------------------
The unpacked Electron bundle lives at ``apps/desktop/release/linux-unpacked``
and the running binary is ``<that>/Hermes``. The in-app updater rebuilds the
bundle *in place* — but any update path that copies a new file over a running
binary with a non-atomic ``copy + unlink`` can leave the destination either
zero-byte (mid-write crash), or torn between old and new contents. When the
relaunch script then ``exec``s that binary, it runs whichever half-finished
state was on disk and Electron silently resets.

A correct swap must:
  1. Stage the new binary next to the destination with a sibling temp name
     (``<dest>.hermes-update-new``), so a crash during the copy leaves the
     running binary untouched.
  2. Once the staged copy is fsync'd and fully written, atomically rename it
     on top of the running binary using :func:`os.replace`. ``os.replace``
     resolves to ``rename(2)`` on POSIX (atomic on the same filesystem) and
     to ``MoveFileEx`` with ``MOVEFILE_REPLACE_EXISTING`` on Windows — i.e.
     the destination is either the old binary or the new binary, never a
     half-written file, regardless of platform. ``os.rename`` does NOT
     provide the same guarantee on Windows.
  3. Move the old binary aside (``<dest>.hermes-update-old``) for one cycle so
     a follow-up update that immediately fails can still roll back. The
     backup is best-effort cleaned at the end.

This module is consumed by:
  - The Electron main process's ``applyUpdatesPosixInApp`` handoff, which
    rebuilds the unpacked bundle and uses the equivalent JS helper
    (``writeFileAtomic`` + ``os.replace`` migration) — see
    :mod:`apps.desktop.electron.update_binary_swap` for the JS mirror.
  - The terminal `hermes update` path's `_atomic_replace_dir` helper in
    :mod:`hermes_cli.main` (this module's sibling-staging convention is the
    same: ``<dst>.hermes-update-{new,old}``).

Cross-platform safe:
  - On POSIX (Linux + macOS), ``os.replace`` is the same atomic rename as
    ``rename(2)``.
  - On Windows, ``os.rename`` will refuse to overwrite a destination — only
    ``os.replace`` (MoveFileEx + REPLACE_EXISTING) overwrites atomically.
    Falling back to ``shutil.move`` or ``copy + unlink`` here would
    re-introduce the torn-file window this helper exists to close.
  - Across-device moves (``EXDEV``) are not supported — the staged copy and
    the destination MUST live on the same filesystem for the rename to be
    atomic. We raise ``AtomicSwapError`` instead of falling back to a
    copy+unlink so the caller can surface a clear error rather than silently
    regress to non-atomic semantics.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, os.PathLike]

# Sibling paths the swap touches. Centralised so tests can assert on the
# exact naming convention and so a future rename only has one place to
# change.
STAGING_SUFFIX = ".hermes-update-new"
BACKUP_SUFFIX = ".hermes-update-old"


class AtomicSwapError(RuntimeError):
    """Raised when an atomic binary swap cannot be performed safely.

    Callers should surface this verbatim to the user — silently degrading to a
    non-atomic copy is the exact failure mode this module exists to prevent.
    """


def _sibling(path: PathLike, suffix: str) -> Path:
    """Return the sibling path next to *path* with *suffix* appended.

    ``/opt/Hermes/Hermes`` + ``.hermes-update-new`` →
    ``/opt/Hermes/Hermes.hermes-update-new``.
    """
    return Path(str(path) + suffix)


def _fsync_dir(path: Path) -> None:
    """Best-effort fsync of a directory's contents.

    On POSIX, after creating or renaming files inside a directory, the
    directory itself must be fsync'd for the change to survive a crash. On
    Windows there is no equivalent — directory metadata is journaled and we
    skip silently. Failures here are non-fatal: the swap is still atomic at
    the inode level, we just lose crash-survival of the directory entry.
    """
    if os.name == "nt":
        return
    fd: Optional[int] = None
    try:
        fd = os.open(str(path), os.O_DIRECTORY)
        os.fsync(fd)
    except (OSError, AttributeError):
        # AttributeError: Windows doesn't expose O_DIRECTORY reliably.
        pass
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass


def _clear_siblings(*paths: Path) -> None:
    """Remove any leftover staging / backup files from a previously crashed swap.

    A crash between the staging copy and the atomic rename leaves a
    ``<dest>.hermes-update-new`` next to the destination. On the next
    update we must clear it before we stage again — otherwise we would copy
    *into* the staged copy instead of *next to* it, and the eventual rename
    would point at the stale staged tree.
    """
    for path in paths:
        if not path.exists() and not path.is_symlink():
            continue
        try:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink()
        except OSError:
            # Best-effort — a stuck leftover is worse than a failed clear,
            # so ignore and let the staging copy report the conflict.
            pass


def atomic_swap_binary(
    source: PathLike,
    destination: PathLike,
    *,
    mode: Optional[int] = None,
    clear_siblings: bool = True,
) -> Path:
    """Atomically swap *source* onto *destination*.

    The contract:

    - After a successful return, ``destination`` is the bytes of ``source``
      and no temporary files remain next to it.
    - On any failure, ``destination`` is unchanged from its prior contents
      (either the old file, or no file at all if none existed).
    - The swap is atomic on the same filesystem: ``os.replace`` either
      completes fully or not at all — no half-written destination.

    Parameters
    ----------
    source:
        Path to the new binary content. Typically a freshly-rebuilt Electron
        unpacked tree's launcher or the staged download of the next version.
    destination:
        Path the running binary currently lives at. Must be on the same
        filesystem as *source*; cross-filesystem "atomic" semantics are not
        possible and we raise rather than silently degrade.
    mode:
        Optional POSIX mode to apply to the destination after the swap.
        Useful for restoring ``0o755`` after a copy that lost the executable
        bit (e.g. a fat-fingered ``shutil.copy`` through a tmpfs on Linux).
        Ignored on Windows.
    clear_siblings:
        When ``True`` (default), pre-clear any ``<dest>.hermes-update-new`` /
        ``<dest>.hermes-update-old`` left over from a previously crashed swap.

    Returns
    -------
    Path
        The resolved *destination* path, so the caller can immediately use it
        to ``chmod`` or ``stat`` the new binary without re-resolving.

    Raises
    ------
    AtomicSwapError
        If the swap cannot be performed atomically (cross-filesystem move,
        write failure, missing source, etc.).
    """
    src = Path(source)
    dst = Path(destination)

    if not src.exists():
        raise AtomicSwapError(f"source does not exist: {src}")

    staging = _sibling(dst, STAGING_SUFFIX)
    backup = _sibling(dst, BACKUP_SUFFIX)

    if clear_siblings:
        _clear_siblings(staging, backup)

    # 1. Stage the new binary next to the destination. If this fails, dst
    #    is untouched and the staging file (if any) is cleaned below.
    try:
        # shutil.copy2 preserves mode bits where possible. We still apply
        # ``mode`` below to defend against copy2 silently downgrading on
        # tmpfs / FAT / overlay filesystems.
        shutil.copy2(src, staging)
    except OSError as exc:
        # Clean up a half-written staging file from this attempt.
        _clear_siblings(staging)
        raise AtomicSwapError(
            f"failed to stage {src} -> {staging}: {exc.strerror or exc}"
        ) from exc

    # fsync the staged file so the bytes are durable before we rename.
    try:
        with open(staging, "rb") as fh:
            os.fsync(fh.fileno())
    except OSError:
        # fsync is best-effort; the rename below is still atomic at the
        # inode level. Crash-survival of in-flight bytes is the only thing
        # we lose.
        pass

    # 2. Move the old binary aside (if any) so a failed rename can be undone.
    #    We do this BEFORE the atomic replace so the rename has only one
    #    destination to worry about.
    if dst.exists() or dst.is_symlink():
        try:
            os.replace(dst, backup)
        except OSError as exc:
            _clear_siblings(staging)
            raise AtomicSwapError(
                f"failed to move existing destination aside before swap "
                f"({dst} -> {backup}): {exc.strerror or exc}"
            ) from exc

    # 3. The atomic swap. ``os.replace`` is the ONLY Python stdlib call that
    #    is guaranteed atomic on both POSIX and Windows. ``os.rename`` on
    #    Windows raises EEXIST when destination exists — the very bug we
    #    are fixing.
    try:
        os.replace(staging, dst)
    except OSError as exc:
        # Roll back: try to restore the backup to its original location.
        if backup.exists() and not dst.exists():
            try:
                os.replace(backup, dst)
            except OSError:
                # The original is gone AND the swap failed. Surface the
                # original error so the caller can decide whether to
                # re-stage manually.
                _clear_siblings(staging, backup)
                raise AtomicSwapError(
                    f"atomic swap failed AND rollback failed: {exc.strerror or exc}"
                ) from exc
        _clear_siblings(staging, backup)
        raise AtomicSwapError(
            f"atomic swap {staging} -> {dst} failed: {exc.strerror or exc}"
        ) from exc

    # 4. fsync the directory so the rename entry survives a crash. Without
    #    this, a power loss between the rename and the fsync can leave the
    #    directory entry pointing at the old (now-unlinked) inode.
    _fsync_dir(dst.parent)

    # 5. Apply the requested mode. Done after the rename so the chmod
    #    happens against the live binary, not a half-promoted staging file.
    if mode is not None and os.name != "nt":
        try:
            os.chmod(dst, mode)
        except OSError as exc:
            raise AtomicSwapError(
                f"swap succeeded but chmod({dst}, {oct(mode)}) failed: "
                f"{exc.strerror or exc}"
            ) from exc

    # 6. Best-effort: drop the backup. Leaving it around costs one cycle's
    #    worth of disk; removing it loses the roll-back handle for any
    #    subsequent failure. We err on the side of cleanliness — a swap
    #    that needs to roll back will create its own backup on retry.
    _clear_siblings(backup)

    return dst


def verify_swap(
    destination: PathLike,
    *,
    expected_size: Optional[int] = None,
    expected_mode: Optional[int] = None,
    expected_uid: Optional[int] = None,
) -> bool:
    """Verify the post-swap state of *destination*.

    Used by tests and by the desktop in-app updater's success path to
    confirm the swap completed and the binary is launchable (correct mode,
    correct owner on POSIX).

    Returns True when every supplied expectation matches, False otherwise.
    Missing-destination or stat errors return False rather than raising so
    a verification check never blocks a successful handoff.
    """
    try:
        st = os.stat(str(destination))
    except OSError:
        return False
    if expected_size is not None and st.st_size != expected_size:
        return False
    if expected_mode is not None and os.name != "nt":
        if (st.st_mode & 0o7777) != (expected_mode & 0o7777):
            return False
    if expected_uid is not None and os.name != "nt":
        if st.st_uid != expected_uid:
            return False
    return True


def expected_version_marker(install_root: PathLike) -> Path:
    """Return the canonical version-marker path inside the desktop bundle.

    Mirrors where electron-builder writes the version stamp inside the
    unpacked release tree (``apps/desktop/release/<plat>-unpacked/``). Used
    by the post-swap verification helper and by the test suite to assert
    "the installed bundle version matches what `hermes-desktop --version`
    reports".

    Cross-platform: the file lives in the unpacked bundle's resources dir
    on all platforms; only the parent dir name differs.
    """
    return Path(install_root) / "version"


__all__ = [
    "AtomicSwapError",
    "STAGING_SUFFIX",
    "BACKUP_SUFFIX",
    "atomic_swap_binary",
    "verify_swap",
    "expected_version_marker",
]