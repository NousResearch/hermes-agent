"""The pre-PM ``uv``/``uvx`` family an old install left in ``$HERMES_HOME/bin``.

A leftover there shadows the user's own uv (#101269). Identity (native format +
uv's own strings) proves only *what* the file is, never that Hermes installed
this copy — astral's installer wrote no receipt — so deletion is an explicit
user decision (``hermes doctor --fix``); ordinary maintenance only points at it
(:func:`hermes_cli.update_cmd_maint`) and a keep-data uninstall preserves it.
Automatic deletion is limited to POSIX platforms with kernel no-clobber
rename support (:func:`automatic_cleanup`) and every move is no-clobber, so
another writer's file can never be overwritten.
"""

import ctypes
import ctypes.util
import errno
import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path

from hermes_cli.colors import Colors, color

#: Lower bound on a plausible binary per leaf kind (full ``uv`` >1 MiB, thin ``uvx``
#: floor set at 64 KiB, well below its real ~330 KiB to avoid missing a leaner build);
#: a floor is not ownership evidence — see :data:`_IDENTITY_MARKERS`.
_MIN_BINARY_BYTES: dict[str, int] = {"uv": 1 << 20, "uvx": 64 << 10}

#: Native-executable leading bytes: ELF, Mach-O (thin/fat, both endians), PE.
_NATIVE_EXECUTABLE_MAGICS = (
    b"\x7fELF",
    b"\xfe\xed\xfa\xce",
    b"\xfe\xed\xfa\xcf",
    b"\xce\xfa\xed\xfe",
    b"\xcf\xfa\xed\xfe",
    b"\xca\xfe\xba\xbe",
    b"\xbe\xba\xfe\xca",
    b"MZ",
)

#: ASCII strings the release binaries embed: ``uv`` its env-var names and repo slug,
#: the thin ``uvx`` launcher its own messages.
_UV_IDENTITY_MARKERS: tuple[bytes, ...] = (
    b"UV_CACHE_DIR",
    b"UV_PYTHON_INSTALL_DIR",
    b"UV_TOOL_DIR",
    b"UV_TOOL_BIN_DIR",
    b"astral-sh/uv",
)
_UVX_IDENTITY_MARKERS: tuple[bytes, ...] = (
    b"Could not determine the location of the `uvx` binary",
    b"Could not find the `uv` binary at either of:",
)

#: Product identity: a native executable matching that kind's markers (``uvx``
#: accepts ``uv``'s — on Windows it is a copy). This identifies uv, NOT the hand
#: that installed it: astral's unpinned install.sh writes no receipt, so a user's
#: own uv is byte-similar. The verdict feeds a REPORT and an explicit ``--fix``,
#: never an unattended delete.
_IDENTITY_MARKERS: dict[str, tuple[bytes, ...]] = {
    "uv": _UV_IDENTITY_MARKERS,
    "uvx": _UV_IDENTITY_MARKERS + _UVX_IDENTITY_MARKERS,
}

#: Scan geometry: one marker-length of overlap finds a marker split by a chunk edge.
_IDENTITY_SCAN_CHUNK = 1 << 20
_IDENTITY_SCAN_OVERLAP = max(len(m) for m in _IDENTITY_MARKERS["uvx"]) - 1


def _leaf_kind(uv_name: str) -> str:
    """``'uv'`` or ``'uvx'`` for a leaf name, case- and ``.exe``-insensitive."""
    stem = uv_name[:-4] if uv_name.lower().endswith(".exe") else uv_name
    return "uvx" if stem.lower() == "uvx" else "uv"


def _is_native_executable(magic: bytes) -> bool:
    """Whether *magic* (the first bytes of a leaf) is a native-executable format."""
    return any(magic.startswith(prefix) for prefix in _NATIVE_EXECUTABLE_MAGICS)


def _scan_fd_for_uv_identity(fd: int, markers: tuple[bytes, ...]) -> bool:
    """Whether the already-open *fd* contains any of *markers*.

    Reads that descriptor (never a re-open) and treats a read error as no evidence."""
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        overlap = b""
        while True:
            chunk = os.read(fd, _IDENTITY_SCAN_CHUNK)
            if not chunk:
                return False
            window = overlap + chunk
            if any(marker in window for marker in markers):
                return True
            overlap = window[-_IDENTITY_SCAN_OVERLAP:]
    except OSError:
        return False


def _log_warn(msg: str) -> None:
    print(f"{color('⚠', Colors.YELLOW)} {msg}")


class _NoReplaceUnsupported(OSError):
    """The platform exposes no kernel rename that refuses an occupied destination."""


#: Per-platform no-clobber flag: Linux ``renameat2`` RENAME_NOREPLACE (uapi/fs.h,
#: 1 — NOT 2, which is RENAME_EXCHANGE and would swap the leaves); macOS
#: ``renameatx_np`` RENAME_EXCL (<sys/stdio.h>, 4 — note 2 is RENAME_SWAP).
_RENAME_FLAGS = {"linux": 0x00000001, "darwin": 0x00000004}

#: errno values that mean "this kernel/filesystem rejects the no-clobber flag"
#: (as opposed to a transient failure). The set is a flat union, not an OS
#: mapping: Linux ``renameat2`` on a pre-3.15 kernel or a flag-forbidding fs
#: answers EINVAL/ENOSYS, macOS ``renameatx_np`` answers ENOTSUP/EOPNOTSUPP, but
#: any member flips the binding regardless of platform.
_NOREPLACE_UNSUPPORTED_ERRNOS = frozenset(
    err
    for err in (
        errno.EINVAL,
        errno.ENOSYS,
        getattr(errno, "ENOTSUP", None),
        getattr(errno, "EOPNOTSUPP", None),
    )
    if err is not None
)
_noreplace_binding: tuple[str, ctypes._CFuncPtr] | bool | None = None


def _bind_noreplace_rename() -> tuple[str, ctypes._CFuncPtr] | bool:
    """Bind the *at*-fd rename that fails on an occupied destination.

    ``renameat2(RENAME_NOREPLACE)`` on Linux, ``renameatx_np(RENAME_EXCL)`` on
    macOS; both keep the directory anchor. Absence is sticky within the process —
    a fail-closed cleanup then names the manual step instead of retrying leaves."""
    global _noreplace_binding
    if _noreplace_binding is not None:
        return _noreplace_binding
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    except (OSError, TypeError):
        libc = None
    binding: tuple[str, ctypes._CFuncPtr] | bool = False
    if libc is not None:
        if sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
            fn = libc.renameat2
            fn.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
            fn.restype = ctypes.c_int
            binding = ("linux", fn)
        elif sys.platform == "darwin" and hasattr(libc, "renameatx_np"):
            fn = libc.renameatx_np
            fn.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_ulong]
            fn.restype = ctypes.c_int
            binding = ("darwin", fn)
    _noreplace_binding = binding
    return binding


def _rename_noreplace(fd: int, src: str, dst: str) -> None:
    """Rename *src* to *dst* inside the anchored directory, never replacing.

    Raises :class:`FileExistsError` when *dst* is occupied (an atomic decision —
    no pre-check window), :class:`_NoReplaceUnsupported` where the kernel/fs
    rejects the flag (:data:`_NOREPLACE_UNSUPPORTED_ERRNOS`, a platform-agnostic
    set), or another OSError."""
    binding = _bind_noreplace_rename()
    if not binding:
        raise _NoReplaceUnsupported("no rename-with-no-replace on this platform")
    flavor, fn = binding
    rc = fn(fd, os.fsencode(src), fd, os.fsencode(dst), _RENAME_FLAGS[flavor])
    if rc == 0:
        return
    err = ctypes.get_errno()
    if err == errno.EEXIST:
        raise FileExistsError(err, os.strerror(err), dst)
    if err in _NOREPLACE_UNSUPPORTED_ERRNOS:
        # Old kernel (unknown flag) or a fs that forbids the flag: ordinary
        # rename would silently clobber, so disable and fail closed. EPERM is a
        # genuine permission failure, not an unsupported flag — leave the
        # binding usable and let the caller report it.
        globals()["_noreplace_binding"] = False
        raise _NoReplaceUnsupported(err, os.strerror(err), src)
    raise OSError(err, os.strerror(err), src)


#: The pre-PM uv family an old install/`uv self` pair dropped in ``$HERMES_HOME/bin``.
LEGACY_MANAGED_UV_NAMES = ("uv", "uvx", "uv.exe", "uvx.exe")

#: Shared by the anchored probe and doctor's report so the two cannot disagree.
ABSENT, UNPROBEABLE, UNPROVEN, REMOVABLE = "absent", "unprobeable", "unproven", "removable"


def automatic_cleanup() -> bool:
    """Whether an anchored no-clobber cleanup can run right now.

    Directory-fd deletion exists only on POSIX; the no-clobber rename is probed
    lazily so a filesystem that rejects the flag at runtime
    (:data:`_NOREPLACE_UNSUPPORTED_ERRNOS`) flips this off for the rest of the
    process instead of leaving a ``--fix`` that can never succeed."""
    return os.name == "posix" and _bind_noreplace_rename() is not False


#: Bounded wait on PM's install lock, fail-closed on timeout; module-level so tests can shorten it.
_LEGACY_UV_LOCK_TIMEOUT = 10.0


def _pm_install_lock():
    """PM's install lock, or a no-op when no store exists to serialize with.

    Lock the store PM's installs serialize on (``writable_store_root()``); an
    absent store is not created just to lock it."""
    from contextlib import nullcontext

    from pm.paths import writable_store_root
    from pm.store import Store

    root = writable_store_root()
    if not root.is_dir():
        return nullcontext()
    return Store(root).install_lock(timeout=_LEGACY_UV_LOCK_TIMEOUT)


def bin_escapes_home(hermes_home: Path) -> bool:
    """Whether ``<home>/bin`` resolves outside the home — deletion must never follow it.

    Realpath equality: a symlinked *home* still reads as anchored, an absent ``bin``
    is not an escape."""
    bin_dir = hermes_home / "bin"
    if not bin_dir.is_dir():
        return False
    return os.path.normcase(os.path.realpath(bin_dir)) != os.path.normcase(
        os.path.join(os.path.realpath(hermes_home), "bin")
    )


@dataclass(frozen=True)
class _BinAnchor:
    """The ``<home>/bin`` a deletion run is anchored to.

    ``fd`` pins the directory, so a later path swap cannot redirect the unlinks; an
    absent ``bin`` is ``None`` — nothing acquired means nothing to delete."""

    fd: int


def _pin_dir(path: Path) -> int:
    """Open *path* as a directory handle: the anchor later lookups are relative to."""
    return os.open(path, os.O_RDONLY | os.O_DIRECTORY)


def _open_home_bin(hermes_home: Path) -> _BinAnchor | None:
    """Anchor ``<home>/bin``; ``None`` when there is nothing to delete.

    Raises ``OSError`` on a fail-closed platform whose family still exists. The
    home is pinned, then ``bin`` is opened ``O_NOFOLLOW`` relative to it, so a
    mid-check swap of either cannot redirect the deletions."""
    bin_dir = hermes_home / "bin"
    if not bin_dir.is_dir():
        return None  # no anchor: stop, never fall back to path-based deletion
    if not automatic_cleanup():
        # Only where the family exists: a manual step for an absent file is unfixable.
        if bin_escapes_home(hermes_home):
            return None  # behind that link are the USER's own files; doctor names it
        if not any(os.path.lexists(bin_dir / name) for name in LEGACY_MANAGED_UV_NAMES):
            return None
        raise OSError(
            f"automatic legacy-uv cleanup is not supported on this platform — remove the uv family in "
            f"{bin_dir} manually"
        )
    try:
        home_fd = _pin_dir(hermes_home)
        try:
            if bin_escapes_home(hermes_home):
                raise OSError(f"{bin_dir} resolves outside the Hermes home; refusing to delete through it")
            # Path check and pinned identity must agree, or home moved under us.
            pinned = os.fstat(home_fd)
            via_path = os.stat(hermes_home)
            if (pinned.st_dev, pinned.st_ino) != (via_path.st_dev, via_path.st_ino):
                raise OSError(
                    f"{hermes_home} changed while its bin was being checked; refusing to delete"
                )
            return _BinAnchor(
                fd=os.open("bin", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=home_fd)
            )
        finally:
            os.close(home_fd)
    except FileNotFoundError:
        return None  # gone between probe and open


def _classify(kind: str, st: os.stat_result, magic: bytes, has_identity: bool) -> str:
    """:data:`REMOVABLE` only when the leaf is a regular file over *kind*'s floor,
    with native magic and *kind*'s markers; :data:`UNPROVEN` otherwise."""
    if not stat.S_ISREG(st.st_mode) or st.st_size < _MIN_BINARY_BYTES[kind]:
        return UNPROVEN
    if not _is_native_executable(magic):
        return UNPROVEN
    if not has_identity:
        return UNPROVEN
    return REMOVABLE


def _open_verified(name, st: os.stat_result, *, dir_fd: int | None = None) -> int | None:
    """Open *name* ``O_RDONLY|O_NOFOLLOW`` only while it is still the object *st* named.

    The descriptor fstats back to the same device/inode/size before any content is
    read, so a swap between stat and open yields ``None`` instead of another file's
    bytes, and the file type must match on every platform; POSIX additionally
    requires an equal ``st_mode``. The metadata gate
    (regular file, size floor) stays with the caller and runs FIRST — a FIFO/device
    must never reach a blocking open."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(name, flags, dir_fd=dir_fd)
    except OSError:
        return None
    try:
        opened = os.fstat(fd)
    except OSError:
        os.close(fd)
        return None
    # Windows' path-based stat synthesizes execute bits for an ``.exe`` name that
    # the descriptor's stat omits, so a full ``st_mode`` compare would reject an
    # unchanged file. Device/inode/size prove continuity everywhere; the type
    # bits must match everywhere (a dir/symlink/device swap at the name, which
    # Windows' synthesized inode would not separate), and equal mode is a
    # POSIX-only extra.
    if (opened.st_dev, opened.st_ino, opened.st_size) != (st.st_dev, st.st_ino, st.st_size):
        os.close(fd)
        return None
    if stat.S_IFMT(opened.st_mode) != stat.S_IFMT(st.st_mode):
        os.close(fd)
        return None
    if os.name == "posix" and opened.st_mode != st.st_mode:
        os.close(fd)
        return None
    return fd


def _read_magic_and_identity(fd: int, kind: str) -> tuple[bytes, bool]:
    """Magic + marker scan on an already verified descriptor (one open, scan
    skipped once the magic disqualifies the leaf)."""
    try:
        magic = os.read(fd, 8)
    except OSError:
        return b"", False
    return magic, _is_native_executable(magic) and _scan_fd_for_uv_identity(
        fd, _IDENTITY_MARKERS[kind]
    )


def _probe_leaf(anchor: _BinAnchor, uv_binary: Path, uv_name: str) -> tuple[str, os.stat_result | None]:
    """Classify one leaf through the pinned descriptor — the deletion authority.

    Returns the verdict and the probed ``stat`` (``None`` for ABSENT/UNPROBEABLE),
    which :func:`_remove_verified` binds the move to; metadata gates before the
    open, and the open descriptor must fstat back to the probed object."""
    kind = _leaf_kind(uv_name)
    try:
        st = os.stat(uv_name, dir_fd=anchor.fd, follow_symlinks=False)
    except FileNotFoundError:
        return ABSENT, None
    except OSError as e:
        _log_warn(f"Could not inspect {uv_binary}: {e}")
        return UNPROBEABLE, None
    if not stat.S_ISREG(st.st_mode) or st.st_size < _MIN_BINARY_BYTES[kind]:
        return UNPROVEN, st  # no open, no scan: the shape already disqualifies it
    fd = _open_verified(uv_name, st, dir_fd=anchor.fd)
    if fd is None:
        # Unopenable, or another object took the name between stat and open.
        return UNPROBEABLE, None
    try:
        magic, has_identity = _read_magic_and_identity(fd, kind)
    finally:
        os.close(fd)
    return _classify(kind, st, magic, has_identity), st


def _restore_public_name(anchor: _BinAnchor, staging: str, uv_name: str) -> bool:
    """Return a moved-aside object to its public name without overwriting anything.

    A name another writer populated while the leaf was staged stays theirs: the
    object is kept under the quarantine name for manual recovery instead."""
    try:
        _rename_noreplace(anchor.fd, staging, uv_name)
        return True
    except FileExistsError:
        _log_warn(
            f"{uv_name} now holds another file — the moved-aside copy is kept at "
            f"{staging} for you to inspect; move it manually if it is yours"
        )
        return False
    except OSError as e:
        _log_warn(
            f"Could not return {staging} to {uv_name}: {e} — the copy is kept under "
            f"{staging} for you to inspect; move it manually if it is yours"
        )
        return False


def _remove_verified(anchor: _BinAnchor, uv_name: str, expected_st: os.stat_result) -> bool:
    """Move the probed leaf aside and dispose it, never touching another writer's file.

    Every rename is kernel no-clobber: an occupied quarantine name refuses the
    move, and restoring an unexpected object refuses an occupied public name
    (the object stays quarantined). The dispose unlink acts on the name the
    verified inode was actually moved to."""
    staging = f".{uv_name}.hermes-cleanup-{os.getpid()}"
    try:
        _rename_noreplace(anchor.fd, uv_name, staging)
    except FileExistsError:
        # A previous run's quarantine, or a file of the user's, owns the name —
        # the leaf keeps its own, retryable name.
        _log_warn(
            f"Skipping {uv_name}: {staging} already exists in the same directory "
            "(a previous cleanup's leftover, or a file of yours) — remove it and retry"
        )
        return False
    except FileNotFoundError:
        return False  # gone between the probe and the move
    except OSError as e:
        _log_warn(f"Skipping {uv_name}: cannot move it aside safely ({e}) — remove it manually if it is a leftover")
        return False
    try:
        current = os.stat(staging, dir_fd=anchor.fd, follow_symlinks=False)
    except OSError as e:
        _log_warn(f"Skipping {uv_name}: lost track of the moved copy ({e})")
        _restore_public_name(anchor, staging, uv_name)
        return False
    if (current.st_dev, current.st_ino) != (expected_st.st_dev, expected_st.st_ino):
        # The name changed hands between the probe and the move: the staged object
        # is not the one we verified. Return or quarantine it; never dispose.
        _log_warn(f"{uv_name} changed while it was being checked — nothing was deleted")
        _restore_public_name(anchor, staging, uv_name)
        return False
    try:
        os.unlink(staging, dir_fd=anchor.fd)
    except OSError as e:
        # Disposal failed: make the failure retryable from the leaf's own name
        # when the public name is free, else keep the quarantined copy.
        _log_warn(f"Could not remove {uv_name}: {e}")
        _restore_public_name(anchor, staging, uv_name)
        return False
    return True


def classify_leftover(path: Path) -> str:
    """Report-only verdict by plain name; same metadata-first gate as
    :func:`_probe_leaf`, so the doctor never names a leaf the probe would refuse
    and never opens a special file (a blocking FIFO/device) for content I/O."""
    kind = _leaf_kind(path.name)
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return ABSENT
    except OSError:
        return UNPROBEABLE
    if not stat.S_ISREG(st.st_mode) or st.st_size < _MIN_BINARY_BYTES[kind]:
        return UNPROVEN  # no open, no read: links/devices/FIFOs and small files stop here
    fd = _open_verified(path, st)
    if fd is None:
        return UNPROBEABLE
    try:
        magic, has_identity = _read_magic_and_identity(fd, kind)
    finally:
        os.close(fd)
    return _classify(kind, st, magic, has_identity)


def remove_legacy_managed_uv(hermes_home: Path) -> list[Path]:
    """Delete the pre-PM ``uv``/``uvx`` in ``$HERMES_HOME/bin`` — the explicit
    decision path (``hermes doctor --fix``). Best-effort: only a :func:`_classify`
    leaf, only the probed inode, and never by clobbering another writer's file.
    """
    removed: list[Path] = []
    try:
        with _pm_install_lock():
            anchor = _open_home_bin(hermes_home)
            if anchor is None:
                # nothing to delete (absent/escaped bin, or gone mid-check); never
                # race a newly-created link
                return []
            try:
                for uv_name in LEGACY_MANAGED_UV_NAMES:
                    uv_binary = hermes_home / "bin" / uv_name
                    verdict, probed_st = _probe_leaf(anchor, uv_binary, uv_name)
                    if verdict != REMOVABLE or probed_st is None:
                        # Positive whitelist: an unknown verdict can only under-delete.
                        if verdict == UNPROVEN:
                            _log_warn(
                                f"Skipping {uv_binary}: not shaped like a pre-PM uv binary "
                                "(a link, not a native uv binary, under the size floor, or "
                                "without uv's own identity markers) — "
                                "remove it manually if it is a leftover"
                            )
                        continue
                    try:
                        if _remove_verified(anchor, uv_name, probed_st):
                            removed.append(uv_binary)
                    except OSError as e:
                        _log_warn(f"Could not remove {uv_binary}: {e}")
            finally:
                os.close(anchor.fd)
    except OSError as e:
        # Fail closed: leave them for the next run; never abort doctor's own check.
        _log_warn(f"Skipped legacy uv cleanup: {e}")
        return []
    return removed
