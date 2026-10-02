"""The pre-PM ``uv``/``uvx`` family an old install left in ``$HERMES_HOME/bin``.

Dead weight — PM stages uv in its own store, off PATH — and load-bearing while
it lasts: ``install.ps1`` prepends that directory to the Windows User PATH, so
a leftover ``uv.exe`` shadows the user's own uv in every shell (#101269).
One implementation is shared by uninstall, ``hermes update``'s self-heal and
``hermes doctor --fix`` so the three cannot drift.

Cleanup is POSIX-only (:data:`AUTOMATIC_CLEANUP`): leaf deletion needs the
``dir_fd`` anchor, so other platforms fail closed — the doctor names a manual
step instead of a retry that could never succeed. Two residuals stay
informational by design, never findings: an ``UNPROVEN`` leaf (wrong shape —
it may still shadow PATH, but nothing proves it is Hermes' to delete) and a
``bin`` resolving outside the home (the files behind it are the user's).
"""

import os
import stat
from dataclasses import dataclass
from pathlib import Path

from hermes_cli.colors import Colors, color

#: The smallest uv release binary (uv 0.5's x86_64 Linux build) — a size floor
#: that keeps user wrapper scripts and tiny shims out of the deletion set: the
#: pre-PM installer staged astral's standalone builds, which are all far larger.
MIN_UV_BINARY_BYTES = 1 << 20


def _log_warn(msg: str) -> None:
    print(f"{color('⚠', Colors.YELLOW)} {msg}")


#: The pre-PM uv family an old install/`uv self` pair dropped in ``$HERMES_HOME/bin``.
LEGACY_MANAGED_UV_NAMES = ("uv", "uvx", "uv.exe", "uvx.exe")

#: Leaf verdicts: only :data:`REMOVABLE` (a proven pre-PM standalone binary) deletes.
#: The anchored cleanup probe and the doctor's report classification share this one
#: vocabulary, so the two can never disagree about what counts as Hermes's.
ABSENT, UNPROBEABLE, UNPROVEN, REMOVABLE = "absent", "unprobeable", "unproven", "removable"

#: Only POSIX exposes ``dir_fd``, the retained deletion anchor this implementation
#: requires; every other platform therefore fails closed by design.
AUTOMATIC_CLEANUP = os.name == "posix"

#: Bounded wait on PM's install lock before touching those binaries: legacy migration is a
#: shared mutation under the one install lock, fail-closed on timeout. Module-level so
#: tests can shorten it.
_LEGACY_UV_LOCK_TIMEOUT = 10.0


def _pm_install_lock():
    """PM's install lock, only where a store already exists to serialize with.

    ``install_lock()`` targets ``writable_store_root()`` — the root PM's installs
    lock; ``store_root()`` would be a different file on a sealed payload. And
    locking an absent store would create the store dir during an uninstall, so
    an absent root means nobody to serialize with.
    """
    from contextlib import nullcontext

    from pm.paths import writable_store_root
    from pm.store import Store

    root = writable_store_root()
    if not root.is_dir():
        return nullcontext()
    return Store(root).install_lock(timeout=_LEGACY_UV_LOCK_TIMEOUT)


def bin_escapes_home(hermes_home: Path) -> bool:
    """Whether ``<home>/bin`` resolves somewhere other than the home's own ``bin``.

    A symlinked (or junctioned) ``bin`` means deletion would traverse the parent
    and hit the user's real files — the cleanup must never do that. Identity is
    realpath equality: a symlinked home still reads as anchored; an absent bin
    is not an escape (there is nothing to anchor).
    """
    bin_dir = hermes_home / "bin"
    if not bin_dir.is_dir():
        return False
    return os.path.normcase(os.path.realpath(bin_dir)) != os.path.normcase(
        os.path.join(os.path.realpath(hermes_home), "bin")
    )


@dataclass(frozen=True)
class _BinAnchor:
    """The ``<home>/bin`` a deletion run is anchored to.

    ``fd`` is the POSIX directory handle the probes and leaf unlinks are relative
    to, so the directory identity is pinned when it is opened and a later swap of
    the *path* cannot redirect them. An *absent* ``bin`` is not representable —
    :func:`_open_home_bin` returns ``None`` instead, because "nothing was
    acquired" means "delete nothing".
    """

    fd: int


def _pin_dir(path: Path) -> int:
    """Open *path* itself as a directory handle (POSIX).

    The handle is the anchor every later lookup is relative to. A symlinked
    *path* is followed — a symlinked home IS the home — but once pinned, no
    later swap of the path can re-point anything derived from this handle.
    """
    return os.open(path, os.O_RDONLY | os.O_DIRECTORY)


def _open_home_bin(hermes_home: Path) -> _BinAnchor | None:
    """Anchor ``<home>/bin`` for the deletions below; ``None`` when there is nothing to delete.

    POSIX pins the chain instead of trusting a path twice: home is opened first,
    containment confirmed against the pinned identity, then ``bin`` opened
    ``O_NOFOLLOW`` RELATIVE to that handle — a linked ``bin`` fails the open
    (``ELOOP``) and a home swapped between check and open is refused by the
    identity re-verification. Without ``dir_fd``, an eligible family raises the
    ``OSError`` callers turn into a manual-step skip; linked or absent bins
    return ``None``.
    """
    bin_dir = hermes_home / "bin"
    if not bin_dir.is_dir():
        return None  # no anchor: callers must stop, not fall back to path-based deletion
    if not AUTOMATIC_CLEANUP:
        # Warn only where the family exists — "remove it manually" for a file
        # that is not there is a finding no action can clear.
        if bin_escapes_home(hermes_home):
            return None  # behind that link are the USER's own files; doctor names it
        if not any(os.path.lexists(bin_dir / name) for name in LEGACY_MANAGED_UV_NAMES):
            return None
        raise OSError(
            f"automatic legacy-uv cleanup is POSIX-only — remove the uv family in "
            f"{bin_dir} manually on this platform"
        )
    try:
        # Every decision below is relative to this handle, so a later path swap
        # cannot re-point them; a symlinked home is followed (it IS the home).
        home_fd = _pin_dir(hermes_home)
        try:
            if bin_escapes_home(hermes_home):
                raise OSError(f"{bin_dir} resolves outside the Hermes home; refusing to delete through it")
            # The check read the path; the handle holds the identity — a
            # disagreement means the home was swapped inside that window.
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
        return None  # gone between the probe and the open: nothing to anchor


def _classify(st: os.stat_result) -> str:
    """Shape verdict for one probed leaf: :data:`REMOVABLE` only for a proven
    pre-PM binary, :data:`UNPROVEN` otherwise. The pre-PM installer staged real
    standalone builds (never scripts or links) and left no receipt, so shape is
    the only ownership evidence — anything else surfaces for manual review."""
    if not stat.S_ISREG(st.st_mode) or st.st_size < MIN_UV_BINARY_BYTES:
        return UNPROVEN
    return REMOVABLE


def _probe_leaf(anchor: _BinAnchor, uv_binary: Path, uv_name: str) -> str:
    """Classify one leaf through the pinned descriptor — the deletion authority.

    A probe error is what the unlink would hit, so it is logged once here and
    the caller skips the name."""
    try:
        st = os.stat(uv_name, dir_fd=anchor.fd, follow_symlinks=False)
    except FileNotFoundError:
        return ABSENT
    except OSError as e:
        _log_warn(f"Could not inspect {uv_binary}: {e}")
        return UNPROBEABLE
    return _classify(st)


def classify_leftover(path: Path) -> str:
    """Shape verdict for ``path`` by plain name — report-only.

    The doctor classifies with this so its "removable" list never names a file
    the anchored probe (:func:`_probe_leaf`) would refuse."""
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return ABSENT
    except OSError:
        return UNPROBEABLE
    return _classify(st)


def remove_legacy_managed_uv(hermes_home: Path) -> list[Path]:
    """Delete the pre-PM ``uv``/``uvx`` binaries in ``$HERMES_HOME/bin``, best-effort.

    Ownership proof is shape only (:func:`_classify`): a wrapper, symlink or
    small script is skipped and left for manual review; launchers and the
    user's own scripts stay. A linked or absent ``bin`` refuses the run; on a
    fail-closed platform the manual step is logged instead.
    """
    removed: list[Path] = []
    try:
        with _pm_install_lock():
            anchor = _open_home_bin(hermes_home)
            if anchor is None:
                return []  # bin was absent; do not race a newly-created link.
            try:
                for uv_name in LEGACY_MANAGED_UV_NAMES:
                    uv_binary = hermes_home / "bin" / uv_name
                    verdict = _probe_leaf(anchor, uv_binary, uv_name)
                    if verdict != REMOVABLE:
                        # Positive whitelist, not an exclusion list: only a proven
                        # pre-PM binary is ever unlinked, so a future verdict (or a
                        # classification that changes shape) can only under-delete.
                        if verdict == UNPROVEN:
                            _log_warn(
                                f"Skipping {uv_binary}: not shaped like a pre-PM uv binary "
                                "(a link, or under the size floor) — remove it manually if it "
                                "is a leftover"
                            )
                        # ABSENT: nothing there. UNPROBEABLE: the probe logged why.
                        continue
                    try:
                        os.unlink(uv_name, dir_fd=anchor.fd)
                        removed.append(uv_binary)
                    except Exception as e:
                        _log_warn(f"Could not remove {uv_binary}: {e}")
            finally:
                os.close(anchor.fd)
    except Exception as e:
        # Fail closed either way — a lock timeout, a read-only store, or an
        # undecodable store-root probe leaves the binaries for the next run
        # (doctor/update retry); nothing here may abort doctor's own check.
        _log_warn(f"Skipped legacy uv cleanup: {e}")
        return []
    return removed
