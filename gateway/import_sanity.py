"""Detect source code swapped under a running gateway process.

The gateway is long-lived; Python caches imported modules in ``sys.modules``.
If the checkout changes while the process runs — an update flow, a concurrent
git operation, a stray test suite — the process keeps executing the OLD
modules while lazy imports of changed files fail with ``ImportError``. The
process looks healthy (platforms stay connected, polling works) yet lazy
imports fail unpredictably, and fixing the on-disk source does not help: only
a restart does, and nothing in the logs points there.

The check is a two-time proof (#96464):

1. At import time — for a gateway process that is effectively process birth,
   because ``gateway/__init__`` imports this module before ``run.py``'s body
   runs — capture a *birth receipt*: the HEAD commit (read straight from the
   ``.git`` files, no subprocess) and the on-disk identity of every repo
   module already loaded at that moment.
2. :func:`startup_import_smoke`, run at the end of gateway startup: eagerly
   import the late-loaded critical modules, then compare the world against
   the birth receipt. A checkout that moved during the startup window — a
   cached OLD module in ``sys.modules`` while the file on disk is already NEW
   — cannot masquerade as healthy: the receipt still describes the OLD
   generation, so the mismatch fails the comparison. Only when it passes does
   the smoke seed the runtime snapshot: the on-disk identity of every loaded
   module living inside this repository (a ``sys.modules`` sweep, not a
   hand-picked list).
3. :func:`verify_runtime_snapshots` — called periodically. Re-stats the
   snapshotted files; any mtime/size drift from the snapshot means the
   checkout moved under the live process and is reported exactly once per
   file with a restart instruction, while the return value stays False for as
   long as the drift persists.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys

logger = logging.getLogger(__name__)

# Modules the gateway only imports lazily (per-message agent turns, cron job
# execution) — eagerly imported by the smoke so a swap that already happened
# during the launch window surfaces immediately, and so they are guaranteed
# to be part of the runtime snapshot. Tests pass their own tuple explicitly.
_MONITORED_MODULES = ("run_agent", "tools.terminal_tool")

# Runtime snapshot populated only by a startup smoke whose receipt comparison
# passed: {module_name: (path, st_mtime_ns, st_size)}.
_snapshots: dict[str, tuple[str, int, int]] = {}
_reported: set[str] = set()

# Birth receipt: (HEAD commit or None, {module_name: (path, mtime_ns, size)}).
# Captured once, at this module's import — the earliest observation it can
# make. Overridable in tests to stage specific startup-window scenarios.
_BIRTH: tuple[str | None, dict[str, tuple[str, int, int]]] | None = None


def _repo_root() -> str:
    """Repository root (parent of the ``gateway`` package), resolved."""
    return os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def _read_git_dir(root: str) -> str | None:
    """Resolve the .git dir for a plain checkout or a linked worktree."""
    dot = os.path.join(root, ".git")
    if os.path.isdir(dot):
        return dot
    if os.path.isfile(dot):
        # Linked worktree pointer: "gitdir: /path/to/main/.git/worktrees/<n>"
        try:
            with open(dot, encoding="utf-8") as fh:
                first = fh.readline().strip()
        except OSError:
            return None
        if first.startswith("gitdir:"):
            return first[len("gitdir:"):].strip() or None
    return None


def _resolve_git_ref(git_dir: str, ref: str) -> str | None:
    """Resolve a ref name to a commit id via loose refs and packed-refs."""
    common = git_dir
    commondir = os.path.join(git_dir, "commondir")
    if os.path.isfile(commondir):
        try:
            with open(commondir, encoding="utf-8") as fh:
                rel = fh.read().strip()
        except OSError:
            rel = ""
        if rel:
            common = rel if os.path.isabs(rel) else os.path.normpath(
                os.path.join(git_dir, rel)
            )
    # Per-worktree refs live under git_dir; shared refs under the common dir.
    for base in (git_dir, common):
        path = os.path.join(base, *ref.split("/"))
        try:
            with open(path, encoding="utf-8") as fh:
                value = fh.read().strip()
        except OSError:
            continue
        if value.startswith("ref:"):  # nested symref — don't guess
            return None
        return value or None
    packed = os.path.join(common, "packed-refs")
    try:
        with open(packed, encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) == 2 and parts[1] == ref:
                    return parts[0]
    except OSError:
        pass
    return None


def _read_head_commit(root: str) -> str | None:
    """Best-effort HEAD commit id through plain file reads (no subprocess).

    Covers the layouts a gateway process actually meets: a plain checkout,
    a linked worktree, detached HEAD, and refs packed into packed-refs.
    Returns None on anything unexpected — a missing receipt disables the
    startup comparison rather than guessing (same conservatism as the probe:
    no proof, no verdict).
    """
    git_dir = _read_git_dir(root)
    if not git_dir:
        return None
    try:
        with open(os.path.join(git_dir, "HEAD"), encoding="utf-8") as fh:
            head = fh.read().strip()
    except OSError:
        return None
    if head.startswith("ref:"):
        return _resolve_git_ref(git_dir, head[4:].strip())
    return head or None  # detached HEAD carries the raw commit id


def _iter_loaded_repo_modules(root: str):
    """Yield (name, path) for every loaded module whose file lives under root.

    Sweeping sys.modules instead of a hand-picked list means any repo file
    touched by a later checkout swap (tools.environments.*, registry, cron,
    plugins...) is covered without maintaining an inventory that would rot.
    """
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if not path:
            continue
        try:
            resolved = os.path.realpath(path)
        except OSError:
            continue
        if resolved == root or resolved.startswith(root + os.sep):
            yield name, path


def _capture_birth_receipt(root: str) -> tuple[str | None, dict[str, tuple[str, int, int]]]:
    """Snapshot the source generation as of NOW (= this module's import).

    The HEAD commit catches checkout-level switches (update flows, rebases,
    resets); the per-module stats of whatever repo modules are already loaded
    close the only pre-receipt blind spot (the gateway package init itself).
    Modules imported AFTER this moment are covered by the other half of the
    proof: the startup smoke compares against this receipt before seeding.
    """
    modules: dict[str, tuple[str, int, int]] = {}
    for name, path in _iter_loaded_repo_modules(root):
        try:
            st = os.stat(path)
        except OSError:
            continue
        modules[name] = (path, st.st_mtime_ns, st.st_size)
    return _read_head_commit(root), modules


_BIRTH = _capture_birth_receipt(_repo_root())


def _snapshot_module(name: str) -> tuple[int, int] | None:
    """Return (st_mtime_ns, st_size) for an imported module, or None.

    Never raises — a module vanishing between the sys.modules lookup and the
    stat is a skip, not a canary failure (TOCTOU window is nanoseconds).
    """
    module = sys.modules.get(name)
    path = getattr(module, "__file__", None)
    if not path:
        return None
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size)


def _snapshot_loaded_repo_modules(root: str) -> None:
    """Record the on-disk identity of every loaded repo module as baseline."""
    for name, path in _iter_loaded_repo_modules(root):
        snap = _snapshot_module(name)
        if snap is not None:
            _snapshots[name] = (path, snap[0], snap[1])
            _reported.discard(name)


def _verify_birth_receipt() -> bool:
    """Compare the import-time receipt against the world as it is NOW.

    This is the second observation of the two-time proof. ``import_module``
    returns the cached object for anything already in ``sys.modules`` and
    never re-reads disk, so without an earlier receipt a swap that happened
    during the startup window would be adopted as the healthy baseline. The
    receipt was minted before the late imports ran, so cached-OLD-module +
    NEW-file-on-disk necessarily mismatches it here.

    The HEAD leg follows the same law as the per-module leg: a birth commit
    that cannot be re-read at smoke time (ref lock, repack, mid-transition)
    is INCONCLUSIVE — unhealthy, no seeding, WARNING instead of a
    confirmed-swap ERROR — and a later smoke can recover once the ref is
    readable again. A missing second observation is not proof of equality:
    modules first imported after the receipt have no birth stat, so the HEAD
    leg is their only first observation and must not vanish silently.
    """
    receipt = _BIRTH
    if receipt is None:  # receipt capture failed — disable, don't guess
        return True
    commit, modules = receipt
    healthy = True
    if commit:
        now_commit = _read_head_commit(_repo_root())
        if now_commit is None:
            healthy = False
            logger.warning(
                "Import sanity check INCONCLUSIVE: the birth receipt captured "
                "HEAD %s but the current HEAD cannot be read — cannot verify "
                "the checkout generation, runtime snapshot not seeded; a "
                "later smoke can recover once the ref is readable again",
                commit[:12],
            )
        elif commit != now_commit:
            healthy = False
            logger.error(
                "Import sanity check FAILED: the checkout changed during the "
                "startup window (HEAD %s -> %s) — this process may be running a "
                "mix of pre- and post-change code; restart the gateway to load a "
                "consistent tree",
                commit[:12],
                now_commit[:12],
            )
    for name, (path, mtime_ns, size) in modules.items():
        try:
            st = os.stat(path)
        except FileNotFoundError:
            healthy = False
            logger.error(
                "Import sanity check FAILED: source for %s vanished during "
                "the startup window (%s) — the checkout moved under this "
                "process while it was launching; restart the gateway before "
                "trusting it",
                name,
                path,
            )
            continue
        except OSError as exc:
            # Transient stat failure: can't confirm the world matches the
            # receipt, so don't seed a baseline from an unverified state.
            logger.warning(
                "Cannot stat %s (%s) while comparing the startup receipt — "
                "treating as unhealthy, runtime snapshot not seeded",
                path,
                exc,
            )
            healthy = False
            continue
        if (st.st_mtime_ns, st.st_size) != (mtime_ns, size):
            healthy = False
            logger.error(
                "Import sanity check FAILED: %s changed on disk during the "
                "startup window (%s) — modules imported before the change "
                "hold OLD code in sys.modules while later imports read NEW "
                "files; restart the gateway",
                name,
                path,
            )
    return healthy


def startup_import_smoke(
    modules: tuple[str, ...] = _MONITORED_MODULES,
    snapshot_root: str | None = None,
) -> bool:
    """Eagerly import the lazy critical modules, then seed the runtime snapshot.

    The order is the proof: receipt was captured at import time, the eager
    imports happen now, and only if the world still matches the receipt does
    the snapshot become the ongoing baseline. Returns True when every
    monitored module imported cleanly AND the receipt comparison passed;
    False means the gateway is already in a broken state and an ERROR has
    been logged. Never raises — the canary must not block startup.
    """
    healthy = True
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - report, don't crash startup
            healthy = False
            logger.error(
                "Import sanity check FAILED for %s: %s: %s — the environment is "
                "broken or the source tree changed under this process since it "
                "started; lazy imports (cron jobs, agent turns) will keep failing "
                "until the gateway is restarted",
                name,
                type(exc).__name__,
                exc,
            )
    healthy = _verify_birth_receipt() and healthy
    # Seed the ongoing baseline ONLY from a verified state: seeding after a
    # detected swap would adopt the NEW generation as "healthy" and mute the
    # runtime canary for the exact mixed state it exists to catch.
    if healthy:
        _snapshot_loaded_repo_modules(snapshot_root or _repo_root())
        logger.info(
            "Import sanity check OK (%d repo modules snapshotted; eager-imported: %s)",
            len(_snapshots),
            ", ".join(sorted(modules)) or "none",
        )
    return healthy


def verify_runtime_snapshots() -> bool:
    """Re-stat snapshotted modules; report on-disk drift exactly once per file.

    Returns True only while every snapshotted file still matches disk — the
    ERROR is logged once, but the return value stays False for as long as the
    drift persists, so callers reading the bool cannot mistake "already
    reported" for "recovered". A mismatch means the checkout was switched
    under the live process — the process runs mixed/stale code until it is
    restarted (#96464).
    """
    healthy = True
    for name, (path, mtime_ns, size) in list(_snapshots.items()):
        try:
            st = os.stat(path)
        except FileNotFoundError:
            # File vanished (checkout switched / branch deleted): same class
            # of incident — the on-disk world no longer matches the process.
            if name not in _reported:
                _reported.add(name)
                logger.error(
                    "Source file for %s disappeared (%s) after this process "
                    "started — the checkout was switched under a running "
                    "gateway; restart required to pick up consistent code",
                    name,
                    path,
                )
            healthy = False
            continue
        except OSError as exc:
            # Transient filesystem trouble (permissions, NFS stale handle) is
            # indistinguishable from a moved checkout at this layer — warn at
            # a level that won't be mistaken for a confirmed swap, and keep
            # returning False so the condition can't be silently ignored.
            logger.warning(
                "Cannot stat source file for %s (%s): %s — transient filesystem "
                "error or the checkout moved; will re-check next tick",
                name,
                path,
                exc,
            )
            healthy = False
            continue
        if (st.st_mtime_ns, st.st_size) != (mtime_ns, size):
            if name not in _reported:
                _reported.add(name)
                logger.error(
                    "Source file for %s changed on disk after this process started "
                    "(%s) — the checkout was switched under a running gateway; this "
                    "process keeps the OLD code in memory while later imports read "
                    "the NEW files, so it runs mixed/stale code with undefined "
                    "behavior until the gateway is restarted",
                    name,
                    path,
                )
            healthy = False
    return healthy
