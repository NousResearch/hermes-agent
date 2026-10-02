"""Bounded home maintenance after an installed revision changes.

A per-install, per-profile record and lock remain because config migrations
must not retry a broken migration on every launch, and the full SQLite
integrity guard is too expensive to run every time. Steps still own their
idempotence and rollback; failure is recorded until the next revision (or
an explicit post-update run). PM owns runtime diagnosis, not this record.
"""
from __future__ import annotations

from pm.environments import install_state_dir
import json
import logging
import os
import subprocess
import time
from pathlib import Path

logger = logging.getLogger(__name__)

RECORD_SCHEMA_VERSION = 1
LOCK_STALE_SECONDS = 600



# ---------------------------------------------------------------------------
# current identity
# ---------------------------------------------------------------------------

def read_git_head(root: Path) -> str | None:
    """The commit SHA of the checkout at ``root``.

    Asks git, rather than reimplementing it: parsing ``.git`` by hand
    (worktree gitfiles, symbolic HEAD, packed-refs, commondir) is a
    reimplementation of ``git rev-parse HEAD`` that reftable breaks
    wholesale. The managed git comes first, a PATH git second; with
    neither, the answer is None and boot carries on — fail-open.

    Cost: one ~10ms subprocess. Only checkouts pay it — a sealed tree
    reads its stamp and never gets here — and it happens once per boot.
    """
    git = _git_binary()
    if git is None:
        return None
    try:
        out = subprocess.run(
            [git, "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    sha = out.stdout.strip()
    return sha if len(sha) >= 7 else None


def _git_binary() -> str | None:
    """The git to run, or None when this machine has no usable one.

    pm's pinned git first (the win32 Git-for-Windows package — where it
    is installed, it is the one whose bash contract the rest of the tree
    already trusts), then a PATH git. The imports are local: this module
    is loaded early enough that a module-level import would widen the
    boot import graph for a lookup most platforms answer from PATH.
    """
    try:
        from pm import installed_package

        installed = installed_package("git")
        if installed is not None and installed.binary is not None:
            return str(installed.binary)
    except Exception as exc:  # noqa: BLE001 — boot must not die on a lookup
        logger.debug("pm git lookup failed: %s", exc)
    import shutil

    return shutil.which("git")


def current_install_identity(project_root: Path) -> str | None:
    """What code this install is: stamp commit for sealed trees, git HEAD
    for checkouts, None for broken trees (never bootstrap, never write)."""
    from hermes_cli.steward import read_install_stamp

    root = Path(project_root)
    if (root / ".git").exists():
        return read_git_head(root)
    stamp = read_install_stamp(root)
    commit = stamp.get("commit")
    if isinstance(commit, str) and len(commit) >= 7:
        return commit
    # A tagless/commitless stamp is a broken artifact; the tag alone is
    # accepted as a weaker identity (bundled artifacts always carry one).
    tag = stamp.get("tag")
    return tag if isinstance(tag, str) and tag else None


# ---------------------------------------------------------------------------
# the per-install state folder: installs/<SHA16>/ under the DEFAULT home
# ---------------------------------------------------------------------------

def record_path(project_root: Path) -> Path:
    """Each profile completes its own home maintenance for this installation."""
    from hermes_cli.profiles import get_active_profile_name

    name = get_active_profile_name() or "default"
    return install_state_dir(project_root) / "bootstrap" / f"{name}.json"


def read_last_known(path: Path) -> dict:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_record(path: Path, identity: str, results: dict) -> None:
    payload = {
        "schemaVersion": RECORD_SCHEMA_VERSION,
        "identity": identity,
        "bootstrappedAt": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "results": results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def needs_bootstrap(project_root: Path) -> str | None:
    """The new identity when this install changed since its last bootstrap,
    else None. None identity (broken tree) never bootstraps."""
    identity = current_install_identity(project_root)
    if not identity:
        return None
    known = read_last_known(record_path(project_root))
    if known.get("identity") == identity:
        return None
    return identity


# ---------------------------------------------------------------------------
# single-flight lock
# ---------------------------------------------------------------------------

class _RecordLock:
    """O_CREAT|O_EXCL existence-as-mutex next to a record file.

    Losers skip (boot never waits on another process's bootstrap; the steps
    are idempotent, so a botched winner only costs redundant work later).
    A stale lock — older than LOCK_STALE_SECONDS — is broken and re-tried
    once: a crashed winner died before its record write, so re-running is
    correct.
    """

    def __init__(self, record: Path):
        self.path = record.with_name(record.name + ".lock")
        self.acquired = False

    def _try_create(self) -> bool:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        except OSError:
            return False
        try:
            os.write(fd, json.dumps({"pid": os.getpid(), "startedAt": time.time()}).encode("utf-8"))
        finally:
            os.close(fd)
        return True

    def _is_stale(self) -> bool:
        try:
            body = json.loads(self.path.read_text(encoding="utf-8-sig"))
            started = float(body.get("startedAt", 0))
        except (OSError, ValueError):
            # Unreadable lock: age it by mtime instead.
            try:
                started = self.path.stat().st_mtime
            except OSError:
                return False
        return (time.time() - started) > LOCK_STALE_SECONDS

    def acquire(self) -> bool:
        if self._try_create():
            self.acquired = True
            return True
        if self._is_stale():
            try:
                self.path.unlink()
            except OSError:
                return False
            if self._try_create():
                self.acquired = True
                return True
        return False

    def release(self) -> None:
        if not self.acquired:
            return
        try:
            self.path.unlink()
        except OSError:
            pass
        self.acquired = False


# ---------------------------------------------------------------------------
# the boot entry point
# ---------------------------------------------------------------------------

def run_boot_bootstrap(project_root: Path) -> dict:
    """Bound home maintenance to one attempt per installed revision."""
    from hermes_cli import post_update

    identity = needs_bootstrap(project_root)
    if not identity:
        return {"home": "skipped"}
    record = record_path(project_root)
    lock = _RecordLock(record)
    if not lock.acquire():
        return {"home": "lost-race"}
    try:
        # A previous holder may have finished after our first read.
        if read_last_known(record).get("identity") == identity:
            return {"home": "done-by-other"}
        logger.info("home maintenance: code changed to %s, running steps", identity[:12])
        results = post_update.run_steps(post_update.BOOT_HOME_STEPS)
        _write_record(record, identity, results)
        return {"home": results}
    finally:
        lock.release()


def maybe_run_boot_bootstrap(project_root: Path) -> None:
    """The one call boot paths use. Never raises: a bootstrap problem must
    not stop the gateway/serve/CLI from starting."""
    try:
        run_boot_bootstrap(Path(project_root))
    except Exception as exc:
        logger.warning("boot bootstrap failed (continuing boot): %s", exc)
