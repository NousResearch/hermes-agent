"""Bounded home maintenance after an installed revision changes.

A per-install, per-profile record and lock remain because the full SQLite
integrity guard is too expensive to run every time. Steps still own their
idempotence and rollback; failure is recorded until the next revision (or
an explicit post-update run). The one exception is a config migration that
stopped at a failing step: the steps before it are kept, and the rest is
retried on every start until it succeeds (one warning per start). PM owns
runtime diagnosis, not this record.

A home whose profile a plugin runs elsewhere (``hermes_cli.profiles_run_elsewhere``)
is written by the backend the plugin runs, so no host process maintains it:
nothing is run and nothing is recorded, and the first start after the marker
is gone does this revision's maintenance.
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
    except Exception as exc:
        logger.debug("pm git lookup failed: %s", exc)
    try:
        from hermes_platform.resolver.git import resolve_usable_git
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.split(".", 1)[0] == "hermes_platform":
            return None
        raise

    return resolve_usable_git()


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


def owed_config_migration(known: dict) -> dict | None:
    """The record's ``migrate_config`` result when that migration stopped at a failing step
    (``failed_step``; ``failed_warning`` says why), else None. *known*: :func:`read_last_known`."""
    owed = (known.get("results") or {}).get("migrate_config") or {}
    return owed if owed.get("failed_step") else None


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

def _home_runs_elsewhere() -> bool:
    """Whether a plugin runs this process's profile elsewhere, with one log line when it does.

    Asked only when maintenance is due (a new revision, or a step still owed), before the lock:
    the usual start, whose record says this revision is done, never gets here. The backend the
    plugin runs cannot list the marker folder and is answered False, like every reader of it.
    """
    from hermes_constants import get_hermes_home
    from hermes_cli import profiles_run_elsewhere

    entry = profiles_run_elsewhere.run_elsewhere_for_home(get_hermes_home())
    if entry is None:
        return False
    if entry.folder_error is not None:
        why = f"{entry.path.parent} cannot be read, so a plugin may run profile '{entry.profile}' elsewhere"
    else:
        owner = f"the '{entry.plugin}' plugin" if entry.plugin is not None else "a plugin"
        why = f"{owner} runs profile '{entry.profile}' elsewhere"
    logger.info("home maintenance: %s, so this host leaves its home alone; it is owed until the profile "
                "runs here", why)
    return True


def run_boot_bootstrap(project_root: Path) -> dict:
    """Bound home maintenance to one attempt per installed revision. A home whose profile a
    plugin runs elsewhere gets none from this host and no record: it stays owed."""
    from hermes_cli import post_update

    identity = current_install_identity(project_root)
    if not identity:
        return {"home": "skipped"}
    record = record_path(project_root)
    known = read_last_known(record)
    if known.get("identity") == identity:
        outcome = _retry_owed_migration(record, identity, known)
        if outcome == "skipped":
            _recheck_default_plugins(post_update)
        return {"home": outcome}
    if _home_runs_elsewhere():
        return {"home": "runs-elsewhere"}
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


def _recheck_default_plugins(post_update) -> None:
    """Every start of an already-maintained revision re-asks the default-plugin step: the live catalog
    can add a default without a new revision. The step is cheap when this revision and catalog are
    settled and only starts a detached child otherwise, so a start never waits on the network. It is
    not recorded: the revision's maintenance record stays as it was."""
    step = dict(post_update.BOOT_HOME_STEPS).get("install_default_plugins")
    if step is None or _home_runs_elsewhere():
        return
    try:
        step()
    except (Exception, SystemExit):  # health: allow BLE001 -- a recheck never stops a start (ConfigBackendUnavailable is a SystemExit)
        logger.debug("default plugin recheck failed", exc_info=True)


def _owed_steps(known: dict) -> list[str]:
    """Steps of this revision still owed: a config migration that stopped at a failing step, and any
    step whose result says ``owed`` (default plugins a failed fetch left uninstalled)."""
    results = known.get("results") or {}
    owed = [name for name, result in results.items() if isinstance(result, dict) and result.get("owed")]
    return owed + (["migrate_config"] if owed_config_migration(known) is not None else [])


def _retry_owed_migration(record: Path, identity: str, known: dict):
    """This revision's maintenance is done, but a step is still owed (:func:`_owed_steps`): rerun those
    steps on every start until they succeed. A config migration warns once per start; the default-plugin
    step backs off itself, on one failure stamp for every home and plugin. The other steps stay bounded
    to one attempt per revision."""
    from hermes_cli import post_update

    if not _owed_steps(known):
        return "skipped"
    if _home_runs_elsewhere():
        return "runs-elsewhere"
    lock = _RecordLock(record)
    if not lock.acquire():
        return "lost-race"
    try:
        known = read_last_known(record)
        names = _owed_steps(known)
        if not names or known.get("identity") != identity:
            return "skipped"
        owed = owed_config_migration(known)
        retried = post_update.run_steps([step for step in post_update.BOOT_HOME_STEPS if step[0] in names])
        outcome = retried.get("migrate_config") or {}
        if owed is not None and not outcome.get("ok") and not outcome.get("failed_step"):
            # The retry raised (a full disk, a lock): the step is still owed, so the next start
            # retries it too. Only a retry that succeeded clears it.
            retried["migrate_config"] = {**outcome, "failed_step": owed["failed_step"],
                                         "failed_warning": outcome.get("error") or owed.get("failed_warning")}
        for name in names:
            result = retried.get(name)
            if name != "migrate_config" and isinstance(result, dict) and not result.get("ok"):
                retried[name] = {**result, "owed": True}  # a retry that raised keeps the step owed
        _write_record(record, identity, {**known["results"], **retried})
        return retried
    finally:
        lock.release()


def maybe_run_boot_bootstrap(project_root: Path) -> None:
    """The one call boot paths use. Never raises: a bootstrap problem must
    not stop the gateway/serve/CLI from starting."""
    try:
        run_boot_bootstrap(Path(project_root))
    except Exception as exc:
        logger.warning("boot bootstrap failed (continuing boot): %s", exc)
