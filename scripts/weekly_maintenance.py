#!/usr/bin/env python3
"""Weekly maintenance for the active Hermes profile.

Runs the safe-by-default housekeeping pass that long-lived Hermes
installs benefit from once a week:

  * ``state.db``      — WAL checkpoint (TRUNCATE) + ``VACUUM`` so the
                        SQLite file actually shrinks after the gateway's
                        opportunistic prune deleted rows.
  * snapshot retention — drop ``state-snapshots/`` entries older than the
                        configured horizon (default 90 days).
  * log rotation       — gzip ``logs/*.log`` files larger than 10 MB and
                        prune compressed logs older than 90 days.

Why this script exists
----------------------

Hermes runs idempotent maintenance opportunistically at startup
(``HermesState.maybe_auto_prune_and_vacuum``), but a long-running gateway
that almost never restarts can drift far past the maintenance window. A
small wrapper that the user can drop into ``cron`` / launchd / systemd
keeps the file size bounded without depending on restarts.

Critical: profile awareness (#24035)
------------------------------------

The previous community version of this script computed paths from
``os.path.expanduser("~")`` directly, which silently no-op'd users on
non-default profiles (``hermes -p work``) — VACUUM ran on the wrong
database and the active profile's state.db kept growing.

This version resolves every path through ``hermes_constants``:

  * ``get_hermes_home()`` — honours ``$HERMES_HOME`` and the active
    profile so the script always operates on the database the running
    Hermes process is actually using.
  * ``get_hermes_dir(new, old)`` — preserves the project's
    backward-compatibility migration rules where applicable.

Usage
-----

::

    # Default — run all phases, log to stderr.
    python scripts/weekly_maintenance.py

    # Dry-run — show what would happen without touching anything.
    python scripts/weekly_maintenance.py --dry-run

    # Only one phase.
    python scripts/weekly_maintenance.py --only vacuum
    python scripts/weekly_maintenance.py --only snapshots
    python scripts/weekly_maintenance.py --only logs

    # Tighter retention (default: 90 days).
    python scripts/weekly_maintenance.py --retention-days 30

The script honours ``$HERMES_HOME`` so that the same binary works for
every profile:

::

    HERMES_HOME=~/.hermes-work python scripts/weekly_maintenance.py
    hermes -p work shell -- python scripts/weekly_maintenance.py
"""

import argparse
import gzip
import json
import logging
import os
import shutil
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional


# ---------------------------------------------------------------------------
# Path resolution — every Hermes path MUST go through hermes_constants
# (#24035).  We bend over backwards to make this importable from
# ``~/.hermes/scripts/`` (where the community script lives) too — that's
# why we add the source checkout to ``sys.path`` defensively before the
# import attempt.
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent

_CANDIDATE_REPO_ROOTS = (
    _HERE.parent,                              # repo checkout: <repo>/scripts/
    _HERE.parent / "hermes-agent",             # ~/.hermes/scripts/ → ~/.hermes/hermes-agent
    Path.home() / ".hermes" / "hermes-agent",  # last-resort
)

for _candidate in _CANDIDATE_REPO_ROOTS:
    if (_candidate / "hermes_constants.py").is_file():
        _root_str = str(_candidate)
        if _root_str not in sys.path:
            sys.path.insert(0, _root_str)
        break

try:
    from hermes_constants import get_hermes_home  # type: ignore[import-not-found]
except Exception as exc:  # pragma: no cover - depends on user environment
    print(
        "weekly_maintenance: cannot import hermes_constants — make sure "
        "the hermes-agent checkout is reachable from this script. "
        f"({exc.__class__.__name__}: {exc})",
        file=sys.stderr,
    )
    raise


logger = logging.getLogger("hermes.weekly_maintenance")


# ---------------------------------------------------------------------------
# Resolved paths
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MaintenancePaths:
    """Container for every path the script touches.

    Constructed by :func:`resolve_paths` so the resolution logic is
    testable in isolation. Every field is derived from
    ``hermes_constants.get_hermes_home()`` — never from
    ``Path.home() / ".hermes"`` (#24035).
    """

    home: Path
    state_db: Path
    snapshots_dir: Path
    logs_dir: Path

    def to_json(self) -> str:
        """Serialise for ``--dry-run`` output."""
        return json.dumps(
            {
                "home": str(self.home),
                "state_db": str(self.state_db),
                "snapshots_dir": str(self.snapshots_dir),
                "logs_dir": str(self.logs_dir),
            },
            indent=2,
        )


def resolve_paths(home: Optional[Path] = None) -> MaintenancePaths:
    """Resolve every maintenance path from the active profile.

    *home* override exists for tests; production callers pass nothing
    and we read from :func:`hermes_constants.get_hermes_home`.
    """
    base = (home or get_hermes_home()).resolve()
    return MaintenancePaths(
        home=base,
        state_db=base / "state.db",
        snapshots_dir=base / "state-snapshots",
        logs_dir=base / "logs",
    )


# ---------------------------------------------------------------------------
# Phase results
# ---------------------------------------------------------------------------


@dataclass
class PhaseResult:
    name: str
    skipped: bool = False
    error: Optional[str] = None
    details: List[str] = field(default_factory=list)

    def add(self, line: str) -> None:
        self.details.append(line)

    def fail(self, msg: str) -> None:
        self.error = msg


# ---------------------------------------------------------------------------
# Phase: VACUUM state.db
# ---------------------------------------------------------------------------


def vacuum_state_db(paths: MaintenancePaths, *, dry_run: bool) -> PhaseResult:
    """Run ``WAL checkpoint(TRUNCATE)`` + ``VACUUM`` against the active state.db.

    Skips cleanly when:
      - the file does not exist (fresh install / wrong profile),
      - the file is currently locked by a live process (we never wait;
        a lock means the gateway is running and will eventually do its
        own opportunistic VACUUM at next restart).
    """
    result = PhaseResult(name="vacuum_state_db")
    db = paths.state_db

    if not db.exists():
        result.skipped = True
        result.add(f"state.db not found: {db} (skipping)")
        return result

    size_before = db.stat().st_size
    result.add(f"state.db = {db} ({_human_bytes(size_before)})")

    if dry_run:
        result.skipped = True
        result.add("dry-run — would WAL checkpoint(TRUNCATE) + VACUUM")
        return result

    try:
        # ``timeout=0`` means "fail fast if the DB is busy". A live
        # gateway with WAL-mode writes will hold an exclusive lock during
        # VACUUM so we never wait — the next restart's opportunistic
        # ``maybe_auto_prune_and_vacuum`` will pick up the slack.
        conn = sqlite3.connect(str(db), timeout=0, isolation_level=None)
        try:
            try:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except sqlite3.Error as exc:
                logger.debug("wal_checkpoint(TRUNCATE) failed: %s", exc)
            conn.execute("VACUUM")
        finally:
            conn.close()
    except sqlite3.OperationalError as exc:
        # Most common: "database is locked" — a live gateway is using it.
        result.fail(f"state.db busy ({exc}); skipping VACUUM")
        return result
    except Exception as exc:  # pragma: no cover - depends on env
        result.fail(f"unexpected VACUUM failure: {exc}")
        return result

    size_after = db.stat().st_size
    saved = size_before - size_after
    result.add(
        f"VACUUM done: {_human_bytes(size_before)} → {_human_bytes(size_after)} "
        f"({_human_bytes(saved)} reclaimed)"
    )
    return result


# ---------------------------------------------------------------------------
# Phase: prune snapshots
# ---------------------------------------------------------------------------


def prune_snapshots(
    paths: MaintenancePaths,
    *,
    retention_days: int,
    dry_run: bool,
    now: Optional[float] = None,
) -> PhaseResult:
    """Drop ``state-snapshots/<id>/`` entries older than *retention_days*.

    Only snapshot directories with a recognisable mtime older than the
    cutoff are removed. Files (e.g. README, manifest stubs) at the
    snapshot-root level are never touched.
    """
    result = PhaseResult(name="prune_snapshots")
    root = paths.snapshots_dir
    if not root.is_dir():
        result.skipped = True
        result.add(f"{root} not present (skipping)")
        return result

    cutoff = (now if now is not None else time.time()) - retention_days * 86400
    removed = 0
    kept = 0
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        try:
            mtime = entry.stat().st_mtime
        except OSError as exc:
            result.add(f"  ! stat failed: {entry.name} ({exc})")
            continue
        if mtime < cutoff:
            removed += 1
            if dry_run:
                result.add(f"  - would remove: {entry.name}")
            else:
                try:
                    shutil.rmtree(entry)
                    result.add(f"  - removed: {entry.name}")
                except Exception as exc:
                    result.add(f"  ! remove failed: {entry.name} ({exc})")
        else:
            kept += 1

    result.add(
        f"snapshots: {removed} removed, {kept} kept "
        f"(retention = {retention_days} days)"
    )
    return result


# ---------------------------------------------------------------------------
# Phase: rotate logs
# ---------------------------------------------------------------------------

_LOG_ROTATE_BYTES = 10 * 1024 * 1024  # 10 MB


def rotate_logs(
    paths: MaintenancePaths,
    *,
    retention_days: int,
    dry_run: bool,
    now: Optional[float] = None,
) -> PhaseResult:
    """Gzip large ``*.log`` files and drop very old compressed logs."""
    result = PhaseResult(name="rotate_logs")
    root = paths.logs_dir
    if not root.is_dir():
        result.skipped = True
        result.add(f"{root} not present (skipping)")
        return result

    rotated = 0
    pruned = 0
    cutoff = (now if now is not None else time.time()) - retention_days * 86400

    for entry in sorted(root.glob("*.log")):
        try:
            size = entry.stat().st_size
        except OSError as exc:
            result.add(f"  ! stat failed: {entry.name} ({exc})")
            continue
        if size < _LOG_ROTATE_BYTES:
            continue
        rotated += 1
        if dry_run:
            result.add(f"  - would rotate: {entry.name} ({_human_bytes(size)})")
            continue
        target = entry.with_suffix(entry.suffix + f".{int(time.time())}.gz")
        try:
            with entry.open("rb") as src, gzip.open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            entry.unlink()
            result.add(f"  - rotated: {entry.name} → {target.name}")
        except Exception as exc:
            result.add(f"  ! rotate failed: {entry.name} ({exc})")

    for entry in sorted(root.glob("*.log.*.gz")):
        try:
            mtime = entry.stat().st_mtime
        except OSError as exc:
            result.add(f"  ! stat failed: {entry.name} ({exc})")
            continue
        if mtime < cutoff:
            pruned += 1
            if dry_run:
                result.add(f"  - would drop old archive: {entry.name}")
            else:
                try:
                    entry.unlink()
                    result.add(f"  - dropped old archive: {entry.name}")
                except Exception as exc:
                    result.add(f"  ! drop failed: {entry.name} ({exc})")

    result.add(
        f"logs: {rotated} rotated, {pruned} archives dropped "
        f"(rotate ≥ {_human_bytes(_LOG_ROTATE_BYTES)}, retention = {retention_days} days)"
    )
    return result


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


_PHASE_REGISTRY = {
    "vacuum":    vacuum_state_db,
    "snapshots": prune_snapshots,
    "logs":      rotate_logs,
}


def run(
    *,
    only: Optional[str] = None,
    retention_days: int = 90,
    dry_run: bool = False,
    home: Optional[Path] = None,
) -> List[PhaseResult]:
    """Execute the requested phase(s) and return per-phase results."""
    paths = resolve_paths(home)
    selected = _select_phases(only)

    print(f"weekly_maintenance: HERMES_HOME = {paths.home}")
    print(f"weekly_maintenance: paths = {paths.to_json()}")
    if dry_run:
        print("weekly_maintenance: DRY-RUN — no files will be modified")

    results: List[PhaseResult] = []
    for name in selected:
        print(f"\n── phase: {name} ──")
        fn = _PHASE_REGISTRY[name]
        if name == "vacuum":
            res = fn(paths, dry_run=dry_run)
        else:
            res = fn(paths, retention_days=retention_days, dry_run=dry_run)
        for line in res.details:
            print(f"  {line}")
        if res.error:
            print(f"  ! {res.error}")
        results.append(res)

    return results


def _select_phases(only: Optional[str]) -> Iterable[str]:
    if not only:
        return list(_PHASE_REGISTRY.keys())
    if only not in _PHASE_REGISTRY:
        raise SystemExit(
            f"--only must be one of {sorted(_PHASE_REGISTRY)}, got {only!r}"
        )
    return [only]


def _human_bytes(n: int) -> str:
    """Compact human-readable byte size."""
    if n < 1024:
        return f"{n} B"
    units = ("KB", "MB", "GB", "TB")
    val = float(n) / 1024.0
    for unit in units:
        if val < 1024 or unit == units[-1]:
            return f"{val:.1f} {unit}"
        val /= 1024.0
    return f"{n} B"  # pragma: no cover - unreachable


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="weekly_maintenance.py",
        description=(
            "Run weekly maintenance against the active Hermes profile "
            "(VACUUM state.db, prune snapshots, rotate logs). "
            "Resolves every path through hermes_constants.get_hermes_home() "
            "so a profile-specific HERMES_HOME is honoured (#24035)."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would happen without modifying anything.",
    )
    parser.add_argument(
        "--only", choices=sorted(_PHASE_REGISTRY), default=None,
        help="Run a single phase instead of all three.",
    )
    parser.add_argument(
        "--retention-days", type=int, default=90,
        help="Days to keep snapshots and rotated logs (default: 90).",
    )
    args = parser.parse_args(argv)

    if args.retention_days <= 0:
        parser.error("--retention-days must be positive")

    logging.basicConfig(
        level=os.environ.get("HERMES_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    results = run(
        only=args.only,
        retention_days=args.retention_days,
        dry_run=args.dry_run,
    )
    return 1 if any(r.error for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
