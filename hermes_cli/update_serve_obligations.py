"""Durable manual-serve handoffs, independent of gateway restart receipts."""

import json
import logging
import math
import os
import sys
import tempfile
from contextlib import suppress
from pathlib import Path

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)


def defer_manual_serve(runtime: dict, *, require_alive: bool = False) -> bool:
    """Transfer an identified manual runtime to its own durable restart reminder.

    Most runtimes carry a numeric ``detail["create_time"]`` incarnation stamp, and the
    reminder is stored per incarnation (``{pid}-{create_time.hex()}.json``) so PID reuse
    can never discharge the wrong process. When the spawn ledger probed the process but
    could not read its incarnation (unreadable ``/proc``, permission error, missing
    psutil) the stamp is an explicit ``None``; that row still describes a live manual
    serve that must survive receipt rotation, so it is stored as a pid-scoped fallback
    (``{pid}-unknown.json``) that is discharged as soon as the PID is provably gone.
    Rejecting ``None`` here used to return ``False`` forever, leaving an immortal row
    in ``pending_manual_serves``. A ``detail`` dict with no ``create_time`` key at all
    carries no identity signal and is still rejected.
    """
    from hermes_cli.process_identity import _pid_alive_matches

    if runtime.get("kind") not in ("serve", "dashboard") or runtime.get("supervisor") != "manual-serve" or runtime.get("restart_via") != "respawn-argv":
        return False
    pid = runtime.get("pid")
    detail = runtime.get("detail")
    if not isinstance(detail, dict):
        return False
    if type(pid) is not int or pid <= 0:
        return False
    created = detail.get("create_time")
    if created is None and "create_time" not in detail:
        return False
    if created is not None and (
        type(created) not in (int, float) or not math.isfinite(created) or created <= 0
    ):
        return False
    try:
        alive = _pid_alive_matches(pid, created)
        if require_alive and alive is not True:
            return False
        if alive is False:
            return True
        directory = get_hermes_home() / "serve_restart_pending"
        directory.mkdir(parents=True, exist_ok=True)
        row = {"kind": runtime["kind"], "profile": runtime.get("profile", "unknown"), "pid": pid, "create_time": created}
        if created is None:
            target = directory / f"{pid}-unknown.json"
        else:
            target = directory / f"{pid}-{float(created).hex()}.json"
        # One immutable file per incarnation avoids read/merge/write races between CLI startups.
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory, delete=False) as handle:
                temporary = Path(handle.name)
                json.dump(row, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        _supersede_sibling_reminders(directory, pid, keep=target.name)
        return True
    except (OSError, ValueError, TypeError) as exc:
        logger.debug("Could not preserve manual serve obligation: %s", exc)
        return False


def _supersede_sibling_reminders(directory: Path, pid: int, *, keep: str) -> None:
    """Remove stale reminder files for ``pid`` left by an earlier incarnation stamp.

    Identity can flip between known and unknown across updates (the ledger reads
    ``/proc`` best-effort), so ``{pid}-{hex}.json`` and ``{pid}-unknown.json`` for the
    same process would otherwise warn side by side. The freshly written file wins;
    siblings are best-effort garbage. Never raises.
    """
    try:
        for sibling in directory.glob(f"{pid}-*.json"):
            if sibling.name != keep:
                with suppress(OSError):
                    sibling.unlink()
    except OSError as exc:
        logger.debug("Could not supersede sibling serve reminders for pid %s: %s", pid, exc)


def retain_receipt_manual_serves(receipt: dict) -> list[dict]:
    """Return transfers still owed so receipt rotation cannot discard failed writes.

    Rows that :func:`defer_manual_serve` persisted to ``serve_restart_pending/``
    (including pid-scoped ``None``-incarnation fallbacks) are discharged here and never
    re-appended, so no receipt row can outlive its process forever.
    """
    plan = receipt.get("plan") or {}
    rows = list(plan.get("runtimes") or []) + list(receipt.get("pending_manual_serves") or [])
    pending = []
    for row in rows:
        if not isinstance(row, dict) or row.get("kind") not in ("serve", "dashboard") or row.get("supervisor") != "manual-serve":
            continue
        if not defer_manual_serve(row) and row not in pending:
            pending.append(row)
    return pending


def warn_pending_manual_serves(*, startup: bool = False, pending_manual: list[dict] | None = None) -> None:
    """Warn about manual debt independently of gateway evidence; optionally reuse a snapshot's failed transfers."""
    from hermes_cli.process_identity import _pid_alive_matches
    from hermes_cli.update_receipt import read_latest_receipt

    stream = sys.stderr if startup else sys.stdout
    if pending_manual is None:
        pending_manual = retain_receipt_manual_serves(read_latest_receipt() or {})
    for row in pending_manual:
        print(f"  ⚠ {row['kind']} [{row.get('profile', 'unknown')}] pid {row.get('pid', 'unknown')}: manual restart reminder could not be saved; restart remains pending in the update receipt.", file=stream)
        print("    Ask its owner to relaunch `hermes serve` / `hermes dashboard`; check reminder storage permissions and free space.", file=stream)
    directory = get_hermes_home() / "serve_restart_pending"
    for path in sorted(directory.glob("*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            # ``create_time`` may be absent or null for pid-scoped fallback reminders
            # (unreadable incarnation at defer time); ``None`` still discharges as soon
            # as the PID is provably gone.
            if _pid_alive_matches(row["pid"], row.get("create_time")) is False:
                path.unlink(missing_ok=True)
                continue
            print(f"  ⚠ {row['kind']} [{row['profile']}] pid {row['pid']}: manual restart still pending; this process may still serve pre-update code.", file=stream)
            print("    Ask its owner to relaunch `hermes serve` / `hermes dashboard` (reconnect Desktop for an SSH backend).", file=stream)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.debug("Could not reconcile manual serve obligation %s: %s", path, exc)
            print(f"  ⚠ Manual serve restart reminder could not be verified: {path.name}", file=stream)
