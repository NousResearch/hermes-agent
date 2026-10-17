"""Durable manual-serve handoffs, independent of gateway restart receipts."""

import json
import logging
import math
import os
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)


def defer_manual_serve(runtime: dict, *, require_alive: bool = False) -> bool:
    """Transfer an identified manual runtime to its own durable restart reminder."""
    from hermes_cli.process_identity import _pid_alive_matches

    if runtime.get("kind") not in ("serve", "dashboard") or runtime.get("supervisor") != "manual-serve" or runtime.get("restart_via") != "respawn-argv":
        return False
    pid = runtime.get("pid")
    detail = runtime.get("detail")
    if not isinstance(detail, dict):
        return False
    created = detail.get("create_time")
    if type(pid) is not int or pid <= 0:
        return False
    identified = type(created) in (int, float) and math.isfinite(created) and created > 0
    if not identified:
        # Without a recorded creation time no durable reminder can be filed (#116507);
        # only a provably dead pid discharges the row, anything less stays pending.
        return not require_alive and _pid_alive_matches(pid, None) is False
    try:
        alive = _pid_alive_matches(pid, created)
        if require_alive and alive is not True:
            return False
        if alive is False:
            return True
        directory = get_hermes_home() / "serve_restart_pending"
        directory.mkdir(parents=True, exist_ok=True)
        row = {"kind": runtime["kind"], "profile": runtime.get("profile", "unknown"), "pid": pid, "create_time": created}
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
        return True
    except (OSError, ValueError, TypeError) as exc:
        logger.debug("Could not preserve manual serve obligation: %s", exc)
        return False


def retain_receipt_manual_serves(receipt: dict) -> list[dict]:
    """Return transfers still owed so receipt rotation cannot discard failed writes.

    ``carried_manual_serves`` is the durable running record's copy of the previous receipt's rows
    (update_receipt.begin): read it too, or the warning vanishes while an update runs and after a
    run killed before its ``plan`` stage.
    """
    plan = receipt.get("plan") or {}
    rows = list(plan.get("runtimes") or []) + list(receipt.get("pending_manual_serves") or [])
    rows += list(receipt.get("carried_manual_serves") or [])
    pending = []
    for row in rows:
        if not isinstance(row, dict) or row.get("kind") not in ("serve", "dashboard") or row.get("supervisor") != "manual-serve":
            continue
        if not defer_manual_serve(row) and row not in pending:
            pending.append(row)
    return pending


def live_manual_serve_reminders() -> tuple[list[dict], list[str]]:
    """``(rows still owed, names of files that could not be read)``.

    A reminder is discharged by the death of the process it names — the only discharge available
    without its owner's cooperation — so a provably dead pid prunes its file here. One rule, shared
    by the startup warning and ``hermes doctor``, so the two cannot disagree about what is owed.
    """
    from hermes_cli.process_identity import _pid_alive_matches

    rows: list[dict] = []
    unreadable: list[str] = []
    for path in sorted((get_hermes_home() / "serve_restart_pending").glob("*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8-sig"))
            if _pid_alive_matches(row["pid"], row["create_time"]) is False:
                path.unlink(missing_ok=True)
                continue
            rows.append(row)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.debug("Could not reconcile manual serve obligation %s: %s", path, exc)
            unreadable.append(path.name)
    return rows, unreadable


def _started_stamp(created: object) -> str:
    """``" (started 2026-10-17 05:12 UTC)"`` for a usable epoch creation time, else ``""``.

    ``bool`` is excluded deliberately: ``create_time`` is a filesystem epoch, and a boolean in its
    place would render 1970.
    """
    if not isinstance(created, (int, float)) or isinstance(created, bool) or not math.isfinite(created):
        return ""
    try:
        return f" (started {datetime.fromtimestamp(float(created), tz=UTC):%Y-%m-%d %H:%M} UTC)"
    except (OSError, OverflowError, ValueError):
        return ""


def describe_manual_serve_reminder(row: dict) -> str:
    """``serve [work] pid 900 (started 2026-10-17 05:12 UTC)`` — the identity an operator acts on."""
    described = f"{row.get('kind', 'serve')} [{row.get('profile', 'unknown')}] pid {row.get('pid', 'unknown')}"
    return described + _started_stamp(row.get("create_time"))


def warn_pending_manual_serves(*, startup: bool = False, pending_manual: list[dict] | None = None) -> None:
    """Warn about manual debt independently of gateway evidence; optionally reuse a snapshot's failed transfers."""
    from hermes_cli.update_receipt import read_latest_receipt

    stream = sys.stderr if startup else sys.stdout
    if pending_manual is None:
        pending_manual = retain_receipt_manual_serves(read_latest_receipt() or {})
    for row in pending_manual:
        print(f"  ⚠ {row['kind']} [{row.get('profile', 'unknown')}] pid {row.get('pid', 'unknown')}: manual restart reminder could not be saved; restart remains pending in the update receipt.", file=stream)
        detail = row.get("detail") if isinstance(row.get("detail"), dict) else {}
        if type(detail.get("create_time")) in (int, float):
            print("    Ask its owner to relaunch `hermes serve` / `hermes dashboard`; check reminder storage permissions and free space.", file=stream)
        else:
            # No usable creation time means identity, not storage, blocked the durable reminder.
            print("    This host could not read the process creation time, so no durable reminder could be filed; ask its owner to relaunch `hermes serve` / `hermes dashboard`, and the warning clears once the pid is confirmed gone.", file=stream)
    rows, unreadable = live_manual_serve_reminders()
    for row in rows:
        print(f"  ⚠ {describe_manual_serve_reminder(row)}: manual restart still pending; this process may still serve pre-update code.", file=stream)
        # No "(reconnect Desktop for an SSH backend)" here: a reminder is filed only for a
        # ``manual-serve`` runtime, and the ledger classifies Desktop's SSH backend
        # (``--ssh-session-token-file``) as ``desktop-ssh``, which defer_manual_serve skips. So no
        # row that reaches this line can show the reader's Desktop is remote — the wording it
        # carried survives where it is true, in report_unaccounted_runtimes' Desktop branch.
        print("    Ask its owner to relaunch `hermes serve` / `hermes dashboard`.", file=stream)
    for name in unreadable:
        print(f"  ⚠ Manual serve restart reminder could not be verified: {name}", file=stream)
