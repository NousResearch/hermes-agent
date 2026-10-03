"""Durable manual-serve handoffs, independent of gateway restart receipts."""

import json
import logging
import math
import os
import sys
import tempfile
from pathlib import Path

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)


def receipt_applied_no_code(receipt: dict) -> bool:
    """True when the receipt's update provably applied no code.

    A failed channel-resolution abort ("No update was applied") and an
    already-up-to-date run finalize with pre SHA == post SHA: the checkout
    never moved, so the receipt's plan inventory describes processes serving
    code that is still current — no restart can be owed for them. A missing
    SHA (identity probe failed) is never provable; callers keep the
    conservative behavior in that case.
    """
    if not isinstance(receipt, dict):
        return False
    pre = receipt.get("pre_update")
    post = receipt.get("post_update")
    pre_sha = pre.get("sha") if isinstance(pre, dict) else None
    post_sha = post.get("sha") if isinstance(post, dict) else None
    return bool(pre_sha) and bool(post_sha) and pre_sha == post_sha


def _manual_row_serves_current_code(row: dict) -> bool | None:
    """True when the row's live process provably runs out of this updater's own checkout.

    ``None`` when unprovable (no psutil, unreadable argv, unknown updater root): callers
    must treat unprovable as "keep the reminder".
    """
    from hermes_cli.update_receipt import _code_root_for_path, _updater_code_root

    updater_root = _updater_code_root()
    if updater_root is None:
        return None
    try:
        import psutil

        cmdline = psutil.Process(int(row["pid"])).cmdline()
    except Exception:
        return None
    for raw in cmdline or []:
        root = _code_root_for_path(raw)
        if root is not None:
            return root == updater_root
    return None


def _receipts_show_apply_since(create_time: float) -> bool | None:
    """True when some receipt finished after ``create_time`` with a moved checkout.

    False when update receipts newer than the process exist and every one of them provably
    applied no code. ``None`` when no update receipt covers the period (none finished after
    the process start, or a newer one could not be classified): absence of evidence must
    never discharge a pending restart. Non-update receipts (pm sync/plugin-check pointers)
    carry no apply evidence and are ignored.
    """
    from datetime import datetime

    from hermes_cli.update_receipt import _receipt_dir

    try:
        covered = False
        for path in _receipt_dir().glob("*.json"):
            try:
                receipt = json.loads(path.read_text(encoding="utf-8-sig"))
                finished_ts = datetime.fromisoformat(receipt["finished_at"]).timestamp()
            except Exception:
                continue
            if not isinstance(receipt, dict) or not isinstance(receipt.get("pre_update"), dict):
                continue
            if finished_ts <= float(create_time):
                continue
            covered = True
            if not receipt_applied_no_code(receipt):
                return True  # the checkout moved (or identity is unprovable) after the start
        return False if covered else None
    except Exception:
        return None


def _row_obligation_discharged(row: dict) -> bool:
    """A durable reminder is unfounded when its process provably serves the current checkout.

    A founded reminder comes from an update that MOVED the checkout after the process
    started (it may hold pre-update modules in memory). When every receipt newer than the
    process is a no-apply run and the process runs out of this updater's own checkout, no
    update it could have missed exists: discharge the stale reminder instead of warning on
    every startup until the pid dies.
    """
    created = row.get("create_time")
    if isinstance(created, bool) or not isinstance(created, (int, float)):
        return False
    if not math.isfinite(created) or created <= 0:
        return False
    if _receipts_show_apply_since(float(created)) is not False:
        return False
    return _manual_row_serves_current_code(row) is True


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
    """Return transfers still owed so receipt rotation cannot discard failed writes."""
    plan = receipt.get("plan") or {}
    # ``plan.runtimes`` is this run's own inventory, founded only when the run moved the
    # checkout; a no-apply receipt (failed channel abort, already up to date) must not found
    # restart debt for processes it never touched. ``pending_manual_serves`` are carry-overs
    # from earlier receipts whose founding update may be several rotations back — still owed.
    rows = [] if receipt_applied_no_code(receipt) else list(plan.get("runtimes") or [])
    rows += list(receipt.get("pending_manual_serves") or [])
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
        detail = row.get("detail") if isinstance(row.get("detail"), dict) else {}
        if type(detail.get("create_time")) in (int, float):
            print("    Ask its owner to relaunch `hermes serve` / `hermes dashboard`; check reminder storage permissions and free space.", file=stream)
        else:
            # No usable creation time means identity, not storage, blocked the durable reminder.
            print("    This host could not read the process creation time, so no durable reminder could be filed; ask its owner to relaunch `hermes serve` / `hermes dashboard`, and the warning clears once the pid is confirmed gone.", file=stream)
    directory = get_hermes_home() / "serve_restart_pending"
    for path in sorted(directory.glob("*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8-sig"))
            alive = _pid_alive_matches(row["pid"], row["create_time"])
            if alive is False or (alive is True and _row_obligation_discharged(row)):
                path.unlink(missing_ok=True)
                continue
            print(f"  ⚠ {row['kind']} [{row['profile']}] pid {row['pid']}: manual restart still pending; this process may still serve pre-update code.", file=stream)
            print("    Ask its owner to relaunch `hermes serve` / `hermes dashboard` (reconnect Desktop for an SSH backend).", file=stream)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.debug("Could not reconcile manual serve obligation %s: %s", path, exc)
            print(f"  ⚠ Manual serve restart reminder could not be verified: {path.name}", file=stream)
