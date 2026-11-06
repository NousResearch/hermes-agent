"""`hermes update --status`: read-only update health report.

The stuck-update doctor. Prints install shape, update-marker verdict (with the
owner's age, not just its pid), armed source-completion tail, and the last
receipt's outcome — then names the exact recovery command for whatever it
finds. Strictly read-only: it never claims the marker or checkout lock (a
diagnostic that mutates the state it inspects cannot be trusted), never
reconciles receipts, and works without a git checkout so bootstrap/workspace
installs can use it too.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any


def _m():
    from hermes_cli import main

    return main


def _format_age(seconds: float) -> str:
    if seconds != seconds or seconds == float("inf"):  # NaN / missing started_at
        return "unknown age"
    if seconds < 90:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m"
    if seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


def _read_marker_state(home: Path) -> dict[str, Any]:
    """Verdict on the shared update marker without touching it."""
    from hermes_cli.update_lock import _parse_marker, _read_bytes, judge_marker, update_marker_path

    try:
        path = update_marker_path()
    except Exception:
        path = home / ".hermes-update-in-progress"
    raw = _read_bytes(path)
    if raw is None:
        return {"present": False, "path": str(path)}
    if not raw:
        return {"present": True, "empty": True, "path": str(path)}
    verdict, owner, run = judge_marker(raw)
    marker = _parse_marker(raw)
    age = time.time() - marker.started_at if marker and marker.started_at else float("inf")
    # Quarantine preview: would the watchdog archive this on the next run?
    # (dry_run — the report never mutates.) install_root is unknown here, so a
    # held-checkout false negative just omits the note; never the reverse.
    try:
        from hermes_cli.update_lock import quarantine_stale_marker

        quarantine_preview = quarantine_stale_marker(dry_run=True)
    except Exception:
        quarantine_preview = None
    return {
        "present": True,
        "path": str(path),
        "verdict": verdict,
        "owner": owner,
        "run": run,
        "age_seconds": age,
        "pid": marker.pid if marker else None,
        "quarantine_eligible": quarantine_preview is not None,
    }


def _read_tail_state(project_root: Path) -> dict[str, Any]:
    """Whether a source-completion tail is still armed for this checkout."""
    try:
        from hermes_cli.venv_sync import completion_pending_path
    except Exception:
        return {"armed": None}
    try:
        pending = completion_pending_path(project_root)
    except Exception:
        return {"armed": None}
    try:
        armed = pending.is_file()
    except OSError:
        return {"armed": None, "path": str(pending)}
    return {"armed": armed, "path": str(pending)}


def _read_last_receipt() -> dict[str, Any] | None:
    """Newest update receipt, or None. Never raises, never reconciles."""
    try:
        from hermes_cli.update_receipt import read_latest_receipt
    except Exception:
        return None
    try:
        record = read_latest_receipt()
    except Exception:
        return None
    return record if isinstance(record, dict) else None


def _receipt_health(record: dict[str, Any]) -> tuple[str, str]:
    """(state, detail) for the report: ok | running | interrupted-like | failed | owed."""
    outcome = record.get("outcome")
    if outcome == "running":
        try:
            from hermes_cli.update_receipt import _owner_alive
        except Exception:
            _owner_alive = None  # type: ignore[assignment]
        alive = _owner_alive(record) if _owner_alive else None
        stages = record.get("stages") or []
        last = stages[-1].get("name") if stages and isinstance(stages[-1], dict) else "start"
        if alive is False:
            return "attention", f"marked running but its process is gone (last stage: {last}) — likely interrupted"
        return "running", f"in flight since {record.get('started_at', '?')} (last stage: {last})"
    if outcome == "interrupted":
        return "attention", f"interrupted: {record.get('stop_reason', 'process exited mid-update')}"
    if outcome in ("committed", "committed_success", "success"):
        followups = record.get("followups") or []
        if followups:
            return "owed", f"succeeded with {len(followups)} follow-up(s) still owed"
        return "ok", "last update completed"
    if outcome in ("failed", "refused"):
        return "attention", f"last update {outcome}: {record.get('stop_reason') or record.get('refusal', 'see receipt')}"
    return "unknown", f"unrecognized outcome {outcome!r}"


def _quarantined_markers(home: Path) -> list[str]:
    try:
        return sorted(p.name for p in home.glob(".hermes-update-in-progress.quarantined-*"))
    except OSError:
        return []


def _cmd_update_status() -> int:
    """Print the update health report. 0 = idle, 3 = attention needed."""
    from hermes_cli.update_inventory import UpdatePlan, _collect_install_shape

    attention = False
    try:
        root_home = _m().PROJECT_ROOT
    except Exception:
        root_home = Path.home() / ".hermes"

    print("☤ Hermes update status")
    print()

    # Install shape: kind, mechanism, checkout HEAD. Never touches the network.
    plan = UpdatePlan()
    try:
        _collect_install_shape(plan)
    except Exception as exc:  # shape probing must not fail the report
        print(f"  install:  could not determine ({exc})")
    else:
        print(f"  install:  {plan.install_method} (updatable in place: {plan.updatable_in_place})")
        print(f"  via:      {plan.update_mechanism}")
        if plan.expected_sha:
            print(f"  checkout: {plan.expected_sha[:12]}")
    print()

    # Marker verdict.
    marker = _read_marker_state(root_home)
    if not marker.get("present"):
        print("  marker:   none — no update in flight")
    elif marker.get("empty"):
        print("  marker:   present but empty (a claim in flight or a crashed writer)")
        attention = True
    else:
        verdict = marker.get("verdict")
        age = _format_age(marker.get("age_seconds", float("inf")))
        if verdict == "live":
            print(f"  marker:   LIVE — owned by pid {marker.get('owner')} ({age} old)")
            if marker.get("run"):
                print(f"  run:      {marker.get('run')}")
        elif verdict == "ours":
            print("  marker:   owned by this process")
        elif verdict == "dead":
            print(f"  marker:   STALE — owner pid {marker.get('pid')} is dead ({age} old)")
            if marker.get("quarantine_eligible"):
                print("  fix:      the next `hermes update` quarantines it automatically and continues;")
                print("            or clear it now only if no updater runs anywhere.")
            else:
                print("  fix:      `hermes update` proceeds once it is eligible (dead + 2x ceiling + checkout free),")
                print("            or clear it only if no updater runs anywhere.")
            attention = True
        else:
            print(f"  marker:   {verdict} ({marker.get('path')})")
            attention = True
    quarantined = _quarantined_markers(root_home)
    for name in quarantined:
        print(f"  archived: {name} (a previous stale marker, kept as evidence)")
    print()

    # Armed tail.
    try:
        project_root = _m().PROJECT_ROOT
    except Exception:
        project_root = root_home
    tail = _read_tail_state(Path(project_root))
    if tail.get("armed") is True:
        print(f"  tail:     ARMED — a source-completion tail still owes work ({tail.get('path')})")
        print("  fix:      finish it with `hermes update`, or clear it only when no updater runs")
        attention = True
    elif tail.get("armed") is False:
        print("  tail:     none armed")
    try:
        from hermes_cli.update_stage import TAIL_STAGE_STALE_SECONDS, read_tail_stage

        stage = read_tail_stage()
    except Exception:
        stage = None
    if stage and stage.get("message"):
        raw_age = stage.get("age_seconds")
        age = _format_age(raw_age if isinstance(raw_age, (int, float)) else float("inf"))
        stale = isinstance(raw_age, (int, float)) and raw_age > TAIL_STAGE_STALE_SECONDS
        state = "STALE" if (stale and stage.get("status") == "running") else str(stage.get("status", "")).upper()
        print(f"  progress: [{state}] {stage.get('message')} ({age} ago)")
        if state == "STALE":
            print("  fix:      no tail progress for a while — the run may be wedged; check the updater's output,")
            print("            then run `hermes update` to let the watchdog clear it if its owner is dead")
            attention = True
    print()

    # Last receipt.
    record = _read_last_receipt()
    if record is None:
        print("  receipt:  none yet — no update has run on this home")
    else:
        state, detail = _receipt_health(record)
        sha = ((record.get("pre_update") or {}).get("sha") or "")[:12]
        print(f"  receipt:  {record.get('started_at', '?')} → {record.get('outcome', '?')}" + (f" (from {sha})" if sha else ""))
        print(f"  health:   {detail}")
        if state in ("attention", "owed", "running"):
            attention = state == "attention"
            if state == "running":
                print("  note:     a live update is in progress; wait for it before starting another")
    print()

    if attention:
        print("  → attention needed (exit 3); see the fix lines above.")
        return 3
    print("  → idle and healthy.")
    return 0
