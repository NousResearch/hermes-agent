"""``hermes heartbeat`` — shell surface over :mod:`hermes_cli.heartbeat`.

Persistence is the existing ``heartbeat:<session_id>`` ``state_meta`` row (``save_heartbeat`` /
``HeartbeatManager``); this module adds no second store. Profile scope comes from ``-p/--profile``
(``HERMES_HOME``), the same way ``hermes cron`` and ``hermes sessions`` resolve it.

A running gateway needs no restart: ``gateway/run_heartbeat_restore.restore_heartbeat_watches`` is
called at startup AND on every poller tick (``gateway/run_goals.py::_start_heartbeat_poller``), so an
active row written here for a gateway-routed session is registered on the next tick.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_cli.heartbeat import (
    MIN_INTERVAL_SECONDS,
    POLL_SECONDS,
    HeartbeatManager,
    HeartbeatState,
    _META_PREFIX,
    format_interval,
    parse_interval,
)


def _db():
    from hermes_cli.goals import _get_session_db

    db = _get_session_db()
    if db is None:
        print("Error: could not open the profile's state.db.", file=sys.stderr)
        sys.exit(1)
    return db


def _session_row(db, session_id: str) -> Optional[Dict[str, Any]]:
    try:
        return db.get_session(session_id)
    except Exception:
        return None


def _route(row: Optional[Dict[str, Any]]) -> str:
    """``telegram dm chat=123 thread=7`` from a sessions row; ``(no session row)`` when missing."""
    if not row:
        return "(no session row)"
    parts = [str(row.get("source") or "?")]
    if row.get("chat_type"):
        parts.append(str(row["chat_type"]))
    if row.get("chat_id"):
        parts.append(f"chat={row['chat_id']}")
    if row.get("thread_id"):
        parts.append(f"thread={row['thread_id']}")
    if row.get("user_id"):
        parts.append(f"user={row['user_id']}")
    return " ".join(parts)


def _gateway_routed(row: Optional[Dict[str, Any]]) -> bool:
    return bool(row and row.get("session_key"))


def _ago(ts: float) -> str:
    if not ts:
        return "never"
    delta = max(0, int(time.time() - ts))
    for unit, suffix in ((86400, "d"), (3600, "h"), (60, "m")):
        if delta >= unit:
            return f"{delta // unit}{suffix} ago"
    return f"{delta}s ago"


def _entry(session_id: str, state: HeartbeatState, row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "session_id": session_id,
        "route": _route(row),
        "platform": (row or {}).get("source"),
        "chat_id": (row or {}).get("chat_id"),
        "thread_id": (row or {}).get("thread_id"),
        "gateway_routed": _gateway_routed(row),
        "interval": format_interval(state.interval_seconds),
        "interval_seconds": state.interval_seconds,
        "status": state.status,
        "fire_count": state.fire_count,
        "last_fired_at": state.last_fired_at or None,
        "prompt": state.prompt,
    }


def _pickup_note(row: Optional[Dict[str, Any]], state: HeartbeatState) -> str:
    if state.status != "active":
        return "Status is not active; the gateway will not fire it until resumed."
    if _gateway_routed(row):
        return (f"A running gateway for this profile registers the watch on its next poll (~{int(POLL_SECONDS)}s); "
                "no restart needed. Not running? `hermes gateway start` restores it at startup.")
    return ("This session has no gateway routing key (CLI/TUI session): it fires only while that session is "
            "open in a CLI process — the gateway ignores it.")


def _print_state(session_id: str, state: HeartbeatState, row: Optional[Dict[str, Any]]) -> None:
    e = _entry(session_id, state, row)
    print(f"Session:   {session_id}")
    print(f"Route:     {e['route']}")
    print(f"Status:    {e['status']}")
    print(f"Interval:  {e['interval']}")
    print(f"Fired:     {e['fire_count']}× (last {_ago(state.last_fired_at)})")
    if state.status == "active":
        next_in = max(0, int((state.last_fired_at or state.created_at) + state.interval_seconds - time.time()))
        print(f"Next due:  ~{format_interval(next_in) if next_in >= 60 else f'{next_in}s'}")
    print(f"Prompt:    {state.prompt}")
    print(f"Pickup:    {_pickup_note(row, state)}")


# ---- subcommands -----------------------------------------------------------------------------

def _cmd_list(args) -> int:
    db = _db()
    entries: List[Dict[str, Any]] = []
    for key, raw in db.list_meta_prefix(_META_PREFIX):
        session_id = key[len(_META_PREFIX):]
        try:
            state = HeartbeatState.from_json(raw)
        except Exception:
            entries.append({"session_id": session_id, "status": "unparseable", "route": _route(_session_row(db, session_id)),
                            "interval": "?", "fire_count": 0, "last_fired_at": None, "prompt": raw[:60]})
            continue
        if state.status == "cleared" and not getattr(args, "all", False):
            continue
        entries.append(_entry(session_id, state, _session_row(db, session_id)))
    entries.sort(key=lambda e: (e["status"] != "active", e["session_id"]))
    if getattr(args, "json", False):
        print(json.dumps(entries, indent=2, ensure_ascii=False))
        return 0
    if not entries:
        print("No heartbeats in this profile." + ("" if getattr(args, "all", False) else " (use --all to include cleared)"))
        return 0
    header = f"{'Session':<28} {'Status':<8} {'Every':<7} {'Fired':<6} {'Last':<10} {'Route':<34} Prompt"
    print(header)
    print("─" * 120)
    for e in entries:
        prompt = e["prompt"].replace("\n", " ")
        print(f"{e['session_id']:<28} {e['status']:<8} {e['interval']:<7} {e['fire_count']:<6} "
              f"{_ago(e['last_fired_at'] or 0):<10} {e['route'][:34]:<34} {prompt[:40]}")
    return 0


def _cmd_set(args) -> int:
    db = _db()
    row = _session_row(db, args.session_id)
    if row is None:
        print(f"Error: no session '{args.session_id}' in this profile. "
              "Find ids with `hermes sessions list --source telegram` (add -p <profile> for a named profile).",
              file=sys.stderr)
        return 1
    seconds = parse_interval(args.every)
    if seconds is None:
        print(f"Error: '{args.every}' is not an interval (try 30m, 2h, 'every 90 minutes').", file=sys.stderr)
        return 1
    if seconds == -1:
        print(f"Error: interval must be at least {MIN_INTERVAL_SECONDS}s.", file=sys.stderr)
        return 1
    if args.prompt_file:
        try:
            prompt = Path(args.prompt_file).expanduser().read_text(encoding="utf-8-sig")
        except OSError as exc:
            print(f"Error: cannot read --prompt-file: {exc}", file=sys.stderr)
            return 1
    else:
        prompt = args.prompt
    mgr = HeartbeatManager(args.session_id)
    try:
        state = mgr.set(prompt, seconds)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"♥ Heartbeat set (every {format_interval(state.interval_seconds)}).")
    _print_state(args.session_id, state, row)
    return 0


def _require(session_id: str):
    db = _db()
    mgr = HeartbeatManager(session_id)
    if not mgr.has_heartbeat():
        print(f"No heartbeat on session {session_id}.", file=sys.stderr)
        return None, None, db
    return mgr, _session_row(db, session_id), db


def _cmd_status(args) -> int:
    mgr, row, _db_ = _require(args.session_id)
    if mgr is None:
        return 1
    if getattr(args, "json", False):
        print(json.dumps(_entry(args.session_id, mgr.state, row), indent=2, ensure_ascii=False))
    else:
        _print_state(args.session_id, mgr.state, row)
    return 0


def _cmd_pause(args) -> int:
    mgr, row, _db_ = _require(args.session_id)
    if mgr is None:
        return 1
    mgr.pause()
    print("⏸ Heartbeat paused.")
    _print_state(args.session_id, mgr.state, row)
    return 0


def _cmd_resume(args) -> int:
    mgr, row, _db_ = _require(args.session_id)
    if mgr is None:
        return 1
    mgr.resume()
    print("♥ Heartbeat resumed.")
    _print_state(args.session_id, mgr.state, row)
    return 0


def _cmd_clear(args) -> int:
    mgr, _row, _db_ = _require(args.session_id)
    if mgr is None:
        return 1
    mgr.clear()
    print(f"Heartbeat cleared on {args.session_id}. A running gateway drops the watch on its next poll.")
    return 0


_HEARTBEAT_SUBCOMMANDS = {
    "list": _cmd_list, "set": _cmd_set, "status": _cmd_status,
    "pause": _cmd_pause, "resume": _cmd_resume, "clear": _cmd_clear,
}


def heartbeat_command(args) -> int:
    """Handle ``hermes heartbeat`` subcommands (returns the exit code)."""
    subcmd = getattr(args, "heartbeat_command", None) or "list"
    handler = _HEARTBEAT_SUBCOMMANDS.get(subcmd)
    if handler is None:
        print(f"Unknown heartbeat command: {subcmd}\nUsage: hermes heartbeat [list|set|status|pause|resume|clear]",
              file=sys.stderr)
        return 1
    return handler(args)
