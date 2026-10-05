"""One-shot queue/steer wrappers over the live owner's RPC primitives."""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from contextlib import closing

from hermes_cli.active_sessions import active_session_registry_snapshot
from hermes_cli.shared_session_attach import discover_attach_url
from hermes_state import SessionDB, _default_db_path


def resolve_live_session(target: str, owners: list[dict]) -> str:
    # Live sessions can still be lazy (no history row yet). Registry IDs are
    # durable IDs; prompt.submit instead addresses metadata.live_session_id.
    if any(owner["session_id"] == target for owner in owners):
        return target
    if _default_db_path().exists():
        with closing(SessionDB(read_only=True)) as db:
            if db.get_session(target) is not None:
                return target
            matches = db._read_all("SELECT id FROM sessions WHERE title = ?", (target,))
    else:
        matches = []
    live_owners = [owner for owner in owners
                   if (owner.get("metadata") or {}).get("live_session_id") == target]
    if len(live_owners) > 1:
        ids = ", ".join(owner["session_id"] for owner in live_owners)
        raise ValueError(f"Ambiguous live/UI session ID; retry with a durable session ID: {ids}")
    if live_owners:
        return live_owners[0]["session_id"]
    if not matches:
        raise ValueError("No session matches the exact durable ID, live/UI ID or name.")
    if len(matches) != 1:
        ids = ", ".join(row["id"] for row in matches)
        raise ValueError(f"Ambiguous session name; retry with a session ID: {ids}")
    return matches[0]["id"]


def submit_live_prompt(url: str, session_id: str, text: str, action: str) -> dict:
    from websockets.sync.client import connect

    request_id = "ingress-1"
    with connect(url, proxy=None, open_timeout=3, close_timeout=1) as socket:
        socket.send(json.dumps({"jsonrpc": "2.0", "id": request_id, "method": "prompt.submit" if action == "queue" else "session.steer",
                                "params": {"session_id": session_id, "text": text,
                                           **({"queued": True} if action == "queue" else {})}}))
        deadline = time.monotonic() + 10
        while True:
            reply = json.loads(socket.recv(timeout=max(0, deadline - time.monotonic())))
            if reply.get("id") == request_id:
                # Match /steer's next-turn behavior for an explicitly rejected correction.
                # Only explicit rejection permits fallback; transport errors never retry.
                if action == "steer" and reply.get("result", {}).get("status") == "rejected":
                    action = "queue"
                    request_id = "ingress-next-turn"
                    socket.send(json.dumps({"jsonrpc": "2.0", "id": request_id,
                                            "method": "prompt.submit", "params": {
                                                "session_id": session_id, "text": text, "queued": True}}))
                    continue
                return reply


def send_live_message(target: str, text: str, action: str) -> int:
    try:
        owners = active_session_registry_snapshot(strict=True)
        session_id = resolve_live_session(target, owners)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except (OSError, RuntimeError, sqlite3.Error):
        print("Error: Session lookup failed in the active profile.", file=sys.stderr)
        return 1
    owners = [owner for owner in owners if owner["session_id"] == session_id]
    if not owners:
        print(f"Error: Session {session_id} exists but is not live. Resume it first with:\n"
              f"  hermes --resume {session_id}", file=sys.stderr)
        return 1
    if len(owners) != 1:
        print("Error: Session owner identity is ambiguous; no submission was attempted.", file=sys.stderr)
        return 1
    live_id = (owners[0].get("metadata") or {}).get("live_session_id")
    if not live_id:
        print("Error: This live session does not support CLI submission.", file=sys.stderr)
        return 1
    try:
        url = discover_attach_url(session_id)
    except (ValueError, OSError, RuntimeError):
        # Discovery and transport errors may contain credentials. Never render
        # their text or a traceback, including server-supplied RPC error bodies.
        print("Error: Live session attachment failed; use the window where it is open.", file=sys.stderr)
        return 1
    if url is None:
        print("Error: Session is no longer live; no submission was attempted.", file=sys.stderr)
        return 1
    try:
        reply = submit_live_prompt(url, live_id, text, action)
    except Exception:
        print("Error: Submission was not confirmed. Check the owning session before retrying.", file=sys.stderr)
        return 1
    if "error" in reply:
        if action == "steer" and reply["error"]["code"] == 4010:
            print("Error: Steering is unavailable for this session (RPC 4010). No prompt was queued. "
                  "Use hermes queue to submit a next turn.", file=sys.stderr)
            return 1
        print(f"Error: Live session rejected the prompt (RPC code {reply['error']['code']}).", file=sys.stderr)
        return 1
    status = reply["result"].get("status")
    if status not in ("queued", "streaming"):
        print("Error: Live session did not accept the input.", file=sys.stderr)
        return 1
    print(f"Accepted {action} for session {session_id} ({status}).")
    return 0
