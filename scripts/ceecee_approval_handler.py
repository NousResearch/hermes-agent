#!/usr/bin/env python3
"""Discord approval handler for CeeCee content drafts.

Polls #content for !approve/!reject/!amend commands and updates the
content_engine SQLite database. Silent when no commands found.
"""
import json
import os
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

CHANNEL_ID = "1507448580649123900"

# P13 isolation: paths derive from HERMES_HOME (env-overridable) so local
# disposable runs never touch /home/kensei/.hermes or the production
# content_engine DB. REPO_ROOT lets the DB default resolve to the active
# checkout rather than a hardcoded absolute path.
_HERMES_HOME = Path(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")))
_REPO_ROOT = Path(os.environ.get("KENSEI_REPO_ROOT", "/home/kensei/repos/KenseiAgent"))
DB_PATH = Path(os.environ.get("CEECEE_DB_PATH", str(_REPO_ROOT / "content_engine" / "db" / "content_engine.db")))
STATE_PATH = _HERMES_HOME / "state" / "ceecee-approval-state.json"
DOTENV_PATH = _HERMES_HOME / ".env"

# P13 isolation: when --dry-run is passed, every write path (Discord API
# POST/PUT, SQLite UPDATE/commit, state file save, reaction add) is
# suppressed. Read paths (load_state, parse_command, Discord GET for
# message polling) run unchanged so the classification/decision logic is
# still exercised.
_DRY_RUN = False


def get_token() -> str:
    """Read DISCORD_BOT_TOKEN from .env."""
    if DOTENV_PATH.exists():
        for line in DOTENV_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("DISCORD_BOT_TOKEN="):
                return line[len("DISCORD_BOT_TOKEN="):].strip().strip('"').strip("'")
    return os.environ.get("DISCORD_BOT_TOKEN", "")


def discord_api(endpoint: str, data: dict | None = None) -> dict | list | None:
    """Call the Discord REST API. Returns parsed JSON or None on error."""
    if _DRY_RUN and data is not None:
        return None  # dry-run: suppress write (POST) calls; reads (GET) still run
    token = get_token()
    if not token:
        return None
    url = f"https://discord.com/api/v10{endpoint}"
    headers = {"Authorization": f"Bot {token}", "Content-Type": "application/json"}
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(
        url, data=body, headers=headers, method="POST" if data else "GET"
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code == 429:
            return None
        return None
    except Exception:
        return None


def load_state() -> dict:
    """Load processed message IDs from state file."""
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, Exception):
            pass
    return {"processed": [], "last": 0}


def save_state(st: dict) -> None:
    """Persist processed message IDs to state file."""
    if _DRY_RUN:
        return
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(st), encoding="utf-8")


def parse_command(text: str) -> dict | None:
    """Parse a Discord command string.

    Returns dict with keys: command, draft_id, args or None.
    """
    m = re.match(
        r"^!(approve|reject|amend)\s+(\S+)\s*(.*)",
        text.strip(),
        re.IGNORECASE,
    )
    if not m:
        return None
    return {
        "command": m.group(1).lower(),
        "draft_id": m.group(2),
        "args": m.group(3).strip(),
    }


def update_draft_status(draft_id: str, command: str, args: str) -> str | None:
    """Update the draft status in the content_engine database.

    Returns a human-readable result message or None if draft not found.
    """
    if not DB_PATH.exists():
        return "Database not found"

    conn = sqlite3.connect(str(DB_PATH))
    try:
        row = conn.execute(
            "SELECT id, status FROM drafts WHERE id = ?", (draft_id,)
        ).fetchone()
        if not row:
            return None

        # P13 dry-run: compute the result message but skip the UPDATE + commit.
        if _DRY_RUN:
            now = datetime.now(timezone.utc).isoformat()
            if command == "approve":
                return f"Approved {draft_id} (dry-run)"
            elif command == "reject":
                return f"Rejected {draft_id} (dry-run)" + (f" — {args}" if args else "")
            elif command == "amend":
                return f"Amended {draft_id} (dry-run)" + (f" — {args}" if args else "")
            return f"Unknown command: {command}"

        now = datetime.now(timezone.utc).isoformat()

        # G03 audit path: record every approval decision through the
        # approval_state ledger (append-only log + transition guard). If the
        # transition is illegal (e.g. approving an already-published draft),
        # fail loudly instead of silently overwriting state.
        try:
            sys.path.insert(0, str(_REPO_ROOT / "content_engine"))
            from approval_state import ApprovalError, ApprovalLedger

            ledger = ApprovalLedger(db_path=str(DB_PATH))
            ledger.decide(draft_id, command, actor="sahil", comment=args)
        except ApprovalError as exc:
            return f"Blocked: {exc}"

        if command == "approve":
            conn.execute(
                "UPDATE drafts SET status = 'approved', approved_at = ? WHERE id = ?",
                (now, draft_id),
            )
            msg = f"Approved {draft_id}"

        elif command == "reject":
            conn.execute(
                "UPDATE drafts SET status = 'rejected', rejected_at = ? WHERE id = ?",
                (now, draft_id),
            )
            msg = f"Rejected {draft_id}"
            if args:
                msg += f" — {args}"

        elif command == "amend":
            conn.execute(
                "UPDATE drafts SET status = 'amended' WHERE id = ?",
                (draft_id,),
            )
            msg = f"Amended {draft_id}"
            if args:
                msg += f" — {args}"

        else:
            return f"Unknown command: {command}"

        conn.commit()
        return msg
    finally:
        conn.close()


def add_reaction(message_id: str, emoji: str = "✅") -> bool:
    """Add a reaction emoji to a Discord message (PUT endpoint)."""
    if _DRY_RUN:
        return False
    token = get_token()
    if not token:
        return False
    url = (
        f"https://discord.com/api/v10/channels/{CHANNEL_ID}"
        f"/messages/{message_id}/reactions/{emoji}/@me"
    )
    headers = {"Authorization": f"Bot {token}"}
    req = urllib.request.Request(url, data=b"", headers=headers, method="PUT")
    try:
        with urllib.request.urlopen(req, timeout=10):
            return True
    except urllib.error.HTTPError:
        return False
    except Exception:
        return False


def main() -> None:
    global _DRY_RUN
    if "--dry-run" in sys.argv:
        _DRY_RUN = True
    st = load_state()
    processed = set(st.get("processed", []))

    messages = discord_api(
        f"/channels/{CHANNEL_ID}/messages?limit=20"
    )
    if not messages or not isinstance(messages, list):
        return

    for msg in messages:
        mid = msg.get("id")
        if mid in processed:
            continue

        content = msg.get("content", "")
        cmd = parse_command(content)
        if not cmd:
            continue

        draft_id = cmd["draft_id"]
        result = update_draft_status(draft_id, cmd["command"], cmd["args"])

        if result is None:
            add_reaction(mid, "❓")
        else:
            add_reaction(mid, "✅")
            print(result)

        processed.add(mid)

    st["processed"] = list(processed)
    st["last"] = int(time.time())
    save_state(st)


if __name__ == "__main__":
    main()
