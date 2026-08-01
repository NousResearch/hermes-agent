#!/usr/bin/env python3
"""KenseiAgent adapter for the mashup proposal approval handler (thin).

All gate logic, pitch-queue handling, typed results and atomic state live in
the `research-mashup-pipeline` package core (`mashup.proposal_approval_handler`,
`mashup.core`). This adapter supplies ONLY the Hermes-specific surface:

- Discord bot ID + channel ID (env-driven, defaults to Kensei runtime);
- Discord API polling via the bot token from $HERMES_HOME/.env;
- the `hermes kanban create` subprocess (host command).

The same conformance suite runs against the package core and this adapter.
"""
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Package core is the single source of truth for gates + state.
_PKG = os.environ.get("MASHUP_PKG", "/home/kensei/research-mashup-pipeline/src")
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from mashup import core  # noqa: E402
from mashup.proposal_approval_handler import (  # noqa: E402
    DiscordApprovalHandler,
    create_kanban_triage,
    enqueue_pitch,
    find_idea_card,
)

TZ = timezone(timedelta(hours=int(os.environ.get("PROPOSAL_TZ_OFFSET", "1"))))
KENSEI_ID = os.environ.get("KENSEI_BOT_ID", "")
CHANNEL_ID = os.environ.get("PROPOSAL_CHANNEL_ID", "")

# Keep parse_command available for the conformance suite + live cron.
def parse_command(content: str):
    """Parse !approve/!reject/!pitch from a Discord message."""
    approve = re.search(r"!approve\s+([a-zA-Z0-9][-a-zA-Z0-9._]+)", content)
    if approve:
        return ("approve", approve.group(1).strip().rstrip("."))
    reject = re.search(r"!reject\s+([a-zA-Z0-9][-a-zA-Z0-9._]+)", content)
    if reject:
        return ("reject", reject.group(1).strip().rstrip("."))
    pitch = re.search(r"!pitch\s+([a-zA-Z0-9][-a-zA-Z0-9._]+)", content)
    if pitch:
        return ("pitch", pitch.group(1).strip().rstrip("."))
    return (None, None)


def get_token() -> str:
    dotenv = core.HERMES_HOME / ".env"
    if dotenv.exists():
        for line in dotenv.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("DISCORD_BOT_TOKEN="):
                key = line[len("DISCORD_BOT_TOKEN="):].strip().strip(chr(34)).strip(chr(39))
                return key
    return os.environ.get("DISCORD_BOT_TOKEN", "")


def discord_api(endpoint: str, data=None):
    token = get_token()
    if not token:
        return {"_err": "no token"}
    import urllib.error
    import urllib.request
    url = f"https://discord.com/api/v10{endpoint}"
    hdrs = {"Authorization": f"Bot {token}", "Content-Type": "application/json"}
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(url, data=body, headers=hdrs, method="POST" if data else "GET")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code == 429:
            return {"_skip": f"rate: {e.headers.get('Retry-After', 5)}"}
        return {"_err": f"HTTP {e.code}"}
    except Exception as e:
        return {"_err": str(type(e).__name__)}


def main():
    """Live cron entrypoint: poll Discord, enforce gates via core."""
    import argparse
    parser = argparse.ArgumentParser(description="Kensei proposal approval handler (adapter)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print("[dry-run] would poll #research-ops and process !approve/!reject/!pitch")
        return 0

    h = datetime.now(TZ).hour
    if h < 7 or h >= 22:
        return 0

    st = core.read_json_durable(core.STATE_FILE, {"approved": [], "rejected": [], "last_msg_id": None})
    approved = set(st.get("approved", []))
    rejected = set(st.get("rejected", []))
    last_msg_id = st.get("last_msg_id")

    msgs = discord_api(f"/channels/{CHANNEL_ID}/messages?limit=30")
    if not msgs or not isinstance(msgs, list):
        return 0

    new_last = last_msg_id
    actions = []

    for msg in msgs:
        mid = msg.get("id")
        if last_msg_id and int(mid) <= int(last_msg_id):
            continue
        ref = msg.get("message_reference")
        if not ref:
            continue
        ref_mid = ref.get("message_id")
        if msg.get("author", {}).get("bot", False):
            continue
        content = msg.get("content", "")
        action, slug = parse_command(content)
        if not action or not slug:
            continue
        ref_msg = discord_api(f"/channels/{CHANNEL_ID}/messages/{ref_mid}")
        if not ref_msg or ref_msg.get("_err"):
            continue
        if ref_msg.get("author", {}).get("id") != KENSEI_ID:
            continue
        slug = slug.lower()
        if slug in approved or slug in rejected:
            continue

        if action == "approve":
            # Gate 3 enforced in core; typed result, never false success.
            res = create_kanban_triage(slug, "", ref_msg.get("content", ""))
            if res.ok:
                approved.add(slug)
                actions.append(f"approved: {slug} -> kanban {res.task_id}")
            else:
                actions.append(f"FAILED approve: {slug} — {res.error}")
                print(f"approval-failure: {slug} — {res.error}")
        elif action == "reject":
            rejected.add(slug)
            actions.append(f"rejected: {slug}")
        elif action == "pitch":
            res = enqueue_pitch(slug)
            if res.ok:
                actions.append(f"pitched: {slug} (pitch requested; worker paused by default)")
            else:
                actions.append(f"FAILED pitch: {slug} — {res.error}")

        if new_last is None or int(mid) > int(new_last):
            new_last = mid

    if actions:
        st["approved"] = list(approved)
        st["rejected"] = list(rejected)
        st["last_msg_id"] = new_last
        core.atomic_write(core.STATE_FILE, json.dumps(st, indent=2))
        for a in actions:
            print(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
