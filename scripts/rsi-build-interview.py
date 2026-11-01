#!/usr/bin/env python3
"""Build a per-profile RSI interview prompt (corrections + audit evidence)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

STORE = Path.home() / ".hermes" / "rsi"
BASE = STORE / "interview-prompt.txt"
CORRECTIONS = STORE / "corrections.yaml"
AUDIT = STORE / "audit" / "latest.json"
OPEN = {"unverified", "regressed", "ineffective"}


def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(path.read_text()) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: rsi-build-interview.py PROFILE", file=sys.stderr)
        sys.exit(2)
    profile = sys.argv[1].strip()
    base = BASE.read_text() if BASE.exists() else ""
    items = (load_yaml(CORRECTIONS).get("corrections") or [])
    open_ones = [
        c for c in items
        if isinstance(c, dict)
        and str(c.get("agent") or "") == profile
        and str(c.get("status") or "unverified") in OPEN
    ]
    lines = [base.rstrip(), "", f"Your profile name is `{profile}`."]
    if open_ones:
        lines.append("OPEN CORRECTIONS to verify (do not invent ids):")
        for c in open_ones:
            lines.append(
                f"- id={c.get('id')} status={c.get('status')} "
                f"problem={c.get('problem')} fix={c.get('fix')}"
            )
    else:
        lines.append("OPEN CORRECTIONS: none.")

    audit = {}
    if AUDIT.exists():
        try:
            audit = (json.loads(AUDIT.read_text()).get("profiles") or {}).get(profile) or {}
        except Exception:
            audit = {}
    sessions = audit.get("sessions") or []
    sess_fail = audit.get("session_failures") or []
    cron_f = audit.get("cron_failures") or []
    kan_f = audit.get("kanban_failures") or []
    lines.append("")
    lines.append("EVIDENCE RSI already read (since last tick). You must account for every failed item. Omitting it is a reporting failure.")
    lines.append(f"chats_seen={len(sessions)} session_failures={len(sess_fail)} cron_failures={len(cron_f)} kanban_failures={len(kan_f)}")
    for s in sess_fail:
        lines.append(
            f"- session id={s.get('id')} source={s.get('source')} "
            f"end={s.get('end_reason')} title={s.get('title')} hits={s.get('fail_hits')}"
        )
    for c in cron_f:
        lines.append(
            f"- cron execution_id={c.get('execution_id')} name={c.get('name')} "
            f"status={c.get('status')} error={c.get('error')}"
        )
    for k in kan_f:
        lines.append(
            f"- kanban task_id={k.get('task_id')} outcome={k.get('outcome')} error={k.get('error')}"
        )
    if not (sess_fail or cron_f or kan_f):
        lines.append("- no failed runs in the audit window. Still report any you remember.")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
