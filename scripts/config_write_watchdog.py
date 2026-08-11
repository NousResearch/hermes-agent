#!/usr/bin/env python3
"""R4 — Config-write audit watchdog.

Detects writes to ~/.hermes/config.yaml and ~/.hermes/profiles/*/config.yaml
by scanning file mtimes against a persisted snapshot, logs to the
profile-activity-ledger, and prints a notification for cron delivery.

Triggered by the 2026-08-07 incident: a bot rewrote 61 live profiles + 51
repo agent configs with no audit trail. This makes config writes visible.

Designed for no_agent cron (stdout = notification, silent when nothing
changed). Also logs every change into the ledger DB for forensic queries.

Usage:
    python3 config_write_watchdog.py            # check + log + notify
    python3 config_write_watchdog.py --snapshot # (re)build baseline only
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import sqlite3
import sys
from pathlib import Path

HERMES_HOME = Path(os.environ.get("HERMES_HOME", "/home/kensei/.hermes")).resolve()
STATE_FILE = HERMES_HOME / "governance" / "config-write-watchdog-state.json"
LEDGER_DB = HERMES_HOME / "governance" / "profile-activity-ledger.sqlite"

# What we watch: root config + every profile config (mode 600, no secrets in
# output — we only record path, hash, mtime, and a one-line change summary).
WATCH_GLOBS = [
    (HERMES_HOME / "config.yaml", "root-config"),
    (HERMES_HOME / "profiles" / "*" / "config.yaml", "profile-config"),
]


def _file_fingerprint(path: Path) -> dict:
    try:
        st = path.stat()
        h = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        return {
            "mtime": int(st.st_mtime),
            "size": st.st_size,
            "sha16": h,
        }
    except FileNotFoundError:
        return None
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return {"error": str(exc)}


def _load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {}


def _save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    tmp.replace(STATE_FILE)


def _log_to_ledger(path: str, kind: str, summary: str) -> None:
    """Append an activity event to the profile-activity-ledger."""
    try:
        conn = sqlite3.connect(LEDGER_DB, timeout=10)
        conn.execute(
            """CREATE TABLE IF NOT EXISTS activity_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT,
                occurred_at INTEGER,
                source TEXT,
                actor_profile TEXT,
                target_profile TEXT,
                event_type TEXT,
                object_type TEXT,
                object_id TEXT,
                board TEXT
            )"""
        )
        conn.execute(
            """INSERT INTO activity_events
               (event_id, occurred_at, source, actor_profile, target_profile,
                event_type, object_type, object_id, board, payload_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                f"cfgwatch-{int(dt.datetime.now().timestamp() * 1000)}",
                int(dt.datetime.now().timestamp()),
                "config-write-watchdog",
                None,
                kind,
                "config.write",
                "config",
                path,
                "ops",
                json.dumps({"summary": summary}),
                int(dt.datetime.now().timestamp()),
            ),
        )
        conn.commit()
        conn.close()
    except Exception as exc:  # pragma: no cover
        print(f"⚠ ledger write failed: {exc}", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", action="store_true", help="rebuild baseline only")
    ap.add_argument("--debug", action="store_true", help="print resolved watch paths")
    args = ap.parse_args()

    state = _load_state()
    changes = []

    for glob, kind in WATCH_GLOBS:
        pattern = str(glob)
        if "*" in pattern:
            # Glob pattern (e.g. ~/.hermes/profiles/*/config.yaml).
            # Split at the first wildcard: base = dir before *, rest = pattern.
            star = pattern.index("*")
            base = Path(pattern[:star])
            sub = pattern[star:]
            paths = sorted(base.glob(sub.lstrip("/")))
            if args.debug:
                print(f"[debug] glob base={base} sub={sub} -> {len(paths)} paths")
        else:
            paths = [Path(pattern)]
            if args.debug:
                print(f"[debug] exact path={pattern}")
        for p in paths:
            if not p.is_file():
                if args.debug:
                    print(f"[debug] skip non-file: {p}")
                continue
            fp = _file_fingerprint(p)
            if fp is None:
                continue
            key = str(p)
            prev = state.get(key)
            if args.snapshot:
                state[key] = fp
                continue
            if prev is None:
                # New file (first sighting) — baseline it, no alert.
                state[key] = fp
                continue
            if prev.get("mtime") != fp["mtime"] or prev.get("sha16") != fp["sha16"]:
                changes.append((key, kind, prev, fp))

    if args.snapshot:
        _save_state(state)
        print(f"snapshot saved: {len(state)} files")
        return 0

    if not changes:
        _save_state(state)
        return 0  # silent — nothing to report

    # Log + notify
    lines = []
    for key, kind, prev, fp in changes:
        summary = f"mtime {prev.get('mtime')}→{fp.get('mtime')} size {prev.get('size')}→{fp.get('size')} sha {prev.get('sha16')}→{fp.get('sha16')}"
        _log_to_ledger(key, kind, summary)
        lines.append(f"• {kind}: {key}\n    {summary}")
        state[key] = fp
    _save_state(state)

    print(f"⚠ CONFIG WRITE DETECTED — {len(changes)} file(s) changed under {HERMES_HOME}")
    print("\n".join(lines))
    print("\nCheck ~/.hermes/governance/profile-activity-ledger.sqlite (event_type=config.write) for the trail.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
