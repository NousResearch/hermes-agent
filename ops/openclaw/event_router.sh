#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENCLAW_ROOT:-/home/yeqiuqiu/clawd-architect}"
STATE_FILE="${OPENCLAW_EVENT_ROUTER_STATE_FILE:-$ROOT/state/cron_watchdog/event_fingerprints.json}"
EVENTS_JSONL_FILE="${OPENCLAW_EVENT_ROUTER_EVENTS_JSONL_FILE:-$ROOT/state/continuity/events/event_router_events.jsonl}"
CONTINUITY_DB_PATH="${OPENCLAW_EVENT_ROUTER_DB_PATH:-$ROOT/state/continuity/continuity_os.sqlite}"
PERSIST_ENABLED=1
SOURCE=""
KEY=""
SEVERITY="info"
SUMMARY=""
EVIDENCE_REF=""
FINGERPRINT_INPUT=""
COOLDOWN_SEC="${OPENCLAW_EVENT_ROUTER_DEFAULT_COOLDOWN_SEC:-1800}"

usage() {
  cat <<'EOF'
Usage:
  event_router.sh --source <source> --key <key> [options]

Options:
  --source <source>              Event source namespace (required)
  --key <key>                    Event key (required)
  --severity <info|warn|critical>
  --summary <text>
  --evidence-ref <path-or-ref>
  --fingerprint-input <text>     Explicit fingerprint seed (default derives from key/severity/summary/evidence)
  --cooldown-sec <n>             Cooldown before unchanged event can emit again (default: 1800)
  --state-file <path>            Override dedupe state file path
  --events-jsonl-file <path>     Append routed events to JSONL log (default under state/continuity/events)
  --db-path <path>               Persist routed events into continuity SQLite (default state/continuity/continuity_os.sqlite)
  --no-persist                   Disable JSONL/DB persistence for this invocation
  -h, --help

Exit codes:
  0  -> emitted (new fingerprint or cooldown elapsed)
  20 -> suppressed (unchanged and still within cooldown)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source)
      SOURCE="${2:-}"; shift 2 ;;
    --key)
      KEY="${2:-}"; shift 2 ;;
    --severity)
      SEVERITY="${2:-}"; shift 2 ;;
    --summary)
      SUMMARY="${2:-}"; shift 2 ;;
    --evidence-ref)
      EVIDENCE_REF="${2:-}"; shift 2 ;;
    --fingerprint-input)
      FINGERPRINT_INPUT="${2:-}"; shift 2 ;;
    --cooldown-sec)
      COOLDOWN_SEC="${2:-}"; shift 2 ;;
    --state-file)
      STATE_FILE="${2:-}"; shift 2 ;;
    --events-jsonl-file)
      EVENTS_JSONL_FILE="${2:-}"; shift 2 ;;
    --db-path)
      CONTINUITY_DB_PATH="${2:-}"; shift 2 ;;
    --no-persist)
      PERSIST_ENABLED=0; shift ;;
    -h|--help)
      usage
      exit 0 ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

if [[ -z "$SOURCE" || -z "$KEY" ]]; then
  echo "--source and --key are required" >&2
  exit 2
fi

case "$SEVERITY" in
  info|warn|critical) ;;
  *)
    echo "invalid --severity: $SEVERITY (expected info|warn|critical)" >&2
    exit 2 ;;
esac

if ! [[ "$COOLDOWN_SEC" =~ ^[0-9]+$ ]]; then
  echo "--cooldown-sec must be an integer" >&2
  exit 2
fi

if [[ -z "$FINGERPRINT_INPUT" ]]; then
  FINGERPRINT_INPUT="source=${SOURCE}|key=${KEY}|severity=${SEVERITY}|summary=${SUMMARY}|evidence=${EVIDENCE_REF}"
fi

python3 - "$STATE_FILE" "$SOURCE" "$KEY" "$SEVERITY" "$SUMMARY" "$EVIDENCE_REF" "$FINGERPRINT_INPUT" "$COOLDOWN_SEC" "$EVENTS_JSONL_FILE" "$CONTINUITY_DB_PATH" "$PERSIST_ENABLED" <<'PY'
import datetime as dt
import hashlib
import json
import os
import pathlib
import sqlite3
import sys

state_file = pathlib.Path(sys.argv[1])
source = sys.argv[2]
key = sys.argv[3]
severity = sys.argv[4]
summary = sys.argv[5]
evidence_ref = sys.argv[6]
fingerprint_input = sys.argv[7]
cooldown_sec = max(0, int(sys.argv[8]))
events_jsonl_file = pathlib.Path(sys.argv[9])
continuity_db_path = pathlib.Path(sys.argv[10])
persist_enabled = bool(int(sys.argv[11]))

now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
now_epoch = int(now.timestamp())
now_iso = now.isoformat().replace("+00:00", "Z")

fingerprint = hashlib.sha256(fingerprint_input.encode("utf-8")).hexdigest()
route_key = f"{source}|{key}"


def load_state() -> dict:
    if not state_file.exists():
        return {
            "schema_version": "openclaw.event_router.v1",
            "events": {},
            "updated_epoch": 0,
            "updated_iso": "",
        }
    try:
        raw = json.loads(state_file.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            raw.setdefault("schema_version", "openclaw.event_router.v1")
            raw.setdefault("events", {})
            return raw
    except Exception:
        pass
    return {
        "schema_version": "openclaw.event_router.v1",
        "events": {},
        "updated_epoch": 0,
        "updated_iso": "",
    }


def atomic_write(path: pathlib.Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, path)


def append_jsonl(path: pathlib.Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")


def persist_event_db(path: pathlib.Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.executescript(
        """
CREATE TABLE IF NOT EXISTS continuity_events (
  event_id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  source TEXT NOT NULL,
  event_key TEXT NOT NULL,
  severity TEXT NOT NULL CHECK(severity IN ('info','warn','critical')),
  fingerprint TEXT NOT NULL,
  emitted INTEGER NOT NULL,
  changed INTEGER NOT NULL,
  cooldown_elapsed INTEGER NOT NULL,
  suppress_reason TEXT,
  summary TEXT,
  evidence_ref TEXT,
  route_key TEXT NOT NULL,
  state_file TEXT,
  metadata_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_continuity_events_route_created ON continuity_events(route_key, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_continuity_events_key_created ON continuity_events(event_key, created_at DESC);
"""
    )
    cur.execute(
        """
INSERT OR REPLACE INTO continuity_events (
  event_id, created_at, source, event_key, severity, fingerprint,
  emitted, changed, cooldown_elapsed, suppress_reason, summary,
  evidence_ref, route_key, state_file, metadata_json
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
""",
        (
            event.get("event_id"),
            event.get("created_at"),
            event.get("source"),
            event.get("key"),
            event.get("severity"),
            event.get("fingerprint"),
            int(bool(event.get("emit"))),
            int(bool(event.get("changed"))),
            int(bool(event.get("cooldown_elapsed"))),
            event.get("suppress_reason"),
            event.get("summary"),
            event.get("evidence_ref"),
            event.get("route_key"),
            event.get("state_file"),
            json.dumps(event, ensure_ascii=False, sort_keys=True),
        ),
    )
    con.commit()
    con.close()


state = load_state()
events = state.setdefault("events", {})
prev = events.get(route_key) if isinstance(events, dict) else None
if not isinstance(prev, dict):
    prev = {}

prev_fingerprint = str(prev.get("last_fingerprint") or "")
last_emitted_epoch = int(prev.get("last_emitted_epoch") or 0)

changed = fingerprint != prev_fingerprint
cooldown_elapsed = True if last_emitted_epoch <= 0 else (now_epoch - last_emitted_epoch) >= cooldown_sec
emit = changed or cooldown_elapsed

suppress_reason = None
if not emit:
    suppress_reason = "unchanged_within_cooldown"

updated = {
    "route_key": route_key,
    "source": source,
    "key": key,
    "last_fingerprint": fingerprint,
    "last_seen_epoch": now_epoch,
    "last_seen_iso": now_iso,
    "last_emitted_epoch": now_epoch if emit else last_emitted_epoch,
    "last_emitted_iso": now_iso if emit else (prev.get("last_emitted_iso") or ""),
    "emit_count": int(prev.get("emit_count") or 0) + (1 if emit else 0),
    "suppress_count": int(prev.get("suppress_count") or 0) + (0 if emit else 1),
    "last_severity": severity,
    "last_summary": summary,
    "last_evidence_ref": evidence_ref,
    "cooldown_sec": cooldown_sec,
}

events[route_key] = updated
state["updated_epoch"] = now_epoch
state["updated_iso"] = now_iso

atomic_write(state_file, json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

event_id_seed = f"{route_key}|{fingerprint}|{now_iso}|emit={int(emit)}|emit_count={updated.get('emit_count')}|suppress_count={updated.get('suppress_count')}"
event_id = f"evt_{hashlib.sha256(event_id_seed.encode('utf-8')).hexdigest()[:20]}"

result = {
    "ok": True,
    "event_id": event_id,
    "emit": emit,
    "changed": changed,
    "cooldown_elapsed": cooldown_elapsed,
    "suppress_reason": suppress_reason,
    "state_file": str(state_file),
    "source": source,
    "key": key,
    "severity": severity,
    "summary": summary,
    "evidence_ref": evidence_ref,
    "fingerprint": fingerprint,
    "route_key": route_key,
    "timestamp": now_iso,
    "created_at": now_iso,
}

persist_errors = []
if persist_enabled:
    try:
        append_jsonl(events_jsonl_file, result)
        result["events_jsonl_file"] = str(events_jsonl_file)
    except Exception as exc:
        persist_errors.append(f"jsonl_append_failed:{exc}")

    try:
        persist_event_db(continuity_db_path, result)
        result["db_path"] = str(continuity_db_path)
    except Exception as exc:
        persist_errors.append(f"db_persist_failed:{exc}")

if persist_errors:
    result["persist_errors"] = persist_errors

print(json.dumps(result, ensure_ascii=False))

if emit:
    raise SystemExit(0)
raise SystemExit(20)
PY