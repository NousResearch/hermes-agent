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
        raw = path.read_bytes()
        h = hashlib.sha256(raw).hexdigest()[:16]
        # Model-config fingerprint: hash only the model/provider routing keys.
        # A file whose sha changes but model_sig doesn't = cosmetic rewrite
        # (schema stamp, skills, budgets) — NOT a model configuration change.
        import yaml as _yaml
        try:
            cfg = _yaml.safe_load(raw) or {}
            routing = {
                "model": cfg.get("model"),
                "fallback_providers": cfg.get("fallback_providers"),
                "providers": cfg.get("providers"),
                "credential_pool_strategies": cfg.get("credential_pool_strategies"),
            }
            msig = hashlib.sha256(
                json.dumps(routing, sort_keys=True, default=str).encode()
            ).hexdigest()[:16]
        except Exception:
            msig = "unparsed"
        return {
            "mtime": int(st.st_mtime),
            "size": st.st_size,
            "sha16": h,
            "model_sig": msig,
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
                board TEXT,
                payload_json TEXT NOT NULL DEFAULT '{}',
                created_at INTEGER NOT NULL
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
    for key, kind, prev, fp in changes:
        summary = f"mtime {prev.get('mtime')}→{fp.get('mtime')} size {prev.get('size')}→{fp.get('size')} sha {prev.get('sha16')}→{fp.get('sha16')}"
        _log_to_ledger(key, kind, summary)
        state[key] = fp
    _save_state(state)

    # ---- Clean output: aggregate, don't dump ----
    from collections import Counter

    # Split: model-routing changes (CRITICAL — Sahil's authorization required)
    # vs cosmetic rewrites (schema stamps, skills, budgets).
    model_changes = [
        (k, kd, p, f) for (k, kd, p, f) in changes
        if p.get("model_sig") not in (None, "unparsed")
        and p.get("model_sig") != f.get("model_sig")
    ]
    cosmetic_changes = [c for c in changes if c not in model_changes]

    # 1. Batch detection: files sharing a new mtime = one writer,
    #    almost always an authorized stamp/rollout, not per-file tampering.
    mtime_buckets: dict[int, list[str]] = {}
    for key, kind, prev, fp in cosmetic_changes:
        mtime_buckets.setdefault(fp.get("mtime", 0), []).append(key)
    batch_groups = [k for k, v in mtime_buckets.items() if len(v) >= 5]
    batch_files = {p for m in batch_groups for p in mtime_buckets[m]}
    singles = [(k, kd) for (k, kd, _, _) in cosmetic_changes if k not in batch_files]

    def short(path: str) -> str:
        if path == str(HERMES_HOME / "config.yaml"):
            return "root"
        return path.replace(f"{HERMES_HOME}/profiles/", "").replace("/config.yaml", "")

    # P3 — approval-token gate: a model-routing change is expected only if
    # Sahil dropped ~/.hermes/governance/model-change-approvals/<label>.approved
    APPROVALS_DIR = HERMES_HOME / "governance" / "model-change-approvals"

    def _approval_label(path: str) -> str:
        if path == str(HERMES_HOME / "config.yaml"):
            return "root"
        return path.replace(f"{HERMES_HOME}/profiles/", "").replace("/config.yaml", "")

    def _has_approval(path: str) -> bool:
        token = APPROVALS_DIR / f"{_approval_label(path)}.approved"
        try:
            content = token.read_text().strip().lower()
            return content.startswith("approved")
        except OSError:
            return False

    def _approval_age_days(path: str) -> float:
        token = APPROVALS_DIR / f"{_approval_label(path)}.approved"
        try:
            return (dt.datetime.now() - dt.datetime.fromtimestamp(token.stat().st_mtime)).days
        except OSError:
            return -1

    unauthorized = []
    authorized = []
    for entry in model_changes:
        (authorized if _has_approval(entry[0]) else unauthorized).append(entry)

    # P5 — timestamped JSON report (audit-safe links; WFA provenance pattern)
    try:
        report_dir = HERMES_HOME / "governance" / "config-write-reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        report_path = report_dir / f"config-write-{stamp}.json"
        report = {
            "generated_at": stamp,
            "model_changes": [
                {"profile": _approval_label(k), "from_sig": p.get("model_sig"),
                 "to_sig": f.get("model_sig"), "approved": _has_approval(k)}
                for k, _, p, f in model_changes
            ],
            "cosmetic_changes": [
                {"file": k, "kind": kd, "from_sha": p.get("sha16"), "to_sha": f.get("sha16")}
                for k, kd, p, f in cosmetic_changes
            ],
        }
        latest = report_dir / "latest.json"
        report_path.write_text(json.dumps(report, indent=2))
        latest.write_text(json.dumps(report, indent=2))
    except OSError as exc:
        report_path = None
        print(f"⚠ report write failed: {exc}", file=sys.stderr)

    if unauthorized:
        print(f"🔴 MODEL CONFIG CHANGE — {len(unauthorized)} profile(s) changed model/provider routing WITHOUT approval token:")
        for key, kind, prev, fp in unauthorized[:10]:
            print(f"  • {short(key)} ({prev.get('model_sig','?')[:8]}→{fp.get('model_sig','?')[:8]})")
        if len(unauthorized) > 10:
            print(f"  ... +{len(unauthorized)-10} more")
        print("  MODEL CONFIGS ARE IMMUTABLE without Sahil's explicit authorization.")
        print("  Authorized change pending? Drop governance/model-change-approvals/<profile>.approved ('approved — <your note>').")
        print("  Otherwise verify source and roll back.")

    if authorized:
        print(f"🟢 Model config change — {len(authorized)} profile(s), approval token present:")
        for key, kind, prev, fp in authorized[:10]:
            age = _approval_age_days(key)
            print(f"  • {short(key)} ({prev.get('model_sig','?')[:8]}→{fp.get('model_sig','?')[:8]}, approval {age:.0f}d old)")

    # P4 — cosmetic changes go to ledger + JSON report only; Discord carries
    # zero cosmetic noise unless it's a single-file (non-batch) change.
    if singles:
        print(f"⚙ Config change: {len(singles)} single-file edit(s) — no model routing touched:")
        for key, kind in singles[:5]:
            print(f"  • {short(key)} ({kind})")
    if report_path is not None and (cosmetic_changes or model_changes):
        print(f"  Report: {report_path} (latest.json kept current)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
