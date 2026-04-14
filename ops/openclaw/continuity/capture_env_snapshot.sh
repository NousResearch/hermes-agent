#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENCLAW_ROOT:-/home/yeqiuqiu/clawd-architect}"
GROUND_TRUTH_SCRIPT="$ROOT/ops/openclaw/snapshot_ground_truth.sh"

if [[ ! -x "$GROUND_TRUTH_SCRIPT" ]]; then
  echo "ERROR: missing executable ground-truth snapshot script: $GROUND_TRUTH_SCRIPT" >&2
  exit 1
fi

"$GROUND_TRUTH_SCRIPT" >/dev/null

python3 - "$ROOT" <<'PY'
import datetime as dt
import hashlib
import json
import os
import pathlib
import sys
from typing import Any, Dict, Tuple

root = pathlib.Path(sys.argv[1]).resolve()

gt_latest_path = root / "state" / "ground_truth" / "latest.json"
if not gt_latest_path.exists():
    raise SystemExit(f"ground-truth latest pointer not found: {gt_latest_path}")

gt_latest = json.loads(gt_latest_path.read_text(encoding="utf-8"))
gt_snapshot_rel = str(gt_latest.get("snapshot_path") or "")
if not gt_snapshot_rel:
    raise SystemExit("ground-truth latest pointer missing snapshot_path")

gt_snapshot_path = (root / gt_snapshot_rel).resolve()
if not gt_snapshot_path.exists():
    raise SystemExit(f"ground-truth snapshot missing: {gt_snapshot_path}")

gt_snapshot = json.loads(gt_snapshot_path.read_text(encoding="utf-8"))

continuity_dir = root / "state" / "continuity"
snapshots_dir = continuity_dir / "snapshots"
latest_dir = continuity_dir / "latest"
snapshots_dir.mkdir(parents=True, exist_ok=True)
latest_dir.mkdir(parents=True, exist_ok=True)


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0)


def atomic_write(path: pathlib.Path, text: str) -> None:
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def to_rel(path: pathlib.Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except Exception:
        return str(path)


def next_append_only_path(base_id: str) -> Tuple[str, pathlib.Path]:
    p = snapshots_dir / f"{base_id}.json"
    if not p.exists():
        return base_id, p
    for i in range(1, 1000):
        sid = f"{base_id}_{i:02d}"
        c = snapshots_dir / f"{sid}.json"
        if not c.exists():
            return sid, c
    raise RuntimeError("unable to allocate append-only env snapshot file")


now = now_utc()
ts_iso = now.isoformat().replace("+00:00", "Z")
ts_file = now.strftime("%Y%m%dT%H%M%SZ")
base_id = f"env_{ts_file}"
env_snapshot_id, env_snapshot_path = next_append_only_path(base_id)

env_snapshot: Dict[str, Any] = {
    "schema_version": "continuity.env_snapshot.v1",
    "snapshot_id": env_snapshot_id,
    "created_at": ts_iso,
    "workspace": str(root),
    "source_ground_truth": {
        "latest_path": "state/ground_truth/latest.json",
        "snapshot_id": gt_latest.get("snapshot_id"),
        "snapshot_path": gt_snapshot_rel,
        "snapshot_sha256": gt_latest.get("snapshot_sha256"),
        "snapshot_ts_utc": gt_latest.get("snapshot_ts_utc"),
    },
    "repo_state": gt_snapshot.get("git_state"),
    "gateway": gt_snapshot.get("gateway"),
    "cron_summary": {
        "total": ((gt_snapshot.get("cron") or {}).get("total")),
        "jobs": ((gt_snapshot.get("cron") or {}).get("jobs") or []),
    },
    "sessions_summary": gt_snapshot.get("sessions"),
    "process_summary": gt_snapshot.get("process_ports"),
    "watchdog_state": {
        "state_refs": sorted(((gt_snapshot.get("watchdog_state") or {}).get("state_files") or {}).keys()),
        "telemetry_metrics": ((gt_snapshot.get("watchdog_state") or {}).get("telemetry_metrics") or {}),
    },
    "autopilot_state": gt_snapshot.get("autopilot_state"),
    "anomalies": gt_snapshot.get("anomalies") or [],
}

payload = json.dumps(env_snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
env_sha = hashlib.sha256(payload.encode("utf-8")).hexdigest()
atomic_write(env_snapshot_path, payload)

latest_payload = {
    "schema_version": "continuity.env_snapshot_latest.v1",
    "updated_at": ts_iso,
    "env_snapshot_id": env_snapshot_id,
    "env_snapshot_path": to_rel(env_snapshot_path),
    "env_snapshot_sha256": env_sha,
    "source_ground_truth_snapshot_id": gt_latest.get("snapshot_id"),
    "source_ground_truth_snapshot_path": gt_snapshot_rel,
    "source_ground_truth_snapshot_sha256": gt_latest.get("snapshot_sha256"),
}
atomic_write(
    latest_dir / "env_snapshot_latest.json",
    json.dumps(latest_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
)

print(
    json.dumps(
        {
            "ok": True,
            "env_snapshot_id": env_snapshot_id,
            "env_snapshot_path": to_rel(env_snapshot_path),
            "latest_path": "state/continuity/latest/env_snapshot_latest.json",
            "env_snapshot_sha256": env_sha,
        },
        ensure_ascii=False,
    )
)
PY
