#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENCLAW_ROOT:-/home/yeqiuqiu/clawd-architect}"
CHECKPOINT_REF=""
HANDOVER_OUT="${OPENCLAW_CONTEXT_HANDOVER_COMPAT_PATH:-$ROOT/reports/handover_context_latest.md}"
SKIP_RENDER=0
ACTION_TOKEN=""
ALLOW_LEGACY_ANCHOR="${OPENCLAW_TRUTH_ANCHOR_ALLOW_LEGACY:-0}"
MUTATION_TICKET=""
declare -a MUTATION_ATTESTATIONS=()
declare -a MUTATION_ATTESTATION_OBJECTS=()

usage() {
  cat <<'EOF'
Usage: sync_latest_artifacts.sh [options]

Keep continuity latest artifacts aligned with deterministic runtime truth.

Options:
  --checkpoint <path-or-id>   Optional checkpoint JSON path or checkpoint id.
                              If omitted, use continuity latest pointer.
  --handover-out <path>       Compatibility handover markdown path.
                              Default: reports/handover_context_latest.md
  --skip-render               Do not re-render compatibility handover markdown.
  --action-token <value>      Canonical mutation token for direct entrypoint use.
  --truth-anchor <value>      Legacy alias of --action-token.
  --allow-legacy-anchor       Allow legacy anchor-only token mode for direct token validation.
  --mutation-ticket <value>   Authority ticket JSON string, @path, or path (high-risk token path).
  --attestation <name>        Satisfied authority attestation (repeatable).
  --attestation-object <value> Structured attestation JSON string, @path, or path (repeatable).
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint)
      CHECKPOINT_REF="${2:-}"; shift 2 ;;
    --handover-out)
      HANDOVER_OUT="${2:-}"; shift 2 ;;
    --skip-render)
      SKIP_RENDER=1; shift ;;
    --action-token|--truth-anchor)
      ACTION_TOKEN="${2:-}"; shift 2 ;;
    --allow-legacy-anchor)
      ALLOW_LEGACY_ANCHOR=1; shift ;;
    --mutation-ticket)
      MUTATION_TICKET="${2:-}"; shift 2 ;;
    --attestation)
      MUTATION_ATTESTATIONS+=("${2:-}"); shift 2 ;;
    --attestation-object)
      MUTATION_ATTESTATION_OBJECTS+=("${2:-}"); shift 2 ;;
    -h|--help)
      usage
      exit 0 ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

guard_args=(
  --script "sync_latest_artifacts.sh"
  --risk-tier "high"
  --mutation-operation "sync_latest_artifacts:promote"
)
if [[ -n "$ACTION_TOKEN" ]]; then
  guard_args+=(--action-token "$ACTION_TOKEN")
fi
if [[ "$ALLOW_LEGACY_ANCHOR" == "1" ]]; then
  guard_args+=(--allow-legacy-anchor)
fi
if [[ -n "$MUTATION_TICKET" ]]; then
  guard_args+=(--mutation-ticket "$MUTATION_TICKET")
fi
for att in "${MUTATION_ATTESTATIONS[@]}"; do
  if [[ -n "${att:-}" ]]; then
    guard_args+=(--attestation "$att")
  fi
done
for att_obj in "${MUTATION_ATTESTATION_OBJECTS[@]}"; do
  if [[ -n "${att_obj:-}" ]]; then
    guard_args+=(--attestation-object "$att_obj")
  fi
done
"$ROOT/ops/openclaw/continuity/mutator_ingress_guard.sh" "${guard_args[@]}"

python3 - "$ROOT" "$CHECKPOINT_REF" "$HANDOVER_OUT" "$SKIP_RENDER" <<'PY'
import datetime as dt
import hashlib
import json
import os
import pathlib
import subprocess
import sys
from typing import Any, Dict, Optional

root = pathlib.Path(sys.argv[1]).resolve()
checkpoint_ref = (sys.argv[2] or "").strip()
handover_out = pathlib.Path(sys.argv[3])
skip_render = bool(int(sys.argv[4]))
if not handover_out.is_absolute():
    handover_out = (root / handover_out).resolve()

latest_dir = root / "state" / "continuity" / "latest"
bridge_path = latest_dir / "runtime_truth_bridge.json"
latest_pointer_path = latest_dir / "latest_pointer.json"


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def atomic_write(path: pathlib.Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, path)


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: pathlib.Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def to_rel(path: pathlib.Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except Exception:
        return str(path)


def resolve_checkpoint_path(ref: str) -> pathlib.Path:
    if ref:
        cand = pathlib.Path(ref)
        if not cand.is_absolute():
            cand = (root / ref).resolve()
        if cand.exists():
            return cand
        by_id = root / "state" / "continuity" / "checkpoints" / f"{ref}.json"
        if by_id.exists():
            return by_id.resolve()
        raise SystemExit(f"checkpoint not found: {ref}")

    if latest_pointer_path.exists():
        pointer = load_json(latest_pointer_path)
        rel = str(pointer.get("json_path") or "")
        if rel:
            p = (root / rel).resolve()
            if p.exists():
                return p

    surface = latest_dir / "handover_latest.json"
    if surface.exists():
        try:
            surface_obj = load_json(surface)
            checkpoint = surface_obj.get("checkpoint") if isinstance(surface_obj.get("checkpoint"), dict) else {}
            rel = str(checkpoint.get("path") or "").strip()
            if rel:
                p = (root / rel).resolve()
                if p.exists():
                    return p
        except Exception:
            pass
        return surface.resolve()

    raise SystemExit("no continuity checkpoint found")


checkpoint_path = resolve_checkpoint_path(checkpoint_ref)
checkpoint = load_json(checkpoint_path)
checkpoint_id = (((checkpoint.get("metadata") or {}).get("checkpoint_id")) or checkpoint_path.stem)
checkpoint_sha_actual = sha256_file(checkpoint_path)
checkpoint_rel = to_rel(checkpoint_path)

latest_pointer: Dict[str, Any] = {}
pointer_matches_checkpoint = False
pointer_sha_match = None
if latest_pointer_path.exists():
    latest_pointer = load_json(latest_pointer_path)
    pointer_id = str(latest_pointer.get("checkpoint_id") or "")
    pointer_path = str(latest_pointer.get("json_path") or "")
    pointer_matches_checkpoint = pointer_id == checkpoint_id and pointer_path == checkpoint_rel
    if pointer_matches_checkpoint:
        pointer_sha = str(latest_pointer.get("json_sha256") or "")
        pointer_sha_match = pointer_sha == checkpoint_sha_actual
        if not pointer_sha_match:
            raise SystemExit("latest_pointer json_sha256 mismatch for resolved checkpoint")

env_latest_path = latest_dir / "env_snapshot_latest.json"
env_latest: Dict[str, Any] = load_json(env_latest_path) if env_latest_path.exists() else {}
env_snapshot_rel = str(env_latest.get("env_snapshot_path") or "")
env_snapshot_sha_expected = str(env_latest.get("env_snapshot_sha256") or "")
env_snapshot_sha_actual = None
env_snapshot_exists = False
if env_snapshot_rel:
    env_snapshot_path = (root / env_snapshot_rel).resolve()
    env_snapshot_exists = env_snapshot_path.exists()
    if env_snapshot_exists:
        env_snapshot_sha_actual = sha256_file(env_snapshot_path)

gt_latest_path = root / "state" / "ground_truth" / "latest.json"
gt_latest: Dict[str, Any] = load_json(gt_latest_path) if gt_latest_path.exists() else {}
gt_snapshot_rel = str(gt_latest.get("snapshot_path") or "")
gt_snapshot_sha_expected = str(gt_latest.get("snapshot_sha256") or "")
gt_snapshot_sha_actual = None
gt_snapshot_exists = False
if gt_snapshot_rel:
    gt_snapshot_path = (root / gt_snapshot_rel).resolve()
    gt_snapshot_exists = gt_snapshot_path.exists()
    if gt_snapshot_exists:
        gt_snapshot_sha_actual = sha256_file(gt_snapshot_path)

captured_env = str(((checkpoint.get("state_capture") or {}).get("env_snapshot_path")) or "")
captured_gt = str(((checkpoint.get("state_capture") or {}).get("ground_truth_snapshot_path")) or "")

render_result: Optional[Dict[str, Any]] = None
if not skip_render:
    render_script = root / "ops" / "openclaw" / "continuity" / "render_context_handover_compat.sh"
    cp = subprocess.run(
        [
            str(render_script),
            "--checkpoint",
            checkpoint_id,
            "--out",
            str(handover_out),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if cp.returncode != 0:
        stderr = (cp.stderr or cp.stdout or "render_context_handover_compat failed").strip()
        raise SystemExit(stderr[:240])
    raw = (cp.stdout or "{}").strip()
    try:
        render_result = json.loads(raw)
    except Exception:
        render_result = {"raw": raw}

bridge = {
    "schema_version": "continuity.runtime_truth_bridge.v1",
    "updated_at": now_iso(),
    "checkpoint": {
      "checkpoint_id": checkpoint_id,
      "json_path": checkpoint_rel,
      "json_sha256": checkpoint_sha_actual,
      "captured_env_snapshot_path": captured_env,
      "captured_ground_truth_snapshot_path": captured_gt,
    },
    "latest_pointer": {
      "path": to_rel(latest_pointer_path),
      "exists": latest_pointer_path.exists(),
      "matches_checkpoint": pointer_matches_checkpoint,
      "sha_match": pointer_sha_match,
      "checkpoint_id": latest_pointer.get("checkpoint_id") if latest_pointer else None,
      "json_path": latest_pointer.get("json_path") if latest_pointer else None,
      "json_sha256": latest_pointer.get("json_sha256") if latest_pointer else None,
    },
    "env_snapshot_latest": {
      "path": to_rel(env_latest_path),
      "exists": env_latest_path.exists(),
      "env_snapshot_path": env_snapshot_rel,
      "env_snapshot_sha256_expected": env_snapshot_sha_expected or None,
      "env_snapshot_sha256_actual": env_snapshot_sha_actual,
      "sha_match": (env_snapshot_sha_expected == env_snapshot_sha_actual) if env_snapshot_sha_expected and env_snapshot_sha_actual else None,
      "matches_checkpoint_capture": bool(captured_env and captured_env == env_snapshot_rel),
    },
    "ground_truth_latest": {
      "path": to_rel(gt_latest_path),
      "exists": gt_latest_path.exists(),
      "snapshot_id": gt_latest.get("snapshot_id") if gt_latest else None,
      "snapshot_path": gt_snapshot_rel or None,
      "snapshot_sha256_expected": gt_snapshot_sha_expected or None,
      "snapshot_sha256_actual": gt_snapshot_sha_actual,
      "sha_match": (gt_snapshot_sha_expected == gt_snapshot_sha_actual) if gt_snapshot_sha_expected and gt_snapshot_sha_actual else None,
      "matches_checkpoint_capture": bool(captured_gt and captured_gt == gt_snapshot_rel),
    },
    "compat_handover": {
      "path": to_rel(handover_out),
      "rendered": not skip_render,
      "render_result": render_result,
    },
}

atomic_write(bridge_path, json.dumps(bridge, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

print(
    json.dumps(
        {
            "ok": True,
            "checkpoint_id": checkpoint_id,
            "runtime_truth_bridge": to_rel(bridge_path),
            "compat_handover": to_rel(handover_out),
            "pointer_matches_checkpoint": pointer_matches_checkpoint,
            "ground_truth_matches_checkpoint_capture": bridge["ground_truth_latest"]["matches_checkpoint_capture"],
        },
        ensure_ascii=False,
    )
)
PY
