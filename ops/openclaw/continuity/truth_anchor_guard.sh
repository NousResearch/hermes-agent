#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENCLAW_ROOT:-/home/yeqiuqiu/clawd-architect}"
TRUTH_ANCHOR=""
CURRENT_PATH="$ROOT/state/continuity/current.json"
JSON_OUT=0
REFRESH_IF_MISSING=1
ALLOW_LEGACY_ANCHOR="${OPENCLAW_TRUTH_ANCHOR_ALLOW_LEGACY:-0}"

usage() {
  cat <<'EOF'
Usage: truth_anchor_guard.sh --action-token <value> [options]

Validate a caller-provided mutation token against state/continuity/current.json.
Default policy requires coherence-aware action_token values whenever coherence metadata exists.

Preferred token format (from continuity_current .action_token):
  snapshot_id=<...>;journal_offset=<...>;pointer_hash=<...>;coherence_tuple_hash=<...>;
  policy_signature=<...>;coherence_build_generation_id=<...>;coherence_valid_until=<ISO8601>

Legacy anchor tokens are accepted only with explicit override.

Options:
  --action-token <value>        Required mutation token from operator context
  --truth-anchor <value>        Legacy alias of --action-token
  --allow-legacy-anchor         Allow legacy anchor-only tokens for break-glass use
                                (also set OPENCLAW_TRUTH_ANCHOR_ALLOW_LEGACY=1)
  --current-path <path>         Override continuity/current path
  --no-refresh                  Do not auto-refresh continuity/current when missing
  --json                        Emit JSON result
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --action-token|--truth-anchor)
      TRUTH_ANCHOR="${2:-}"; shift 2 ;;
    --allow-legacy-anchor)
      ALLOW_LEGACY_ANCHOR=1; shift ;;
    --current-path)
      CURRENT_PATH="${2:-}"; shift 2 ;;
    --no-refresh)
      REFRESH_IF_MISSING=0; shift ;;
    --json)
      JSON_OUT=1; shift ;;
    -h|--help)
      usage
      exit 0 ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

if [[ -z "$TRUTH_ANCHOR" ]]; then
  echo "missing --action-token/--truth-anchor" >&2
  exit 2
fi

python3 - "$ROOT" "$CURRENT_PATH" "$TRUTH_ANCHOR" "$JSON_OUT" "$REFRESH_IF_MISSING" "$ALLOW_LEGACY_ANCHOR" <<'PY'
import datetime as dt
import json
import os
import pathlib
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote

root = pathlib.Path(sys.argv[1]).resolve()
current_path = pathlib.Path(sys.argv[2]).resolve()
provided = str(sys.argv[3] or "").strip()
json_out = bool(int(sys.argv[4]))
refresh_if_missing = bool(int(sys.argv[5]))
allow_legacy_anchor = bool(int(sys.argv[6]))

continuity_path = root / "ops" / "openclaw" / "continuity"
if str(continuity_path) not in sys.path:
    sys.path.insert(0, str(continuity_path))

try:
    from fixed_now import now_ts as _helper_now_ts  # type: ignore
except Exception:  # pragma: no cover - optional helper in minimal test roots
    _helper_now_ts = None


def now_utc() -> dt.datetime:
    if callable(_helper_now_ts):
        try:
            return dt.datetime.fromtimestamp(int(_helper_now_ts()), tz=dt.timezone.utc)
        except Exception:
            pass
    return dt.datetime.now(dt.timezone.utc)


def emit(payload: Dict[str, object], code: int) -> None:
    if json_out:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        status = "PASS" if payload.get("ok") else "BLOCK"
        mode = payload.get("mode")
        suffix = f" ({mode})" if mode else ""
        print(f"{status}: truth_anchor_check{suffix}")
        if not payload.get("ok"):
            hint = payload.get("hint")
            if hint:
                print(str(hint))
    raise SystemExit(code)


def parse_iso(raw: str):
    txt = str(raw or "").strip()
    if not txt:
        return None
    if txt.endswith("Z"):
        txt = txt[:-1] + "+00:00"
    try:
        out = dt.datetime.fromisoformat(txt)
    except Exception:
        return None
    if out.tzinfo is None:
        out = out.replace(tzinfo=dt.timezone.utc)
    return out.astimezone(dt.timezone.utc)


def parse_json_object_output(raw: Any) -> Optional[Dict[str, Any]]:
    text = str(raw or "")
    if not text.strip():
        return None
    try:
        payload = json.loads(text)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def parse_kv_token(raw: str) -> Dict[str, str]:
    token_map: Dict[str, str] = {}
    for part in raw.split(";"):
        piece = part.strip()
        if not piece or "=" not in piece:
            continue
        k, v = piece.split("=", 1)
        key = k.strip()
        if not key:
            continue
        token_map[key] = unquote(v.strip())
    return token_map


def parse_pipe_token(raw: str) -> Dict[str, str]:
    items = [x.strip() for x in raw.split("|")]
    if len(items) == 3:
        return {
            "snapshot_id": items[0],
            "journal_offset": items[1],
            "pointer_hash": items[2],
        }
    if len(items) == 4:
        return {
            "snapshot_id": items[0],
            "journal_offset": items[1],
            "pointer_hash": items[2],
            "coherence_tuple_hash": items[3],
        }
    return {}


def classify_token(raw: str) -> Tuple[str, Dict[str, str]]:
    kv = parse_kv_token(raw)
    if kv:
        return "kv", kv
    pipe = parse_pipe_token(raw)
    if pipe:
        return "pipe", pipe
    return "scalar", {}


current: Dict[str, object] = {}
current_parse_error: Optional[str] = None

if current_path.exists():
    try:
        loaded_current = json.loads(current_path.read_text(encoding="utf-8"))
    except Exception as exc:
        current_parse_error = str(exc)
    else:
        if isinstance(loaded_current, dict):
            current = loaded_current
        else:
            current_parse_error = "not_object"

if not current and refresh_if_missing:
    cp = subprocess.run(
        [str(root / "ops" / "openclaw" / "continuity" / "continuity_current.sh"), "--json"],
        text=True,
        capture_output=True,
        check=False,
    )
    nonzero_payload = parse_json_object_output(cp.stdout)
    if cp.returncode != 0:
        if isinstance(nonzero_payload, dict):
            # continuity_current fail-close contract may intentionally return a
            # structured degraded payload with nonzero exit status.
            current = nonzero_payload
        else:
            emit(
                {
                    "ok": False,
                    "error": "continuity_current_refresh_failed",
                    "stderr": (cp.stderr or cp.stdout or "").strip()[:400],
                    "current_path": str(current_path),
                },
                1,
            )
    else:
        try:
            refreshed_payload = json.loads(cp.stdout or "{}")
        except Exception as exc:
            emit({"ok": False, "error": "continuity_current_parse_failed", "detail": str(exc)}, 1)
        if not isinstance(refreshed_payload, dict):
            emit(
                {
                    "ok": False,
                    "error": "continuity_current_parse_failed",
                    "detail": "not_object",
                },
                1,
            )
        current = refreshed_payload

if not current:
    if current_parse_error is not None:
        emit(
            {
                "ok": False,
                "error": "continuity_current_parse_failed",
                "detail": current_parse_error,
                "path": str(current_path),
            },
            1,
        )
    emit(
        {
            "ok": False,
            "error": "continuity_current_missing",
            "path": str(current_path),
        },
        1,
    )

anchor = current.get("truth_anchor") or {}
coherence = current.get("coherence") or {}

expected = {
    "snapshot_id": str(anchor.get("snapshot_id") or "").strip(),
    "journal_offset": str(anchor.get("journal_offset") or "").strip(),
    "pointer_hash": str(anchor.get("pointer_hash") or "").strip(),
    "coherence_tuple_hash": str(coherence.get("tuple_hash") or "").strip(),
    "policy_signature": str((((coherence.get("policy") or {}).get("signature") or "")).strip()),
    "coherence_build_generation_id": str(coherence.get("build_generation_id") or "").strip(),
    "coherence_valid_until": str(coherence.get("valid_until") or "").strip(),
}

required_action_fields = [
    "snapshot_id",
    "journal_offset",
    "pointer_hash",
    "coherence_tuple_hash",
    "policy_signature",
    "coherence_build_generation_id",
    "coherence_valid_until",
]

strict_required = bool(expected.get("coherence_tuple_hash"))
missing_current_action_fields = [k for k in required_action_fields if not expected.get(k)]
if strict_required and missing_current_action_fields:
    emit(
        {
            "ok": False,
            "mode": "strict",
            "error": "current_action_token_fields_missing",
            "missing_fields": missing_current_action_fields,
            "hint": "Refresh continuity_current/continuity_now surfaces before mutating",
        },
        1,
    )

# legacy token acceptance set
accepted_legacy = set()
for key in ("snapshot_id", "journal_offset", "pointer_hash", "coherence_tuple_hash"):
    token = expected.get(key, "")
    if token:
        accepted_legacy.add(token)
if expected["snapshot_id"] and expected["journal_offset"] and expected["pointer_hash"]:
    accepted_legacy.add(f"{expected['snapshot_id']}|{expected['journal_offset']}|{expected['pointer_hash']}")
    accepted_legacy.add(
        "snapshot_id="
        + expected["snapshot_id"]
        + ";journal_offset="
        + expected["journal_offset"]
        + ";pointer_hash="
        + expected["pointer_hash"]
    )
if expected["snapshot_id"] and expected["journal_offset"] and expected["pointer_hash"] and expected["coherence_tuple_hash"]:
    accepted_legacy.add(
        f"{expected['snapshot_id']}|{expected['journal_offset']}|{expected['pointer_hash']}|{expected['coherence_tuple_hash']}"
    )
    accepted_legacy.add(
        "snapshot_id="
        + expected["snapshot_id"]
        + ";journal_offset="
        + expected["journal_offset"]
        + ";pointer_hash="
        + expected["pointer_hash"]
        + ";coherence_tuple_hash="
        + expected["coherence_tuple_hash"]
    )

token_style, token_map = classify_token(provided)
has_action_fields = any(
    key in token_map
    for key in (
        "coherence_tuple_hash",
        "policy_signature",
        "coherence_build_generation_id",
        "coherence_valid_until",
    )
)
legacy_fields = {"snapshot_id", "journal_offset", "pointer_hash", "coherence_tuple_hash"}
legacy_structural_match = False
if token_style in {"kv", "pipe"} and token_map:
    keys = {k for k in token_map.keys() if k in legacy_fields}
    if keys and keys == set(token_map.keys()):
        legacy_structural_match = all(str(token_map.get(k) or "").strip() == str(expected.get(k) or "").strip() for k in keys)

if token_style == "scalar" and provided in accepted_legacy:
    if strict_required and not allow_legacy_anchor:
        emit(
            {
                "ok": False,
                "mode": "strict",
                "error": "coherence_action_token_required",
                "provided": provided,
                "expected": expected,
                "hint": "Run: bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh current --refresh --json and reuse .action_token",
            },
            1,
        )
    if allow_legacy_anchor:
        emit(
            {
                "ok": True,
                "mode": "legacy_override",
                "provided": provided,
                "allow_legacy_anchor": True,
                "truth_anchor": {
                    "snapshot_id": expected["snapshot_id"] or None,
                    "journal_offset": expected["journal_offset"] or None,
                    "pointer_hash": expected["pointer_hash"] or None,
                    "coherence_tuple_hash": expected["coherence_tuple_hash"] or None,
                },
            },
            0,
        )

# action-token path
if token_style in {"kv", "pipe"} and (has_action_fields or strict_required):
    missing = [k for k in required_action_fields if not str(token_map.get(k) or "").strip()]
    if missing:
        emit(
            {
                "ok": False,
                "mode": "strict",
                "error": "action_token_missing_fields",
                "missing_fields": missing,
                "required_fields": required_action_fields,
                "hint": "Run: bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh current --refresh --json and pass .action_token",
            },
            1,
        )

    mismatches: Dict[str, Dict[str, str]] = {}
    for key in required_action_fields:
        got = str(token_map.get(key) or "").strip()
        exp = str(expected.get(key) or "").strip()
        if got != exp:
            mismatches[key] = {"provided": got, "expected": exp}

    valid_until_raw = str(token_map.get("coherence_valid_until") or "").strip()
    valid_until_dt = parse_iso(valid_until_raw)
    if valid_until_dt is None:
        emit(
            {
                "ok": False,
                "mode": "strict",
                "error": "action_token_invalid_valid_until",
                "coherence_valid_until": valid_until_raw,
                "hint": "Use the exact .action_token from continuity_current output",
            },
            1,
        )

    now = now_utc()
    if now > valid_until_dt:
        emit(
            {
                "ok": False,
                "mode": "strict",
                "error": "action_token_expired",
                "now": now.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
                "coherence_valid_until": valid_until_raw,
                "hint": "Refresh continuity_current and retry with the new .action_token",
            },
            1,
        )

    if mismatches:
        emit(
            {
                "ok": False,
                "mode": "strict",
                "error": "action_token_mismatch",
                "mismatches": mismatches,
                "hint": "Continuity tuple drifted (anchor/policy/coherence changed). Refresh and retry with latest .action_token",
            },
            1,
        )

    emit(
        {
            "ok": True,
            "mode": "strict",
            "provided": provided,
            "truth_anchor": {
                "snapshot_id": expected["snapshot_id"] or None,
                "journal_offset": expected["journal_offset"] or None,
                "pointer_hash": expected["pointer_hash"] or None,
                "coherence_tuple_hash": expected["coherence_tuple_hash"] or None,
            },
            "checked_fields": required_action_fields,
            "coherence_valid_until": valid_until_raw,
        },
        0,
    )

# fallback legacy path
legacy_match = provided in accepted_legacy or legacy_structural_match
if legacy_match and (allow_legacy_anchor or not strict_required):
    emit(
        {
            "ok": True,
            "mode": "legacy_override" if strict_required else "legacy",
            "provided": provided,
            "allow_legacy_anchor": allow_legacy_anchor,
            "truth_anchor": {
                "snapshot_id": expected["snapshot_id"] or None,
                "journal_offset": expected["journal_offset"] or None,
                "pointer_hash": expected["pointer_hash"] or None,
                "coherence_tuple_hash": expected["coherence_tuple_hash"] or None,
            },
        },
        0,
    )

error = "truth_anchor_mismatch"
hint = (
    "Run: bash /home/yeqiuqiu/clawd-architect/ops/openclaw/continuity.sh current --refresh --json and reuse .action_token"
)
if strict_required and not allow_legacy_anchor:
    error = "coherence_action_token_required"

emit(
    {
        "ok": False,
        "mode": "strict" if strict_required else "legacy",
        "error": error,
        "provided": provided,
        "strict_required": strict_required,
        "allow_legacy_anchor": allow_legacy_anchor,
        "hint": hint,
    },
    1,
)
PY
