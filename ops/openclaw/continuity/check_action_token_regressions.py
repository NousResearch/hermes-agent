#!/usr/bin/env python3
"""Local regression harness for action_token / policy freshness durability edges.

Covers debt cases:
- token expiry
- mismatch edge cases
- stale policy/coherence without anchor movement
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

from fixed_now import now_ts, ts_to_iso_utc

ROOT = Path(__file__).resolve().parents[3]
GUARD = ROOT / "ops" / "openclaw" / "continuity" / "truth_anchor_guard.sh"

REQUIRED_FIELDS = [
    "snapshot_id",
    "journal_offset",
    "pointer_hash",
    "coherence_tuple_hash",
    "policy_signature",
    "coherence_build_generation_id",
    "coherence_valid_until",
]
ANCHOR_FIELDS = ["snapshot_id", "journal_offset", "pointer_hash"]
BASE_NOW_TS = now_ts()


def iso_utc(delta_sec: int) -> str:
    return ts_to_iso_utc(BASE_NOW_TS + int(delta_sec))


def encode_action_token(parts: dict[str, str], include_fields: list[str] | None = None) -> str:
    fields = include_fields or REQUIRED_FIELDS
    rows: list[str] = []
    for key in fields:
        txt = str(parts.get(key, "") or "").strip()
        if not txt:
            continue
        rows.append(f"{key}={quote(txt, safe='')}")
    return ";".join(rows)


def run_guard(current_payload: dict[str, Any], token: str, *, fixed_now_ts: int | None = None) -> tuple[int, dict[str, Any], str]:
    with tempfile.TemporaryDirectory(prefix="action_token_regressions_") as td:
        current_path = Path(td) / "current.json"
        current_path.write_text(json.dumps(current_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        env = {**os.environ, "OPENCLAW_AUTOPILOT_FIXED_NOW_TS": str(int(fixed_now_ts if fixed_now_ts is not None else BASE_NOW_TS))}
        cp = subprocess.run(
            [
                "bash",
                str(GUARD),
                "--truth-anchor",
                token,
                "--current-path",
                str(current_path),
                "--no-refresh",
                "--json",
            ],
            text=True,
            capture_output=True,
            check=False,
            env=env,
        )

    stdout = (cp.stdout or "").strip()
    try:
        payload = json.loads(stdout) if stdout else {}
    except Exception:
        payload = {"_parse_error": "stdout_not_json", "stdout": stdout, "stderr": (cp.stderr or "").strip()}
    return cp.returncode, payload, (cp.stderr or "").strip()


def base_current(*, tuple_hash: str, policy_signature: str, generation: str, valid_until: str) -> dict[str, Any]:
    return {
        "truth_anchor": {
            "snapshot_id": "snapshot-20260310T000000Z",
            "journal_offset": "8421",
            "pointer_hash": "ptr-sha-abc123",
        },
        "coherence": {
            "tuple_hash": tuple_hash,
            "policy": {"signature": policy_signature},
            "build_generation_id": generation,
            "valid_until": valid_until,
        },
    }


def token_parts_from_current(cur: dict[str, Any]) -> dict[str, str]:
    anchor = cur.get("truth_anchor") or {}
    coherence = cur.get("coherence") or {}
    return {
        "snapshot_id": str(anchor.get("snapshot_id") or ""),
        "journal_offset": str(anchor.get("journal_offset") or ""),
        "pointer_hash": str(anchor.get("pointer_hash") or ""),
        "coherence_tuple_hash": str(coherence.get("tuple_hash") or ""),
        "policy_signature": str(((coherence.get("policy") or {}).get("signature") or "")),
        "coherence_build_generation_id": str(coherence.get("build_generation_id") or ""),
        "coherence_valid_until": str(coherence.get("valid_until") or ""),
    }


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def scenario_baseline_strict_ok() -> None:
    current = base_current(
        tuple_hash="tuple-live",
        policy_signature="policy-live",
        generation="gen-live",
        valid_until=iso_utc(600),
    )
    token = encode_action_token(token_parts_from_current(current))
    rc, payload, _stderr = run_guard(current, token)
    assert_true(rc == 0, f"expected rc=0, got rc={rc}")
    assert_true(payload.get("ok") is True, f"expected ok=true, got payload={payload}")
    assert_true(payload.get("mode") == "strict", f"expected mode=strict, got payload={payload}")


def scenario_token_expiry_rejected() -> None:
    current = base_current(
        tuple_hash="tuple-live",
        policy_signature="policy-live",
        generation="gen-live",
        valid_until=iso_utc(600),
    )
    expired = token_parts_from_current(current)
    expired["coherence_valid_until"] = iso_utc(-120)
    token = encode_action_token(expired)

    rc, payload, _stderr = run_guard(current, token)
    assert_true(rc == 1, f"expected rc=1, got rc={rc}")
    assert_true(payload.get("error") == "action_token_expired", f"expected action_token_expired, got payload={payload}")


def scenario_fixed_now_clock_authority_expiry() -> None:
    current = base_current(
        tuple_hash="tuple-live",
        policy_signature="policy-live",
        generation="gen-live",
        valid_until=iso_utc(600),
    )
    token = encode_action_token(token_parts_from_current(current))

    rc, payload, _stderr = run_guard(current, token, fixed_now_ts=BASE_NOW_TS + 900)
    assert_true(rc == 1, f"expected rc=1 when fixed-now exceeds valid_until, got rc={rc}")
    assert_true(payload.get("error") == "action_token_expired", f"expected action_token_expired, got payload={payload}")


def scenario_mismatch_policy_signature_only() -> None:
    current = base_current(
        tuple_hash="tuple-live",
        policy_signature="policy-live",
        generation="gen-live",
        valid_until=iso_utc(600),
    )
    altered = token_parts_from_current(current)
    altered["policy_signature"] = "policy-stale"
    token = encode_action_token(altered)

    rc, payload, _stderr = run_guard(current, token)
    assert_true(rc == 1, f"expected rc=1, got rc={rc}")
    assert_true(payload.get("error") == "action_token_mismatch", f"expected action_token_mismatch, got payload={payload}")

    mismatches = payload.get("mismatches") or {}
    assert_true(set(mismatches.keys()) == {"policy_signature"}, f"expected only policy_signature mismatch, got {mismatches}")


def scenario_missing_required_field() -> None:
    current = base_current(
        tuple_hash="tuple-live",
        policy_signature="policy-live",
        generation="gen-live",
        valid_until=iso_utc(600),
    )
    parts = token_parts_from_current(current)
    include_fields = [x for x in REQUIRED_FIELDS if x != "coherence_build_generation_id"]
    token = encode_action_token(parts, include_fields=include_fields)

    rc, payload, _stderr = run_guard(current, token)
    assert_true(rc == 1, f"expected rc=1, got rc={rc}")
    assert_true(payload.get("error") == "action_token_missing_fields", f"expected action_token_missing_fields, got payload={payload}")
    missing = payload.get("missing_fields") or []
    assert_true("coherence_build_generation_id" in missing, f"expected missing coherence_build_generation_id, got {missing}")


def scenario_stale_coherence_without_anchor_movement() -> None:
    """Anchor tuple is unchanged, coherence policy/generation drifts => reject stale token."""
    current = base_current(
        tuple_hash="tuple-new",
        policy_signature="policy-new",
        generation="gen-new",
        valid_until=iso_utc(600),
    )

    stale_parts = token_parts_from_current(current)
    stale_parts["coherence_tuple_hash"] = "tuple-old"
    stale_parts["policy_signature"] = "policy-old"
    stale_parts["coherence_build_generation_id"] = "gen-old"
    token = encode_action_token(stale_parts)

    rc, payload, _stderr = run_guard(current, token)
    assert_true(rc == 1, f"expected rc=1, got rc={rc}")
    assert_true(payload.get("error") == "action_token_mismatch", f"expected action_token_mismatch, got payload={payload}")

    mismatches = payload.get("mismatches") or {}
    mismatch_keys = set(mismatches.keys())
    expected = {"coherence_tuple_hash", "policy_signature", "coherence_build_generation_id"}
    assert_true(expected.issubset(mismatch_keys), f"expected coherence mismatch keys {expected}, got {mismatch_keys}")
    assert_true(not (set(ANCHOR_FIELDS) & mismatch_keys), f"anchor unexpectedly mismatched: {mismatch_keys}")


SCENARIOS: list[tuple[str, Callable[[], None]]] = [
    ("baseline_strict_ok", scenario_baseline_strict_ok),
    ("token_expiry_rejected", scenario_token_expiry_rejected),
    ("fixed_now_clock_authority_expiry", scenario_fixed_now_clock_authority_expiry),
    ("mismatch_policy_signature_only", scenario_mismatch_policy_signature_only),
    ("missing_required_field", scenario_missing_required_field),
    ("stale_coherence_without_anchor_movement", scenario_stale_coherence_without_anchor_movement),
]


def main() -> int:
    if not GUARD.exists():
        print(json.dumps({"ok": False, "error": "truth_anchor_guard_missing", "path": str(GUARD)}, indent=2))
        return 2

    results: list[dict[str, Any]] = []
    failed = 0

    for name, fn in SCENARIOS:
        try:
            fn()
            results.append({"name": name, "ok": True})
            print(f"PASS {name}")
        except Exception as exc:
            failed += 1
            results.append({"name": name, "ok": False, "error": str(exc)})
            print(f"FAIL {name}: {exc}")

    summary = {
        "ok": failed == 0,
        "total": len(results),
        "failed": failed,
        "results": results,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
