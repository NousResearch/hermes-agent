#!/usr/bin/env python3
"""Contract-bound wrapper for gtc latest schema fail-close regressions.

Purpose:
- run canonical `check_gtc_latest_schema_regressions.py` harness,
- enforce required scenario coverage/min-count contract,
- emit strict required-check provenance metadata for verify_then_resume.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from strict_required_check_contracts import required_check_provenance, strict_required_check_contract

ROOT = Path(__file__).resolve().parents[3]
UPSTREAM_HARNESS = ROOT / "ops" / "openclaw" / "continuity" / "check_gtc_latest_schema_regressions.py"

CONTRACT = strict_required_check_contract("gtc_latest_schema_failclose")
CHECK_ID = CONTRACT.check_id
HARNESS_ID = CONTRACT.harness
SUMMARY_SOURCE = CONTRACT.summary_source
SUMMARY_SCHEMA_VERSION = CONTRACT.summary_schema_version


def _extract_summary_json(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
    if not raw:
        return None

    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass

    for marker in ("\n{", "{"):
        idx = raw.rfind(marker)
        if idx < 0:
            continue
        candidate = raw[idx + 1 :] if marker == "\n{" else raw[idx:]
        try:
            payload = json.loads(candidate)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload

    return None


def _tail(text: str, *, max_lines: int = 40, max_chars: int = 4000) -> str:
    rows = (text or "").splitlines()
    if len(rows) > max_lines:
        rows = rows[-max_lines:]
    trimmed = "\n".join(rows)
    if len(trimmed) > max_chars:
        trimmed = trimmed[-max_chars:]
    return trimmed


def _required_check_provenance() -> dict[str, Any]:
    return required_check_provenance(CHECK_ID)


def main() -> int:
    if not UPSTREAM_HARNESS.exists():
        summary = {
            "ok": False,
            "harness": HARNESS_ID,
            "source": SUMMARY_SOURCE,
            "summary_schema_version": SUMMARY_SCHEMA_VERSION,
            "required_check_provenance": _required_check_provenance(),
            "error": "upstream_harness_missing",
            "upstream_harness": str(UPSTREAM_HARNESS),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 1

    cp = subprocess.run(
        [sys.executable, str(UPSTREAM_HARNESS)],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "OPENCLAW_ROOT": str(ROOT)},
    )

    upstream_summary = _extract_summary_json(cp.stdout or "")
    parse_error = not isinstance(upstream_summary, dict)

    results = []
    total: int | None = None
    failed: int | None = None
    fail_fast: Any = None
    upstream_ok = False

    if isinstance(upstream_summary, dict):
        upstream_ok = bool(upstream_summary.get("ok"))
        rows = upstream_summary.get("results")
        if isinstance(rows, list):
            results = [row for row in rows if isinstance(row, dict)]
        total_raw = upstream_summary.get("total")
        if isinstance(total_raw, int):
            total = total_raw
        failed_raw = upstream_summary.get("failed")
        if isinstance(failed_raw, int):
            failed = failed_raw
        fail_fast = upstream_summary.get("fail_fast")

    if total is None:
        total = len(results)
    if failed is None:
        failed = sum(1 for row in results if row.get("ok") is not True)

    scenario_names = {
        str(row.get("name") or "").strip()
        for row in results
        if str(row.get("name") or "").strip()
    }
    missing_required_scenarios = [name for name in CONTRACT.scenario_names if name not in scenario_names]

    contract_violations: list[dict[str, Any]] = []
    if parse_error:
        contract_violations.append({"error": "upstream_summary_unreadable"})
    if cp.returncode != 0:
        contract_violations.append({"error": "upstream_returncode_nonzero", "upstream_returncode": cp.returncode})
    if not upstream_ok:
        contract_violations.append({"error": "upstream_ok_false"})
    if total < CONTRACT.minimum_result_count:
        contract_violations.append(
            {
                "error": "minimum_result_count_unmet",
                "minimum_result_count": CONTRACT.minimum_result_count,
                "actual_result_count": total,
            }
        )
    if missing_required_scenarios:
        contract_violations.append(
            {
                "error": "missing_required_scenarios",
                "missing_required_scenarios": missing_required_scenarios,
            }
        )

    summary: dict[str, Any] = {
        "ok": not contract_violations,
        "total": total,
        "failed": failed,
        "results": results,
        "harness": HARNESS_ID,
        "source": SUMMARY_SOURCE,
        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
        "required_check_provenance": _required_check_provenance(),
        "upstream_harness": str(UPSTREAM_HARNESS),
        "upstream_returncode": cp.returncode,
        "upstream_ok": upstream_ok,
        "required_scenarios": list(CONTRACT.scenario_names),
        "minimum_result_count": CONTRACT.minimum_result_count,
        "fail_fast": fail_fast,
    }

    if contract_violations:
        summary["contract_violations"] = contract_violations
        summary["upstream_stdout_tail"] = _tail(cp.stdout or "")
        summary["upstream_stderr_tail"] = _tail(cp.stderr or "")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not contract_violations else 1


if __name__ == "__main__":
    raise SystemExit(main())
