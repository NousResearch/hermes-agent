#!/usr/bin/env python3
"""Regression harness: queue cooldown projection + claim gating for autopilot backoff tasks."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from fixed_now import now_ts
from strict_required_check_contracts import (
    required_check_provenance,
    strict_required_check_contract,
)

ROOT = Path(__file__).resolve().parents[3]
QUEUE_SYNC = ROOT / "ops" / "openclaw" / "continuity" / "queue_sync_from_autopilot_json.sh"
QUEUE_ARB = ROOT / "ops" / "openclaw" / "continuity" / "queue_arbitrator.sh"

CONTRACT = strict_required_check_contract("queue_cooldown_authority_regressions")
CHECK_ID = CONTRACT.check_id
HARNESS_ID = CONTRACT.harness
SUMMARY_SOURCE = CONTRACT.summary_source
SUMMARY_SCHEMA_VERSION = CONTRACT.summary_schema_version
LEGACY_SCHEMA_VERSION = "queue.cooldown.authority.regressions.v1"
REQUIRED_SCENARIO_NAMES = list(CONTRACT.scenario_names)


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _run(cmd: list[str], *, env: dict[str, str] | None = None, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True, env=env, cwd=str(cwd or ROOT), check=False)


def _parse_json(cp: subprocess.CompletedProcess[str], *, label: str) -> dict:
    raw = (cp.stdout or "").strip()
    _assert(bool(raw), f"{label}: expected JSON stdout, got empty output (stderr={cp.stderr!r})")
    try:
        payload = json.loads(raw)
    except Exception as exc:
        raise AssertionError(f"{label}: failed to parse JSON stdout: {exc}; raw={raw!r}") from exc
    _assert(isinstance(payload, dict), f"{label}: expected object payload, got {type(payload).__name__}")
    return payload


def _sync_state(state_path: Path, db_path: Path, *, env: dict[str, str]) -> dict:
    cp = _run(
        [
            "bash",
            str(QUEUE_SYNC),
            "--json",
            str(state_path),
            "--db",
            str(db_path),
            "--source",
            "autopilot",
        ],
        env=env,
    )
    _assert(cp.returncode == 0, f"queue_sync failed: rc={cp.returncode} stderr={cp.stderr} stdout={cp.stdout}")
    payload = _parse_json(cp, label="queue_sync")
    _assert(payload.get("ok") is True, f"queue_sync not ok: {payload}")
    return payload


def _query_one(db_path: Path, sql: str, params: tuple = ()):  # noqa: ANN001
    con = sqlite3.connect(str(db_path))
    try:
        row = con.execute(sql, params).fetchone()
    finally:
        con.close()
    return row


def scenario_cooldown_projection_and_claim_gate(tmp: Path) -> dict[str, Any]:
    db_path = tmp / "continuity.sqlite"
    state_path = tmp / "hl_terminal_v1.json"

    clock_now_ts = now_ts()
    future_ts = clock_now_ts + 300

    state = {
        "paused": False,
        "cycle": 1,
        "max_cycles": 3,
        "repo": {"path": str(ROOT)},
        "active": None,
        "steps": [
            {
                "id": "sync_spec_context",
                "title": "Sync spec context",
                "kind": "shell",
                "status": "queued",
                "attempts": 0,
                "max_attempts": 3,
                "cmd": "echo noop",
                "next_after_ts": future_ts,
                "last_error": None,
            }
        ],
    }
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    mut_env = {
        **os.environ,
        "OPENCLAW_ROOT": str(ROOT),
        "OPENCLAW_INTERNAL_MUTATION": "1",
        "OPENCLAW_INTERNAL_MUTATION_CALLSITE": "check_queue_cooldown_authority_regressions.py",
        "OPENCLAW_AUTOPILOT_FIXED_NOW_TS": str(clock_now_ts),
    }

    _sync_state(state_path, db_path, env=mut_env)

    row = _query_one(
        db_path,
        "SELECT status, cooldown_until FROM work_queue WHERE task_id = ?",
        ("autopilot:sync_spec_context",),
    )
    _assert(row is not None, "missing autopilot task row after queue_sync")
    _assert(str(row[0]) == "QUEUED", f"unexpected status after queue_sync: {row}")
    cooldown_until = str(row[1] or "").strip()
    _assert(bool(cooldown_until), f"expected cooldown_until to be projected for future next_after_ts: {row}")

    ready_cp = _run(["bash", str(QUEUE_ARB), "ready-list", "--db", str(db_path), "--json"], env=mut_env)
    _assert(ready_cp.returncode == 0, f"ready-list failed: {ready_cp.stderr}")
    ready_payload = _parse_json(ready_cp, label="ready-list")
    ready_ids = {str(item.get("task_id") or "") for item in (ready_payload.get("items") or []) if isinstance(item, dict)}
    _assert("autopilot:sync_spec_context" not in ready_ids, f"cooldown task should be excluded from ready-list: {ready_payload}")

    claim_cp = _run(
        [
            "bash",
            str(QUEUE_ARB),
            "claim",
            "--db",
            str(db_path),
            "--agent",
            "queue_cooldown_regression",
            "--actor-role",
            "planner",
            "--task-id",
            "autopilot:sync_spec_context",
            "--json",
        ],
        env=mut_env,
    )
    _assert(claim_cp.returncode == 1, f"expected deferred claim rc=1 during cooldown, got rc={claim_cp.returncode}")
    claim_payload = _parse_json(claim_cp, label="claim_cooldown")
    _assert(claim_payload.get("ok") is False, f"expected claim failure during cooldown: {claim_payload}")
    _assert(claim_payload.get("error") == "no_claimable_task", f"unexpected claim error: {claim_payload}")
    skipped = claim_payload.get("skipped") if isinstance(claim_payload.get("skipped"), list) else []
    first = skipped[0] if skipped and isinstance(skipped[0], dict) else {}
    _assert(first.get("reason") == "cooldown_active", f"expected cooldown_active skip reason: {claim_payload}")
    retry_after_sec = int(first.get("retry_after_sec") or 0)
    _assert(retry_after_sec > 0, f"expected positive retry_after_sec for cooldown skip: {claim_payload}")

    state["steps"][0]["next_after_ts"] = clock_now_ts - 5
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _sync_state(state_path, db_path, env=mut_env)

    row_after = _query_one(
        db_path,
        "SELECT status, cooldown_until FROM work_queue WHERE task_id = ?",
        ("autopilot:sync_spec_context",),
    )
    _assert(row_after is not None, "missing task row after cooldown clear sync")
    _assert(str(row_after[0]) == "QUEUED", f"unexpected status after cooldown clear: {row_after}")
    _assert(not str(row_after[1] or "").strip(), f"expected cooldown_until cleared when next_after is due/past: {row_after}")

    claim_ready_cp = _run(
        [
            "bash",
            str(QUEUE_ARB),
            "claim",
            "--db",
            str(db_path),
            "--agent",
            "queue_cooldown_regression",
            "--actor-role",
            "planner",
            "--task-id",
            "autopilot:sync_spec_context",
            "--json",
        ],
        env=mut_env,
    )
    _assert(claim_ready_cp.returncode == 0, f"expected claim success after cooldown elapsed: {claim_ready_cp.stderr}\n{claim_ready_cp.stdout}")
    claim_ready_payload = _parse_json(claim_ready_cp, label="claim_ready")
    _assert(claim_ready_payload.get("ok") is True, f"expected successful claim payload: {claim_ready_payload}")
    claimed = claim_ready_payload.get("claimed") if isinstance(claim_ready_payload.get("claimed"), dict) else {}
    _assert(claimed.get("task_id") == "autopilot:sync_spec_context", f"unexpected claimed task: {claim_ready_payload}")

    return {
        "task_id": "autopilot:sync_spec_context",
        "projected_cooldown_until": cooldown_until,
        "deferred_retry_after_sec": retry_after_sec,
        "claimed_task_id": claimed.get("task_id"),
    }


def scenario_fixed_now_clock_authority(tmp: Path) -> dict[str, Any]:
    """OPENCLAW_AUTOPILOT_FIXED_NOW_TS must gate cooldown checks independent of host wall clock."""
    db_path = tmp / "continuity_fixed_now.sqlite"
    state_path = tmp / "hl_terminal_fixed_now.json"

    fixed_now_ts = 1700000000
    future_ts = fixed_now_ts + 300

    state = {
        "paused": False,
        "cycle": 1,
        "max_cycles": 3,
        "repo": {"path": str(ROOT)},
        "active": None,
        "steps": [
            {
                "id": "sync_spec_context",
                "title": "Sync spec context",
                "kind": "shell",
                "status": "queued",
                "attempts": 0,
                "max_attempts": 3,
                "cmd": "echo noop",
                "next_after_ts": future_ts,
                "last_error": None,
            }
        ],
    }
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    mut_env = {
        **os.environ,
        "OPENCLAW_ROOT": str(ROOT),
        "OPENCLAW_INTERNAL_MUTATION": "1",
        "OPENCLAW_INTERNAL_MUTATION_CALLSITE": "check_queue_cooldown_authority_regressions.py:fixed_now_authority",
        "OPENCLAW_AUTOPILOT_FIXED_NOW_TS": str(fixed_now_ts),
    }

    _sync_state(state_path, db_path, env=mut_env)

    ready_cp = _run(["bash", str(QUEUE_ARB), "ready-list", "--db", str(db_path), "--json"], env=mut_env)
    _assert(ready_cp.returncode == 0, f"ready-list failed: {ready_cp.stderr}")
    ready_payload = _parse_json(ready_cp, label="ready-list_fixed_now")
    ready_ids = {str(item.get("task_id") or "") for item in (ready_payload.get("items") or []) if isinstance(item, dict)}
    _assert("autopilot:sync_spec_context" not in ready_ids, f"fixed-now cooldown task should not be ready: {ready_payload}")

    claim_cp = _run(
        [
            "bash",
            str(QUEUE_ARB),
            "claim",
            "--db",
            str(db_path),
            "--agent",
            "queue_cooldown_regression_fixed_now",
            "--actor-role",
            "planner",
            "--task-id",
            "autopilot:sync_spec_context",
            "--json",
        ],
        env=mut_env,
    )
    _assert(claim_cp.returncode == 1, f"expected deferred claim under fixed-now cooldown, got rc={claim_cp.returncode}")
    claim_payload = _parse_json(claim_cp, label="claim_cooldown_fixed_now")
    _assert(claim_payload.get("ok") is False, f"expected claim failure under fixed-now cooldown: {claim_payload}")
    skipped = claim_payload.get("skipped") if isinstance(claim_payload.get("skipped"), list) else []
    first = skipped[0] if skipped and isinstance(skipped[0], dict) else {}
    _assert(first.get("reason") == "cooldown_active", f"expected cooldown_active under fixed-now cooldown: {claim_payload}")

    retry_after_sec = int(first.get("retry_after_sec") or 0)
    _assert(retry_after_sec > 0, f"expected positive retry_after_sec under fixed-now cooldown: {claim_payload}")

    return {
        "fixed_now_ts": fixed_now_ts,
        "deferred_retry_after_sec": retry_after_sec,
        "task_id": "autopilot:sync_spec_context",
    }


SCENARIOS = [
    ("cooldown_projection_and_claim_gate", scenario_cooldown_projection_and_claim_gate),
    ("fixed_now_clock_authority", scenario_fixed_now_clock_authority),
]


def _append_result(
    results: list[dict[str, Any]],
    *,
    name: str,
    expectation: str,
    details: dict[str, Any] | None = None,
) -> None:
    row: dict[str, Any] = {
        "name": name,
        "ok": True,
        "expectation": expectation,
    }
    if isinstance(details, dict) and details:
        row["details"] = details
    results.append(row)


def _required_check_provenance() -> dict[str, object]:
    return required_check_provenance(CHECK_ID)


def _error_summary(error: str, *, results: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    rows = list(results or [])
    return {
        "ok": False,
        "check_id": CHECK_ID,
        "harness": HARNESS_ID,
        "source": SUMMARY_SOURCE,
        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
        "schema_version": LEGACY_SCHEMA_VERSION,
        "required_check_provenance": _required_check_provenance(),
        "error": error,
        "total": len(rows),
        "passed": sum(1 for row in rows if bool(row.get("ok"))),
        "failed": sum(1 for row in rows if not bool(row.get("ok"))),
        "results": rows,
    }


def main() -> int:
    if not QUEUE_SYNC.exists() or not QUEUE_ARB.exists():
        missing = []
        if not QUEUE_SYNC.exists():
            missing.append(str(QUEUE_SYNC))
        if not QUEUE_ARB.exists():
            missing.append(str(QUEUE_ARB))
        print(
            json.dumps(
                _error_summary("queue_runtime_scripts_missing")
                | {
                    "missing": missing,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    results: list[dict[str, Any]] = []
    try:
        with tempfile.TemporaryDirectory(prefix="queue_cooldown_authority_") as td:
            tmp = Path(td)
            for name, fn in SCENARIOS:
                details = fn(tmp)
                _append_result(
                    results,
                    name=name,
                    expectation="queue cooldown projection and claim gating stay fail-closed until cooldown elapses",
                    details=details,
                )

        implemented_names = sorted(str(row.get("name") or "") for row in results)
        if implemented_names != sorted(REQUIRED_SCENARIO_NAMES):
            raise RuntimeError(
                "scenario contract mismatch for "
                f"{CHECK_ID}: implemented={implemented_names} expected={sorted(REQUIRED_SCENARIO_NAMES)}"
            )

        summary = {
            "ok": all(bool(row.get("ok")) for row in results),
            "check_id": CHECK_ID,
            "harness": HARNESS_ID,
            "source": SUMMARY_SOURCE,
            "summary_schema_version": SUMMARY_SCHEMA_VERSION,
            "schema_version": LEGACY_SCHEMA_VERSION,
            "required_check_provenance": _required_check_provenance(),
            "total": len(results),
            "passed": sum(1 for row in results if bool(row.get("ok"))),
            "failed": sum(1 for row in results if not bool(row.get("ok"))),
            "results": results,
        }
        for row in results:
            print(f"PASS: {row.get('name')}")
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if bool(summary.get("ok")) else 1
    except Exception as exc:
        print(json.dumps(_error_summary(str(exc), results=results), ensure_ascii=False, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
