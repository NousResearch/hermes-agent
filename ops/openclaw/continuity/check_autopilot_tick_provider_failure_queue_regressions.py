#!/usr/bin/env python3
"""Deterministic integration harness for tick provider-failure -> queue/handoff persistence."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TICK_SCRIPT = ROOT / "ops" / "autopilot" / "bin" / "hl_autopilot_tick.sh"
INIT_DB = ROOT / "ops" / "openclaw" / "continuity" / "init_db.sh"

import sys

sys.path.insert(0, str(ROOT / "src"))
from walletdb.provider_failure import (  # noqa: E402
    PROVIDER_FAILURE_SUMMARY_SCHEMA_VERSION,
    validate_provider_failure_summary,
)

PROVIDER_SERVICE_UNAVAILABLE_LOG = (
    "provider_error: OpenAIResponsesError status=503 service unavailable upstream timeout\n"
    "request_id=req_harness_should_be_redacted\n"
)
PLAIN_UNKNOWN_PROVIDER_LOG = "Codex request failed (unknown_error).\n"


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _run(cmd: list[str], *, env: dict[str, str] | None = None, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True, env=env, cwd=str(cwd or ROOT), check=False)


def _init_db(db_path: Path) -> None:
    env = {**os.environ, **{"OPENCLAW_CONTINUITY_DB_PATH": str(db_path)}}
    cp = _run(["bash", str(INIT_DB)], env=env)
    _assert(cp.returncode == 0, f"init_db failed: {cp.stderr}")


def _open(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    return con


def _scenario_tick_nonzero_provider_exit_persists_blocked_handoff(
    tmp: Path,
    *,
    log_text: str,
    db_stem: str,
    expected_hint: str | None = None,
    forbidden_substrings: list[str] | None = None,
) -> None:
    db_path = tmp / f"{db_stem}.sqlite"
    _init_db(db_path)

    con = _open(db_path)
    now = "2026-03-10T00:00:00Z"
    con.execute(
        """
INSERT INTO work_queue (
  task_id, source, title, acceptance_criteria, status, role_required, assigned_agent,
  retry_count, max_retries, last_error_log, cooldown_until, created_at, updated_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
""",
        (
            "autopilot:audit_alignment",
            "autopilot",
            "Audit alignment",
            "n/a",
            "RUNNING",
            "planner",
            "tick_provider_failure_regression",
            1,
            2,
            None,
            None,
            now,
            now,
        ),
    )
    con.commit()
    con.close()

    fake_root = tmp / "sandbox" / "ops" / "autopilot"
    tick_copy = fake_root / "bin" / "hl_autopilot_tick.sh"
    state_path = fake_root / "state" / "hl_terminal_v1.json"
    runs_dir = fake_root / "runs"
    tick_copy.parent.mkdir(parents=True, exist_ok=True)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(TICK_SCRIPT, tick_copy)

    run_tag = "20260310T130000Z_audit_alignment"
    log_path = runs_dir / f"{run_tag}.log"
    exit_path = runs_dir / f"{run_tag}.exit"
    log_path.write_text(log_text, encoding="utf-8")
    exit_path.write_text("7\n", encoding="utf-8")

    state = {
        "paused": False,
        "cycle": 1,
        "max_cycles": 3,
        "repo": {"path": str(ROOT)},
        "steps": [
            {
                "id": "audit_alignment",
                "title": "Audit alignment",
                "kind": "agent",
                "status": "running",
                "attempts": 1,
                "max_attempts": 2,
                "last_started_ts": 1773100000,
                "last_finished_ts": None,
            }
        ],
        "active": {
            "pid": 999999,
            "step_id": "audit_alignment",
            "log_path": str(log_path),
            "start_ts": 1773100000,
            "exit_code_path": str(exit_path),
            "delegated_contract_enabled": False,
        },
    }
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    env = {
        **os.environ,
        "OPENCLAW_ROOT": str(ROOT),
        "OPENCLAW_CONTINUITY_DB_PATH": str(db_path),
        "OPENCLAW_AUTOPILOT_USE_QUEUE_ARBITRATOR": "1",
        "OPENCLAW_AUTOPILOT_QUEUE_AGENT": "tick_provider_failure_regression",
        "OPENCLAW_INTERNAL_MUTATION": "1",
        "OPENCLAW_INTERNAL_MUTATION_CALLSITE": "check_autopilot_tick_provider_failure_queue_regressions.py:tick",
    }
    cp = _run(["bash", str(tick_copy)], env=env, cwd=tmp)
    _assert(cp.returncode == 0, f"tick script failed: {cp.stderr}\n{cp.stdout}")

    updated_state = json.loads(state_path.read_text(encoding="utf-8"))
    step = (updated_state.get("steps") or [None])[0]
    _assert(isinstance(step, dict), f"missing step in updated state: {updated_state}")
    _assert(step.get("status") == "blocked", f"expected blocked step after retry exhaustion: {step}")

    step_last_error = str(step.get("last_error") or "")
    if expected_hint is not None:
        _assert(step_last_error == expected_hint, f"unexpected step last_error hint: {step_last_error}")

    summary = step.get("delegated_gate_summary") if isinstance(step.get("delegated_gate_summary"), dict) else None
    _assert(isinstance(summary, dict), f"expected provider gate summary in state: {step}")
    _assert(summary.get("schema_version") == PROVIDER_FAILURE_SUMMARY_SCHEMA_VERSION, f"unexpected schema version: {summary}")
    verdict = validate_provider_failure_summary(summary, strict=True)
    _assert(bool(verdict.get("ok") is True), f"invalid provider summary schema in state: {verdict}")
    _assert(summary.get("queue_reason") == "autopilot_provider_failure_retry_exhausted", f"unexpected queue reason in state summary: {summary}")

    con = _open(db_path)
    tr = con.execute(
        """
SELECT reason, to_status
FROM task_transitions
WHERE task_id = 'autopilot:audit_alignment'
ORDER BY created_at DESC, rowid DESC
LIMIT 1
"""
    ).fetchone()
    hp = con.execute(
        """
SELECT from_role, to_role, failure_signature, gate_metadata_json
FROM task_handoff_packets
WHERE task_id = 'autopilot:audit_alignment'
ORDER BY created_at DESC, rowid DESC
LIMIT 1
"""
    ).fetchone()
    queue_row = con.execute(
        """
SELECT last_error_log
FROM work_queue
WHERE task_id = 'autopilot:audit_alignment'
LIMIT 1
"""
    ).fetchone()
    con.close()

    _assert(tr is not None, "expected transition row for autopilot:audit_alignment")
    _assert(tr["to_status"] == "BLOCKED", f"expected BLOCKED transition status: {dict(tr)}")
    _assert(tr["reason"] == "autopilot_provider_failure_retry_exhausted", f"unexpected transition reason: {dict(tr)}")

    _assert(hp is not None, "expected handoff packet row for BLOCKED transition")
    _assert(hp["from_role"] == "planner", f"unexpected handoff from_role: {dict(hp)}")
    _assert(hp["to_role"] == "sre_watchdog", f"unexpected handoff to_role: {dict(hp)}")
    _assert(queue_row is not None, "expected work_queue row for autopilot:audit_alignment")
    gate_meta = json.loads(str(hp["gate_metadata_json"] or "{}"))
    persisted_summary = gate_meta.get("gate_summary") if isinstance(gate_meta, dict) else None
    _assert(isinstance(persisted_summary, dict), f"expected gate_summary in handoff metadata: {gate_meta}")
    persisted_verdict = validate_provider_failure_summary(persisted_summary, strict=True)
    _assert(bool(persisted_verdict.get("ok") is True), f"invalid persisted provider summary schema: {persisted_verdict}")
    _assert(
        persisted_summary.get("queue_reason") == "autopilot_provider_failure_retry_exhausted",
        f"unexpected persisted queue reason: {persisted_summary}",
    )
    _assert(
        hp["failure_signature"] == persisted_summary.get("summary_signature"),
        f"expected failure_signature to match summary signature: {dict(hp)}",
    )

    queue_last_error = str(queue_row["last_error_log"] or "")
    if expected_hint is not None and queue_last_error:
        _assert(queue_last_error == expected_hint, f"unexpected queue last_error_log hint: {queue_last_error}")

    if forbidden_substrings:
        primary_reason = str(persisted_summary.get("primary_reason") or "")
        for needle in forbidden_substrings:
            _assert(needle not in step_last_error, f"step.last_error leaked '{needle}': {step_last_error}")
            _assert(needle not in queue_last_error, f"work_queue.last_error_log leaked '{needle}': {queue_last_error}")
            _assert(needle not in primary_reason, f"gate_summary.primary_reason leaked '{needle}': {primary_reason}")


def scenario_tick_nonzero_provider_exit_persists_blocked_handoff(tmp: Path) -> None:
    _scenario_tick_nonzero_provider_exit_persists_blocked_handoff(
        tmp,
        log_text=PROVIDER_SERVICE_UNAVAILABLE_LOG,
        db_stem="continuity_service_unavailable",
        forbidden_substrings=["req_harness_should_be_redacted"],
    )


def scenario_tick_plain_unknown_error_is_sanitized_across_state_and_queue_surfaces(tmp: Path) -> None:
    _scenario_tick_nonzero_provider_exit_persists_blocked_handoff(
        tmp,
        log_text=PLAIN_UNKNOWN_PROVIDER_LOG,
        db_stem="continuity_plain_unknown_error",
        expected_hint="stream_error_code=unknown_error",
        forbidden_substrings=["Codex request failed (unknown_error)."],
    )


SCENARIOS = [
    (
        "tick_nonzero_provider_exit_persists_blocked_handoff",
        scenario_tick_nonzero_provider_exit_persists_blocked_handoff,
    ),
    (
        "tick_plain_unknown_error_is_sanitized_across_state_and_queue_surfaces",
        scenario_tick_plain_unknown_error_is_sanitized_across_state_and_queue_surfaces,
    ),
]


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="autopilot_tick_provider_failure_queue_") as td:
        tmp = Path(td)
        passed = 0
        for name, fn in SCENARIOS:
            fn(tmp)
            passed += 1
            print(f"PASS: {name}")
    print(f"SUMMARY: {passed}/{len(SCENARIOS)} scenarios passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
