#!/usr/bin/env python3
"""Focused regressions for delegated gate summary persistence in queue/handoff flows."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import tempfile
from hashlib import sha256
from pathlib import Path
from typing import Any

from fixed_now import now_iso_utc

ROOT = Path(__file__).resolve().parents[3]
INIT_DB = ROOT / "ops" / "openclaw" / "continuity" / "init_db.sh"
QUEUE_ARB = ROOT / "ops" / "openclaw" / "continuity" / "queue_arbitrator.sh"
QUEUE_SYNC = ROOT / "ops" / "openclaw" / "continuity" / "queue_sync_from_autopilot_json.sh"


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


def _now_iso() -> str:
    return now_iso_utc()


def scenario_queue_arbitrator_persists_gate_summary(tmp: Path) -> None:
    db_path = tmp / "continuity.sqlite"
    _init_db(db_path)

    con = _open(db_path)
    cur = con.cursor()
    now = _now_iso()
    cur.execute(
        """
INSERT INTO work_queue (
  task_id, source, title, acceptance_criteria, status, role_required, assigned_agent,
  retry_count, max_retries, last_error_log, cooldown_until, created_at, updated_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
""",
        (
            "autopilot:test_gate_summary",
            "autopilot",
            "Gate summary regression",
            "n/a",
            "RUNNING",
            "planner",
            "tester",
            1,
            3,
            None,
            None,
            now,
            now,
        ),
    )
    con.commit()
    con.close()

    gate_summary = {
        "schema_version": "autopilot.provider_failure_summary.v1",
        "summary_signature": "dgate_4a9ad80575cb7f02f18d",
        "gate_outcome": "REJECTED_RETRY",
        "ingress_classification": "PROVIDER_FAILURE",
        "queue_reason": "autopilot_provider_failure_retry_exhausted",
        "retryable": True,
        "primary_reason": "provider_error: OpenAIResponsesError status=503 service unavailable",
        "provider_failure": {
            "detected": True,
            "provider_family": "openai",
            "classification": "provider_service_unavailable",
            "failure_kind": "service_unavailable",
            "retryable": True,
            "transient": True,
            "status_code": 503,
            "hint": "provider_error: OpenAIResponsesError status=503 service unavailable",
            "signature": "pfail_3ca4f14abecf76eb3f0f",
        },
        "retry_profile": {
            "policy_class": "provider_transient",
            "base_backoff_sec": 180,
            "cap_backoff_sec": 3600,
        },
        "retry_plan": {
            "attempts": 3,
            "next_after_ts": None,
            "retry_exhausted": True,
            "retry_backoff_sec": None,
            "one_shot_retry_budget_consumed": False,
        },
        "source": "step_runtime_exit",
        "step_id": "audit_alignment",
    }
    env = {
        **os.environ,
        "OPENCLAW_INTERNAL_MUTATION": "1",
        "OPENCLAW_INTERNAL_MUTATION_CALLSITE": "check_delegated_gate_summary_queue_regressions.py:queue_arbitrator",
    }
    cp = _run(
        [
            "bash",
            str(QUEUE_ARB),
            "transition",
            "--db",
            str(db_path),
            "--task-id",
            "autopilot:test_gate_summary",
            "--to-status",
            "BLOCKED",
            "--actor-role",
            "planner",
            "--reason",
            "autopilot_provider_failure_retry_exhausted",
            "--gate-summary-json",
            json.dumps(gate_summary, separators=(",", ":"), ensure_ascii=False),
            "--json",
        ],
        env=env,
    )
    _assert(cp.returncode == 0, f"queue_arbitrator transition failed: {cp.stderr}\n{cp.stdout}")
    payload = json.loads(cp.stdout)
    _assert(payload.get("ok") is True, f"transition not ok: {payload}")
    _assert(str(payload.get("handoff_packet_id") or "").startswith("thp_"), f"missing handoff packet id: {payload}")

    con = _open(db_path)
    row = con.execute(
        """
SELECT gate_metadata_json, failure_signature
FROM task_handoff_packets
WHERE task_id = 'autopilot:test_gate_summary'
ORDER BY created_at DESC
LIMIT 1
"""
    ).fetchone()
    con.close()

    _assert(row is not None, "expected handoff packet row")
    gate_meta = json.loads(str(row["gate_metadata_json"] or "{}"))
    summary = gate_meta.get("gate_summary") if isinstance(gate_meta, dict) else None
    _assert(isinstance(summary, dict), f"expected gate_summary object in gate metadata: {gate_meta}")
    _assert(summary.get("summary_signature") == "dgate_4a9ad80575cb7f02f18d", f"unexpected signature: {summary}")
    _assert(summary.get("queue_reason") == "autopilot_provider_failure_retry_exhausted", f"unexpected queue reason: {summary}")
    _assert(isinstance(summary.get("provider_failure"), dict), f"expected provider metadata in gate summary: {summary}")
    _assert(row["failure_signature"] == "dgate_4a9ad80575cb7f02f18d", f"unexpected failure_signature: {row['failure_signature']}")


def scenario_queue_arbitrator_repairs_delegated_artifact_digests(tmp: Path) -> None:
    db_path = tmp / "continuity_arb_digest_repair.sqlite"
    _init_db(db_path)

    packet_path = tmp / "runs" / "20260310T010000Z_audit_alignment.completion_packet.json"
    packet_path.parent.mkdir(parents=True, exist_ok=True)
    packet_path.write_text(json.dumps({"task_id": "autopilot:test_gate_summary_digest_repair", "run_id": "arb_repair"}, ensure_ascii=False) + "\n", encoding="utf-8")
    packet_sha = sha256(packet_path.read_bytes()).hexdigest()

    decision_path = tmp / "state" / "contracts" / "reports" / "autopilot_test_gate_summary_digest_repair" / "arb_repair" / "gate.json"
    decision_path.parent.mkdir(parents=True, exist_ok=True)
    decision_path.write_text(json.dumps({"task_id": "autopilot:test_gate_summary_digest_repair", "gate_outcome": "ACCEPTED"}, ensure_ascii=False) + "\n", encoding="utf-8")
    decision_sha = sha256(decision_path.read_bytes()).hexdigest()

    con = _open(db_path)
    cur = con.cursor()
    now = _now_iso()
    cur.execute(
        """
INSERT INTO work_queue (
  task_id, source, title, acceptance_criteria, status, role_required, assigned_agent,
  retry_count, max_retries, last_error_log, cooldown_until, created_at, updated_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
""",
        (
            "autopilot:test_gate_summary_digest_repair",
            "autopilot",
            "Gate summary digest repair regression",
            "n/a",
            "RUNNING",
            "planner",
            "tester",
            0,
            3,
            None,
            None,
            now,
            now,
        ),
    )
    con.commit()
    con.close()

    gate_summary = {
        "schema_version": "autopilot.delegated_gate_summary.v1",
        "summary_signature": "dgate_arb_digest_repair",
        "queue_reason": "autopilot_step_completed",
        "gate_outcome": "ACCEPTED",
        "step_id": "audit_alignment",
        "decision_path": str(decision_path),
        "decision_sha256": "0" * 64,
        "completion_packet_path": str(packet_path),
        "completion_packet_sha256": "f" * 64,
    }

    env = {
        **os.environ,
        "OPENCLAW_INTERNAL_MUTATION": "1",
        "OPENCLAW_INTERNAL_MUTATION_CALLSITE": "check_delegated_gate_summary_queue_regressions.py:queue_arbitrator_digest_repair",
    }
    cp = _run(
        [
            "bash",
            str(QUEUE_ARB),
            "transition",
            "--db",
            str(db_path),
            "--task-id",
            "autopilot:test_gate_summary_digest_repair",
            "--to-status",
            "REVIEW",
            "--actor-role",
            "planner",
            "--reason",
            "autopilot_step_completed",
            "--gate-summary-json",
            json.dumps(gate_summary, separators=(",", ":"), ensure_ascii=False),
            "--json",
        ],
        env=env,
    )
    _assert(cp.returncode == 0, f"queue_arbitrator digest-repair transition failed: {cp.stderr}\n{cp.stdout}")
    payload = json.loads(cp.stdout or "{}")
    _assert(payload.get("ok") is True, f"expected transition payload ok=true: {payload}")

    repaired_summary = payload.get("gate_summary") if isinstance(payload.get("gate_summary"), dict) else {}
    _assert(repaired_summary.get("completion_packet_sha256") == packet_sha, f"expected repaired packet digest in transition payload: {payload}")
    _assert(repaired_summary.get("decision_sha256") == decision_sha, f"expected repaired decision digest in transition payload: {payload}")

    repair_rows = payload.get("gate_summary_binding_repairs") if isinstance(payload.get("gate_summary_binding_repairs"), list) else []
    repair_fields = sorted(str(item.get("field") or "") for item in repair_rows if isinstance(item, dict))
    _assert(repair_fields == ["completion_packet_sha256", "decision_sha256"], f"expected both digest repairs to be reported: {payload}")

    con = _open(db_path)
    hp = con.execute(
        """
SELECT gate_metadata_json
FROM task_handoff_packets
WHERE task_id = 'autopilot:test_gate_summary_digest_repair'
ORDER BY created_at DESC, rowid DESC
LIMIT 1
"""
    ).fetchone()
    packet_art = con.execute(
        """
SELECT artifact_path, artifact_type, sha256, metadata_json
FROM task_artifacts
WHERE task_id = 'autopilot:test_gate_summary_digest_repair' AND artifact_path = ?
ORDER BY created_at DESC, rowid DESC
LIMIT 1
""",
        (str(packet_path),),
    ).fetchone()
    decision_art = con.execute(
        """
SELECT artifact_path, artifact_type, sha256, metadata_json
FROM task_artifacts
WHERE task_id = 'autopilot:test_gate_summary_digest_repair' AND artifact_path = ?
ORDER BY created_at DESC, rowid DESC
LIMIT 1
""",
        (str(decision_path),),
    ).fetchone()
    con.close()

    _assert(hp is not None, "expected handoff packet in queue_arbitrator digest-repair scenario")
    gate_meta = json.loads(str(hp["gate_metadata_json"] or "{}"))
    persisted_summary = gate_meta.get("gate_summary") if isinstance(gate_meta, dict) else None
    _assert(isinstance(persisted_summary, dict), f"expected gate_summary in handoff metadata: {gate_meta}")
    _assert(persisted_summary.get("completion_packet_sha256") == packet_sha, f"expected repaired packet digest in handoff summary: {persisted_summary}")
    _assert(persisted_summary.get("decision_sha256") == decision_sha, f"expected repaired decision digest in handoff summary: {persisted_summary}")

    _assert(packet_art is not None, "expected completion-packet task_artifacts row at transition time")
    _assert(str(packet_art["sha256"] or "") == packet_sha, f"expected packet sha in transition-time task_artifacts: {dict(packet_art)}")
    packet_meta = json.loads(str(packet_art["metadata_json"] or "{}"))
    _assert(str(packet_meta.get("source") or "") == "delegated_gate_summary", f"expected delegated source metadata for packet artifact: {packet_meta}")
    _assert(str(packet_meta.get("binding") or "") == "completion_packet", f"expected completion_packet binding metadata: {packet_meta}")

    _assert(decision_art is not None, "expected gate-decision task_artifacts row at transition time")
    _assert(str(decision_art["sha256"] or "") == decision_sha, f"expected decision sha in transition-time task_artifacts: {dict(decision_art)}")
    decision_meta = json.loads(str(decision_art["metadata_json"] or "{}"))
    _assert(str(decision_meta.get("source") or "") == "delegated_gate_summary", f"expected delegated source metadata for decision artifact: {decision_meta}")
    _assert(str(decision_meta.get("binding") or "") == "gate_decision", f"expected gate_decision binding metadata: {decision_meta}")


def scenario_queue_arbitrator_drops_unverifiable_delegated_artifact_digests(tmp: Path) -> None:
    db_path = tmp / "continuity_arb_digest_missing.sqlite"
    _init_db(db_path)

    packet_path = tmp / "runs" / "20260310T010000Z_audit_alignment_missing.completion_packet.json"
    decision_path = tmp / "state" / "contracts" / "reports" / "autopilot_test_gate_summary_digest_missing" / "arb_missing" / "gate.json"

    con = _open(db_path)
    cur = con.cursor()
    now = _now_iso()
    cur.execute(
        """
INSERT INTO work_queue (
  task_id, source, title, acceptance_criteria, status, role_required, assigned_agent,
  retry_count, max_retries, last_error_log, cooldown_until, created_at, updated_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
""",
        (
            "autopilot:test_gate_summary_digest_missing",
            "autopilot",
            "Gate summary digest missing regression",
            "n/a",
            "RUNNING",
            "planner",
            "tester",
            0,
            3,
            None,
            None,
            now,
            now,
        ),
    )
    con.commit()
    con.close()

    gate_summary = {
        "schema_version": "autopilot.delegated_gate_summary.v1",
        "summary_signature": "dgate_arb_digest_missing",
        "queue_reason": "autopilot_step_completed",
        "gate_outcome": "ACCEPTED",
        "step_id": "audit_alignment",
        "decision_path": str(decision_path),
        "decision_sha256": "0" * 64,
        "completion_packet_path": str(packet_path),
        "completion_packet_sha256": "f" * 64,
    }

    env = {
        **os.environ,
        "OPENCLAW_INTERNAL_MUTATION": "1",
        "OPENCLAW_INTERNAL_MUTATION_CALLSITE": "check_delegated_gate_summary_queue_regressions.py:queue_arbitrator_digest_missing",
    }
    cp = _run(
        [
            "bash",
            str(QUEUE_ARB),
            "transition",
            "--db",
            str(db_path),
            "--task-id",
            "autopilot:test_gate_summary_digest_missing",
            "--to-status",
            "REVIEW",
            "--actor-role",
            "planner",
            "--reason",
            "autopilot_step_completed",
            "--gate-summary-json",
            json.dumps(gate_summary, separators=(",", ":"), ensure_ascii=False),
            "--json",
        ],
        env=env,
    )
    _assert(cp.returncode == 0, f"queue_arbitrator digest-missing transition failed: {cp.stderr}\n{cp.stdout}")
    payload = json.loads(cp.stdout or "{}")
    _assert(payload.get("ok") is True, f"expected transition payload ok=true: {payload}")

    repaired_summary = payload.get("gate_summary") if isinstance(payload.get("gate_summary"), dict) else {}
    _assert("completion_packet_sha256" not in repaired_summary, f"expected completion digest dropped when file missing: {repaired_summary}")
    _assert("decision_sha256" not in repaired_summary, f"expected decision digest dropped when file missing: {repaired_summary}")

    repair_rows = payload.get("gate_summary_binding_repairs") if isinstance(payload.get("gate_summary_binding_repairs"), list) else []
    by_field = {
        str(item.get("field") or ""): item
        for item in repair_rows
        if isinstance(item, dict)
    }
    _assert(
        "completion_packet_sha256" in by_field and "decision_sha256" in by_field,
        f"expected both digest drops in repair rows: {payload}",
    )
    _assert(
        str((by_field.get("completion_packet_sha256") or {}).get("action") or "") == "dropped_unverifiable_binding",
        f"expected dropped action metadata for completion binding: {repair_rows}",
    )
    _assert(
        str((by_field.get("decision_sha256") or {}).get("action") or "") == "dropped_unverifiable_binding",
        f"expected dropped action metadata for decision binding: {repair_rows}",
    )

    con = _open(db_path)
    hp = con.execute(
        """
SELECT gate_metadata_json
FROM task_handoff_packets
WHERE task_id = 'autopilot:test_gate_summary_digest_missing'
ORDER BY created_at DESC, rowid DESC
LIMIT 1
"""
    ).fetchone()
    art_count = con.execute(
        """
SELECT COUNT(*) AS n
FROM task_artifacts
WHERE task_id = 'autopilot:test_gate_summary_digest_missing'
"""
    ).fetchone()
    con.close()

    _assert(hp is not None, "expected handoff packet in digest-missing scenario")
    gate_meta = json.loads(str(hp["gate_metadata_json"] or "{}"))
    persisted_summary = gate_meta.get("gate_summary") if isinstance(gate_meta, dict) else None
    _assert(isinstance(persisted_summary, dict), f"expected gate_summary in handoff metadata: {gate_meta}")
    _assert("completion_packet_sha256" not in persisted_summary, f"expected dropped packet digest in handoff summary: {persisted_summary}")
    _assert("decision_sha256" not in persisted_summary, f"expected dropped decision digest in handoff summary: {persisted_summary}")
    _assert("completion_packet_path" not in persisted_summary, f"expected dropped packet path in handoff summary: {persisted_summary}")
    _assert("decision_path" not in persisted_summary, f"expected dropped decision path in handoff summary: {persisted_summary}")

    _assert(int(art_count["n"] or 0) == 0, f"expected no delegated task_artifacts projection when bindings are unverifiable: {dict(art_count)}")


def scenario_queue_arbitrator_rejects_invalid_provider_summary(tmp: Path) -> None:
    db_path = tmp / "continuity_invalid.sqlite"
    _init_db(db_path)

    con = _open(db_path)
    cur = con.cursor()
    now = _now_iso()
    cur.execute(
        """
INSERT INTO work_queue (
  task_id, source, title, acceptance_criteria, status, role_required, assigned_agent,
  retry_count, max_retries, last_error_log, cooldown_until, created_at, updated_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
""",
        (
            "autopilot:test_gate_summary_invalid",
            "autopilot",
            "Gate summary invalid regression",
            "n/a",
            "RUNNING",
            "planner",
            "tester",
            1,
            3,
            None,
            None,
            now,
            now,
        ),
    )
    con.commit()
    con.close()

    invalid_summary = {
        "schema_version": "autopilot.provider_failure_summary.v1",
        "queue_reason": "autopilot_provider_failure_retry_exhausted",
        "summary_signature": "not_valid",
    }
    env = {
        **os.environ,
        "OPENCLAW_INTERNAL_MUTATION": "1",
        "OPENCLAW_INTERNAL_MUTATION_CALLSITE": "check_delegated_gate_summary_queue_regressions.py:queue_arbitrator_invalid",
    }
    cp = _run(
        [
            "bash",
            str(QUEUE_ARB),
            "transition",
            "--db",
            str(db_path),
            "--task-id",
            "autopilot:test_gate_summary_invalid",
            "--to-status",
            "BLOCKED",
            "--actor-role",
            "planner",
            "--reason",
            "autopilot_provider_failure_retry_exhausted",
            "--gate-summary-json",
            json.dumps(invalid_summary, separators=(",", ":"), ensure_ascii=False),
            "--json",
        ],
        env=env,
    )
    _assert(cp.returncode != 0, f"expected invalid gate summary rejection, got rc=0: {cp.stdout}")
    payload = json.loads(cp.stdout or "{}")
    _assert(payload.get("error") == "invalid_gate_summary_schema", f"unexpected error payload: {payload}")


def scenario_queue_sync_projects_gate_summary(tmp: Path) -> None:
    db_path = tmp / "continuity_sync.sqlite"
    state_path = tmp / "autopilot_state.json"
    _init_db(db_path)

    packet_path = tmp / "runs" / "20260310T000000Z_audit_alignment.completion_packet.json"
    packet_path.parent.mkdir(parents=True, exist_ok=True)
    packet_path.write_text(json.dumps({"task_id": "autopilot:audit_alignment", "run_id": "run_ok"}, ensure_ascii=False) + "\n", encoding="utf-8")
    packet_sha = sha256(packet_path.read_bytes()).hexdigest()

    decision_path = tmp / "state" / "contracts" / "reports" / "autopilot_audit_alignment" / "run_ok" / "gate.json"
    decision_path.parent.mkdir(parents=True, exist_ok=True)
    decision_path.write_text(json.dumps({"task_id": "autopilot:audit_alignment", "gate_outcome": "ACCEPTED"}, ensure_ascii=False) + "\n", encoding="utf-8")
    decision_sha = sha256(decision_path.read_bytes()).hexdigest()

    def write_state(status: str, queue_reason: str, completion_packet_sha256: str | None = None) -> None:
        state = {
            "paused": False,
            "cycle": 1,
            "max_cycles": 5,
            "repo": {"path": str(ROOT)},
            "steps": [
                {
                    "id": "audit_alignment",
                    "title": "Audit alignment",
                    "kind": "agent",
                    "status": status,
                    "attempts": 0,
                    "max_attempts": 3,
                    "last_started_ts": 1773100000,
                    "last_finished_ts": 1773100060,
                    "delegated_gate_summary": {
                        "schema_version": "autopilot.delegated_gate_summary.v1",
                        "summary_signature": "dgate_sync_projection",
                        "queue_reason": queue_reason,
                        "decision_path": str(decision_path),
                        "completion_packet_path": str(packet_path),
                        "completion_packet_sha256": completion_packet_sha256 or packet_sha,
                    },
                }
            ],
        }
        state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    env = {
        **os.environ,
        "OPENCLAW_CONTINUITY_DB_PATH": str(db_path),
        "OPENCLAW_INTERNAL_MUTATION": "1",
        "OPENCLAW_INTERNAL_MUTATION_CALLSITE": "check_delegated_gate_summary_queue_regressions.py:queue_sync",
    }

    write_state("running", "autopilot_state:running")
    cp1 = _run(["bash", str(QUEUE_SYNC), "--db", str(db_path), "--json", str(state_path)], env=env)
    _assert(cp1.returncode == 0, f"queue_sync first run failed: {cp1.stderr}\n{cp1.stdout}")

    write_state("done", "autopilot_step_completed")
    cp2 = _run(["bash", str(QUEUE_SYNC), "--db", str(db_path), "--json", str(state_path)], env=env)
    _assert(cp2.returncode == 0, f"queue_sync second run failed: {cp2.stderr}\n{cp2.stdout}")

    con = _open(db_path)
    tr = con.execute(
        """
SELECT reason, evidence_ref
FROM task_transitions
WHERE task_id = 'autopilot:audit_alignment'
ORDER BY created_at DESC, rowid DESC
LIMIT 1
"""
    ).fetchone()
    hp = con.execute(
        """
SELECT gate_metadata_json
FROM task_handoff_packets
WHERE task_id = 'autopilot:audit_alignment'
ORDER BY created_at DESC, rowid DESC
LIMIT 1
"""
    ).fetchone()
    art = con.execute(
        """
SELECT artifact_path, sha256
FROM task_artifacts
WHERE task_id = 'autopilot:audit_alignment' AND artifact_path LIKE '%completion_packet.json'
ORDER BY created_at DESC, rowid DESC
LIMIT 1
"""
    ).fetchone()
    decision_art = con.execute(
        """
SELECT artifact_path, sha256
FROM task_artifacts
WHERE task_id = 'autopilot:audit_alignment' AND artifact_path = ?
ORDER BY created_at DESC, rowid DESC
LIMIT 1
""",
        (str(decision_path),),
    ).fetchone()
    con.close()

    _assert(tr is not None, "expected transition row for autopilot:audit_alignment")
    _assert(tr["reason"] == "autopilot_step_completed", f"expected projected queue reason, got: {tr['reason']}")
    _assert(str(decision_path) in str(tr["evidence_ref"] or ""), "expected decision path evidence ref")
    _assert(str(packet_path) in str(tr["evidence_ref"] or ""), "expected completion packet path in transition evidence")

    _assert(hp is not None, "expected handoff packet for projected transition")
    gate_meta = json.loads(str(hp["gate_metadata_json"] or "{}"))
    summary = gate_meta.get("gate_summary") if isinstance(gate_meta, dict) else None
    _assert(isinstance(summary, dict), f"expected gate_summary in queue-sync handoff metadata: {gate_meta}")
    _assert(summary.get("queue_reason") == "autopilot_step_completed", f"expected queue reason in handoff summary: {summary}")
    _assert(summary.get("completion_packet_sha256") == packet_sha, f"expected packet hash in handoff summary: {summary}")
    _assert(summary.get("decision_sha256") == decision_sha, f"expected decision hash in handoff summary: {summary}")

    _assert(art is not None, "expected task_artifacts row for completion packet")
    _assert(str(art["artifact_path"] or "") == str(packet_path), f"unexpected artifact path: {dict(art)}")
    _assert(str(art["sha256"] or "") == packet_sha, f"expected packet hash in task_artifacts: {dict(art)}")

    _assert(decision_art is not None, "expected task_artifacts row for decision artifact")
    _assert(str(decision_art["artifact_path"] or "") == str(decision_path), f"unexpected decision artifact path: {dict(decision_art)}")
    _assert(str(decision_art["sha256"] or "") == decision_sha, f"expected decision hash in task_artifacts: {dict(decision_art)}")


def scenario_queue_sync_repairs_mismatched_completion_packet_digest(tmp: Path) -> None:
    db_path = tmp / "continuity_sync_digest_repair.sqlite"
    state_path = tmp / "autopilot_state_digest_repair.json"
    _init_db(db_path)

    packet_path = tmp / "runs" / "20260310T000000Z_audit_alignment.completion_packet.json"
    packet_path.parent.mkdir(parents=True, exist_ok=True)
    packet_path.write_text(json.dumps({"task_id": "autopilot:audit_alignment", "run_id": "run_repair"}, ensure_ascii=False) + "\n", encoding="utf-8")
    packet_sha = sha256(packet_path.read_bytes()).hexdigest()

    decision_path = tmp / "state" / "contracts" / "reports" / "autopilot_audit_alignment" / "run_repair" / "gate.json"
    decision_path.parent.mkdir(parents=True, exist_ok=True)
    decision_path.write_text(json.dumps({"task_id": "autopilot:audit_alignment", "gate_outcome": "ACCEPTED"}, ensure_ascii=False) + "\n", encoding="utf-8")

    def write_state(status: str, completion_packet_sha256: str) -> None:
        state = {
            "paused": False,
            "cycle": 1,
            "max_cycles": 5,
            "repo": {"path": str(ROOT)},
            "steps": [
                {
                    "id": "audit_alignment",
                    "title": "Audit alignment",
                    "kind": "agent",
                    "status": status,
                    "attempts": 0,
                    "max_attempts": 3,
                    "last_started_ts": 1773100000,
                    "last_finished_ts": 1773100060,
                    "delegated_gate_summary": {
                        "schema_version": "autopilot.delegated_gate_summary.v1",
                        "summary_signature": "dgate_sync_digest_repair",
                        "queue_reason": "autopilot_step_completed",
                        "decision_path": str(decision_path),
                        "completion_packet_path": str(packet_path),
                        "completion_packet_sha256": completion_packet_sha256,
                    },
                }
            ],
        }
        state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    env = {
        **os.environ,
        "OPENCLAW_CONTINUITY_DB_PATH": str(db_path),
        "OPENCLAW_INTERNAL_MUTATION": "1",
        "OPENCLAW_INTERNAL_MUTATION_CALLSITE": "check_delegated_gate_summary_queue_regressions.py:queue_sync_digest_repair",
    }

    write_state("running", packet_sha)
    cp1 = _run(["bash", str(QUEUE_SYNC), "--db", str(db_path), "--json", str(state_path)], env=env)
    _assert(cp1.returncode == 0, f"queue_sync digest-repair pre-run failed: {cp1.stderr}\n{cp1.stdout}")

    write_state("done", "0" * 64)
    cp = _run(["bash", str(QUEUE_SYNC), "--db", str(db_path), "--json", str(state_path)], env=env)
    _assert(cp.returncode == 0, f"queue_sync digest-repair run failed: {cp.stderr}\n{cp.stdout}")
    payload = json.loads(cp.stdout or "{}")
    _assert(payload.get("ok") is True, f"expected digest-repair payload ok=true: {payload}")
    _assert(int(payload.get("gate_summaries_invalid") or 0) >= 1, f"expected digest mismatch issue to be surfaced: {payload}")

    con = _open(db_path)
    hp = con.execute(
        """
SELECT gate_metadata_json
FROM task_handoff_packets
WHERE task_id = 'autopilot:audit_alignment'
ORDER BY created_at DESC, rowid DESC
LIMIT 1
"""
    ).fetchone()
    art = con.execute(
        """
SELECT artifact_path, sha256
FROM task_artifacts
WHERE task_id = 'autopilot:audit_alignment' AND artifact_path = ?
ORDER BY created_at DESC, rowid DESC
LIMIT 1
""",
        (str(packet_path),),
    ).fetchone()
    con.close()

    _assert(hp is not None, "expected handoff packet row in digest-repair scenario")
    gate_meta = json.loads(str(hp["gate_metadata_json"] or "{}"))
    summary = gate_meta.get("gate_summary") if isinstance(gate_meta, dict) else None
    _assert(isinstance(summary, dict), f"expected gate_summary in digest-repair handoff metadata: {gate_meta}")
    _assert(summary.get("completion_packet_sha256") == packet_sha, f"expected repaired completion packet digest in gate summary: {summary}")

    _assert(art is not None, "expected packet artifact row in digest-repair scenario")
    _assert(str(art["sha256"] or "") == packet_sha, f"expected repaired digest to project into task_artifacts: {dict(art)}")


def scenario_queue_sync_drops_unverifiable_delegated_artifact_digests(tmp: Path) -> None:
    db_path = tmp / "continuity_sync_digest_missing.sqlite"
    state_path = tmp / "autopilot_state_digest_missing.json"
    _init_db(db_path)

    packet_path = tmp / "runs" / "20260310T000000Z_audit_alignment_missing.completion_packet.json"
    decision_path = tmp / "state" / "contracts" / "reports" / "autopilot_audit_alignment" / "run_missing" / "gate.json"

    def write_state(status: str, include_summary: bool) -> None:
        step: dict[str, Any] = {
            "id": "audit_alignment",
            "title": "Audit alignment",
            "kind": "agent",
            "status": status,
            "attempts": 0,
            "max_attempts": 3,
            "last_started_ts": 1773100000,
            "last_finished_ts": 1773100060,
        }
        if include_summary:
            step["delegated_gate_summary"] = {
                "schema_version": "autopilot.delegated_gate_summary.v1",
                "summary_signature": "dgate_sync_digest_missing",
                "queue_reason": "autopilot_step_completed",
                "decision_path": str(decision_path),
                "decision_sha256": "0" * 64,
                "completion_packet_path": str(packet_path),
                "completion_packet_sha256": "f" * 64,
            }

        state = {
            "paused": False,
            "cycle": 1,
            "max_cycles": 5,
            "repo": {"path": str(ROOT)},
            "steps": [step],
        }
        state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    env = {
        **os.environ,
        "OPENCLAW_CONTINUITY_DB_PATH": str(db_path),
        "OPENCLAW_INTERNAL_MUTATION": "1",
        "OPENCLAW_INTERNAL_MUTATION_CALLSITE": "check_delegated_gate_summary_queue_regressions.py:queue_sync_digest_missing",
    }

    write_state("running", False)
    cp1 = _run(["bash", str(QUEUE_SYNC), "--db", str(db_path), "--json", str(state_path)], env=env)
    _assert(cp1.returncode == 0, f"queue_sync digest-missing pre-run failed: {cp1.stderr}\n{cp1.stdout}")

    write_state("done", True)
    cp = _run(["bash", str(QUEUE_SYNC), "--db", str(db_path), "--json", str(state_path)], env=env)
    _assert(cp.returncode == 0, f"queue_sync digest-missing run failed: {cp.stderr}\n{cp.stdout}")
    payload = json.loads(cp.stdout or "{}")
    _assert(payload.get("ok") is True, f"expected digest-missing payload ok=true: {payload}")
    _assert(int(payload.get("gate_summaries_invalid") or 0) >= 2, f"expected missing-digest issues to be surfaced: {payload}")

    issue_sample = payload.get("gate_summary_issue_sample") if isinstance(payload.get("gate_summary_issue_sample"), list) else []
    issue_codes = {
        str(item.get("code") or "")
        for item in issue_sample
        if isinstance(item, dict)
    }
    _assert("delegated_completion_packet_path_unverifiable_dropped" in issue_codes, f"expected completion path-drop issue code: {payload}")
    _assert("delegated_decision_path_unverifiable_dropped" in issue_codes, f"expected decision path-drop issue code: {payload}")

    con = _open(db_path)
    hp = con.execute(
        """
SELECT gate_metadata_json
FROM task_handoff_packets
WHERE task_id = 'autopilot:audit_alignment'
ORDER BY created_at DESC, rowid DESC
LIMIT 1
"""
    ).fetchone()
    packet_art = con.execute(
        """
SELECT artifact_path, sha256, metadata_json
FROM task_artifacts
WHERE task_id = 'autopilot:audit_alignment' AND artifact_path = ?
ORDER BY created_at DESC, rowid DESC
LIMIT 1
""",
        (str(packet_path),),
    ).fetchone()
    decision_art = con.execute(
        """
SELECT artifact_path, sha256, metadata_json
FROM task_artifacts
WHERE task_id = 'autopilot:audit_alignment' AND artifact_path = ?
ORDER BY created_at DESC, rowid DESC
LIMIT 1
""",
        (str(decision_path),),
    ).fetchone()
    con.close()

    _assert(hp is not None, "expected handoff packet row in digest-missing scenario")
    gate_meta = json.loads(str(hp["gate_metadata_json"] or "{}"))
    summary = gate_meta.get("gate_summary") if isinstance(gate_meta, dict) else None
    _assert(isinstance(summary, dict), f"expected gate_summary in digest-missing handoff metadata: {gate_meta}")
    _assert("completion_packet_sha256" not in summary, f"expected dropped completion packet digest in gate summary: {summary}")
    _assert("decision_sha256" not in summary, f"expected dropped decision digest in gate summary: {summary}")
    _assert("completion_packet_path" not in summary, f"expected dropped completion packet path in gate summary: {summary}")
    _assert("decision_path" not in summary, f"expected dropped decision path in gate summary: {summary}")

    _assert(packet_art is None, "expected no completion packet artifact ref row when delegated binding path is dropped")
    _assert(decision_art is None, "expected no decision artifact ref row when delegated binding path is dropped")


def scenario_queue_sync_drops_invalid_provider_summary_by_default(tmp: Path) -> None:
    db_path = tmp / "continuity_sync_drop.sqlite"
    state_path = tmp / "autopilot_state_drop.json"
    _init_db(db_path)

    def write_state(status: str, delegated_gate_summary: dict[str, Any] | None) -> None:
        step: dict[str, Any] = {
            "id": "audit_alignment",
            "title": "Audit alignment",
            "kind": "agent",
            "status": status,
            "attempts": 0,
            "max_attempts": 3,
            "last_started_ts": 1773100000,
            "last_finished_ts": 1773100060,
        }
        if isinstance(delegated_gate_summary, dict):
            step["delegated_gate_summary"] = delegated_gate_summary
        state = {
            "paused": False,
            "cycle": 1,
            "max_cycles": 5,
            "repo": {"path": str(ROOT)},
            "steps": [step],
        }
        state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    invalid_provider_summary = {
        "schema_version": "autopilot.provider_failure_summary.v1",
        "queue_reason": "autopilot_provider_failure_retry_exhausted",
        "summary_signature": "bad_signature",
    }

    env = {
        **os.environ,
        "OPENCLAW_CONTINUITY_DB_PATH": str(db_path),
        "OPENCLAW_INTERNAL_MUTATION": "1",
        "OPENCLAW_INTERNAL_MUTATION_CALLSITE": "check_delegated_gate_summary_queue_regressions.py:queue_sync_drop_invalid",
    }

    write_state("running", None)
    cp1 = _run(["bash", str(QUEUE_SYNC), "--db", str(db_path), "--json", str(state_path)], env=env)
    _assert(cp1.returncode == 0, f"queue_sync first run failed: {cp1.stderr}\n{cp1.stdout}")

    write_state("done", invalid_provider_summary)
    cp2 = _run(["bash", str(QUEUE_SYNC), "--db", str(db_path), "--json", str(state_path)], env=env)
    _assert(cp2.returncode == 0, f"queue_sync drop-mode run failed: {cp2.stderr}\n{cp2.stdout}")
    payload2 = json.loads(cp2.stdout)
    _assert(payload2.get("ok") is True, f"queue_sync drop-mode payload not ok: {payload2}")
    _assert(payload2.get("invalid_provider_summary_mode") == "drop", f"unexpected mode payload: {payload2}")
    _assert(int(payload2.get("gate_summaries_invalid") or 0) >= 1, f"expected invalid summary count in payload: {payload2}")

    con = _open(db_path)
    tr = con.execute(
        """
SELECT reason
FROM task_transitions
WHERE task_id = 'autopilot:audit_alignment'
ORDER BY created_at DESC, rowid DESC
LIMIT 1
"""
    ).fetchone()
    hp = con.execute(
        """
SELECT gate_metadata_json
FROM task_handoff_packets
WHERE task_id = 'autopilot:audit_alignment'
ORDER BY created_at DESC, rowid DESC
LIMIT 1
"""
    ).fetchone()
    con.close()

    _assert(tr is not None, "expected transition row in drop-mode scenario")
    _assert(tr["reason"] == "autopilot_state:done", f"expected fallback transition reason after drop: {dict(tr)}")

    _assert(hp is not None, "expected handoff packet row in drop-mode scenario")
    gate_meta = json.loads(str(hp["gate_metadata_json"] or "{}"))
    summary = gate_meta.get("gate_summary") if isinstance(gate_meta, dict) else None
    _assert(summary is None, f"expected dropped invalid provider summary in handoff metadata: {gate_meta}")


def scenario_queue_sync_fail_closes_invalid_provider_summary_when_enabled(tmp: Path) -> None:
    db_path = tmp / "continuity_sync_failclose.sqlite"
    state_path = tmp / "autopilot_state_failclose.json"
    _init_db(db_path)

    state = {
        "paused": False,
        "cycle": 1,
        "max_cycles": 5,
        "repo": {"path": str(ROOT)},
        "steps": [
            {
                "id": "audit_alignment",
                "title": "Audit alignment",
                "kind": "agent",
                "status": "done",
                "attempts": 0,
                "max_attempts": 3,
                "last_started_ts": 1773100000,
                "last_finished_ts": 1773100060,
                "delegated_gate_summary": {
                    "schema_version": "autopilot.provider_failure_summary.v1",
                    "queue_reason": "autopilot_provider_failure_retry_exhausted",
                    "summary_signature": "bad_signature",
                },
            }
        ],
    }
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    env = {
        **os.environ,
        "OPENCLAW_CONTINUITY_DB_PATH": str(db_path),
        "OPENCLAW_INTERNAL_MUTATION": "1",
        "OPENCLAW_INTERNAL_MUTATION_CALLSITE": "check_delegated_gate_summary_queue_regressions.py:queue_sync_failclose_invalid",
    }

    cp = _run(
        [
            "bash",
            str(QUEUE_SYNC),
            "--db",
            str(db_path),
            "--json",
            str(state_path),
            "--invalid-provider-summary-mode",
            "fail_close",
        ],
        env=env,
    )
    _assert(cp.returncode != 0, f"expected fail-close rejection for invalid summary, got rc=0: {cp.stdout}")
    payload = json.loads(cp.stdout or "{}")
    _assert(payload.get("error") == "invalid_provider_failure_summary_schema", f"unexpected fail-close payload: {payload}")
    _assert(payload.get("invalid_provider_summary_mode") == "fail_close", f"missing fail_close mode in payload: {payload}")

    con = _open(db_path)
    queue_rows = con.execute("SELECT COUNT(*) AS n FROM work_queue").fetchone()
    con.close()
    _assert(int(queue_rows["n"] or 0) == 0, "expected fail-close rollback to leave work_queue unchanged")


SCENARIOS = [
    ("queue_arbitrator_persists_gate_summary", scenario_queue_arbitrator_persists_gate_summary),
    ("queue_arbitrator_repairs_delegated_artifact_digests", scenario_queue_arbitrator_repairs_delegated_artifact_digests),
    ("queue_arbitrator_drops_unverifiable_delegated_artifact_digests", scenario_queue_arbitrator_drops_unverifiable_delegated_artifact_digests),
    ("queue_arbitrator_rejects_invalid_provider_summary", scenario_queue_arbitrator_rejects_invalid_provider_summary),
    ("queue_sync_projects_gate_summary", scenario_queue_sync_projects_gate_summary),
    ("queue_sync_repairs_mismatched_completion_packet_digest", scenario_queue_sync_repairs_mismatched_completion_packet_digest),
    ("queue_sync_drops_unverifiable_delegated_artifact_digests", scenario_queue_sync_drops_unverifiable_delegated_artifact_digests),
    ("queue_sync_drops_invalid_provider_summary_by_default", scenario_queue_sync_drops_invalid_provider_summary_by_default),
    ("queue_sync_fail_closes_invalid_provider_summary_when_enabled", scenario_queue_sync_fail_closes_invalid_provider_summary_when_enabled),
]


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="delegated_gate_summary_queue_") as td:
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
