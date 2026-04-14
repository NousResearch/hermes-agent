#!/usr/bin/env python3
"""Regression harness for queue.task artifact-role manifests in GTC refs.

Focus:
- queue.task evidence refs include first-class `task_artifact_manifest` rows.
- projected artifact roles preserve task_artifacts typing (`task_artifact:<type>`).
- metadata and sha hints from task_artifacts survive into GTC refs/linkage.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
GTC_SYNC = ROOT / "ops" / "openclaw" / "continuity" / "gtc_v2_sync.sh"
INIT_DB = ROOT / "ops" / "openclaw" / "continuity" / "init_db.sh"
INGRESS_GUARD = ROOT / "ops" / "openclaw" / "continuity" / "mutator_ingress_guard.sh"


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _copy_exec(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    dst.chmod(dst.stat().st_mode | stat.S_IXUSR)


def _mk_temp_root() -> tuple[Path, Path]:
    td = Path(tempfile.mkdtemp(prefix="gtc_queue_artifact_manifest_"))
    root = td / "root"
    _copy_exec(INIT_DB, root / "ops" / "openclaw" / "continuity" / "init_db.sh")
    _copy_exec(INGRESS_GUARD, root / "ops" / "openclaw" / "continuity" / "mutator_ingress_guard.sh")
    return td, root


def _env(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "OPENCLAW_ROOT": str(root),
            "OPENCLAW_CONTINUITY_DB_PATH": str(root / "state" / "continuity" / "continuity_os.sqlite"),
            "OPENCLAW_GTC_ROOT": str(root / "state" / "gtc-v2"),
            "OPENCLAW_INTERNAL_MUTATION": "1",
            "OPENCLAW_INTERNAL_MUTATION_CALLSITE": "harness:check_gtc_queue_artifact_manifest_regressions",
        }
    )
    return env


def _write(path: Path, content: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _seed_queue_rows(root: Path) -> dict[str, str]:
    task_id = "autopilot:test_artifact_manifest"
    event_id = "tevt_artifact_manifest_seed"
    created_at = "2026-03-10T00:00:00Z"

    run_log_rel = "out/autopilot_artifacts/test/run.log"
    report_rel = "reports/task_manifest.md"
    packet_rel = "state/contracts/reports/autopilot/test_completion_packet.json"

    run_log_sha = _write(root / run_log_rel, "run log payload\n")
    report_sha = _write(root / report_rel, "# Report\n\nTask evidence.\n")
    packet_sha = _write(root / packet_rel, '{"ok":true,"schema_version":"autopilot.completion.v1"}\n')

    db_path = root / "state" / "continuity" / "continuity_os.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    init_cp = subprocess.run(
        ["bash", str(root / "ops" / "openclaw" / "continuity" / "init_db.sh")],
        text=True,
        capture_output=True,
        check=False,
        env=_env(root),
        timeout=30,
    )
    _assert(init_cp.returncode == 0, f"init_db failed: rc={init_cp.returncode} stderr={(init_cp.stderr or '').strip()}")

    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        cur.execute(
            """
INSERT INTO work_queue (
  task_id, source, title, acceptance_criteria, status, role_required,
  assigned_agent, retry_count, max_retries, last_error_log, cooldown_until,
  created_at, updated_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
""",
            (
                task_id,
                "autopilot",
                "Seed artifact manifest test",
                "GTC refs should include typed artifact manifest",
                "DONE",
                "executor",
                "codex-executioner",
                0,
                3,
                None,
                None,
                created_at,
                created_at,
            ),
        )

        evidence_ref = "|".join([run_log_rel, report_rel, packet_rel])
        cur.execute(
            """
INSERT INTO task_transitions (
  event_id, task_id, from_status, to_status, actor_role, reason, evidence_ref, created_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
""",
            (
                event_id,
                task_id,
                "RUNNING",
                "DONE",
                "executor",
                "artifact_manifest_seed",
                evidence_ref,
                created_at,
            ),
        )

        task_artifacts_rows = [
            (
                "tart_manifest_packet",
                task_id,
                "json",
                packet_rel,
                packet_sha,
                json.dumps({"source": "delegated_gate_summary", "completion_packet": True}, ensure_ascii=False, sort_keys=True),
                created_at,
            ),
            (
                "tart_manifest_runlog",
                task_id,
                "run_log",
                run_log_rel,
                run_log_sha,
                None,
                created_at,
            ),
            (
                "tart_manifest_report",
                task_id,
                "markdown",
                report_rel,
                report_sha,
                json.dumps({"source": "task_report"}, ensure_ascii=False, sort_keys=True),
                created_at,
            ),
        ]
        cur.executemany(
            """
INSERT INTO task_artifacts (
  artifact_id, task_id, artifact_type, artifact_path, sha256, metadata_json, created_at
) VALUES (?, ?, ?, ?, ?, ?, ?)
""",
            task_artifacts_rows,
        )
        con.commit()
    finally:
        con.close()

    return {
        "task_id": task_id,
        "event_id": event_id,
        "packet_sha": packet_sha,
        "run_log_sha": run_log_sha,
        "report_sha": report_sha,
    }


def _run_sync(root: Path) -> tuple[int, dict[str, Any], str]:
    cp = subprocess.run(
        ["bash", str(GTC_SYNC), "--skip-schema-gate", "--json"],
        text=True,
        capture_output=True,
        check=False,
        env=_env(root),
        timeout=90,
    )
    stdout = (cp.stdout or "").strip()
    payload: dict[str, Any]
    try:
        payload = json.loads(stdout) if stdout else {}
    except Exception:
        payload = {"_parse_error": "stdout_not_json", "stdout": stdout[:1200], "stderr": (cp.stderr or "")[:400]}
    if not isinstance(payload, dict):
        payload = {"_parse_error": "payload_not_object", "payload_type": type(payload).__name__}
    return cp.returncode, payload, (cp.stderr or "").strip()


def main() -> int:
    td, root = _mk_temp_root()
    try:
        seed = _seed_queue_rows(root)
        rc, payload, stderr = _run_sync(root)

        _assert(rc == 0, f"expected rc=0, got {rc}; stderr={stderr}")
        _assert(payload.get("ok") is True, f"expected sync ok=true, got {payload.get('ok')}")

        db_path = root / "state" / "continuity" / "continuity_os.sqlite"
        con = sqlite3.connect(db_path)
        con.row_factory = sqlite3.Row
        try:
            cur = con.cursor()
            row = cur.execute(
                """
SELECT evidence_id, facts_json, refs_json
FROM gtc_evidence_index
WHERE connector_type = 'queue.task'
ORDER BY monotonic_seq DESC
LIMIT 1
"""
            ).fetchone()
            _assert(row is not None, "expected queue.task evidence row")

            evidence_id = str(row["evidence_id"] or "")
            facts = json.loads(str(row["facts_json"] or "{}"))
            refs = json.loads(str(row["refs_json"] or "{}"))

            _assert(str(refs.get("task_id") or "") == seed["task_id"], f"unexpected task_id in refs: {refs.get('task_id')}")

            manifest = refs.get("task_artifact_manifest") if isinstance(refs.get("task_artifact_manifest"), list) else []
            artifacts = refs.get("artifacts") if isinstance(refs.get("artifacts"), list) else []
            artifact_roles = {str(a.get("role") or "") for a in artifacts if isinstance(a, dict)}

            _assert(len(manifest) >= 3, f"expected >=3 manifest rows, got {len(manifest)}")
            _assert("task_artifact:json" in artifact_roles, f"missing task_artifact:json role: {sorted(artifact_roles)}")
            _assert("task_artifact:run_log" in artifact_roles, f"missing task_artifact:run_log role: {sorted(artifact_roles)}")
            _assert("task_artifact:markdown" in artifact_roles, f"missing task_artifact:markdown role: {sorted(artifact_roles)}")
            _assert("transition_evidence" in artifact_roles, f"missing transition_evidence role: {sorted(artifact_roles)}")

            packet_rows = [
                m
                for m in manifest
                if isinstance(m, dict) and str(m.get("artifact_type") or "") == "json"
            ]
            _assert(packet_rows, "missing json manifest row")
            packet_meta = packet_rows[0].get("metadata") if isinstance(packet_rows[0].get("metadata"), dict) else {}
            _assert(packet_meta.get("completion_packet") is True, f"missing completion_packet metadata in {packet_meta}")

            _assert(int(facts.get("task_artifact_count") or 0) >= 3, f"unexpected task_artifact_count in facts: {facts}")
            _assert(facts.get("has_task_artifact_manifest") is True, f"expected has_task_artifact_manifest=true in facts: {facts}")

            link_row = cur.execute(
                """
SELECT 1
FROM gtc_evidence_artifact
WHERE evidence_id = ? AND role = ?
LIMIT 1
""",
                (evidence_id, "task_artifact:json"),
            ).fetchone()
            _assert(link_row is not None, "expected gtc_evidence_artifact link for task_artifact:json")
        finally:
            con.close()

        print("PASS queue_artifact_manifest_projection")
        print(
            json.dumps(
                {
                    "ok": True,
                    "schema_version": "gtc.queue_artifact_manifest.regressions.v1",
                    "returncode": rc,
                    "task_id": seed["task_id"],
                    "evidence_id": evidence_id,
                    "checks": {
                        "task_artifact_manifest_present": True,
                        "typed_roles_projected": True,
                        "metadata_preserved": True,
                        "artifact_linkage_present": True,
                    },
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "schema_version": "gtc.queue_artifact_manifest.regressions.v1",
                    "error": str(exc),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 1
    finally:
        shutil.rmtree(td, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
