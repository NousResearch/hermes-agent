#!/usr/bin/env python3
"""Regression harness for deterministic GTC incident replay reconstruction.

Focus:
- route/open incident selection resolves correctly from latest surfaces.
- evidence chain reconstructs across operator.actions + queue.task (+ checkpoint links).
- incident-scoped checkpoint expansion excludes unrelated rows; full scope break-glass remains available.
- artifact pack carries typed task artifact roles/manifests from GTC refs/linkage.
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

from strict_required_check_contracts import (
    required_check_provenance,
    strict_required_check_contract,
)

ROOT = Path(__file__).resolve().parents[3]
INIT_DB = ROOT / "ops" / "openclaw" / "continuity" / "init_db.sh"
INGRESS_GUARD = ROOT / "ops" / "openclaw" / "continuity" / "mutator_ingress_guard.sh"
GTC_SYNC = ROOT / "ops" / "openclaw" / "continuity" / "gtc_v2_sync.sh"
GTC_REPLAY = ROOT / "ops" / "openclaw" / "continuity" / "gtc_incident_replay.sh"

CONTRACT = strict_required_check_contract("gtc_incident_replay_verify_gate_posture")
CHECK_ID = CONTRACT.check_id
HARNESS_ID = CONTRACT.harness
SUMMARY_SOURCE = CONTRACT.summary_source
SUMMARY_SCHEMA_VERSION = CONTRACT.summary_schema_version
LEGACY_SCHEMA_VERSION = "gtc.incident_replay.regressions.v1"
REQUIRED_SCENARIO_NAMES = list(CONTRACT.scenario_names)


def _assert(cond: bool, message: str) -> None:
    if not cond:
        raise AssertionError(message)


def _copy_exec(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    dst.chmod(dst.stat().st_mode | stat.S_IXUSR)


def _mk_temp_root() -> tuple[Path, Path]:
    td = Path(tempfile.mkdtemp(prefix="gtc_incident_replay_regressions_"))
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
            "OPENCLAW_INTERNAL_MUTATION_CALLSITE": "harness:check_gtc_incident_replay_regressions",
        }
    )
    return env


def _write(path: Path, content: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _seed_fixture(root: Path) -> dict[str, str]:
    task_id = "autopilot:test_incident_replay"
    route_key = "watchdog.test.incident_replay|task_failed"
    event_key = "task_failed"
    source = "watchdog.test.incident_replay"
    transition_event_id = "tevt_gtc_incident_replay"
    continuity_event_id = "evt_gtc_incident_replay"
    noise_event_id = "evt_gtc_incident_replay_noise"
    created_at = "2026-03-10T09:00:00Z"
    checkpoint_id = "chk_test_incident_replay_001"

    run_log_rel = "out/incident_replay/run.log"
    packet_rel = "state/contracts/reports/autopilot/test_replay_packet.json"
    report_rel = "reports/test_incident_replay.md"
    noise_ref_rel = "state/continuity/latest/noise_event_probe.json"

    run_log_sha = _write(root / run_log_rel, "executor failure trace\n")
    packet_sha = _write(root / packet_rel, '{"schema_version":"autopilot.completion.v1","ok":false,"reason":"unit_test"}\n')
    report_sha = _write(root / report_rel, "# Incident replay fixture\n\nFailure detail.\n")
    _write(root / noise_ref_rel, '{"schema_version":"test.noise.v1","note":"unrelated checkpoint evidence"}\n')

    latest_dir = root / "state" / "continuity" / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    (latest_dir / "latest_pointer.json").write_text(
        json.dumps(
            {
                "checkpoint_id": checkpoint_id,
                "checkpoint_path": "state/continuity/checkpoints/chk_test_incident_replay_001.json",
                "checkpoint_sha256": "deadbeef",
                "generated_at": created_at,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (latest_dir / "verify_last.json").write_text(
        json.dumps(
            {
                "timestamp": created_at,
                "status": "READY",
                "reason": "fixture",
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (latest_dir / "continuity_now_latest.json").write_text(
        json.dumps(
            {
                "generated_at": created_at,
                "verify": {
                    "status": "READY",
                    "timestamp": created_at,
                    "gate_preflight": {
                        "available": True,
                        "strict_autonomy": {
                            "enabled": True,
                            "source": "verify_gate_required_env",
                            "required": True,
                            "override": "disable",
                            "override_denied_if_run": True,
                        },
                        "predicted_gate": {
                            "ready_to_run": False,
                            "predicted_blocker_reason": "strict_autonomy_required_override_denied",
                        },
                    },
                },
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    init_cp = subprocess.run(
        ["bash", str(root / "ops" / "openclaw" / "continuity" / "init_db.sh")],
        text=True,
        capture_output=True,
        check=False,
        env=_env(root),
        timeout=30,
    )
    _assert(init_cp.returncode == 0, f"init_db failed: rc={init_cp.returncode} stderr={(init_cp.stderr or '').strip()}")

    db_path = root / "state" / "continuity" / "continuity_os.sqlite"
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
                "Incident replay fixture task",
                "Replay utility should reconstruct evidence + artifacts",
                "FAILED",
                "executor",
                "codex-executioner",
                1,
                3,
                None,
                None,
                created_at,
                created_at,
            ),
        )

        evidence_ref = "|".join([run_log_rel, packet_rel, report_rel])
        cur.execute(
            """
INSERT INTO task_transitions (
  event_id, task_id, from_status, to_status, actor_role, reason, evidence_ref, created_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
""",
            (
                transition_event_id,
                task_id,
                "RUNNING",
                "FAILED",
                "executor",
                "fixture_failure",
                evidence_ref,
                created_at,
            ),
        )

        cur.executemany(
            """
INSERT INTO task_artifacts (
  artifact_id, task_id, artifact_type, artifact_path, sha256, metadata_json, created_at
) VALUES (?, ?, ?, ?, ?, ?, ?)
""",
            [
                (
                    "tart_incident_packet",
                    task_id,
                    "json",
                    packet_rel,
                    packet_sha,
                    json.dumps(
                        {
                            "completion_packet": True,
                            "source": "delegated_gate_summary",
                            "binding": "completion_packet",
                            "fixture": True,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    created_at,
                ),
                (
                    "tart_incident_run_log",
                    task_id,
                    "run_log",
                    run_log_rel,
                    run_log_sha,
                    None,
                    created_at,
                ),
                (
                    "tart_incident_report",
                    task_id,
                    "markdown",
                    report_rel,
                    report_sha,
                    json.dumps(
                        {
                            "source": "delegated_gate_summary",
                            "binding": "gate_decision",
                            "fixture_report": True,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    created_at,
                ),
            ],
        )

        degraded_gate_summary = {
            "schema_version": "autopilot.delegated_gate_summary.v1",
            "summary_signature": "sig_incident_replay_fixture",
            "queue_reason": "fixture_failure",
            "ingress_classification": "accepted",
            "completion_packet_path": packet_rel,
            "completion_packet_sha256": "0" * 64,
            "decision_path": report_rel,
            "decision_sha256": report_sha,
            "completion_packet_source": "stdout_json",
        }
        cur.execute(
            """
INSERT INTO task_handoff_packets (
  packet_id, task_id, parent_task_id, transition_event_id,
  from_role, to_role, from_status, to_status,
  created_at, evidence_refs_json, gate_metadata_json, task_linkage_json,
  lock_refs_json, next_gate, budget_tokens_used, model_tier, retry_count, failure_signature
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
""",
            (
                "pkt_gtc_incident_replay_fixture",
                task_id,
                None,
                transition_event_id,
                "executor",
                "validator",
                "RUNNING",
                "FAILED",
                created_at,
                json.dumps([packet_rel, report_rel], ensure_ascii=False),
                json.dumps({"gate_summary": degraded_gate_summary}, ensure_ascii=False, sort_keys=True),
                json.dumps({}, ensure_ascii=False),
                json.dumps({}, ensure_ascii=False),
                "continue",
                0,
                "unknown",
                0,
                "sig_incident_replay_fixture",
            ),
        )

        continuity_evidence_ref = "|".join([task_id, packet_rel])
        cur.execute(
            """
INSERT INTO continuity_events (
  event_id, created_at, source, event_key, severity, fingerprint,
  emitted, changed, cooldown_elapsed, suppress_reason, summary,
  evidence_ref, route_key, state_file, metadata_json
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
""",
            (
                continuity_event_id,
                created_at,
                source,
                event_key,
                "critical",
                "fingerprint_incident_replay_fixture",
                1,
                1,
                1,
                None,
                "fixture critical incident for replay",
                continuity_evidence_ref,
                route_key,
                None,
                json.dumps({"fixture": True}, ensure_ascii=False, sort_keys=True),
            ),
        )

        cur.execute(
            """
INSERT INTO continuity_events (
  event_id, created_at, source, event_key, severity, fingerprint,
  emitted, changed, cooldown_elapsed, suppress_reason, summary,
  evidence_ref, route_key, state_file, metadata_json
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
""",
            (
                noise_event_id,
                created_at,
                "watchdog.test.noise",
                "noise_probe",
                "warn",
                "fingerprint_incident_replay_noise",
                1,
                1,
                1,
                None,
                "unrelated warn event sharing checkpoint",
                noise_ref_rel,
                "watchdog.test.noise|noise_probe",
                None,
                json.dumps({"fixture": True, "noise": True}, ensure_ascii=False, sort_keys=True),
            ),
        )

        con.commit()
    finally:
        con.close()

    return {
        "task_id": task_id,
        "route_key": route_key,
        "event_id": continuity_event_id,
        "noise_event_id": noise_event_id,
        "checkpoint_id": checkpoint_id,
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
    try:
        payload = json.loads(stdout) if stdout else {}
    except Exception:
        payload = {"_parse_error": "stdout_not_json", "stdout": stdout[:2000], "stderr": (cp.stderr or "")[:500]}
    if not isinstance(payload, dict):
        payload = {"_parse_error": "payload_not_object", "payload_type": type(payload).__name__}
    return cp.returncode, payload, (cp.stderr or "").strip()


def _run_replay(root: Path, route_key: str, checkpoint_scope: str = "incident") -> tuple[int, dict[str, Any], str]:
    cp = subprocess.run(
        [
            "bash",
            str(GTC_REPLAY),
            "--route-key",
            route_key,
            "--gtc-root",
            str(root / "state" / "gtc-v2"),
            "--db",
            str(root / "state" / "continuity" / "continuity_os.sqlite"),
            "--checkpoint-scope",
            checkpoint_scope,
            "--json",
        ],
        text=True,
        capture_output=True,
        check=False,
        env=_env(root),
        timeout=90,
    )
    stdout = (cp.stdout or "").strip()
    try:
        payload = json.loads(stdout) if stdout else {}
    except Exception:
        payload = {"_parse_error": "stdout_not_json", "stdout": stdout[:3000], "stderr": (cp.stderr or "")[:500]}
    if not isinstance(payload, dict):
        payload = {"_parse_error": "payload_not_object", "payload_type": type(payload).__name__}
    return cp.returncode, payload, (cp.stderr or "").strip()


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
        "results": rows,
    }


def main() -> int:
    if not GTC_REPLAY.exists():
        print(
            json.dumps(
                _error_summary("gtc_incident_replay_missing") | {"path": str(GTC_REPLAY)},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    td, root = _mk_temp_root()
    results: list[dict[str, Any]] = []
    try:
        seed = _seed_fixture(root)

        sync_rc, sync_payload, sync_stderr = _run_sync(root)
        _assert(sync_rc == 0, f"gtc sync failed rc={sync_rc} stderr={sync_stderr}")
        _assert(sync_payload.get("ok") is True, f"gtc sync ok=false payload={sync_payload}")
        gateboard = sync_payload.get("gateboard") if isinstance(sync_payload.get("gateboard"), dict) else {}
        open_incident_count = int(gateboard.get("open_incident_count") or 0)
        _assert(open_incident_count >= 1, f"expected open incident, got payload={sync_payload}")

        replay_rc, replay_payload, replay_stderr = _run_replay(root, seed["route_key"], checkpoint_scope="incident")
        _assert(replay_rc == 0, f"replay failed rc={replay_rc} stderr={replay_stderr}")
        _assert(replay_payload.get("ok") is True, f"replay ok=false payload={replay_payload}")

        incident = replay_payload.get("incident") or {}
        _assert(str(incident.get("route_key") or "") == seed["route_key"], f"unexpected route_key in incident={incident}")
        _append_result(
            results,
            name="route_selection_open_incident",
            expectation="sync/replay selects seeded route from open incident surfaces",
            details={
                "route_key": seed["route_key"],
                "open_incident_count": open_incident_count,
                "incident_id": incident.get("incident_id"),
            },
        )

        latest_paths = sync_payload.get("latest_paths") if isinstance(sync_payload.get("latest_paths"), dict) else {}
        incident_surface_path = root / str(latest_paths.get("incident_replay") or "")
        incident_surface = json.loads(incident_surface_path.read_text(encoding="utf-8")) if incident_surface_path.exists() else {}
        _assert(isinstance(incident_surface, dict), f"incident surface not object: {incident_surface_path}")
        incident_verify = incident_surface.get("verify_gate_preflight") if isinstance(incident_surface.get("verify_gate_preflight"), dict) else {}
        _assert(str(incident_verify.get("mode") or "") == "required", f"incident surface missing verify mode: {incident_verify}")
        _assert(
            str(incident_verify.get("predicted_blocker_reason") or "") == "strict_autonomy_required_override_denied",
            f"incident surface missing predicted blocker: {incident_verify}",
        )

        replay_verify = replay_payload.get("verify_gate_preflight") if isinstance(replay_payload.get("verify_gate_preflight"), dict) else {}
        _assert(str(replay_verify.get("mode") or "") == "required", f"replay bundle missing verify mode: {replay_verify}")
        _assert(
            str(replay_verify.get("predicted_blocker_reason") or "") == "strict_autonomy_required_override_denied",
            f"replay bundle missing predicted blocker: {replay_verify}",
        )
        _append_result(
            results,
            name="verify_gate_preflight_posture_projection",
            expectation="incident surface and replay bundle preserve strict verify-gate predicted blocker posture",
            details={
                "surface_mode": incident_verify.get("mode"),
                "surface_predicted_blocker_reason": incident_verify.get("predicted_blocker_reason"),
                "bundle_mode": replay_verify.get("mode"),
                "bundle_predicted_blocker_reason": replay_verify.get("predicted_blocker_reason"),
            },
        )

        chain = replay_payload.get("evidence_chain") or {}
        connector_counts = chain.get("connector_counts") or {}
        _assert(int(connector_counts.get("operator.actions") or 0) >= 1, f"missing operator.actions in chain: {connector_counts}")
        _assert(int(connector_counts.get("queue.task") or 0) >= 1, f"missing queue.task in chain: {connector_counts}")
        _assert(str(chain.get("checkpoint_scope") or "") == "incident", f"unexpected checkpoint scope={chain}")

        selected_event_ids = {str(x or "") for x in (chain.get("selected_event_ids") or [])}
        _assert(seed["event_id"] in selected_event_ids, f"seed event missing from selected_event_ids={selected_event_ids}")
        _assert(seed["noise_event_id"] not in selected_event_ids, f"noise event leaked into selected_event_ids={selected_event_ids}")

        chain_rows = chain.get("rows") or []

        def _contains_noise(rows: list[Any]) -> bool:
            for row in rows:
                if not isinstance(row, dict):
                    continue
                refs = row.get("refs") or {}
                if not isinstance(refs, dict):
                    continue
                if str(refs.get("continuity_event_id") or "") == seed["noise_event_id"]:
                    return True
            return False

        _assert(not _contains_noise(chain_rows), "incident-scoped replay leaked unrelated checkpoint evidence")
        _append_result(
            results,
            name="incident_scope_checkpoint_filter",
            expectation="incident checkpoint scope excludes unrelated continuity event evidence while preserving connector chain",
            details={
                "checkpoint_scope": chain.get("checkpoint_scope"),
                "selected_event_ids": sorted(selected_event_ids),
                "connector_counts": connector_counts,
            },
        )

        full_rc, full_payload, full_stderr = _run_replay(root, seed["route_key"], checkpoint_scope="full")
        _assert(full_rc == 0, f"full replay failed rc={full_rc} stderr={full_stderr}")
        _assert(full_payload.get("ok") is True, f"full replay ok=false payload={full_payload}")
        full_chain = full_payload.get("evidence_chain") or {}
        _assert(str(full_chain.get("checkpoint_scope") or "") == "full", f"unexpected full checkpoint scope={full_chain}")
        _assert(_contains_noise(full_chain.get("rows") or []), "full checkpoint scope should include unrelated checkpoint evidence")
        _append_result(
            results,
            name="full_scope_break_glass_includes_neighbors",
            expectation="break-glass full checkpoint scope intentionally includes sibling checkpoint evidence",
            details={
                "checkpoint_scope": full_chain.get("checkpoint_scope"),
                "contains_noise_event": True,
            },
        )

        artifact_pack = replay_payload.get("artifact_pack") or {}
        role_counts = artifact_pack.get("role_counts") or {}
        _assert(int(role_counts.get("task_artifact:json") or 0) >= 1, f"missing task_artifact:json in artifact pack: {role_counts}")
        _assert(int(role_counts.get("transition_evidence") or 0) >= 1, f"missing transition_evidence in artifact pack: {role_counts}")

        typed_roles = set(artifact_pack.get("typed_task_roles") or [])
        _assert("task_artifact:json" in typed_roles, f"typed task role missing: {sorted(typed_roles)}")

        packet_entries = [
            row
            for row in (artifact_pack.get("entries") or [])
            if isinstance(row, dict) and str(row.get("role") or "") == "task_artifact:json"
        ]
        _assert(packet_entries, "missing artifact entry for task_artifact:json")
        packet_metadata_samples = packet_entries[0].get("metadata_samples") or []
        _assert(
            any(isinstance(m, dict) and m.get("completion_packet") is True for m in packet_metadata_samples),
            f"expected completion_packet metadata in {packet_metadata_samples}",
        )

        handoff = replay_payload.get("handoff_decisions") if isinstance(replay_payload.get("handoff_decisions"), dict) else {}
        binding_status_counts = handoff.get("binding_status_counts") if isinstance(handoff.get("binding_status_counts"), dict) else {}
        binding_issue_code_counts = handoff.get("binding_issue_code_counts") if isinstance(handoff.get("binding_issue_code_counts"), dict) else {}
        warning_reasons = handoff.get("warning_reasons") if isinstance(handoff.get("warning_reasons"), list) else []
        _assert(int(handoff.get("count") or 0) >= 1, f"expected handoff rows in replay bundle: {handoff}")
        _assert(str(handoff.get("binding_integrity_status") or "") == "degraded", f"expected degraded binding integrity: {handoff}")
        _assert(int(binding_status_counts.get("degraded") or 0) >= 1, f"expected degraded binding status count: {handoff}")
        _assert(int(binding_issue_code_counts.get("binding_sha_mismatch") or 0) >= 1, f"expected binding_sha_mismatch issue count: {handoff}")
        _assert("queue_task_handoff_gate_binding_degraded" in warning_reasons, f"missing warning reason in handoff rollup: {handoff}")

        _append_result(
            results,
            name="typed_artifact_roles_manifested",
            expectation="artifact pack projects typed task artifact roles and downstream handoff binding integrity rollups",
            details={
                "typed_task_roles": sorted(typed_roles),
                "role_counts": role_counts,
                "handoff_binding_integrity_status": handoff.get("binding_integrity_status"),
                "handoff_binding_status_counts": binding_status_counts,
            },
        )

        bundle_paths = replay_payload.get("bundle_paths") or {}
        bundle_json = root / str(bundle_paths.get("json") or "")
        bundle_md = root / str(bundle_paths.get("markdown") or "")
        _assert(bundle_json.exists(), f"bundle json missing: {bundle_json}")
        _assert(bundle_md.exists(), f"bundle md missing: {bundle_md}")
        _append_result(
            results,
            name="bundle_written",
            expectation="incident replay writes deterministic JSON + markdown bundle artifacts",
            details={
                "json": str(bundle_paths.get("json") or ""),
                "markdown": str(bundle_paths.get("markdown") or ""),
            },
        )

        if set(row.get("name") for row in results) != set(REQUIRED_SCENARIO_NAMES):
            raise RuntimeError(
                "scenario contract mismatch for "
                f"{CHECK_ID}: implemented={sorted(row.get('name') for row in results)} expected={sorted(REQUIRED_SCENARIO_NAMES)}"
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
            "route_key": seed["route_key"],
            "incident_id": incident.get("incident_id"),
            "connector_counts": connector_counts,
            "artifact_role_counts": role_counts,
            "bundle_paths": bundle_paths,
            "results": results,
        }

        for row in results:
            print(f"PASS: {row.get('name')}")
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if bool(summary.get("ok")) else 1
    except Exception as exc:
        print(json.dumps(_error_summary(str(exc), results=results), ensure_ascii=False, indent=2, sort_keys=True))
        return 1
    finally:
        shutil.rmtree(td, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
