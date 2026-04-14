#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import jsonschema

ROOT = Path(__file__).resolve().parents[3]
SURFACE = ROOT / "ops" / "openclaw" / "continuity" / "orchestrator_contract_v1_surface.py"

DEFAULT_OUTPUT = (
    ROOT
    / "state"
    / "continuity"
    / "latest"
    / "evidence"
    / "ex_06_orchestrator_contract_bridge_packet_2026-04-03.json"
)

SCHEMAS = {
    "plan": ROOT / "docs" / "ops" / "schemas" / "orchestrator_plan.schema.json",
    "run": ROOT / "docs" / "ops" / "schemas" / "orchestrator_run.schema.json",
    "event": ROOT / "docs" / "ops" / "schemas" / "orchestrator_event_stream.schema.json",
    "replay": ROOT / "docs" / "ops" / "schemas" / "orchestrator_replay_resync.schema.json",
    "bridge": ROOT / "docs" / "ops" / "schemas" / "orchestrator_contract_bridge_packet.schema.json",
}


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _invoke_surface(command: str, request_path: Path, *, state_db: Path, retention_max_events: int) -> dict[str, Any]:
    cp = subprocess.run(
        [
            sys.executable,
            str(SURFACE),
            command,
            "--request",
            str(request_path),
            "--state-db",
            str(state_db),
            "--retention-max-events",
            str(retention_max_events),
        ],
        text=True,
        capture_output=True,
        check=False,
        cwd=str(ROOT),
    )
    if cp.returncode != 0:
        raise RuntimeError(
            f"surface_command_failed:{command}:returncode={cp.returncode}:stdout={cp.stdout.strip()}:stderr={cp.stderr.strip()}"
        )
    return json.loads(cp.stdout)


def _validate(payload: dict[str, Any], schema_path: Path) -> None:
    schema = _read_json(schema_path)
    jsonschema.validate(payload, schema)


def _build_packet() -> dict[str, Any]:
    generated_at = _now_iso()

    with tempfile.TemporaryDirectory(prefix="ex06_bridge_") as td:
        tmp = Path(td)
        state_db = tmp / "runtime.sqlite"

        plan_request = {
            "request_id": "req:orchestrator:plan:ex06:bridge:2026-04-03",
            "generated_at": generated_at,
            "snapshot_ref": {
                "snapshot_id": "snapshot:ex06:bridge:2026-04-03T07:00:00Z:1",
                "manifest_id": "manifest:ex06:bridge:2026-04-03:1",
            },
            "artifacts": [
                {
                    "artifact_kind": "execution_state",
                    "artifact_version": "v1",
                    "parameters": {"lane": "core_execution"},
                }
            ],
        }
        plan_request_path = tmp / "plan_request.json"
        _write_json(plan_request_path, plan_request)
        plan_packet = _invoke_surface("plan", plan_request_path, state_db=state_db, retention_max_events=1)

        run_request = {
            "request_id": "req:orchestrator:run:ex06:bridge:2026-04-03",
            "generated_at": generated_at,
            "plan_id": plan_packet["response"]["plan_id"],
            "idempotency_key": "idem:ex06:bridge:2026-04-03:run:1",
            "dry_run": True,
            "dispatch": {"enabled": False, "target_profile": None},
            "evaluation_gate": {
                "required": True,
                "gate_policy_ref": "docs/ops/evaluation_gated_promotion_loop_v1.md#pre_run_gate",
                "canary_mode": "shadow",
                "attestation_refs": ["state/continuity/latest/mutation_attestation_latest.json"],
            },
            "output_artifacts": [
                {
                    "artifact_kind": "execution_state",
                    "snapshot_id": plan_packet["snapshot_ref"]["snapshot_id"],
                    "manifest_id": plan_packet["snapshot_ref"]["manifest_id"],
                }
            ],
        }
        run_request_path = tmp / "run_request.json"
        _write_json(run_request_path, run_request)
        run_packet = _invoke_surface("run", run_request_path, state_db=state_db, retention_max_events=1)

        event_request = {
            "stream_id": "stream:ex06:contract_bridge",
            "since_seq": 0,
            "event": {
                "event_id": "event:ex06:contract_bridge:0001",
                "type": "artifacts.run.accepted",
                "severity": "info",
                "entity_ref": run_packet["response"]["run_id"],
                "payload_ref": "state/continuity/latest/evidence/ex_06_orchestrator_contract_bridge_packet_2026-04-03.json#orchestrator_run_packet",
                "correlation_id": "flow://ex06/bridge/2026-04-03",
                "idempotency_key": run_packet["idempotency_key"],
                "dedupe_key": "dedupe:ex06:contract_bridge:0001",
            },
        }
        event_request_path = tmp / "event_request.json"
        _write_json(event_request_path, event_request)
        event_packet = _invoke_surface("emit-event", event_request_path, state_db=state_db, retention_max_events=1)

        replay_request = {
            "request_id": "req:orchestrator:replay:ex06:bridge:2026-04-03",
            "generated_at": generated_at,
            "request": {
                "reason": "cursor_gap",
                "from_event_seq": 0,
                "to_event_seq": event_packet["event"]["event_seq"],
                "last_known_snapshot_id": plan_packet["snapshot_ref"]["snapshot_id"],
                "last_applied_event_seq": 0,
                "max_events": 10,
            },
        }
        replay_request_path = tmp / "replay_request.json"
        _write_json(replay_request_path, replay_request)
        replay_packet = _invoke_surface("replay-resync", replay_request_path, state_db=state_db, retention_max_events=1)

    bridge_packet = {
        "schema_version": "clawd.orchestrator.contract_bridge_packet.v1",
        "contract_version": "orchestrator_api_contract_v1",
        "generated_at": generated_at,
        "slice_id": "EX-06",
        "integration_runner": "ops/openclaw/continuity/orchestrator_contract_bridge_packet_v1.py",
        "bridge_rules": {
            "idempotency_key_family": "shared_retry_key",
            "run_conflict_detector": "canonical_request_hash",
            "event_ordering_semantics": "strict_event_seq",
            "replay_retention_cliff_action": "snapshot_reseed_required",
        },
        "orchestrator_plan_packet": plan_packet,
        "orchestrator_run_packet": run_packet,
        "orchestrator_event_stream_packet": event_packet,
        "orchestrator_replay_resync_packet": replay_packet,
        "invariants": {
            "run_event_idempotency_key_match": run_packet["idempotency_key"] == event_packet["event"].get("idempotency_key"),
            "event_seq_matches_cursor": event_packet["event"]["event_seq"] == event_packet["cursor"]["next_after_seq"],
            "replay_retention_cliff_detected": replay_packet["response"]["status"] == "snapshot_reseed_required",
            "replay_requires_snapshot_reseed": replay_packet["response"]["status"] == "snapshot_reseed_required",
            "replay_actions_include_reseed": {"resolve_snapshot", "rebuild_plan"}.issubset(
                set(replay_packet["response"].get("actions") or [])
            ),
        },
    }

    _validate(plan_packet, SCHEMAS["plan"])
    _validate(run_packet, SCHEMAS["run"])
    _validate(event_packet, SCHEMAS["event"])
    _validate(replay_packet, SCHEMAS["replay"])
    _validate(bridge_packet, SCHEMAS["bridge"])

    return bridge_packet


def _parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build EX-06 orchestrator contract bridge packet v1")
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output JSON path")
    ap.add_argument("--json", action="store_true", help="Print JSON summary")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])
    output = Path(args.output).resolve()
    packet = _build_packet()
    _write_json(output, packet)

    summary = {
        "ok": True,
        "schema_version": packet["schema_version"],
        "output": str(output),
        "generated_at": packet["generated_at"],
        "replay_status": packet["orchestrator_replay_resync_packet"]["response"]["status"],
    }
    if args.json:
        sys.stdout.write(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    else:
        sys.stdout.write(f"wrote {output}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
