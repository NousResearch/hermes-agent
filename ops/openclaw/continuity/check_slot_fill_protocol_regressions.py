#!/usr/bin/env python3
"""Regression harness for subagent slot-fill protocol governance.

Focus:
- healthy protocol/workflow fixtures remain green.
- required protocol snippet drift fails closed.
- workflow linkage drift remains warning-only (non-blocking) for operator ergonomics.
- continuity dispatcher keeps routing `slot-fill-check` to canonical check script.
"""

from __future__ import annotations

import json
import os
import shutil
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
SLOT_FILL_CHECK = ROOT / "ops" / "openclaw" / "continuity" / "check_slot_fill_protocol.sh"
DISPATCHER_SRC = ROOT / "ops" / "openclaw" / "continuity.sh"
SLOT_FILL_SCHEMA_SRC = ROOT / "ops" / "openclaw" / "architecture" / "schemas" / "slot_fill_protocol_check.schema.json"

CONTRACT = strict_required_check_contract("slot_fill_protocol_contract_regressions")
CHECK_ID = CONTRACT.check_id
HARNESS_ID = CONTRACT.harness
SUMMARY_SCHEMA_VERSION = CONTRACT.summary_schema_version
SUMMARY_SOURCE = CONTRACT.summary_source
REQUIRED_SCENARIO_NAMES = list(CONTRACT.scenario_names)
LEGACY_SCHEMA_VERSION = "slot_fill_protocol.regressions.v1"


def _assert(cond: bool, message: str) -> None:
    if not cond:
        raise AssertionError(message)


def _copy_exec(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    dst.chmod(dst.stat().st_mode | stat.S_IXUSR)

    if src == SLOT_FILL_CHECK and SLOT_FILL_SCHEMA_SRC.exists():
        schema_dst = dst.parent.parent / "architecture" / "schemas" / SLOT_FILL_SCHEMA_SRC.name
        schema_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SLOT_FILL_SCHEMA_SRC, schema_dst)


def _write_protocol(
    path: Path,
    *,
    include_quick_checklist: bool = True,
    include_execution_tuple_fields: bool = True,
) -> None:
    lines = [
        "# Slot fill protocol",
        "",
        "## main lane orchestration-only for non-trivial slices",
        "Use worker_lane=subagent_default for non-trivial slices; main_session_tiny_exception is tiny-only with delegation_basis.",
        "",
        "## Spawn-before-speak invariant",
        "Always call `sessions_spawn` first when a slot opens.",
        "",
        "## Narration-only acknowledgment",
        "Never emit narration-only acknowledgments before spawn.",
        "",
        "## If spawn blocked",
        "Emit blocker with explicit reason and immediate next action.",
        "",
        "## Delegation trigger rules (explicit)",
        "delegate by default for non-trivial scope/risk; tiny exceptions must cite main_session_tiny_exception + delegation_basis.",
        "",
        "## Stale-worker / closeout-bundle discipline",
        "Record stale_worker_decision and closeout_bundle_ref for stale recovery or completion closeout.",
    ]
    if include_execution_tuple_fields:
        lines.extend(
            [
                "",
                "## Required reporting fields",
                "- execution_mode",
                "- worker_lane",
                "- model_selection",
                "- delegation_basis",
                "- stale_worker_decision",
                "- closeout_bundle_ref",
            ]
        )
    if include_quick_checklist:
        lines.extend(["", "## Quick checklist", "- choose slice", "- spawn", "- report"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_workflow(
    path: Path,
    *,
    include_protocol_reference: bool = True,
    include_spawn_before_speak: bool = True,
    include_execute_modes: bool = True,
    include_execution_tuple_fields: bool = True,
) -> None:
    rows = ["# Workflow"]
    if include_protocol_reference:
        rows.append("Canonical protocol: docs/ops/subagent_slot_fill_protocol_v1.md")
    else:
        rows.append("Canonical protocol: docs/ops/another_protocol.md")

    if include_spawn_before_speak:
        rows.append("Non-negotiable: spawn-before-speak when a slot frees up.")
    else:
        rows.append("Use judgment for operator acknowledgments.")

    if include_execute_modes:
        rows.append("Execution modes: EXECUTE_NOW or PLAN_ONLY.")
    rows.append("main lane orchestration-only for non-trivial slices.")
    rows.append("Delegation trigger rules (explicit): use main_session_tiny_exception only when tiny and include delegation_basis.")
    rows.append("Stale-worker / closeout-bundle discipline with stale_worker_decision + closeout_bundle_ref.")

    if include_execution_tuple_fields:
        rows.append(
            "Execution tuple: execution_mode, worker_lane, model_selection, delegation_basis, stale_worker_decision, closeout_bundle_ref."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _prepare_fixture_root() -> tuple[Path, Path]:
    td = Path(tempfile.mkdtemp(prefix="slot_fill_protocol_regressions_"))
    root = td / "root"

    _copy_exec(SLOT_FILL_CHECK, root / "ops" / "openclaw" / "continuity" / "check_slot_fill_protocol.sh")
    _copy_exec(DISPATCHER_SRC, root / "ops" / "openclaw" / "continuity.sh")
    _write_protocol(root / "docs" / "ops" / "subagent_slot_fill_protocol_v1.md")
    _write_workflow(root / "WORKFLOW_AUTO.md")
    return td, root


def _run_slot_fill_check(root: Path) -> tuple[int, dict[str, Any], str]:
    cp = subprocess.run(
        ["bash", str(root / "ops" / "openclaw" / "continuity" / "check_slot_fill_protocol.sh"), "--json"],
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
        cwd=str(root),
        env={**os.environ, "OPENCLAW_ROOT": str(root)},
    )
    stdout = (cp.stdout or "").strip()
    try:
        payload = json.loads(stdout) if stdout else {}
    except Exception:
        payload = {
            "_parse_error": "stdout_not_json",
            "stdout": stdout[:2000],
            "stderr": (cp.stderr or "")[:500],
        }
    if not isinstance(payload, dict):
        payload = {"_parse_error": "payload_not_object", "payload_type": type(payload).__name__}
    return cp.returncode, payload, (cp.stderr or "").strip()


def _run_dispatcher_slot_fill(root: Path) -> tuple[int, dict[str, Any], str]:
    cp = subprocess.run(
        ["bash", str(root / "ops" / "openclaw" / "continuity.sh"), "slot-fill-check", "--json"],
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
        cwd=str(root),
        env={**os.environ, "OPENCLAW_ROOT": str(root)},
    )
    stdout = (cp.stdout or "").strip()
    try:
        payload = json.loads(stdout) if stdout else {}
    except Exception:
        payload = {
            "_parse_error": "stdout_not_json",
            "stdout": stdout[:2000],
            "stderr": (cp.stderr or "")[:500],
        }
    if not isinstance(payload, dict):
        payload = {"_parse_error": "payload_not_object", "payload_type": type(payload).__name__}
    return cp.returncode, payload, (cp.stderr or "").strip()


def _find_check(payload: dict[str, Any], name: str) -> dict[str, Any] | None:
    checks = payload.get("checks")
    if not isinstance(checks, list):
        return None
    for row in checks:
        if isinstance(row, dict) and str(row.get("name") or "") == name:
            return row
    return None


def _required_check_provenance() -> dict[str, object]:
    return required_check_provenance(CHECK_ID)


def _error_summary(error: str) -> dict[str, Any]:
    return {
        "ok": False,
        "check_id": CHECK_ID,
        "harness": HARNESS_ID,
        "source": SUMMARY_SOURCE,
        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
        "schema_version": LEGACY_SCHEMA_VERSION,
        "required_check_provenance": _required_check_provenance(),
        "error": error,
        "total": 0,
        "passed": 0,
        "results": [],
    }


def main() -> int:
    if not SLOT_FILL_CHECK.exists():
        summary = _error_summary("slot_fill_check_missing")
        summary["path"] = str(SLOT_FILL_CHECK)
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 2

    if not DISPATCHER_SRC.exists():
        summary = _error_summary("continuity_dispatcher_missing")
        summary["path"] = str(DISPATCHER_SRC)
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 2

    td, root = _prepare_fixture_root()
    protocol_path = root / "docs" / "ops" / "subagent_slot_fill_protocol_v1.md"
    workflow_path = root / "WORKFLOW_AUTO.md"

    try:
        results: list[dict[str, Any]] = []

        # Scenario 1: healthy fixture remains green.
        name = "healthy_fixture_ok"
        rc, payload, stderr = _run_slot_fill_check(root)
        _assert(rc == 0, f"healthy fixture expected rc=0 got rc={rc} stderr={stderr}")
        _assert(payload.get("ok") is True, f"healthy fixture expected ok=true payload={payload}")
        _assert(int(payload.get("critical_failures") or 0) == 0, f"critical failures expected 0 payload={payload}")
        _assert(int(payload.get("warn_failures") or 0) == 0, f"warn failures expected 0 payload={payload}")
        contract_info = payload.get("contract") if isinstance(payload.get("contract"), dict) else {}
        _assert(contract_info.get("state_valid") is True, f"summary contract should be schema-valid payload={payload}")
        _assert(
            str(contract_info.get("schema_path") or "").endswith("slot_fill_protocol_check.schema.json"),
            f"unexpected summary contract schema path payload={payload}",
        )
        results.append(
            {
                "name": name,
                "ok": True,
                "expectation": "healthy protocol/workflow fixture remains green",
                "returncode": rc,
                "critical_failures": int(payload.get("critical_failures") or 0),
                "warn_failures": int(payload.get("warn_failures") or 0),
            }
        )

        # Scenario 2: required protocol snippet drift fails closed.
        name = "missing_required_protocol_snippet_failclose"
        _write_protocol(protocol_path, include_quick_checklist=False)

        rc, payload, stderr = _run_slot_fill_check(root)
        _assert(rc != 0, f"missing protocol snippet should fail rc!=0 stderr={stderr}")
        _assert(payload.get("ok") is False, f"missing protocol snippet should set ok=false payload={payload}")
        snippet_check = _find_check(payload, "protocol_required_snippets") or {}
        _assert(snippet_check.get("ok") is False, f"protocol_required_snippets should fail payload={snippet_check}")
        missing = ((snippet_check.get("details") or {}).get("missing") or [])
        _assert("Quick checklist" in missing, f"missing quick checklist not surfaced details={snippet_check}")
        results.append(
            {
                "name": name,
                "ok": True,
                "expectation": "required protocol snippet drift fails closed",
                "returncode": rc,
                "missing_snippets": missing,
            }
        )

        # Reset protocol before workflow scenarios.
        _write_protocol(protocol_path, include_quick_checklist=True)

        # Scenario 3: workflow execution-mode/tuple drift fails closed.
        name = "workflow_execution_tuple_drift_failclose"
        _write_workflow(
            workflow_path,
            include_protocol_reference=True,
            include_spawn_before_speak=True,
            include_execute_modes=False,
            include_execution_tuple_fields=False,
        )

        rc, payload, stderr = _run_slot_fill_check(root)
        _assert(rc != 0, f"workflow execution tuple drift should fail rc!=0 stderr={stderr}")
        _assert(payload.get("ok") is False, f"workflow execution tuple drift should set ok=false payload={payload}")
        mode_check = _find_check(payload, "workflow_declares_execute_now_plan_only") or {}
        tuple_check = _find_check(payload, "workflow_declares_execution_tuple_fields") or {}
        _assert(mode_check.get("ok") is False, f"workflow execution-mode check should fail payload={mode_check}")
        _assert(tuple_check.get("ok") is False, f"workflow execution tuple check should fail payload={tuple_check}")
        results.append(
            {
                "name": name,
                "ok": True,
                "expectation": "workflow execution mode + tuple drift fails closed",
                "returncode": rc,
                "failed_checks": [
                    str(mode_check.get("name") or "workflow_declares_execute_now_plan_only"),
                    str(tuple_check.get("name") or "workflow_declares_execution_tuple_fields"),
                ],
            }
        )

        # Scenario 4: workflow reference drift remains warning-only.
        name = "workflow_reference_drift_warn_only"
        _write_workflow(workflow_path, include_protocol_reference=False, include_spawn_before_speak=True)

        rc, payload, stderr = _run_slot_fill_check(root)
        _assert(rc == 0, f"workflow reference drift should remain rc=0 stderr={stderr}")
        _assert(payload.get("ok") is True, f"workflow reference drift should keep ok=true payload={payload}")
        reference_check = _find_check(payload, "workflow_references_slot_fill_protocol") or {}
        _assert(reference_check.get("ok") is False, f"workflow reference check should fail payload={reference_check}")
        _assert(
            int(payload.get("warn_failures") or 0) >= 1,
            f"workflow reference drift should increment warn_failures payload={payload}",
        )
        results.append(
            {
                "name": name,
                "ok": True,
                "expectation": "workflow reference drift remains warning-only",
                "returncode": rc,
                "warn_failures": int(payload.get("warn_failures") or 0),
            }
        )

        # Reset workflow before dispatcher scenario.
        _write_workflow(workflow_path, include_protocol_reference=True, include_spawn_before_speak=True)

        # Scenario 5: continuity dispatcher routes slot-fill-check command.
        name = "dispatcher_slot_fill_route_ok"
        rc, payload, stderr = _run_dispatcher_slot_fill(root)
        _assert(rc == 0, f"dispatcher slot-fill-check expected rc=0 got rc={rc} stderr={stderr}")
        _assert(payload.get("ok") is True, f"dispatcher slot-fill-check expected ok=true payload={payload}")
        _assert(payload.get("schema_version") == "slot_fill_protocol.check.v1", f"unexpected schema payload={payload}")
        contract_info = payload.get("contract") if isinstance(payload.get("contract"), dict) else {}
        _assert(contract_info.get("state_valid") is True, f"dispatcher summary contract should be schema-valid payload={payload}")
        results.append(
            {
                "name": name,
                "ok": True,
                "expectation": "continuity dispatcher keeps slot-fill-check routing",
                "returncode": rc,
                "schema_version": payload.get("schema_version"),
            }
        )

        if set(row.get("name") for row in results) != set(REQUIRED_SCENARIO_NAMES):
            raise RuntimeError(
                "required-check scenario contract mismatch for "
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
            "check_script": str(SLOT_FILL_CHECK),
            "dispatcher_script": str(DISPATCHER_SRC),
            "total": len(results),
            "passed": sum(1 for row in results if bool(row.get("ok"))),
            "results": results,
        }

        for row in results:
            status = "PASS" if bool(row.get("ok")) else "FAIL"
            print(f"{status}: {row.get('name')}")

        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if bool(summary.get("ok")) else 1
    except Exception as exc:
        summary = _error_summary(str(exc))
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 1
    finally:
        shutil.rmtree(td, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
