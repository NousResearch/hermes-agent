#!/usr/bin/env python3
"""Regression harness for swarm operability contract enforcement.

Focus:
- healthy swarm contract/runbook wiring remains green.
- missing required role fails closed as critical.
- malformed role shape fails closed as critical.
- runbook snippet drift stays warning-only (non-blocking) for operator ergonomics.
"""

from __future__ import annotations

import json
import os
import re
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
SWARM_CHECK = ROOT / "ops" / "openclaw" / "architecture" / "check_swarm_operability.sh"
CONTRACT_SRC = ROOT / "ops" / "openclaw" / "architecture" / "swarm_role_contracts.v1.yaml"
RUNBOOK_SRC = ROOT / "docs" / "ops" / "swarm_operating_contract_runbook_v1.md"
CONTINUITY_DISPATCHER_SRC = ROOT / "ops" / "openclaw" / "continuity.sh"

CONTRACT = strict_required_check_contract("swarm_operability_contract_regressions")
CHECK_ID = CONTRACT.check_id
HARNESS_ID = CONTRACT.harness
SUMMARY_SCHEMA_VERSION = CONTRACT.summary_schema_version
SUMMARY_SOURCE = CONTRACT.summary_source
REQUIRED_SCENARIO_NAMES = list(CONTRACT.scenario_names)
LEGACY_SCHEMA_VERSION = "swarm.operability.regressions.v1"
FAILURE_TAXONOMY_VERSION = "swarm_operability.check_failure.v1"
EXPECTED_EVIDENCE_REFS = {
    "ops/openclaw/architecture/swarm_role_contracts.v1.yaml",
    "docs/ops/swarm_operating_contract_runbook_v1.md",
    "ops/openclaw/continuity.sh",
    "ops/openclaw/architecture/check_swarm_operability.sh",
}


def _assert(cond: bool, message: str) -> None:
    if not cond:
        raise AssertionError(message)


def _read_json(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    obj = json.loads(raw)
    if not isinstance(obj, dict):
        raise AssertionError(f"expected JSON object in {path}")
    return obj


def _write_exec(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _materialize_runbook_command_paths(root: Path, runbook_text: str) -> None:
    for match in re.finditer(r"bash\s+([\w./_-]+\.sh)", runbook_text):
        rel = match.group(1)
        target = (root / rel).resolve()
        if target.exists():
            continue
        _write_exec(target, "#!/usr/bin/env bash\nexit 0\n")


def _prepare_fixture_root() -> tuple[Path, Path]:
    td = Path(tempfile.mkdtemp(prefix="swarm_operability_regressions_"))
    root = td / "root"

    contract_text = CONTRACT_SRC.read_text(encoding="utf-8")
    runbook_text = RUNBOOK_SRC.read_text(encoding="utf-8")

    contract_path = root / "ops" / "openclaw" / "architecture" / "swarm_role_contracts.v1.yaml"
    runbook_path = root / "docs" / "ops" / "swarm_operating_contract_runbook_v1.md"
    continuity_path = root / "ops" / "openclaw" / "continuity.sh"

    contract_path.parent.mkdir(parents=True, exist_ok=True)
    runbook_path.parent.mkdir(parents=True, exist_ok=True)
    continuity_path.parent.mkdir(parents=True, exist_ok=True)

    contract_path.write_text(contract_text, encoding="utf-8")
    runbook_path.write_text(runbook_text, encoding="utf-8")
    shutil.copy2(CONTINUITY_DISPATCHER_SRC, continuity_path)
    continuity_path.chmod(continuity_path.stat().st_mode | stat.S_IXUSR)

    _materialize_runbook_command_paths(root, runbook_text)
    return td, root


def _run_swarm_check(root: Path) -> tuple[int, dict[str, Any], str]:
    cp = subprocess.run(
        ["bash", str(SWARM_CHECK), "--json"],
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
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
    rows = payload.get("checks")
    if not isinstance(rows, list):
        return None
    for row in rows:
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
    try:
        import yaml
    except Exception as exc:
        print(json.dumps(_error_summary(f"pyyaml_missing:{exc}"), ensure_ascii=False, indent=2, sort_keys=True))
        return 2

    if not SWARM_CHECK.exists():
        summary = _error_summary("swarm_check_missing")
        summary["path"] = str(SWARM_CHECK)
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 2

    td, root = _prepare_fixture_root()
    contract_path = root / "ops" / "openclaw" / "architecture" / "swarm_role_contracts.v1.yaml"
    runbook_path = root / "docs" / "ops" / "swarm_operating_contract_runbook_v1.md"

    try:
        results: list[dict[str, Any]] = []

        # Scenario 1: healthy fixture stays green.
        name = "healthy_fixture_ok"
        rc, payload, stderr = _run_swarm_check(root)
        _assert(rc == 0, f"healthy fixture expected rc=0 got rc={rc} stderr={stderr}")
        _assert(payload.get("ok") is True, f"healthy fixture expected ok=true payload={payload}")
        _assert(int(payload.get("critical_failures") or 0) == 0, f"healthy fixture critical failures: {payload}")
        _assert(payload.get("failure_taxonomy_version") == FAILURE_TAXONOMY_VERSION, f"missing failure taxonomy version payload={payload}")
        _assert(payload.get("fail_close_triggered") is False, f"healthy fixture should not trigger fail-close payload={payload}")
        _assert(payload.get("failure_code") is None, f"healthy fixture failure_code should be null payload={payload}")
        evidence_refs = payload.get("evidence_refs") if isinstance(payload.get("evidence_refs"), list) else []
        _assert(EXPECTED_EVIDENCE_REFS.issubset(set(str(x) for x in evidence_refs)), f"evidence refs incomplete payload={payload}")
        contract_info = payload.get("contract") if isinstance(payload.get("contract"), dict) else {}
        _assert(contract_info.get("state_valid") is True, f"summary contract should be schema-valid payload={payload}")
        _assert(
            str(contract_info.get("schema_path") or "").endswith("swarm_operability_check.schema.json"),
            f"unexpected summary contract schema path payload={payload}",
        )
        slot_fill_check = _find_check(payload, "continuity_dispatcher_slot_fill_check") or {}
        _assert(slot_fill_check.get("ok") is True, f"slot-fill-check dispatcher wiring missing payload={payload}")
        results.append(
            {
                "name": name,
                "ok": True,
                "expectation": "healthy contract/runbook fixture remains green",
                "returncode": rc,
                "critical_failures": int(payload.get("critical_failures") or 0),
            }
        )

        # Scenario 2: remove required role -> fail closed.
        name = "missing_required_role_failclose"
        contract_obj = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
        _assert(isinstance(contract_obj, dict), "contract fixture not object")
        roles = contract_obj.get("roles") if isinstance(contract_obj.get("roles"), dict) else {}
        if "validator" in roles:
            roles.pop("validator", None)
        contract_obj["roles"] = roles
        contract_path.write_text(yaml.safe_dump(contract_obj, sort_keys=False), encoding="utf-8")

        rc, payload, stderr = _run_swarm_check(root)
        _assert(rc != 0, f"missing role should fail rc!=0 stderr={stderr}")
        _assert(payload.get("ok") is False, f"missing role should set ok=false payload={payload}")
        _assert(payload.get("fail_close_triggered") is True, f"missing role should trigger fail-close payload={payload}")
        _assert(payload.get("failure_category") == "operability_contract_fail_close", f"missing role should map to fail-close category payload={payload}")
        _assert(str(payload.get("failure_code") or "").startswith("critical:"), f"missing role should emit critical failure code payload={payload}")
        role_check = _find_check(payload, "swarm_roles_present") or {}
        details = role_check.get("details") if isinstance(role_check.get("details"), dict) else {}
        missing = details.get("missing") if isinstance(details.get("missing"), list) else []
        _assert("validator" in missing, f"missing validator not surfaced details={details}")
        results.append(
            {
                "name": name,
                "ok": True,
                "expectation": "required role deletion fails closed and surfaces missing validator",
                "returncode": rc,
                "missing": missing,
            }
        )

        # Reset fixture contract for subsequent scenarios.
        contract_path.write_text(CONTRACT_SRC.read_text(encoding="utf-8"), encoding="utf-8")

        # Scenario 3: malformed role shape (empty required_inputs) -> fail closed.
        name = "malformed_role_shape_failclose"
        contract_obj = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
        _assert(isinstance(contract_obj, dict), "contract fixture not object after reset")
        roles = contract_obj.get("roles") if isinstance(contract_obj.get("roles"), dict) else {}
        _assert("planner" in roles and isinstance(roles.get("planner"), dict), "planner role missing in fixture")
        roles["planner"]["required_inputs"] = []
        contract_obj["roles"] = roles
        contract_path.write_text(yaml.safe_dump(contract_obj, sort_keys=False), encoding="utf-8")

        rc, payload, stderr = _run_swarm_check(root)
        _assert(rc != 0, f"malformed role shape should fail rc!=0 stderr={stderr}")
        _assert(payload.get("ok") is False, f"malformed role shape should set ok=false payload={payload}")
        shape_check = _find_check(payload, "swarm_role_shape") or {}
        _assert(shape_check.get("ok") is False, f"swarm_role_shape should fail payload={shape_check}")
        failure_codes = payload.get("failure_codes") if isinstance(payload.get("failure_codes"), dict) else {}
        critical_codes = failure_codes.get("critical") if isinstance(failure_codes.get("critical"), list) else []
        _assert(any("swarm_role_shape" in str(code) for code in critical_codes), f"critical taxonomy bucket missing swarm_role_shape payload={payload}")
        results.append(
            {
                "name": name,
                "ok": True,
                "expectation": "malformed role shape fails closed via swarm_role_shape",
                "returncode": rc,
                "swarm_role_shape_ok": shape_check.get("ok"),
            }
        )

        # Reset fixture contract before warning-only scenario.
        contract_path.write_text(CONTRACT_SRC.read_text(encoding="utf-8"), encoding="utf-8")

        # Scenario 4: runbook snippet drift triggers warning but not hard failure.
        name = "runbook_snippet_drift_warn_only"
        runbook_text = RUNBOOK_SRC.read_text(encoding="utf-8")
        _assert("queue_arbitrator.sh handoffs --json" in runbook_text, "fixture runbook missing expected snippet")
        mutated = runbook_text.replace("queue_arbitrator.sh handoffs --json", "queue_arbitrator.sh handoffz --json", 1)
        runbook_path.write_text(mutated, encoding="utf-8")
        _materialize_runbook_command_paths(root, mutated)

        rc, payload, stderr = _run_swarm_check(root)
        _assert(rc == 0, f"warn-only runbook drift should remain rc=0 stderr={stderr}")
        _assert(payload.get("ok") is True, f"warn-only runbook drift should keep ok=true payload={payload}")
        _assert(payload.get("fail_close_triggered") is False, f"warn-only drift must not trigger fail-close payload={payload}")
        _assert(payload.get("failure_category") == "operability_warning_boundary", f"warn-only drift should map warning category payload={payload}")
        _assert(str(payload.get("failure_code") or "").startswith("warn:"), f"warn-only drift should emit warn failure_code payload={payload}")
        snippet_check = _find_check(payload, "runbook_required_command_snippets") or {}
        _assert(snippet_check.get("ok") is False, f"expected runbook snippet warn failure payload={snippet_check}")
        snippet_details = snippet_check.get("details") if isinstance(snippet_check.get("details"), dict) else {}
        missing_snippets = snippet_details.get("missing") if isinstance(snippet_details.get("missing"), list) else []
        _assert(
            "queue_arbitrator.sh handoffs --json" in missing_snippets,
            f"missing snippet contract not surfaced details={snippet_details}",
        )
        results.append(
            {
                "name": name,
                "ok": True,
                "expectation": "runbook snippet drift remains warning-only (no hard fail)",
                "returncode": rc,
                "missing_snippets": missing_snippets,
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
            "check_script": str(SWARM_CHECK),
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
