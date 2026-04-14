#!/usr/bin/env python3
"""XB-403 downstream eval/release gate pack runtime."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:  # pragma: no cover
    from jsonschema import Draft202012Validator, FormatChecker
except Exception:  # pragma: no cover
    Draft202012Validator = None
    FormatChecker = None

from release_evidence_ladder_gate import evaluate_bundle as evaluate_release_bundle
from domain_release_evidence_extension_gate import evaluate as evaluate_domain_extension

SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_REPO_ROOT = SCRIPT_PATH.parent.parent
DEFAULT_PACK_SCHEMA_PATH = DEFAULT_REPO_ROOT / "docs" / "ops" / "schemas" / "xb_403_downstream_eval_gate_pack.schema.json"
DEFAULT_DOMAIN_RELEASE_SCHEMA_PATH = (
    DEFAULT_REPO_ROOT / "docs" / "ops" / "schemas" / "domain_release_evidence_bundle.schema.json"
)

REQUIRED_FIXTURE_FAMILIES = {"F8", "F9", "F10", "F11", "F12", "FX"}


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_repo_path(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    else:
        path = path.resolve()
    return path


def is_within(root: Path, target: Path) -> bool:
    try:
        target.relative_to(root)
        return True
    except Exception:
        return False


def _gate_result(gate: str, ok: bool, reason: Optional[str] = None, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    row = {"gate": gate, "status": "pass" if ok else "fail"}
    if reason:
        row["reason"] = reason
    if details is not None:
        row["details"] = details
    return row


def gate_pack_schema(pack: Any, schema_path: Path) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    if Draft202012Validator is None or FormatChecker is None:
        return False, "pack_schema_gate_unavailable", {"error": "jsonschema_validator_unavailable"}
    if not schema_path.exists() or not schema_path.is_file():
        return False, "pack_schema_gate_unavailable", {"error": "schema_missing", "schema_path": str(schema_path)}

    try:
        schema_doc = load_json_file(schema_path)
    except Exception as exc:
        return False, "pack_schema_gate_unavailable", {"error": "schema_unreadable", "detail": str(exc)}

    validator = Draft202012Validator(schema_doc, format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(pack), key=lambda e: (list(e.absolute_path), str(e.message)))
    if not errors:
        return True, None, {"schema_path": str(schema_path)}

    err = errors[0]
    return False, "pack_schema_invalid", {
        "error": "schema_validation_failed",
        "path": "/".join(str(p) for p in err.absolute_path),
        "message": str(err.message),
    }


def _resolve_pack_ref(repo_root: Path, pack: Dict[str, Any], key: str) -> Tuple[bool, Optional[str], Dict[str, Any], Optional[Path]]:
    raw = str(pack.get(key) or "").strip()
    if not raw:
        return False, "pack_ref_unresolved", {"error": "missing_ref", "key": key}, None
    path = resolve_repo_path(repo_root, raw)
    if not is_within(repo_root, path):
        return False, "pack_ref_unresolved", {"error": "ref_outside_repo", "key": key, "path": raw}, None
    if not path.exists() or not path.is_file():
        return False, "pack_ref_unresolved", {"error": "ref_missing", "key": key, "path": raw}, None
    return True, None, {"key": key, "path": str(path)}, path


def gate_replay_fixtures(payload: Any) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    if not isinstance(payload, dict):
        return False, "replay_fixture_blocked", {"error": "not_object"}

    if str(payload.get("decision") or "") != "PASS":
        return False, "replay_fixture_blocked", {
            "error": "decision_not_pass",
            "decision": payload.get("decision"),
        }

    families = payload.get("fixture_families") if isinstance(payload.get("fixture_families"), list) else []
    family_status = {
        str((row or {}).get("family") or ""): str((row or {}).get("status") or "")
        for row in families
        if isinstance(row, dict)
    }

    missing = sorted(REQUIRED_FIXTURE_FAMILIES - set(family_status.keys()))
    if missing:
        return False, "replay_fixture_blocked", {"error": "fixture_family_missing", "missing": missing}

    not_pass = sorted([family for family, status in family_status.items() if status != "pass"])
    if not_pass:
        return False, "replay_fixture_blocked", {"error": "fixture_family_not_pass", "families": not_pass}

    negative = payload.get("batch1_negative_regressions") if isinstance(payload.get("batch1_negative_regressions"), list) else []
    if not negative:
        return False, "replay_fixture_blocked", {"error": "negative_regressions_missing"}

    mismatches: List[Dict[str, Any]] = []
    for row in negative:
        if not isinstance(row, dict):
            mismatches.append({"error": "row_invalid"})
            continue
        expected = str(row.get("expected_decision") or "")
        actual = str(row.get("actual_decision") or "")
        if expected != actual:
            mismatches.append(
                {
                    "scenario": row.get("scenario"),
                    "expected_decision": expected,
                    "actual_decision": actual,
                }
            )

    if mismatches:
        return False, "replay_fixture_blocked", {"error": "negative_regression_mismatch", "rows": mismatches}

    return True, None, {
        "family_count": len(families),
        "required_families": sorted(REQUIRED_FIXTURE_FAMILIES),
        "negative_regression_count": len(negative),
    }


def gate_benchmark_scorecard(payload: Any) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    if not isinstance(payload, dict):
        return False, "benchmark_scorecard_blocked", {"error": "not_object"}

    if str(payload.get("decision") or "") != "PASS":
        return False, "benchmark_scorecard_blocked", {
            "error": "decision_not_pass",
            "decision": payload.get("decision"),
        }

    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), list) else []
    if not metrics:
        return False, "benchmark_scorecard_blocked", {"error": "metrics_missing"}

    blocked = [
        {
            "metric_id": row.get("metric_id"),
            "status": row.get("status"),
        }
        for row in metrics
        if isinstance(row, dict) and str(row.get("status") or "") != "pass"
    ]
    if blocked:
        return False, "benchmark_scorecard_blocked", {"error": "metric_not_pass", "rows": blocked}

    return True, None, {"metric_count": len(metrics)}


def gate_rollback_simulation(payload: Any, repo_root: Path) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    if not isinstance(payload, dict):
        return False, "rollback_simulation_blocked", {"error": "not_object"}

    if str(payload.get("decision") or "") != "PASS":
        return False, "rollback_simulation_blocked", {
            "error": "decision_not_pass",
            "decision": payload.get("decision"),
        }

    scenarios = payload.get("scenarios") if isinstance(payload.get("scenarios"), list) else []
    if not scenarios:
        return False, "rollback_simulation_blocked", {"error": "scenarios_missing"}

    not_pass = [
        {"scenario": row.get("scenario"), "status": row.get("status")}
        for row in scenarios
        if isinstance(row, dict) and str(row.get("status") or "") != "pass"
    ]
    if not_pass:
        return False, "rollback_simulation_blocked", {"error": "scenario_not_pass", "rows": not_pass}

    refs = payload.get("gate_decision_refs") if isinstance(payload.get("gate_decision_refs"), dict) else {}
    release_ref = str(refs.get("release_ladder") or "").strip()
    domain_ref = str(refs.get("domain_extension") or "").strip()
    if not release_ref or not domain_ref:
        return False, "rollback_simulation_blocked", {"error": "gate_decision_refs_missing"}

    for name, token in [("release_ladder", release_ref), ("domain_extension", domain_ref)]:
        path = resolve_repo_path(repo_root, token)
        if not is_within(repo_root, path) or not path.exists() or not path.is_file():
            return False, "rollback_simulation_blocked", {
                "error": "gate_decision_ref_unresolved",
                "ref_name": name,
                "path": token,
            }

    return True, None, {
        "scenario_count": len(scenarios),
        "gate_decision_refs": {
            "release_ladder": release_ref,
            "domain_extension": domain_ref,
        },
    }


def gate_release_id_parity(pack: Dict[str, Any], release_bundle: Dict[str, Any], replay: Dict[str, Any], benchmark: Dict[str, Any], rollback: Dict[str, Any]) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    expected = str(pack.get("expected_release_id") or "").strip()
    ids = {
        "release_bundle": str(release_bundle.get("release_id") or "").strip(),
        "replay_results": str(replay.get("release_id") or "").strip(),
        "benchmark_scorecard": str(benchmark.get("release_id") or "").strip(),
        "rollback_report": str(rollback.get("release_id") or "").strip(),
    }

    values = {token for token in ids.values() if token}
    if len(values) != 1:
        return False, "release_id_mismatch", {"release_ids": ids}

    canonical = next(iter(values))
    if expected and expected != canonical:
        return False, "release_id_mismatch", {
            "expected_release_id": expected,
            "observed_release_id": canonical,
            "release_ids": ids,
        }

    return True, None, {"release_id": canonical}


def evaluate(
    pack: Any,
    *,
    pack_path: Path,
    repo_root: Path,
    pack_schema_path: Path,
    domain_release_schema_path: Path,
) -> Dict[str, Any]:
    gates: List[Dict[str, Any]] = []
    blocked = False
    block_gate: Optional[str] = None
    block_reason: Optional[str] = None

    pack_dict = pack if isinstance(pack, dict) else {}

    def _blocked_gate(name: str) -> None:
        gates.append({"gate": name, "status": "skipped", "reason": "blocked_by_previous_gate"})

    # 1) schema gate
    ok, reason, details = gate_pack_schema(pack, pack_schema_path)
    gates.append(_gate_result("pack_schema", ok, reason, details))
    if not ok:
        blocked = True
        block_gate = "pack_schema"
        block_reason = reason

    # 2) resolve refs
    resolved: Dict[str, Path] = {}
    for key, gate_name in [
        ("release_bundle_ref", "pack_release_bundle_ref"),
        ("deterministic_replay_fixture_results_ref", "pack_replay_ref"),
        ("domain_benchmark_scorecard_ref", "pack_benchmark_ref"),
        ("rollback_simulation_report_ref", "pack_rollback_ref"),
    ]:
        if blocked:
            _blocked_gate(gate_name)
            continue
        rok, rreason, rdetails, rpath = _resolve_pack_ref(repo_root, pack_dict, key)
        gates.append(_gate_result(gate_name, rok, rreason, rdetails))
        if not rok or rpath is None:
            blocked = True
            block_gate = gate_name
            block_reason = rreason
            continue
        resolved[key] = rpath

    release_bundle: Dict[str, Any] = {}
    replay_results: Dict[str, Any] = {}
    benchmark_scorecard: Dict[str, Any] = {}
    rollback_report: Dict[str, Any] = {}
    load_details: Dict[str, Any] = {}

    # 3) load artifacts
    for key, gate_name in [
        ("release_bundle_ref", "load_release_bundle"),
        ("deterministic_replay_fixture_results_ref", "load_replay_results"),
        ("domain_benchmark_scorecard_ref", "load_benchmark_scorecard"),
        ("rollback_simulation_report_ref", "load_rollback_report"),
    ]:
        if blocked:
            _blocked_gate(gate_name)
            continue
        try:
            obj = load_json_file(resolved[key])
        except Exception as exc:
            gates.append(_gate_result(gate_name, False, "pack_ref_unresolved", {"error": str(exc), "path": str(resolved[key])}))
            blocked = True
            block_gate = gate_name
            block_reason = "pack_ref_unresolved"
            continue
        if not isinstance(obj, dict):
            gates.append(_gate_result(gate_name, False, "pack_ref_unresolved", {"error": "artifact_not_object", "path": str(resolved[key])}))
            blocked = True
            block_gate = gate_name
            block_reason = "pack_ref_unresolved"
            continue
        load_details[key] = {
            "path": str(resolved[key]),
            "sha256": hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest(),
        }
        gates.append(_gate_result(gate_name, True, details=load_details[key]))
        if key == "release_bundle_ref":
            release_bundle = obj
        elif key == "deterministic_replay_fixture_results_ref":
            replay_results = obj
        elif key == "domain_benchmark_scorecard_ref":
            benchmark_scorecard = obj
        elif key == "rollback_simulation_report_ref":
            rollback_report = obj

    # 4) release ladder gate
    release_gate_result: Optional[Dict[str, Any]] = None
    if blocked:
        _blocked_gate("release_ladder")
    else:
        release_gate_result = evaluate_release_bundle(
            release_bundle,
            bundle_path=resolved["release_bundle_ref"],
            repo_root=repo_root,
            schema_path=domain_release_schema_path,
        )
        gate_ok = str(release_gate_result.get("decision") or "") == "PASS"
        gates.append(
            _gate_result(
                "release_ladder",
                gate_ok,
                None if gate_ok else "release_ladder_blocked",
                {
                    "decision": release_gate_result.get("decision"),
                    "block_gate": release_gate_result.get("block_gate"),
                    "block_reason": release_gate_result.get("block_reason"),
                },
            )
        )
        if not gate_ok:
            blocked = True
            block_gate = "release_ladder"
            block_reason = "release_ladder_blocked"

    # 5) domain extension gate
    domain_gate_result: Optional[Dict[str, Any]] = None
    if blocked:
        _blocked_gate("domain_extension")
    else:
        domain_gate_result = evaluate_domain_extension(
            release_bundle,
            bundle_path=resolved["release_bundle_ref"],
            repo_root=repo_root,
            schema_path=domain_release_schema_path,
        )
        gate_ok = str(domain_gate_result.get("decision") or "") == "PASS"
        gates.append(
            _gate_result(
                "domain_extension",
                gate_ok,
                None if gate_ok else "domain_extension_blocked",
                {
                    "decision": domain_gate_result.get("decision"),
                    "block_gate": domain_gate_result.get("block_gate"),
                    "block_reason": domain_gate_result.get("block_reason"),
                },
            )
        )
        if not gate_ok:
            blocked = True
            block_gate = "domain_extension"
            block_reason = "domain_extension_blocked"

    # 6) replay fixtures
    if blocked:
        _blocked_gate("replay_fixtures")
    else:
        gok, greason, gdetails = gate_replay_fixtures(replay_results)
        gates.append(_gate_result("replay_fixtures", gok, greason, gdetails))
        if not gok:
            blocked = True
            block_gate = "replay_fixtures"
            block_reason = greason

    # 7) benchmark scorecard
    if blocked:
        _blocked_gate("benchmark_scorecard")
    else:
        gok, greason, gdetails = gate_benchmark_scorecard(benchmark_scorecard)
        gates.append(_gate_result("benchmark_scorecard", gok, greason, gdetails))
        if not gok:
            blocked = True
            block_gate = "benchmark_scorecard"
            block_reason = greason

    # 8) rollback simulation
    if blocked:
        _blocked_gate("rollback_simulation")
    else:
        gok, greason, gdetails = gate_rollback_simulation(rollback_report, repo_root)
        gates.append(_gate_result("rollback_simulation", gok, greason, gdetails))
        if not gok:
            blocked = True
            block_gate = "rollback_simulation"
            block_reason = greason

    # 9) release-id parity
    if blocked:
        _blocked_gate("release_id_parity")
    else:
        gok, greason, gdetails = gate_release_id_parity(
            pack_dict,
            release_bundle,
            replay_results,
            benchmark_scorecard,
            rollback_report,
        )
        gates.append(_gate_result("release_id_parity", gok, greason, gdetails))
        if not gok:
            blocked = True
            block_gate = "release_id_parity"
            block_reason = greason

    return {
        "schema": "clawd.xb_403_downstream_eval_release_gate.decision.v1",
        "slice_id": "XB-403",
        "evaluated_at": now_iso(),
        "decision": "BLOCK" if blocked else "PASS",
        "block_gate": block_gate,
        "block_reason": block_reason,
        "pack": {
            "path": str(pack_path),
            "sha256": hashlib.sha256(json.dumps(pack_dict, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest(),
        },
        "artifacts": load_details,
        "release_id": (
            str(release_bundle.get("release_id") or "")
            if isinstance(release_bundle, dict)
            else None
        ),
        "release_ladder": {
            "decision": None if release_gate_result is None else release_gate_result.get("decision"),
            "block_gate": None if release_gate_result is None else release_gate_result.get("block_gate"),
            "block_reason": None if release_gate_result is None else release_gate_result.get("block_reason"),
        },
        "domain_extension": {
            "decision": None if domain_gate_result is None else domain_gate_result.get("decision"),
            "block_gate": None if domain_gate_result is None else domain_gate_result.get("block_gate"),
            "block_reason": None if domain_gate_result is None else domain_gate_result.get("block_reason"),
        },
        "gates": gates,
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="XB-403 downstream eval/release gate")
    ap.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT))
    ap.add_argument("--pack-schema-path", default=str(DEFAULT_PACK_SCHEMA_PATH))
    ap.add_argument("--domain-release-schema-path", default=str(DEFAULT_DOMAIN_RELEASE_SCHEMA_PATH))
    ap.add_argument("--pack", required=True)
    ap.add_argument("--json", action="store_true")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    repo_root = Path(args.repo_root).expanduser().resolve()
    pack_schema_path = Path(args.pack_schema_path).expanduser().resolve()
    domain_release_schema_path = Path(args.domain_release_schema_path).expanduser().resolve()
    pack_path = Path(args.pack).expanduser().resolve()

    try:
        pack = load_json_file(pack_path)
    except Exception as exc:
        result = {
            "schema": "clawd.xb_403_downstream_eval_release_gate.decision.v1",
            "slice_id": "XB-403",
            "evaluated_at": now_iso(),
            "decision": "BLOCK",
            "block_gate": "pack_schema",
            "block_reason": "pack_schema_invalid",
            "pack": {"path": str(pack_path), "sha256": None},
            "artifacts": {},
            "release_id": None,
            "release_ladder": {"decision": None, "block_gate": None, "block_reason": None},
            "domain_extension": {"decision": None, "block_gate": None, "block_reason": None},
            "gates": [
                _gate_result("pack_schema", False, "pack_schema_invalid", {"error": str(exc)}),
                {"gate": "pack_release_bundle_ref", "status": "skipped", "reason": "blocked_by_previous_gate"},
                {"gate": "pack_replay_ref", "status": "skipped", "reason": "blocked_by_previous_gate"},
                {"gate": "pack_benchmark_ref", "status": "skipped", "reason": "blocked_by_previous_gate"},
                {"gate": "pack_rollback_ref", "status": "skipped", "reason": "blocked_by_previous_gate"},
                {"gate": "load_release_bundle", "status": "skipped", "reason": "blocked_by_previous_gate"},
                {"gate": "load_replay_results", "status": "skipped", "reason": "blocked_by_previous_gate"},
                {"gate": "load_benchmark_scorecard", "status": "skipped", "reason": "blocked_by_previous_gate"},
                {"gate": "load_rollback_report", "status": "skipped", "reason": "blocked_by_previous_gate"},
                {"gate": "release_ladder", "status": "skipped", "reason": "blocked_by_previous_gate"},
                {"gate": "domain_extension", "status": "skipped", "reason": "blocked_by_previous_gate"},
                {"gate": "replay_fixtures", "status": "skipped", "reason": "blocked_by_previous_gate"},
                {"gate": "benchmark_scorecard", "status": "skipped", "reason": "blocked_by_previous_gate"},
                {"gate": "rollback_simulation", "status": "skipped", "reason": "blocked_by_previous_gate"},
                {"gate": "release_id_parity", "status": "skipped", "reason": "blocked_by_previous_gate"},
            ],
        }
    else:
        result = evaluate(
            pack,
            pack_path=pack_path,
            repo_root=repo_root,
            pack_schema_path=pack_schema_path,
            domain_release_schema_path=domain_release_schema_path,
        )

    rc = 0 if result.get("decision") == "PASS" else 2
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
