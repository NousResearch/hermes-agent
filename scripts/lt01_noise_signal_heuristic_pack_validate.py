#!/usr/bin/env python3
"""Validate LT-01 noise-vs-signal heuristic pack contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

try:  # pragma: no cover
    from jsonschema import Draft202012Validator, FormatChecker
except Exception:  # pragma: no cover
    Draft202012Validator = None
    FormatChecker = None


def _json_ptr(parts: Iterable[Any]) -> str:
    seq = list(parts or [])
    if not seq:
        return "$"
    return "$/" + "/".join(str(part) for part in seq)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _issue(code: str, path: str, message: str) -> Dict[str, str]:
    return {"code": code, "path": path, "message": message}


def _semantic_checks(payload: Dict[str, Any], repo_root: Path) -> List[Dict[str, str]]:
    issues: List[Dict[str, str]] = []

    heuristics = payload.get("heuristics") if isinstance(payload.get("heuristics"), list) else []
    heuristic_ids: List[str] = []
    heuristic_id_set: Set[str] = set()

    for idx, item in enumerate(heuristics):
        if not isinstance(item, dict):
            continue
        heuristic_id = str(item.get("heuristic_id") or "").strip()
        if not heuristic_id:
            continue

        if heuristic_id in heuristic_id_set:
            issues.append(
                _issue(
                    "duplicate_heuristic_id",
                    _json_ptr(["heuristics", idx, "heuristic_id"]),
                    f"heuristic_id '{heuristic_id}' is duplicated",
                )
            )
        heuristic_id_set.add(heuristic_id)
        heuristic_ids.append(heuristic_id)

        signal_indicator = str(item.get("signal_indicator") or "").strip().lower()
        noise_indicator = str(item.get("noise_indicator") or "").strip().lower()
        if signal_indicator and noise_indicator and signal_indicator == noise_indicator:
            issues.append(
                _issue(
                    "signal_noise_indicator_identical",
                    _json_ptr(["heuristics", idx]),
                    f"heuristic '{heuristic_id}' has identical signal_indicator and noise_indicator",
                )
            )

        suppression = item.get("suppression") if isinstance(item.get("suppression"), dict) else {}
        suppression_counter = str(suppression.get("suppression_counter") or "").strip()
        expected_prefix = heuristic_id.lower() + "_"
        if suppression_counter and not suppression_counter.startswith(expected_prefix):
            issues.append(
                _issue(
                    "suppression_counter_prefix_mismatch",
                    _json_ptr(["heuristics", idx, "suppression", "suppression_counter"]),
                    f"suppression_counter '{suppression_counter}' must start with '{expected_prefix}'",
                )
            )

        evidence_source = item.get("evidence_source") if isinstance(item.get("evidence_source"), dict) else {}
        source_path_raw = str(evidence_source.get("source_path") or "").strip()
        if source_path_raw:
            source_path = Path(source_path_raw)
            source_abs = source_path if source_path.is_absolute() else repo_root / source_path
            if not source_abs.exists():
                issues.append(
                    _issue(
                        "evidence_source_path_missing",
                        _json_ptr(["heuristics", idx, "evidence_source", "source_path"]),
                        f"evidence_source.source_path does not exist: {source_path_raw}",
                    )
                )

    integration_targets = payload.get("integration_targets") if isinstance(payload.get("integration_targets"), list) else []
    referenced_ids: Set[str] = set()
    for tidx, target in enumerate(integration_targets):
        if not isinstance(target, dict):
            continue
        ids = target.get("heuristic_ids") if isinstance(target.get("heuristic_ids"), list) else []
        for hidx, heuristic_id in enumerate(ids):
            hid = str(heuristic_id or "").strip()
            if not hid:
                continue
            referenced_ids.add(hid)
            if hid not in heuristic_id_set:
                issues.append(
                    _issue(
                        "integration_target_unknown_heuristic",
                        _json_ptr(["integration_targets", tidx, "heuristic_ids", hidx]),
                        f"integration target references unknown heuristic_id '{hid}'",
                    )
                )

    for idx, heuristic_id in enumerate(heuristic_ids):
        if heuristic_id not in referenced_ids:
            issues.append(
                _issue(
                    "heuristic_unmapped_to_target",
                    _json_ptr(["heuristics", idx, "heuristic_id"]),
                    f"heuristic_id '{heuristic_id}' is not mapped by any integration_targets entry",
                )
            )

    rubric = payload.get("scoring_rubric") if isinstance(payload.get("scoring_rubric"), dict) else {}
    bands = rubric.get("bands") if isinstance(rubric.get("bands"), list) else []
    seen_scores: Set[int] = set()
    for bidx, band in enumerate(bands):
        if not isinstance(band, dict):
            continue
        score = band.get("score")
        if isinstance(score, int):
            if score in seen_scores:
                issues.append(
                    _issue(
                        "duplicate_score_band",
                        _json_ptr(["scoring_rubric", "bands", bidx, "score"]),
                        f"score band '{score}' appears more than once",
                    )
                )
            seen_scores.add(score)

    if seen_scores and seen_scores != {0, 1, 2, 3, 4}:
        issues.append(
            _issue(
                "score_band_set_incomplete",
                _json_ptr(["scoring_rubric", "bands"]),
                "scoring_rubric.bands must include each score exactly once from 0..4",
            )
        )

    return issues


def _schema_errors(payload: Dict[str, Any], validator: Optional[Draft202012Validator]) -> List[Dict[str, str]]:
    if validator is None:
        return []
    errors = sorted(validator.iter_errors(payload), key=lambda item: list(item.path))
    result: List[Dict[str, str]] = []
    for err in errors:
        result.append(
            _issue(
                "schema_validation_failed",
                _json_ptr(err.path),
                err.message,
            )
        )
    return result


def _validate_pack(pack_path: Path, validator: Optional[Draft202012Validator], repo_root: Path) -> Dict[str, Any]:
    try:
        payload = _load_json(pack_path)
    except Exception as exc:
        return {
            "pack_path": str(pack_path),
            "ok": False,
            "error": "load_failed",
            "message": str(exc),
            "issues": [],
        }

    if not isinstance(payload, dict):
        return {
            "pack_path": str(pack_path),
            "ok": False,
            "error": "invalid_payload",
            "message": "pack root must be a JSON object",
            "issues": [],
        }

    issues: List[Dict[str, str]] = []
    issues.extend(_schema_errors(payload, validator))
    if not any(issue.get("code") == "schema_validation_failed" for issue in issues):
        issues.extend(_semantic_checks(payload, repo_root=repo_root))

    if issues:
        return {
            "pack_path": str(pack_path),
            "pack_id": payload.get("pack_id"),
            "ok": False,
            "error": "validation_failed",
            "issues": issues,
        }

    return {
        "pack_path": str(pack_path),
        "pack_id": payload.get("pack_id"),
        "ok": True,
        "heuristic_count": len(payload.get("heuristics") or []),
        "integration_target_count": len(payload.get("integration_targets") or []),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate LT-01 noise-signal heuristic packs")
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("docs/ops/schemas/noise_signal_heuristic_pack.v1.schema.json"),
        help="Path to schema JSON",
    )
    parser.add_argument(
        "--pack",
        type=Path,
        nargs="+",
        required=True,
        help="One or more heuristic pack JSON files",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    validator: Optional[Draft202012Validator] = None
    if Draft202012Validator is not None:
        schema_path = args.schema if args.schema.is_absolute() else repo_root / args.schema
        schema_payload = _load_json(schema_path)
        validator = Draft202012Validator(schema_payload, format_checker=FormatChecker())

    results: List[Dict[str, Any]] = []
    for pack in args.pack:
        pack_path = pack if pack.is_absolute() else repo_root / pack
        results.append(_validate_pack(pack_path, validator, repo_root=repo_root))

    failed = [item for item in results if not bool(item.get("ok"))]
    output = {
        "ok": len(failed) == 0,
        "checked": len(results),
        "passed": len(results) - len(failed),
        "failed": len(failed),
        "validator": "lt01_noise_signal_heuristic_pack_validate.v1",
        "results": results,
    }

    print(json.dumps(output, indent=2))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
