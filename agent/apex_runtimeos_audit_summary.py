"""APEX RuntimeOS audit summary.

Reads sanitized checkpoint JSONL produced by ``agent.apex_runtimeos_audit`` and
returns compact aggregate health metrics. Bad lines are counted, not fatal.
No prompts, messages, local paths, raw errors, or credentials are read or
exposed by this module.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional


def default_audit_path() -> Path:
    configured = os.environ.get("APEX_RUNTIMEOS_AUDIT_PATH", "").strip()
    if configured:
        return Path(configured).expanduser()
    base = Path(os.environ.get("HERMES_APEX_RUNTIMEOS_AUDIT_DIR", str(Path.home() / ".hermes" / "apex_runtimeos_audit")))
    return base / "checkpoints.jsonl"


def _new_group() -> Dict[str, Any]:
    return {"count": 0, "blocking": 0, "status": Counter(), "avg_elapsed_ms": 0.0, "_elapsed_total": 0.0}


def _safe_model(value: Any) -> str:
    text = str(value or "unknown")
    return text[:120]


def summarize_audit(path: Optional[Path] = None, *, limit: int = 10000) -> Dict[str, Any]:
    audit_path = path or default_audit_path()
    summary: Dict[str, Any] = {
        "schema": "ApexRuntimeOSAuditSummary/v1",
        "audit_path_exists": audit_path.exists(),
        "records": 0,
        "bad_lines": 0,
        "stages": {},
        "organs": {},
        "models": {},
        "sessions": {},
        "blocking_records": 0,
        "recommendations": {
            "count": 0,
            "status": {},
            "codes": {},
            "severity": {},
            "mutates_runtime": 0,
            "applied": 0,
            "auto_control": {"enabled_count": 0, "blocked": 0, "allowed": 0, "min_severity": {}},
        },
        "recommendation_gate": "OK",
    }
    if not audit_path.exists():
        return summary

    stage_groups = defaultdict(_new_group)
    organ_groups = defaultdict(_new_group)
    model_groups = defaultdict(_new_group)
    session_groups = defaultdict(_new_group)

    with audit_path.open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            if idx >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                summary["bad_lines"] += 1
                continue
            checkpoint = record.get("checkpoint")
            if not isinstance(checkpoint, Mapping):
                summary["bad_lines"] += 1
                continue
            summary["records"] += 1
            stage = str(record.get("stage") or checkpoint.get("stage") or "unknown")
            session_id = str(record.get("session_id") or "")[:120]
            is_blocking = bool(checkpoint.get("blocking"))
            if is_blocking:
                summary["blocking_records"] += 1

            for group in (stage_groups[stage], session_groups[session_id]):
                group["count"] += 1
                group["blocking"] += int(is_blocking)

            results = checkpoint.get("results")
            if not isinstance(results, Mapping):
                continue
            for organ, item in results.items():
                if not isinstance(item, Mapping):
                    continue
                status = str(item.get("status") or "UNKNOWN")
                elapsed = float(item.get("elapsed_ms") or 0.0)
                output = item.get("output") if isinstance(item.get("output"), Mapping) else {}
                model = _safe_model(output.get("model")) if isinstance(output, Mapping) else "unknown"
                control = item.get("control") if isinstance(item.get("control"), Mapping) else None
                if isinstance(control, Mapping):
                    auto = summary["recommendations"]["auto_control"]
                    if control.get("auto_control_enabled"):
                        auto["enabled_count"] += 1
                    if control.get("blocking") or control.get("action") == "block":
                        auto["blocked"] += 1
                    else:
                        auto["allowed"] += 1
                    min_severity = str(control.get("min_severity") or "unknown")
                    auto["min_severity"][min_severity] = auto["min_severity"].get(min_severity, 0) + 1
                recommendation = output.get("recommendation") if isinstance(output, Mapping) else None
                for group in (stage_groups[stage], organ_groups[str(organ)], model_groups[model], session_groups[session_id]):
                    group["status"][status] += 1
                    group["_elapsed_total"] += elapsed
                if isinstance(recommendation, Mapping):
                    summary["recommendations"]["count"] += 1
                    summary["recommendations"]["status"][str(recommendation.get("severity") or "info")] = summary["recommendations"]["status"].get(str(recommendation.get("severity") or "info"), 0) + 1
                    code = str(recommendation.get("code") or "unknown")
                    summary["recommendations"]["codes"][code] = summary["recommendations"]["codes"].get(code, 0) + 1
                    summary["recommendations"]["severity"][str(recommendation.get("severity") or "info")] = summary["recommendations"]["severity"].get(str(recommendation.get("severity") or "info"), 0) + 1
                    summary["recommendations"]["mutates_runtime"] += int(bool(recommendation.get("mutates_runtime")))
                    summary["recommendations"]["applied"] += int(bool(recommendation.get("applied")))

    def finalize(groups: Mapping[str, Dict[str, Any]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, value in groups.items():
            count = int(value.get("count") or sum(value["status"].values()) or 0)
            status_total = sum(value["status"].values()) or 1
            out[key] = {
                "count": count,
                "blocking": int(value.get("blocking") or 0),
                "status": dict(value["status"]),
                "avg_elapsed_ms": round(float(value.get("_elapsed_total") or 0.0) / status_total, 3),
            }
        return out

    summary["stages"] = finalize(stage_groups)
    summary["organs"] = finalize(organ_groups)
    summary["models"] = finalize(model_groups)
    summary["sessions"] = finalize(session_groups)
    rec = summary["recommendations"]
    if rec["status"].get("error"):
        summary["recommendation_gate"] = "REVIEW"
    elif rec["status"].get("warn"):
        summary["recommendation_gate"] = "WATCH"
    else:
        summary["recommendation_gate"] = "OK"
    return summary


def render_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# APEX RuntimeOS Audit Summary",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| records | {summary.get('records', 0)} |",
        f"| bad lines | {summary.get('bad_lines', 0)} |",
        f"| blocking records | {summary.get('blocking_records', 0)} |",
        f"| recommendations | {summary.get('recommendations', {}).get('count', 0)} |",
        f"| recommendation gate | {summary.get('recommendation_gate', 'OK')} |",
        f"| auto control blocked | {summary.get('recommendations', {}).get('auto_control', {}).get('blocked', 0)} |",
        f"| audit path exists | {summary.get('audit_path_exists', False)} |",
        "",
        "## Recommendations",
        "",
        "| Severity | Count |",
        "|---|---:|",
    ]
    recommendations = summary.get("recommendations") or {}
    for severity, count in sorted((recommendations.get("severity") or {}).items()):
        lines.append(f"| {severity} | {count} |")
    if not (recommendations.get("severity") or {}):
        lines.append("| - | 0 |")
    lines.extend([
        "",
        "## Organs",
        "",
        "| Organ | Count | Blocking | Status | Avg ms |",
        "|---|---:|---:|---|---:|",
    ])
    for organ, data in sorted((summary.get("organs") or {}).items()):
        lines.append(f"| {organ} | {data.get('count', 0)} | {data.get('blocking', 0)} | {data.get('status', {})} | {data.get('avg_elapsed_ms', 0)} |")
    lines.extend(["", "## Stages", "", "| Stage | Count | Blocking | Status | Avg ms |", "|---|---:|---:|---|---:|"])
    for stage, data in sorted((summary.get("stages") or {}).items()):
        lines.append(f"| {stage} | {data.get('count', 0)} | {data.get('blocking', 0)} | {data.get('status', {})} | {data.get('avg_elapsed_ms', 0)} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize APEX RuntimeOS checkpoint audit JSONL")
    parser.add_argument("--path", default="", help="audit JSONL path; defaults to RuntimeOS audit path")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    parser.add_argument("--limit", type=int, default=10000, help="max lines to read")
    args = parser.parse_args()
    summary = summarize_audit(Path(args.path).expanduser() if args.path else None, limit=args.limit)
    print(json.dumps(summary, ensure_ascii=False, indent=2) if args.json else render_markdown(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
