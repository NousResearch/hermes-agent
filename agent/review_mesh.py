from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent.review_memory import get_review_memory_paths, lookup_review_memory
from agent.subagent_profiles import get_review_mesh_specialist_profile

SEVERITY_ORDER = {
    "critical": 0,
    "high": 1,
    "medium": 2,
    "low": 3,
    "info": 4,
}

_SPECIALIST_TOOLSETS = {
    "testing": ["terminal", "file"],
    "security": ["terminal", "file"],
    "performance": ["terminal", "file"],
    "maintainability": ["terminal", "file"],
    "red_team": ["terminal", "file", "web"],
}

_KEYWORD_RULES = {
    "testing": [
        "test", "pytest", "unittest", "regression", "coverage", "assert", "spec",
    ],
    "security": [
        "auth", "oauth", "jwt", "token", "secret", "password", "permission", "rbac",
        "sql", "xss", "csrf", "security", "crypto", "encrypt", "decrypt", "session",
    ],
    "performance": [
        "performance", "latency", "slow", "optimize", "throughput", "query", "cache",
        "n+1", "loop", "memory", "cpu", "hot path",
    ],
}


@dataclass(frozen=True)
class ReviewMeshRequest:
    goal: str
    context: Optional[str] = None
    touched_paths: List[str] = field(default_factory=list)
    enable_red_team: bool = False
    explicit_specialists: List[str] = field(default_factory=list)
    project_root: Optional[str] = None


@dataclass(frozen=True)
class ReviewMeshPlan:
    specialists: List[str]
    activation_reasons: Dict[str, List[str]]



def _normalize_text(*parts: Optional[str]) -> str:
    return "\n".join(part for part in parts if part).lower()



def _path_matches(paths: List[str], specialist: str) -> List[str]:
    reasons: List[str] = []
    for path in paths:
        lowered = path.lower()
        if specialist == "testing" and ("test" in lowered or lowered.endswith(("_test.py", ".spec.ts", ".test.ts", ".test.js"))):
            reasons.append(f"path:{path}")
        elif specialist == "security" and any(token in lowered for token in ("auth", "security", "secret", "token", "permission", "login", "session")):
            reasons.append(f"path:{path}")
        elif specialist == "performance" and any(token in lowered for token in ("perf", "cache", "query", "index", "worker", "queue", "stream")):
            reasons.append(f"path:{path}")
    return reasons



def plan_review_mesh(request: ReviewMeshRequest) -> ReviewMeshPlan:
    text = _normalize_text(request.goal, request.context, "\n".join(request.touched_paths))
    activation_reasons: Dict[str, List[str]] = {}

    specialists: List[str] = []

    testing_reasons = _path_matches(request.touched_paths, "testing")
    if any(keyword in text for keyword in _KEYWORD_RULES["testing"]):
        testing_reasons.append("keyword_match")
    if testing_reasons:
        specialists.append("testing")
        activation_reasons["testing"] = sorted(set(testing_reasons))

    for specialist in ("security", "performance"):
        reasons = _path_matches(request.touched_paths, specialist)
        if any(keyword in text for keyword in _KEYWORD_RULES[specialist]):
            reasons.append("keyword_match")
        if reasons:
            specialists.append(specialist)
            activation_reasons[specialist] = sorted(set(reasons))

    specialists.append("maintainability")
    activation_reasons["maintainability"] = ["always_on"]

    for specialist in request.explicit_specialists:
        if specialist not in specialists and specialist in _SPECIALIST_TOOLSETS:
            specialists.append(specialist)
            activation_reasons[specialist] = ["explicit_specialist"]

    if request.enable_red_team and "red_team" not in specialists:
        specialists.append("red_team")
        activation_reasons["red_team"] = ["explicitly_requested"]

    return ReviewMeshPlan(specialists=specialists, activation_reasons=activation_reasons)



def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 2:
            return "\n".join(lines[1:-1]).strip()
    return stripped



def _coerce_payload(raw_summary: str) -> Dict[str, Any]:
    cleaned = _strip_code_fences(raw_summary)
    if not cleaned:
        return {"summary": "", "findings": []}
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return {"summary": cleaned, "findings": []}
    if isinstance(parsed, list):
        return {"summary": "", "findings": parsed}
    if isinstance(parsed, dict):
        findings = parsed.get("findings")
        if not isinstance(findings, list):
            parsed = dict(parsed)
            parsed["findings"] = []
        return parsed
    return {"summary": cleaned, "findings": []}



def normalize_severity(value: Any) -> str:
    text = str(value or "medium").strip().lower()
    mapping = {
        "blocker": "critical",
        "critical": "critical",
        "severe": "high",
        "urgent": "high",
        "high": "high",
        "moderate": "medium",
        "medium": "medium",
        "warn": "medium",
        "warning": "medium",
        "minor": "low",
        "low": "low",
        "informational": "info",
        "info": "info",
    }
    return mapping.get(text, "medium")



def _normalize_confidence(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.5
    return max(0.0, min(1.0, round(number, 2)))



def normalize_specialist_payload(specialist: str, payload: Dict[str, Any], task_index: int) -> Dict[str, Any]:
    summary = str(payload.get("summary") or payload.get("overview") or "").strip()
    normalized_findings: List[Dict[str, Any]] = []
    for raw in payload.get("findings", []):
        if not isinstance(raw, dict):
            continue
        title = str(raw.get("title") or raw.get("headline") or raw.get("name") or "Untitled finding").strip()
        file_path = raw.get("file_path") or raw.get("path")
        line_start = raw.get("line_start") or raw.get("line")
        line_end = raw.get("line_end") or raw.get("end_line") or line_start
        finding = {
            "specialist": specialist,
            "category": str(raw.get("category") or specialist).strip() or specialist,
            "severity": normalize_severity(raw.get("severity") or raw.get("level")),
            "confidence": _normalize_confidence(raw.get("confidence")),
            "title": title,
            "summary": str(raw.get("summary") or raw.get("details") or raw.get("description") or "").strip(),
            "recommendation": str(raw.get("recommendation") or raw.get("fix") or raw.get("action") or "").strip(),
            "evidence": str(raw.get("evidence") or raw.get("excerpt") or "").strip(),
            "file_path": str(file_path).strip() if file_path else None,
            "line_start": int(line_start) if isinstance(line_start, int) or (isinstance(line_start, str) and line_start.isdigit()) else None,
            "line_end": int(line_end) if isinstance(line_end, int) or (isinstance(line_end, str) and line_end.isdigit()) else None,
            "source_task_index": task_index,
        }
        normalized_findings.append(finding)
    return {
        "specialist": specialist,
        "summary": summary,
        "findings": normalized_findings,
    }



def parse_specialist_summary(specialist: str, raw_summary: str, task_index: int) -> Dict[str, Any]:
    return normalize_specialist_payload(
        specialist=specialist,
        payload=_coerce_payload(raw_summary or ""),
        task_index=task_index,
    )



def aggregate_review_findings(
    specialist_runs: List[Dict[str, Any]],
    *,
    project_root: Optional[str] = None,
) -> Dict[str, Any]:
    findings: List[Dict[str, Any]] = []
    suppressed_findings: List[Dict[str, Any]] = []
    severity_counts: Dict[str, int] = {}
    specialists_run: List[str] = []
    raw_finding_count = 0
    memory_hit_count = 0

    for run in specialist_runs:
        specialist = run.get("specialist")
        if specialist:
            specialists_run.append(specialist)
        for finding in run.get("findings", []):
            if not isinstance(finding, dict):
                continue
            raw_finding_count += 1
            severity = normalize_severity(finding.get("severity"))
            finding = dict(finding)
            finding["severity"] = severity
            memory_match = lookup_review_memory(finding, project_root=project_root)
            finding["fingerprint"] = memory_match["fingerprint"]
            finding["review_memory"] = {
                key: value
                for key, value in memory_match.items()
                if key != "records"
            }
            if memory_match["match_count"]:
                memory_hit_count += 1
            if memory_match["suppress"]:
                finding["suppressed_by_review_memory"] = True
                suppressed_findings.append(finding)
                continue
            findings.append(finding)
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

    sort_key = lambda item: (
        SEVERITY_ORDER.get(item.get("severity", "medium"), 99),
        -float(item.get("confidence", 0.0)),
        item.get("title", ""),
    )
    findings.sort(key=sort_key)
    suppressed_findings.sort(key=sort_key)
    highest_severity = findings[0]["severity"] if findings else "info"
    return {
        "findings": findings,
        "suppressed_findings": suppressed_findings,
        "severity_counts": severity_counts,
        "highest_severity": highest_severity,
        "specialists_run": specialists_run,
        "finding_count": len(findings),
        "raw_finding_count": raw_finding_count,
        "suppressed_count": len(suppressed_findings),
        "review_memory_hit_count": memory_hit_count,
        "review_memory": {
            **get_review_memory_paths(project_root),
            "matched_finding_count": memory_hit_count,
        },
    }



def build_specialist_task(specialist: str, request: ReviewMeshRequest, plan: ReviewMeshPlan) -> Dict[str, Any]:
    touched_paths = request.touched_paths or []
    touched_block = "\n".join(f"- {path}" for path in touched_paths) if touched_paths else "- (not provided)"
    activation = ", ".join(plan.activation_reasons.get(specialist, [])) or "always_on"
    specialist_label = specialist.replace("_", "-")
    goal = f"Structured {specialist_label} review for: {request.goal}"
    subagent_profile = get_review_mesh_specialist_profile(specialist)
    context_parts = [
        request.context.strip() if request.context else "",
        "",
        f"You are the {specialist_label} specialist in a structured review mesh.",
        f"Activation reason: {activation}",
        f"Use the {subagent_profile.id} subagent profile while executing this review.",
        "Touched paths:",
        touched_block,
        "",
        "Return strict JSON with this shape:",
        '{"summary":"short specialist summary","findings":[{"severity":"critical|high|medium|low|info","title":"...","summary":"...","recommendation":"...","evidence":"...","file_path":"optional/path.py","line_start":123,"line_end":123,"confidence":0.0,"category":"optional"}]}',
        "If you find nothing material, return an empty findings array.",
        "Do not wrap the JSON in markdown fences.",
    ]
    return {
        "specialist": specialist,
        "subagent_profile": subagent_profile.id,
        "goal": goal,
        "context": "\n".join(part for part in context_parts if part is not None).strip(),
        "toolsets": list(_SPECIALIST_TOOLSETS[specialist]),
    }
