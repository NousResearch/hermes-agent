"""Worker Output Contract v2 parsing and prompt helpers."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional


OUTPUT_CONTRACT_VERSION = 2
EVIDENCE_BLOCK_RE = re.compile(
    r"```(?:json)?\s*(?:DEV_WORKER_EVIDENCE|dev_worker_evidence)?\s*(\{.*?\})\s*```",
    re.IGNORECASE | re.DOTALL,
)
VALID_VERIFICATION_STATUSES = {"passed", "failed", "not_run", "partial"}


WORKER_OUTPUT_CONTRACT_V2_PROMPT = """\
## Worker Output Contract v2

Return a short human-readable summary, then a fenced JSON evidence block, then
the exact final marker line if this task requires one.

Use this evidence block shape:
```json DEV_WORKER_EVIDENCE
{
  "summary": "What you concluded or changed.",
  "findings": ["Concrete finding or result."],
  "files_read": ["path/or/file.ext"],
  "files_changed": [],
  "commands_run": ["command --if-any"],
  "verification": {
    "status": "passed",
    "evidence": ["What proves the result."]
  },
  "unresolved_gaps": [],
  "confidence": 0.86,
  "final_marker": null
}
```

Rules:
- Use valid JSON in the fenced block.
- Use verification.status as one of: passed, failed, not_run, partial.
- If no files, commands, changes, gaps, or evidence exist, use an empty array.
- If a final marker is required, set final_marker to that exact marker and make
  the final output line exactly: FINAL_MARKER: <marker>.
"""


def append_worker_output_contract(prompt: str) -> str:
    """Append the v2 output contract unless the prompt already contains it."""

    text = str(prompt or "").strip()
    if "Worker Output Contract v2" in text or "DEV_WORKER_EVIDENCE" in text:
        return text
    if not text:
        return WORKER_OUTPUT_CONTRACT_V2_PROMPT.strip()
    return f"{text}\n\n{WORKER_OUTPUT_CONTRACT_V2_PROMPT.strip()}"


def parse_worker_output_contract(text: Optional[str]) -> Dict[str, Any]:
    """Parse Worker Output Contract v2 evidence from text.

    The parser is intentionally warning-first: callers can surface missing or
    invalid evidence without treating otherwise valid completions as failures.
    """

    value = str(text or "")
    empty = _empty_result("missing", "Worker output did not include a DEV_WORKER_EVIDENCE JSON block.")
    if not value.strip():
        return empty
    match = _find_evidence_block(value)
    if not match:
        if "DEV_WORKER_EVIDENCE" in value.upper():
            return _empty_result("invalid", "Worker evidence block marker was present but no valid JSON object could be extracted.")
        return empty
    raw_json = match.group(1).strip()
    try:
        payload = json.loads(raw_json)
    except Exception as exc:
        result = _empty_result("invalid", f"Worker evidence block is not valid JSON: {exc}")
        result["output_contract_raw"] = raw_json
        return result
    if not isinstance(payload, dict):
        result = _empty_result("invalid", "Worker evidence block must be a JSON object.")
        result["output_contract_raw"] = raw_json
        return result
    return normalize_worker_evidence(payload, raw_json=raw_json)


def normalize_worker_evidence(payload: Dict[str, Any], *, raw_json: Optional[str] = None) -> Dict[str, Any]:
    warnings: list[str] = []
    summary = _string(payload.get("summary"))
    findings = _string_list(payload.get("findings"))
    files_read = _string_list(payload.get("files_read"))
    files_changed = _string_list(payload.get("files_changed"))
    commands_run = _string_list(payload.get("commands_run"))
    unresolved_gaps = _string_list(payload.get("unresolved_gaps"))
    verification = payload.get("verification") if isinstance(payload.get("verification"), dict) else {}
    verification_status = _string((verification or {}).get("status")).lower() or "not_run"
    if verification_status not in VALID_VERIFICATION_STATUSES:
        warnings.append("verification.status is not one of passed, failed, not_run, partial")
        verification_status = "not_run"
    verification_evidence = _string_list((verification or {}).get("evidence"))
    confidence = _confidence(payload.get("confidence"), warnings)
    final_marker = _string(payload.get("final_marker")) or None

    if not summary:
        warnings.append("summary is missing")
    if not isinstance(payload.get("findings"), list):
        warnings.append("findings must be an array")
    if not isinstance(payload.get("files_read"), list):
        warnings.append("files_read must be an array")
    if not isinstance(payload.get("files_changed"), list):
        warnings.append("files_changed must be an array")
    if not isinstance(payload.get("commands_run"), list):
        warnings.append("commands_run must be an array")
    if not isinstance(payload.get("unresolved_gaps"), list):
        warnings.append("unresolved_gaps must be an array")
    if not isinstance((verification or {}).get("evidence"), list):
        warnings.append("verification.evidence must be an array")

    status = "warning" if warnings else "ok"
    result = {
        "output_contract_version": OUTPUT_CONTRACT_VERSION,
        "output_contract_status": status,
        "output_contract_warning": "; ".join(warnings) if warnings else None,
        "structured_summary": summary or None,
        "findings": findings,
        "files_read": files_read,
        "files_changed": files_changed,
        "commands_run": commands_run,
        "verification_status": verification_status,
        "verification_evidence": verification_evidence,
        "unresolved_gaps": unresolved_gaps,
        "worker_confidence": confidence,
        "final_marker": final_marker,
    }
    if raw_json is not None:
        result["output_contract_raw"] = raw_json
    return result


def worker_output_contract_score(fields: Dict[str, Any], *, required_marker: Optional[str] = None) -> float:
    """Return a compact 0-1 score for structured evidence compliance."""

    status = str(fields.get("output_contract_status") or "").lower()
    if status in {"missing", "invalid"}:
        return 0.0
    score = 0.0
    score += 0.2 if fields.get("structured_summary") else 0.0
    score += 0.2 if fields.get("findings") else 0.0
    score += 0.2 if fields.get("verification_status") in VALID_VERIFICATION_STATUSES else 0.0
    score += 0.2 if fields.get("verification_evidence") else 0.0
    if required_marker:
        score += 0.2 if fields.get("final_marker") == required_marker else 0.0
    else:
        score += 0.2
    if status == "warning":
        score = min(score, 0.75)
    return round(max(0.0, min(1.0, score)), 3)


def output_contract_fields_from_event(event: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(event, dict):
        return {}
    keys = (
        "output_contract_version",
        "output_contract_status",
        "output_contract_warning",
        "structured_summary",
        "findings",
        "files_changed",
        "commands_run",
        "verification_status",
        "unresolved_gaps",
        "worker_confidence",
        "final_marker",
        "output_contract_score",
    )
    result = {key: event.get(key) for key in keys if key in event}
    if "verification_evidence" in event:
        result["verification_evidence"] = event.get("verification_evidence") or []
    return result


def _find_evidence_block(text: str) -> Optional[re.Match[str]]:
    matches = list(EVIDENCE_BLOCK_RE.finditer(text))
    if matches:
        preferred = [match for match in matches if "DEV_WORKER_EVIDENCE" in match.group(0).upper()]
        return preferred[-1] if preferred else matches[-1]
    return None


def _empty_result(status: str, warning: str) -> Dict[str, Any]:
    return {
        "output_contract_version": OUTPUT_CONTRACT_VERSION,
        "output_contract_status": status,
        "output_contract_warning": warning,
        "structured_summary": None,
        "findings": [],
        "files_read": [],
        "files_changed": [],
        "commands_run": [],
        "verification_status": None,
        "verification_evidence": [],
        "unresolved_gaps": [],
        "worker_confidence": None,
        "final_marker": None,
    }


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        text = _string(item)
        if text and text not in result:
            result.append(text)
    return result


def _confidence(value: Any, warnings: list[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        warnings.append("confidence must be numeric")
        return None
    if number < 0 or number > 1:
        warnings.append("confidence must be between 0 and 1")
        return max(0.0, min(1.0, number))
    return round(number, 3)
