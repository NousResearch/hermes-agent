"""Offline reliability checks for Money Flow Radar report artifacts.

The harness evaluates already-produced report payloads. It does not fetch
market data, place orders, size portfolios, or decide whether a user should
act on a watchlist item.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from enum import Enum
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Optional, Sequence


REQUIRED_REVIEWED_REPORTS = 14
SOURCE_STALE_WARN_AFTER = timedelta(hours=36)
SOURCE_STALE_FAIL_AFTER = timedelta(days=7)


class ReliabilityStatus(str, Enum):
    """Status values used by reliability checks and aggregate results."""

    OK = "OK"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass(frozen=True)
class ReliabilityCheck:
    """Single reliability check outcome."""

    name: str
    status: ReliabilityStatus
    detail: str
    remediation: str
    category: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "detail": _redact_text(self.detail),
            "remediation": _redact_text(self.remediation),
            "category": self.category,
        }


@dataclass(frozen=True)
class MoneyFlowReliabilityResult:
    """Structured Money Flow Radar reliability result."""

    overall_status: ReliabilityStatus
    checks: list[ReliabilityCheck]
    summary: dict[str, int]
    report_id: Optional[str]
    report_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_status": self.overall_status.value,
            "summary": dict(self.summary),
            "report_id": self.report_id,
            "report_fingerprint": self.report_fingerprint,
            "checks": [check.to_dict() for check in self.checks],
        }


_SECRET_KEY_RE = re.compile(
    r"\b[A-Z0-9_]*(?:API_KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|PRIVATE_KEY|ACCESS_KEY)\b\s*[:=]\s*([^\s,;]+)",
    re.IGNORECASE,
)
_SECRET_TOKEN_RE = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{12,}|ghp_[A-Za-z0-9_]{12,}|xox[baprs]-[A-Za-z0-9-]{12,})\b"
)
_PRIVATE_KEY_BLOCK_RE = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
    re.DOTALL,
)

_SOURCE_TIMESTAMP_KEYS = (
    "as_of",
    "fresh_at",
    "fetched_at",
    "retrieved_at",
    "last_updated",
    "updated_at",
    "published_at",
    "timestamp",
)
_SOURCE_AGE_KEYS = ("age_hours", "lag_hours", "staleness_hours")
_SOURCE_STATUS_KEYS = ("status", "freshness_status", "state", "quality")
_SOURCE_FAIL_STATUSES = {"error", "failed", "failure", "missing", "unavailable", "invalid"}
_SOURCE_WARN_STATUSES = {"stale", "delayed", "degraded", "partial", "unknown"}

_PROVENANCE_KEYS = {
    "as_of",
    "citation",
    "citations",
    "provenance",
    "published_at",
    "retrieved_at",
    "source",
    "source_id",
    "source_ids",
    "source_path",
    "source_url",
    "sources",
    "url",
    "urls",
}

_WATCHLIST_BOUNDARY_RE = re.compile(
    r"\b("
    r"monitor|monitoring|watch|watchlist|observe|tracking|review|evidence|trigger|risk|"
    r"user decides|user decision|decision remains with the user|not investment advice|"
    r"not a recommendation|watchlist only|for review"
    r")\b",
    re.IGNORECASE,
)
_WATCHLIST_FORBIDDEN_PATTERNS = (
    ("buy/sell/hold", re.compile(r"\b(buy|sell|hold)\b(?!-side)", re.IGNORECASE)),
    (
        "position sizing",
        re.compile(
            r"\b(position\s*(size|sizing)|size\s+the\s+position|portfolio\s+weight|"
            r"allocate|allocation|add\s+\d+%)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "order timing",
        re.compile(
            r"\b(entry|exit|enter|at the open|before close|market order|limit order|"
            r"stop[- ]loss|take profit|target price)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "direct positioning",
        re.compile(r"\b(go long|go short|short it|accumulate|trim|overweight|underweight)\b", re.IGNORECASE),
    ),
)

_FINGERPRINT_EXCLUDED_KEYS = {"generated_at"}


def load_money_flow_report(path: str | Path) -> dict[str, Any]:
    """Load a Money Flow Radar report JSON object from disk."""

    report_path = Path(path)
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Money Flow Radar report is not valid JSON: {report_path}") from exc
    if not isinstance(data, dict):
        raise ValueError("Money Flow Radar report JSON must contain an object at the top level.")
    return data


def evaluate_money_flow_report_path(
    path: str | Path,
    *,
    now: Optional[datetime] = None,
) -> MoneyFlowReliabilityResult:
    """Load a report JSON path and evaluate it."""

    return evaluate_money_flow_report(load_money_flow_report(path), now=now)


def evaluate_money_flow_report(
    report: Mapping[str, Any],
    *,
    now: Optional[datetime] = None,
    stale_warn_after: timedelta = SOURCE_STALE_WARN_AFTER,
    stale_fail_after: timedelta = SOURCE_STALE_FAIL_AFTER,
    required_reviewed_reports: int = REQUIRED_REVIEWED_REPORTS,
) -> MoneyFlowReliabilityResult:
    """Evaluate a report payload without external side effects."""

    if not isinstance(report, Mapping):
        raise TypeError("report must be a mapping")

    evaluated_at = _coerce_datetime(now) or datetime.now(timezone.utc)
    checks = [
        _check_identity(report, evaluated_at),
        _check_source_freshness(report, evaluated_at, stale_warn_after, stale_fail_after),
        _check_regime_rotation_evidence(report),
        _check_contradictions(report),
        _check_watchlist_safety(report),
        _check_calibration_gate(report, required_reviewed_reports),
    ]

    fingerprint = fingerprint_money_flow_report(report)
    checks.append(
        _check(
            "report_fingerprint",
            ReliabilityStatus.OK,
            f"Deterministic report fingerprint is available: sha256:{fingerprint}.",
            "Store this fingerprint with reviewed artifacts when reproducibility comparisons are needed.",
            "reproducibility",
        )
    )

    checks.append(
        _check(
            "result_json_serializable",
            ReliabilityStatus.OK,
            "Normalized reliability result can be encoded as JSON.",
            "No remediation required.",
            "serialization",
        )
    )
    result = _build_result(report, checks, fingerprint)
    try:
        json.dumps(result.to_dict(), sort_keys=True)
    except (TypeError, ValueError) as exc:
        checks[-1] = _check(
            "result_json_serializable",
            ReliabilityStatus.FAIL,
            f"Normalized reliability result could not be encoded as JSON: {exc}.",
            "Ensure checks only contain JSON-serializable scalar, list, and mapping values.",
            "serialization",
        )
        result = _build_result(report, checks, fingerprint)
    return result


def fingerprint_money_flow_report(report: Mapping[str, Any]) -> str:
    """Return a deterministic hash of a report excluding volatile fields."""

    normalized = _normalize_for_json(report, exclude_keys=_FINGERPRINT_EXCLUDED_KEYS)
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _check_identity(report: Mapping[str, Any], now: datetime) -> ReliabilityCheck:
    report_id = _optional_str(report.get("report_id"))
    generated_at = _parse_datetime(report.get("generated_at"))
    missing = []
    if not report_id:
        missing.append("report_id")
    if generated_at is None:
        missing.append("generated_at")
    if missing:
        return _check(
            "report_identity",
            ReliabilityStatus.FAIL,
            f"Missing or unparseable identity field(s): {', '.join(missing)}.",
            "Provide a stable report_id and ISO-8601 generated_at timestamp.",
            "identity",
        )
    if generated_at > now + timedelta(minutes=5):
        return _check(
            "report_identity",
            ReliabilityStatus.WARN,
            f"Report {report_id} generated_at is in the future relative to evaluation time.",
            "Check clock skew before promoting or comparing this report.",
            "identity",
        )
    return _check(
        "report_identity",
        ReliabilityStatus.OK,
        f"Report {report_id} has a parseable generated_at timestamp.",
        "No remediation required.",
        "identity",
    )


def _check_source_freshness(
    report: Mapping[str, Any],
    now: datetime,
    stale_warn_after: timedelta,
    stale_fail_after: timedelta,
) -> ReliabilityCheck:
    sources = _extract_source_entries(report)
    if not sources:
        return _check(
            "source_freshness",
            ReliabilityStatus.FAIL,
            "No source_freshness or sources entries are visible in the report.",
            "Include source freshness metadata with source name, status, and as_of or retrieved_at timestamp.",
            "freshness",
        )

    failed: list[str] = []
    warned: list[str] = []
    unknown_age: list[str] = []
    for source in sources:
        name = _source_name(source)
        status = _source_status(source)
        if status in _SOURCE_FAIL_STATUSES:
            failed.append(f"{name} status={status}")
            continue
        if status in _SOURCE_WARN_STATUSES:
            warned.append(f"{name} status={status}")

        age = _source_age(source, now)
        if age is None:
            unknown_age.append(name)
            continue
        age_hours = age.total_seconds() / 3600
        if age > stale_fail_after:
            failed.append(f"{name} age={age_hours:.1f}h")
        elif age > stale_warn_after:
            warned.append(f"{name} age={age_hours:.1f}h")

    if failed:
        return _check(
            "source_freshness",
            ReliabilityStatus.FAIL,
            f"{len(failed)} source freshness issue(s) exceed fail criteria: {_join_limited(failed)}.",
            "Refresh failed or severely stale sources before promoting the report.",
            "freshness",
        )
    if warned or unknown_age:
        detail_parts = []
        if warned:
            detail_parts.append(f"warning criteria: {_join_limited(warned)}")
        if unknown_age:
            detail_parts.append(f"not age-verifiable: {_join_limited(unknown_age)}")
        return _check(
            "source_freshness",
            ReliabilityStatus.WARN,
            f"{len(sources)} source freshness record(s) visible, but {'; '.join(detail_parts)}.",
            "Add parseable timestamps and refresh stale or degraded sources before expansion.",
            "freshness",
        )
    return _check(
        "source_freshness",
        ReliabilityStatus.OK,
        f"{len(sources)} source freshness record(s) are visible and within thresholds.",
        "No remediation required.",
        "freshness",
    )


def _check_regime_rotation_evidence(report: Mapping[str, Any]) -> ReliabilityCheck:
    missing_sections = [field for field in ("regime", "rotation") if not _is_nonempty(report.get(field))]
    missing_evidence: list[str] = []
    missing_provenance: list[str] = []
    for topic in ("regime", "rotation"):
        evidence = _find_topic_evidence(report, topic)
        if not _is_nonempty(evidence):
            missing_evidence.append(topic)
        elif not _has_provenance(evidence):
            missing_provenance.append(topic)

    if missing_sections or missing_evidence or missing_provenance:
        parts = []
        if missing_sections:
            parts.append(f"missing section(s): {', '.join(missing_sections)}")
        if missing_evidence:
            parts.append(f"missing evidence: {', '.join(missing_evidence)}")
        if missing_provenance:
            parts.append(f"missing provenance: {', '.join(missing_provenance)}")
        return _check(
            "regime_rotation_evidence",
            ReliabilityStatus.FAIL,
            "; ".join(parts) + ".",
            "Attach source-backed evidence with visible provenance for both regime and rotation claims.",
            "evidence",
        )
    return _check(
        "regime_rotation_evidence",
        ReliabilityStatus.OK,
        "Regime and rotation sections have evidence with visible provenance.",
        "No remediation required.",
        "evidence",
    )


def _check_contradictions(report: Mapping[str, Any]) -> ReliabilityCheck:
    if "contradictions" not in report:
        return _check(
            "contradictions_visible",
            ReliabilityStatus.FAIL,
            "The contradictions field is missing.",
            "Include contradictions as an empty list when none are currently known.",
            "evidence",
        )
    contradictions = report.get("contradictions")
    if contradictions is None:
        return _check(
            "contradictions_visible",
            ReliabilityStatus.WARN,
            "The contradictions field is present but null.",
            "Use an empty list for no contradictions or provide explicit contradiction records.",
            "evidence",
        )
    count = _collection_count(contradictions)
    return _check(
        "contradictions_visible",
        ReliabilityStatus.OK,
        f"Contradictions field is visible with {count} item(s).",
        "No remediation required.",
        "evidence",
    )


def _check_watchlist_safety(report: Mapping[str, Any]) -> ReliabilityCheck:
    if "watchlist" not in report:
        return _check(
            "watchlist_safety_boundary",
            ReliabilityStatus.FAIL,
            "The watchlist field is missing.",
            "Include a watchlist field, even if empty, and keep language observational.",
            "safety",
        )
    watchlist = report.get("watchlist")
    if not _is_nonempty(watchlist):
        return _check(
            "watchlist_safety_boundary",
            ReliabilityStatus.OK,
            "Watchlist field is present and empty; no action language was emitted.",
            "No remediation required.",
            "safety",
        )

    text = _flatten_text(watchlist)
    forbidden = _forbidden_watchlist_labels(text)
    if forbidden:
        return _check(
            "watchlist_safety_boundary",
            ReliabilityStatus.FAIL,
            f"Watchlist contains forbidden recommendation language ({', '.join(forbidden)}): {_excerpt(text)}",
            "Rewrite watchlist items as monitoring evidence only; remove buy/sell/hold, sizing, and order-timing terms.",
            "safety",
        )
    if not _WATCHLIST_BOUNDARY_RE.search(text):
        return _check(
            "watchlist_safety_boundary",
            ReliabilityStatus.WARN,
            f"Watchlist does not visibly state a monitoring or user-decision boundary: {_excerpt(text)}",
            "Add monitoring/watchlist language and state that decisions remain with the user.",
            "safety",
        )
    return _check(
        "watchlist_safety_boundary",
        ReliabilityStatus.OK,
        "Watchlist language stays observational and preserves the user-decision boundary.",
        "No remediation required.",
        "safety",
    )


def _check_calibration_gate(report: Mapping[str, Any], required_reviewed_reports: int) -> ReliabilityCheck:
    ledger = report.get("review_ledger_summary", report.get("review_ledger"))
    if ledger is None:
        return _check(
            "calibration_gate",
            ReliabilityStatus.WARN,
            "No review ledger summary is visible; expansion eligibility cannot be determined.",
            f"Review at least {required_reviewed_reports} reports and include reviewed report count before expansion.",
            "calibration",
        )
    reviewed_count = _extract_reviewed_count(ledger)
    expansion_eligible = _extract_expansion_eligible(ledger)
    if reviewed_count is None:
        return _check(
            "calibration_gate",
            ReliabilityStatus.WARN,
            "Review ledger summary is present but reviewed report count is not parseable.",
            f"Include reviewed_reports or reviewed_count with a target of {required_reviewed_reports}.",
            "calibration",
        )
    if reviewed_count < required_reviewed_reports and expansion_eligible is True:
        return _check(
            "calibration_gate",
            ReliabilityStatus.FAIL,
            f"Expansion eligibility is claimed with only {reviewed_count}/{required_reviewed_reports} reviewed reports.",
            "Do not mark expansion eligible until the calibration gate has enough reviewed reports.",
            "calibration",
        )
    if reviewed_count < required_reviewed_reports:
        return _check(
            "calibration_gate",
            ReliabilityStatus.WARN,
            f"Calibration gate is not ready: {reviewed_count}/{required_reviewed_reports} reports reviewed.",
            "Keep the harness in review/calibration mode until the required report count is reached.",
            "calibration",
        )
    return _check(
        "calibration_gate",
        ReliabilityStatus.OK,
        f"Calibration gate count is ready with {reviewed_count}/{required_reviewed_reports} reviewed reports.",
        "No remediation required.",
        "calibration",
    )


def _build_result(
    report: Mapping[str, Any],
    checks: list[ReliabilityCheck],
    fingerprint: str,
) -> MoneyFlowReliabilityResult:
    summary = {status.value: 0 for status in ReliabilityStatus}
    for check in checks:
        summary[check.status.value] += 1
    if summary[ReliabilityStatus.FAIL.value]:
        overall = ReliabilityStatus.FAIL
    elif summary[ReliabilityStatus.WARN.value]:
        overall = ReliabilityStatus.WARN
    else:
        overall = ReliabilityStatus.OK
    return MoneyFlowReliabilityResult(
        overall_status=overall,
        checks=list(checks),
        summary=summary,
        report_id=_optional_str(report.get("report_id")),
        report_fingerprint=fingerprint,
    )


def _check(
    name: str,
    status: ReliabilityStatus,
    detail: str,
    remediation: str,
    category: str,
) -> ReliabilityCheck:
    return ReliabilityCheck(
        name=name,
        status=status,
        detail=_redact_text(detail),
        remediation=_redact_text(remediation),
        category=category,
    )


def _extract_source_entries(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if "source_freshness" in report:
        entries.extend(_coerce_source_collection(report.get("source_freshness"), "source_freshness"))
    if "sources" in report:
        entries.extend(_coerce_source_collection(report.get("sources"), "source"))
    return [entry for entry in entries if entry]


def _coerce_source_collection(value: Any, default_name: str) -> list[dict[str, Any]]:
    if not _is_nonempty(value):
        return []
    if isinstance(value, Mapping):
        if _looks_like_single_source(value):
            return [_normalize_source(default_name, value)]
        entries = []
        for key, item in value.items():
            if isinstance(item, Mapping):
                entries.append(_normalize_source(str(key), item))
            else:
                entries.append({"name": str(key), "as_of": item})
        return entries
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        entries = []
        for index, item in enumerate(value, start=1):
            if isinstance(item, Mapping):
                entries.append(_normalize_source(f"{default_name}_{index}", item))
            else:
                entries.append({"name": f"{default_name}_{index}", "as_of": item})
        return entries
    return [{"name": default_name, "as_of": value}]


def _looks_like_single_source(value: Mapping[str, Any]) -> bool:
    keys = {str(key).lower() for key in value}
    return bool(
        keys.intersection(_SOURCE_TIMESTAMP_KEYS)
        or keys.intersection(_SOURCE_AGE_KEYS)
        or keys.intersection(_SOURCE_STATUS_KEYS)
        or "name" in keys
    )


def _normalize_source(default_name: str, source: Mapping[str, Any]) -> dict[str, Any]:
    normalized = {str(key): value for key, value in source.items()}
    normalized.setdefault("name", default_name)
    return normalized


def _source_name(source: Mapping[str, Any]) -> str:
    return _redact_text(_optional_str(source.get("name") or source.get("id") or source.get("source")) or "source")


def _source_status(source: Mapping[str, Any]) -> Optional[str]:
    for key in _SOURCE_STATUS_KEYS:
        value = source.get(key)
        if value is not None:
            return str(value).strip().lower()
    return None


def _source_age(source: Mapping[str, Any], now: datetime) -> Optional[timedelta]:
    for key in _SOURCE_AGE_KEYS:
        value = source.get(key)
        number = _coerce_float(value)
        if number is not None:
            return timedelta(hours=max(0.0, number))
    minutes = _coerce_float(source.get("age_minutes"))
    if minutes is not None:
        return timedelta(minutes=max(0.0, minutes))
    for key in _SOURCE_TIMESTAMP_KEYS:
        parsed = _parse_datetime(source.get(key))
        if parsed is not None:
            return max(timedelta(0), now - parsed)
    return None


def _find_topic_evidence(report: Mapping[str, Any], topic: str) -> Any:
    direct_key = f"{topic}_evidence"
    if _is_nonempty(report.get(direct_key)):
        return report.get(direct_key)
    section = report.get(topic)
    if isinstance(section, Mapping) and _is_nonempty(section):
        for key in ("evidence", "provenance", "sources", "citations"):
            if _is_nonempty(section.get(key)):
                return section
    evidence = report.get("evidence")
    if isinstance(evidence, Mapping):
        for key in (topic, direct_key, f"{topic}_signals"):
            if _is_nonempty(evidence.get(key)):
                return evidence.get(key)
    if isinstance(evidence, Sequence) and not isinstance(evidence, (str, bytes, bytearray)):
        for item in evidence:
            if isinstance(item, Mapping):
                fields = " ".join(
                    str(item.get(key, "")) for key in ("topic", "category", "name", "claim", "section")
                )
                if topic.lower() in fields.lower():
                    return item
    return None


def _has_provenance(value: Any, *, depth: int = 0) -> bool:
    if depth > 5 or not _is_nonempty(value):
        return False
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).lower()
            if key_text in _PROVENANCE_KEYS and _is_nonempty(item):
                return True
            if _has_provenance(item, depth=depth + 1):
                return True
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_has_provenance(item, depth=depth + 1) for item in value)
    return False


def _forbidden_watchlist_labels(text: str) -> list[str]:
    labels = []
    for label, pattern in _WATCHLIST_FORBIDDEN_PATTERNS:
        if pattern.search(text):
            labels.append(label)
    return labels


def _extract_reviewed_count(ledger: Any) -> Optional[int]:
    if isinstance(ledger, Mapping):
        for key in (
            "reviewed_reports",
            "reviewed_count",
            "reviewed",
            "completed_reviews",
            "calibrated_reports",
        ):
            number = _coerce_int(ledger.get(key))
            if number is not None:
                return number
        reports = ledger.get("reports")
        if isinstance(reports, Sequence) and not isinstance(reports, (str, bytes, bytearray)):
            return len(reports)
        for value in ledger.values():
            number = _extract_reviewed_count(value)
            if number is not None:
                return number
    if isinstance(ledger, Sequence) and not isinstance(ledger, (str, bytes, bytearray)):
        return len(ledger)
    if isinstance(ledger, bool):
        return None
    if isinstance(ledger, (int, float)):
        return int(ledger)
    if isinstance(ledger, str):
        match = re.search(r"(\d+)\s*(?:reviewed|reports|reviews)", ledger, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def _extract_expansion_eligible(ledger: Any) -> Optional[bool]:
    if not isinstance(ledger, Mapping):
        return None
    for key in ("expansion_eligible", "eligible_for_expansion", "expansion_ready"):
        if key in ledger:
            return _coerce_bool(ledger.get(key))
    gate = ledger.get("calibration_gate")
    if isinstance(gate, Mapping):
        return _extract_expansion_eligible(gate)
    return None


def _collection_count(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, Mapping):
        return len(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return len(value)
    if isinstance(value, str):
        return 1 if value.strip() else 0
    return 1


def _flatten_text(value: Any) -> str:
    parts: list[str] = []

    def collect(item: Any) -> None:
        if item is None:
            return
        if isinstance(item, Mapping):
            for child in item.values():
                collect(child)
            return
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            for child in item:
                collect(child)
            return
        parts.append(str(item))

    collect(value)
    return " ".join(part.strip() for part in parts if part.strip())


def _normalize_for_json(value: Any, *, exclude_keys: set[str]) -> Any:
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            key_text = str(key)
            if key_text.lower() in exclude_keys:
                continue
            normalized[key_text] = _normalize_for_json(item, exclude_keys=exclude_keys)
        return normalized
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalize_for_json(item, exclude_keys=exclude_keys) for item in value]
    if isinstance(value, datetime):
        return _format_datetime(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _parse_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return _ensure_utc(value)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value, tz=timezone.utc)
        except (OSError, OverflowError, ValueError):
            return None
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if re.match(r"^\d{4}-\d{2}-\d{2}$", text):
        text = f"{text}T00:00:00+00:00"
    elif text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return _ensure_utc(datetime.fromisoformat(text))
    except ValueError:
        return None


def _coerce_datetime(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    return _parse_datetime(value)


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _format_datetime(value: datetime) -> str:
    return _ensure_utc(value).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _is_nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping):
        return any(_is_nonempty(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_is_nonempty(item) for item in value)
    return True


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_float(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> Optional[int]:
    number = _coerce_float(value)
    if number is None:
        return None
    return int(number)


def _coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "y", "1", "ready", "eligible"}:
            return True
        if lowered in {"false", "no", "n", "0", "not_ready", "ineligible"}:
            return False
    return None


def _join_limited(items: Sequence[str], *, limit: int = 5) -> str:
    visible = list(items[:limit])
    suffix = "" if len(items) <= limit else f", +{len(items) - limit} more"
    return ", ".join(_redact_text(item) for item in visible) + suffix


def _excerpt(text: str, *, limit: int = 180) -> str:
    clean = " ".join(_redact_text(text).split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3].rstrip() + "..."


def _redact_text(text: str) -> str:
    redacted = _PRIVATE_KEY_BLOCK_RE.sub("[REDACTED PRIVATE KEY]", str(text))
    redacted = _SECRET_KEY_RE.sub(lambda m: m.group(0).replace(m.group(1), "[REDACTED]"), redacted)
    redacted = _SECRET_TOKEN_RE.sub("[REDACTED]", redacted)
    return redacted


__all__ = [
    "MoneyFlowReliabilityResult",
    "REQUIRED_REVIEWED_REPORTS",
    "ReliabilityCheck",
    "ReliabilityStatus",
    "evaluate_money_flow_report",
    "evaluate_money_flow_report_path",
    "fingerprint_money_flow_report",
    "load_money_flow_report",
]
