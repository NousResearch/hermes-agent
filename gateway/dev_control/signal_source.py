"""Production signal sources for the Hermes Dev back gate."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Protocol

from gateway.dev_control.product_events import DevProductEventStore
from gateway.dev_control.reliability import DevReliabilityStore, scorecard, weakest_categories


DEFAULT_WINDOW_DAYS = 7


@dataclass(frozen=True)
class SignalWindow:
    start: float
    end: float

    @classmethod
    def last_days(cls, days: Optional[float] = None, *, now: Optional[float] = None) -> "SignalWindow":
        end = float(now or time.time())
        day_count = float(days or _env_float("HERMES_DEV_SIGNAL_WINDOW_DAYS", DEFAULT_WINDOW_DAYS))
        return cls(start=end - max(day_count, 0.1) * 86400, end=end)

    @property
    def days(self) -> float:
        return max((self.end - self.start) / 86400, 0.1)


class SignalSource(Protocol):
    def fetch_clusters(self, window: SignalWindow, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        ...


class DeterministicSignalSource:
    """Deterministic clusters over canonical Hermes subagent events."""

    def __init__(self, event_store: Any, *, thresholds: Optional[Dict[str, Any]] = None):
        self.event_store = event_store
        self.thresholds = {**default_thresholds(), **(thresholds or {})}

    def fetch_clusters(self, window: SignalWindow, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        filters = filters or {}
        project_id = str(filters.get("project_id") or "").strip()
        events = [
            event for event in self.event_store.list_events(limit=int(filters.get("limit") or 2000))
            if float(event.get("created_at") or 0) >= window.start
            and float(event.get("created_at") or 0) <= window.end
            and _matches_project(event, project_id)
            and _is_agent_system_event(event)
        ]
        clusters: list[Dict[str, Any]] = []
        clusters.extend(_status_clusters(events, window, self.thresholds))
        clusters.extend(_verification_clusters(events, window, self.thresholds))
        clusters.extend(_numeric_threshold_clusters(events, window, self.thresholds))
        clusters.extend(_outlier_clusters(events, window, self.thresholds))
        clusters = sorted(_dedupe_clusters(clusters), key=lambda item: (-int(item.get("count") or 0), item.get("key") or ""))
        return {
            "source": "deterministic",
            "clusters": clusters,
            "warnings": [],
            "analyzed_event_count": len(events),
        }


class LaminarSignalSource:
    """Laminar SQL-backed source. Hermes owns canonical proposal state."""

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        thresholds: Optional[Dict[str, Any]] = None,
    ):
        self.base_url = (base_url or os.getenv("HERMES_LAMINAR_BASE_URL") or "http://127.0.0.1:5667").rstrip("/")
        self.api_key = api_key or os.getenv("HERMES_LAMINAR_API_KEY") or os.getenv("LAMINAR_PROJECT_API_KEY") or ""
        self.timeout_seconds = float(timeout_seconds or _env_float("HERMES_LAMINAR_SQL_TIMEOUT_SECONDS", 8.0))
        self.thresholds = {**default_thresholds(), **(thresholds or {})}

    def fetch_clusters(self, window: SignalWindow, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        filters = filters or {}
        warnings: list[str] = []
        try:
            rows = self._query(
                """
                SELECT
                    attributes['cluster_key'] AS key,
                    any(attributes['cluster_title']) AS title,
                    count() AS count,
                    groupArray(trace_id) AS evidence_refs,
                    groupArray(coalesce(output, name, '')) AS sample_summaries
                FROM spans
                WHERE start_time >= {start_time:DateTime64(3)}
                  AND start_time < {end_time:DateTime64(3)}
                  AND attributes['hermes_signal_domain'] = {domain:String}
                GROUP BY key
                HAVING count >= {min_count:UInt64}
                ORDER BY count DESC
                LIMIT 50
                """,
                parameters={
                    "start_time": window.start,
                    "end_time": window.end,
                    "domain": str(filters.get("domain") or "agent-system"),
                    "min_count": int(filters.get("min_count") or 1),
                },
            )
        except Exception as exc:
            return {
                "source": "laminar",
                "clusters": [],
                "warnings": [f"Laminar SQL unavailable: {exc}"],
                "analyzed_event_count": 0,
            }
        clusters = [_cluster_from_laminar_row(row, window) for row in rows]
        return {
            "source": "laminar",
            "clusters": [cluster for cluster in clusters if cluster],
            "warnings": warnings,
            "analyzed_event_count": sum(int(row.get("count") or 0) for row in rows if isinstance(row, dict)),
        }

    def _query(self, query: str, *, parameters: Dict[str, Any]) -> list[Dict[str, Any]]:
        body = json.dumps({"query": query, "parameters": parameters}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/v1/sql/query",
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}),
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            text = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(text or f"HTTP {exc.code}") from exc
        data = payload.get("data") or payload.get("rows") or payload.get("result") or payload.get("results") or []
        if isinstance(data, dict):
            data = data.get("rows") or data.get("data") or []
        if not isinstance(data, list):
            raise RuntimeError("Laminar SQL response did not contain rows.")
        return [row for row in data if isinstance(row, dict)]


class ProductSignalSource:
    """Clusters shipped-product error/crash events stored by Hermes."""

    def __init__(self, product_event_store: DevProductEventStore, *, thresholds: Optional[Dict[str, Any]] = None):
        self.product_event_store = product_event_store
        self.thresholds = {**default_thresholds(), **(thresholds or {})}

    def fetch_clusters(self, window: SignalWindow, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        filters = filters or {}
        events = self.product_event_store.list_events(
            start=window.start,
            end=window.end,
            event_type=str(filters.get("type") or "").strip() or None,
            limit=int(filters.get("limit") or 2000),
        )
        min_count = int(filters.get("min_count") or self.thresholds["product_error_min_count"])
        clusters = [
            _product_cluster(signature, items, window)
            for signature, items in _group_product_events(events).items()
            if sum(int(item.get("count") or 0) for item in items) >= min_count
        ]
        clusters = sorted(clusters, key=lambda item: (-int(item.get("count") or 0), item.get("key") or ""))
        return {
            "source": "product",
            "clusters": clusters,
            "warnings": [],
            "analyzed_event_count": sum(int(item.get("count") or 0) for item in events),
        }


class ReliabilitySignalSource:
    """Clusters weak Dev reliability categories into advisory improvement signals."""

    def __init__(self, reliability_store: DevReliabilityStore, *, thresholds: Optional[Dict[str, Any]] = None):
        self.reliability_store = reliability_store
        self.thresholds = {**default_thresholds(), **(thresholds or {})}

    def fetch_clusters(self, window: SignalWindow, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        filters = filters or {}
        outcomes = self.reliability_store.list_outcomes(
            start=window.start,
            end=window.end,
            limit=int(filters.get("limit") or 5000),
        )
        card = scorecard(outcomes, now=window.end)
        target_success_rate = float(
            filters.get("target_success_rate") or self.thresholds["reliability_target_success_rate"]
        )
        rows = [
            row for row in weakest_categories(card.get("categories") or [], limit=int(filters.get("category_limit") or 25))
            if _reliability_category_needs_work(row, target_success_rate=target_success_rate)
        ]
        clusters = [
            _reliability_cluster(row, outcomes, window, target_success_rate=target_success_rate)
            for row in rows
        ]
        return {
            "source": "reliability",
            "clusters": [cluster for cluster in clusters if cluster],
            "warnings": [],
            "analyzed_event_count": len(outcomes),
        }


def default_thresholds() -> Dict[str, Any]:
    return {
        "status_min_count": _env_int("HERMES_DEV_SIGNAL_STATUS_MIN_COUNT", 2),
        "verification_min_count": _env_int("HERMES_DEV_SIGNAL_VERIFICATION_MIN_COUNT", 2),
        "contract_score_threshold": _env_float("HERMES_DEV_SIGNAL_CONTRACT_SCORE_THRESHOLD", 0.75),
        "contract_score_min_count": _env_int("HERMES_DEV_SIGNAL_CONTRACT_SCORE_MIN_COUNT", 3),
        "worker_confidence_threshold": _env_float("HERMES_DEV_SIGNAL_WORKER_CONFIDENCE_THRESHOLD", 0.60),
        "worker_confidence_min_count": _env_int("HERMES_DEV_SIGNAL_WORKER_CONFIDENCE_MIN_COUNT", 3),
        "outlier_min_sample": _env_int("HERMES_DEV_SIGNAL_OUTLIER_MIN_SAMPLE", 5),
        "outlier_ratio": _env_float("HERMES_DEV_SIGNAL_OUTLIER_RATIO", 2.0),
        "proposal_aging_days": _env_int("HERMES_DEV_SIGNAL_PROPOSAL_AGING_DAYS", 14),
        "product_error_min_count": _env_int("HERMES_PRODUCT_SIGNAL_MIN_COUNT", 1),
        "reliability_target_success_rate": _env_float("HERMES_DEV_RELIABILITY_SIGNAL_TARGET_SUCCESS_RATE", 0.95),
    }


def cluster_rate(cluster: Optional[Dict[str, Any]], window: SignalWindow) -> float:
    if not cluster:
        return 0.0
    return round(float(cluster.get("count") or 0) / window.days, 4)


def _status_clusters(events: list[Dict[str, Any]], window: SignalWindow, thresholds: Dict[str, Any]) -> list[Dict[str, Any]]:
    grouped: Dict[str, list[Dict[str, Any]]] = {}
    for event in events:
        status = str(event.get("status") or "").lower()
        if status in {"failed", "needs_review", "error", "timeout", "timed_out"}:
            grouped.setdefault(status, []).append(event)
    return [
        _cluster(
            key=f"terminal_status:{status}",
            title=f"Repeated {status.replace('_', ' ')} Dev worker outcomes",
            events=items,
            window=window,
            metric_name="terminal_status",
            metric_value=status,
        )
        for status, items in grouped.items()
        if len(items) >= int(thresholds["status_min_count"])
    ]


def _verification_clusters(events: list[Dict[str, Any]], window: SignalWindow, thresholds: Dict[str, Any]) -> list[Dict[str, Any]]:
    grouped: Dict[str, list[Dict[str, Any]]] = {}
    for event in events:
        verdict = str(event.get("verification_verdict") or event.get("verification_status") or "").lower()
        if verdict in {"failed", "partial", "unverifiable", "needs_review"}:
            grouped.setdefault(verdict, []).append(event)
    return [
        _cluster(
            key=f"verification:{verdict}",
            title=f"Repeated {verdict.replace('_', ' ')} verification outcomes",
            events=items,
            window=window,
            metric_name="verification_verdict",
            metric_value=verdict,
        )
        for verdict, items in grouped.items()
        if len(items) >= int(thresholds["verification_min_count"])
    ]


def _numeric_threshold_clusters(events: list[Dict[str, Any]], window: SignalWindow, thresholds: Dict[str, Any]) -> list[Dict[str, Any]]:
    clusters = []
    low_contract = [_with_float(event, "output_contract_score") for event in events]
    low_contract = [item for item in low_contract if item[1] is not None and item[1] < float(thresholds["contract_score_threshold"])]
    if len(low_contract) >= int(thresholds["contract_score_min_count"]):
        clusters.append(_cluster(
            key="quality:low_output_contract_score",
            title="Repeated low worker output contract scores",
            events=[item[0] for item in low_contract],
            window=window,
            metric_name="output_contract_score",
            metric_value=round(sum(item[1] for item in low_contract if item[1] is not None) / len(low_contract), 3),
        ))
    low_confidence = [_with_float(event, "worker_confidence") for event in events]
    low_confidence = [item for item in low_confidence if item[1] is not None and item[1] < float(thresholds["worker_confidence_threshold"])]
    if len(low_confidence) >= int(thresholds["worker_confidence_min_count"]):
        clusters.append(_cluster(
            key="quality:low_worker_confidence",
            title="Repeated low worker confidence",
            events=[item[0] for item in low_confidence],
            window=window,
            metric_name="worker_confidence",
            metric_value=round(sum(item[1] for item in low_confidence if item[1] is not None) / len(low_confidence), 3),
        ))
    return clusters


def _outlier_clusters(events: list[Dict[str, Any]], window: SignalWindow, thresholds: Dict[str, Any]) -> list[Dict[str, Any]]:
    clusters = []
    for field, title in (("cost_usd", "Cost outliers in Dev worker runs"), ("duration_seconds", "Duration outliers in Dev worker runs")):
        values = [_with_float(event, field) for event in events]
        values = [item for item in values if item[1] is not None and item[1] > 0]
        if len(values) < int(thresholds["outlier_min_sample"]):
            continue
        numbers = sorted(float(item[1]) for item in values if item[1] is not None)
        median = _percentile(numbers, 0.5)
        p95 = _percentile(numbers, 0.95)
        if median > 0 and p95 >= median * float(thresholds["outlier_ratio"]):
            cutoff = median * float(thresholds["outlier_ratio"])
            clusters.append(_cluster(
                key=f"outlier:{field}",
                title=title,
                events=[event for event, value in values if value is not None and value >= cutoff],
                window=window,
                metric_name=field,
                metric_value={"median": round(median, 4), "p95": round(p95, 4)},
            ))
    return clusters


def _cluster(*, key: str, title: str, events: list[Dict[str, Any]], window: SignalWindow, metric_name: str, metric_value: Any) -> Dict[str, Any]:
    evidence = [_event_ref(event) for event in events[:10]]
    return {
        "key": key,
        "title": title,
        "count": len(events),
        "rate_per_day": cluster_rate({"count": len(events)}, window),
        "evidence_refs": evidence,
        "sample_summaries": [str(event.get("summary") or event.get("message") or event.get("goal") or "")[:240] for event in events[:5]],
        "metrics": {
            metric_name: metric_value,
            "window_start": window.start,
            "window_end": window.end,
        },
        "query_descriptor": {
            "source": "deterministic",
            "cluster_key": key,
            "metric": metric_name,
        },
    }


def _cluster_from_laminar_row(row: Dict[str, Any], window: SignalWindow) -> Optional[Dict[str, Any]]:
    key = str(row.get("key") or "").strip()
    if not key:
        return None
    refs = row.get("evidence_refs") or row.get("trace_ids") or []
    summaries = row.get("sample_summaries") or []
    return {
        "key": key,
        "title": str(row.get("title") or key).strip(),
        "count": int(row.get("count") or 0),
        "rate_per_day": cluster_rate({"count": int(row.get("count") or 0)}, window),
        "evidence_refs": refs if isinstance(refs, list) else [str(refs)],
        "sample_summaries": summaries if isinstance(summaries, list) else [str(summaries)],
        "metrics": {"window_start": window.start, "window_end": window.end, **(row.get("metrics") if isinstance(row.get("metrics"), dict) else {})},
        "query_descriptor": {"source": "laminar", "cluster_key": key},
    }


def _group_product_events(events: list[Dict[str, Any]]) -> Dict[str, list[Dict[str, Any]]]:
    grouped: Dict[str, list[Dict[str, Any]]] = {}
    for event in events:
        signature = str(event.get("signature") or "").strip()
        if signature:
            grouped.setdefault(signature, []).append(event)
    return grouped


def _product_cluster(signature: str, events: list[Dict[str, Any]], window: SignalWindow) -> Dict[str, Any]:
    count = sum(int(event.get("count") or 0) for event in events)
    first = events[0] if events else {}
    context = first.get("context") if isinstance(first.get("context"), dict) else {}
    event_type = str(first.get("type") or "product.error")
    location = context.get("endpoint") or context.get("route") or context.get("location") or context.get("screen") or context.get("flow") or "unknown location"
    title = f"Repeated {event_type.replace('product.', '').replace('_', ' ')} at {location}"
    versions = sorted({str(event.get("app_version") or "") for event in events if event.get("app_version")})
    screens = sorted({str((event.get("context") or {}).get("screen") or "") for event in events if (event.get("context") or {}).get("screen")})
    return {
        "key": f"product:{event_type}:{signature}",
        "title": title,
        "count": count,
        "rate_per_day": cluster_rate({"count": count}, window),
        "evidence_refs": [_product_event_ref(event) for event in events[:10]],
        "sample_summaries": [
            str(event.get("message_redacted") or event_type)[:240]
            for event in events[:5]
        ],
        "metrics": {
            "product_event_type": event_type,
            "signature": signature,
            "affected_versions": versions,
            "affected_screens": screens,
            "window_start": window.start,
            "window_end": window.end,
        },
        "query_descriptor": {
            "source": "product",
            "cluster_key": f"product:{event_type}:{signature}",
            "signature": signature,
            "type": event_type,
        },
    }


def _product_event_ref(event: Dict[str, Any]) -> Dict[str, Any]:
    event_id = event.get("event_id")
    return {
        "kind": "product_event",
        "event_id": event_id,
        "uri": f"hermes://product-events/{event_id}" if event_id else None,
        "signature": event.get("signature"),
        "type": event.get("type"),
        "received_at": event.get("received_at"),
        "count": event.get("count"),
    }


def _reliability_category_needs_work(category: Dict[str, Any], *, target_success_rate: float) -> bool:
    tier = str(category.get("tier") or "unproven").lower()
    if tier == "trusted":
        return False
    if int(category.get("escape_count") or 0) > 0:
        return True
    success_rate = category.get("success_rate")
    return success_rate is None or float(success_rate) < target_success_rate


def _reliability_cluster(
    category: Dict[str, Any],
    outcomes: list[Dict[str, Any]],
    window: SignalWindow,
    *,
    target_success_rate: float,
) -> Dict[str, Any]:
    key_category = str(category.get("category") or "").strip()
    if not key_category:
        return {}
    related = [outcome for outcome in outcomes if outcome.get("category") == key_category]
    evidence_outcomes = [
        outcome for outcome in related
        if outcome.get("escaped") or not bool(outcome.get("success"))
    ]
    dominant_failure_mode = _dominant_reliability_failure_mode(evidence_outcomes)
    evidence_refs = [
        {
            "kind": "reliability_outcome",
            "outcome_id": outcome.get("outcome_id"),
            "uri": f"hermes://dev-reliability/outcomes/{outcome.get('outcome_id')}" if outcome.get("outcome_id") else None,
            "plan_id": outcome.get("plan_id"),
            "task_id": outcome.get("task_id"),
            "category": key_category,
            "reason": _reliability_failure_reason(outcome),
        }
        for outcome in evidence_outcomes[:10]
    ]
    count = int(category.get("failure_count") or 0) + int(category.get("escape_count") or 0)
    count = max(count, len(evidence_refs), 1)
    return {
        "key": f"reliability:{key_category}",
        "title": f"Low reliability: {key_category}",
        "count": count,
        "rate_per_day": cluster_rate({"count": count}, window),
        "evidence_refs": evidence_refs,
        "sample_summaries": [
            f"{key_category}: tier={category.get('tier')}, success_rate={category.get('success_rate')}, escapes={category.get('escape_count')}"
        ],
        "metrics": {
            "success_rate": category.get("success_rate"),
            "target_success_rate": target_success_rate,
            "escape_rate": category.get("escape_rate"),
            "escape_count": category.get("escape_count"),
            "tier": category.get("tier"),
            "sample_count": category.get("sample_count"),
            "dominant_failure_mode": dominant_failure_mode,
        },
        "query_descriptor": {
            "source": "reliability",
            "category": key_category,
            "target_success_rate": target_success_rate,
        },
    }


def _dominant_reliability_failure_mode(outcomes: list[Dict[str, Any]]) -> str:
    counts: Dict[str, int] = {}
    for outcome in outcomes:
        reason = _reliability_failure_reason(outcome)
        counts[reason] = counts.get(reason, 0) + 1
    if not counts:
        return "insufficient evidence"
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _reliability_failure_reason(outcome: Dict[str, Any]) -> str:
    if outcome.get("escaped"):
        return "escaped incident/product signal"
    if str(outcome.get("verification_verdict") or "").lower() not in {"verified", "passed"}:
        return "verification not passing"
    if str(outcome.get("ci_state") or "").lower() != "success":
        return "ci not green"
    if str(outcome.get("code_review_verdict") or "").lower() != "approved":
        return "code review not approved"
    if int(outcome.get("rework_count") or 0) > 0:
        return "rework required"
    return "task did not meet success definition"


def _event_ref(event: Dict[str, Any]) -> Dict[str, Any]:
    event_id = event.get("event_id")
    return {
        "kind": "subagent_event",
        "event_id": event_id,
        "uri": f"hermes://subagent-events/{event_id}" if event_id is not None else None,
        "ao_session_id": event.get("ao_session_id"),
        "plan_id": event.get("launch_plan_id"),
        "task_id": event.get("launch_task_id"),
        "created_at": event.get("created_at"),
    }


def _dedupe_clusters(clusters: Iterable[Dict[str, Any]]) -> list[Dict[str, Any]]:
    by_key: Dict[str, Dict[str, Any]] = {}
    for cluster in clusters:
        key = str(cluster.get("key") or "")
        if key and key not in by_key:
            by_key[key] = cluster
    return list(by_key.values())


def _matches_project(event: Dict[str, Any], project_id: str) -> bool:
    if not project_id:
        return True
    return project_id in {
        str(event.get("project_id") or ""),
        str(event.get("runtime_project_id") or ""),
        str(event.get("ao_project_id") or ""),
    }


def _is_agent_system_event(event: Dict[str, Any]) -> bool:
    if event.get("launch_plan_id") or event.get("launch_task_id") or event.get("ao_session_id"):
        return True
    runtime = str(event.get("runtime") or "").lower()
    return runtime in {"ao", "openhands", "fixture"}


def _with_float(event: Dict[str, Any], key: str) -> tuple[Dict[str, Any], Optional[float]]:
    try:
        value = event.get(key)
        return event, float(value) if value is not None else None
    except Exception:
        return event, None


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    index = min(len(values) - 1, max(0, int(round((len(values) - 1) * percentile))))
    return values[index]


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default
