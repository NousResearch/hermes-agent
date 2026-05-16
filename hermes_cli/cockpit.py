"""Read-only lane schema helpers for BIF-493 cockpit events.

This module intentionally contains pure display/normalization helpers only.  It
must not persist sessions, enqueue work, write trajectories, or create files.
"""

from __future__ import annotations

import base64
from collections import deque
import copy
from dataclasses import dataclass
import hashlib
import queue
import re
import threading
import time
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional

COCKPIT_SCHEMA_VERSION = 1
OPERATOR_CURRENT_ALIAS = "%operator/current"
OPERATOR_CURRENT_TITLE = "Biff chat"

_SECRET_KEY_RE = re.compile(
    r"(api[_-]?key|token|secret|password|passwd|authorization|credential|session[_-]?key)",
    re.IGNORECASE,
)
_SECRET_VALUE_RE = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]+|xox[baprs]-[A-Za-z0-9_-]+|gh[pousr]_[A-Za-z0-9_]+)\b"
)
_SECRET_ASSIGNMENT_RE = re.compile(
    r"\b(?:api[_-]?key|token|secret|password|passwd|authorization|credential|session[_-]?key)\s*[:=]\s*[^\s,;]+",
    re.IGNORECASE,
)
_LONG_NUMERIC_ID_RE = re.compile(r"\b\d{7,}\b")
_URL_RE = re.compile(r"\bhttps?://[^\s<>\"')]+", re.IGNORECASE)
_HOSTPORT_RE = re.compile(
    r"\b(?:localhost|(?:\d{1,3}\.){3}\d{1,3}|[A-Za-z0-9.-]+\.[A-Za-z]{2,}):\d+(?:/[^\s<>\"')]+)?",
    re.IGNORECASE,
)
_BARE_DOMAIN_RE = re.compile(
    r"\b(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,}\b",
    re.IGNORECASE,
)
_ENDPOINT_PATH_RE = re.compile(r"(?<!\w)/(?:api|v\d+|rpc|graphql|webhook|cockpit)(?:/[A-Za-z0-9._~:-]+)+")
_PLATFORM_ID_RE = re.compile(r"(?<![A-Z0-9])[CUTDG](?=[A-Z0-9]*\d)[A-Z0-9]{5,}(?![A-Z0-9])", re.IGNORECASE)
_GATEWAY_SESSION_PHRASE_RE = re.compile(
    r"\b(?:gateway\s+session\s+key|gateway[_-]?session[_-]?key|session[_-]?key)(?:\s*[=:]\s*\S+)?",
    re.IGNORECASE,
)
_INTERNAL_PROMPT_PREFIXES = (
    '[IMPORTANT: The user has invoked the "',
    "[IMPORTANT: The user has invoked the '",
    "[CONTEXT COMPACTION",
)

_EVENT_TYPE_MAP = {
    "status": "lane.status",
    "lane.status": "lane.status",
    "message.delta": "lane.message.delta",
    "lane.message.delta": "lane.message.delta",
    "message.final": "lane.message.final",
    "lane.message.final": "lane.message.final",
    "tool.start": "lane.tool.start",
    "lane.tool.start": "lane.tool.start",
    "tool.complete": "lane.tool.complete",
    "lane.tool.complete": "lane.tool.complete",
    "reasoning.available": "lane.reasoning.available",
    "lane.reasoning.available": "lane.reasoning.available",
    "approval.request": "lane.approval.request",
    "lane.approval.request": "lane.approval.request",
    "error": "lane.error",
    "lane.error": "lane.error",
}

_ALLOWED_PAYLOAD_FIELDS = {
    "lane.status": ("lane_id", "status", "updated_at"),
    "lane.message.delta": ("lane_id", "role", "text_delta", "status", "updated_at"),
    "lane.message.final": ("lane_id", "role", "text", "status", "updated_at"),
    "lane.tool.start": ("lane_id", "tool_name", "status", "duration_ms", "redacted_summary"),
    "lane.tool.complete": ("lane_id", "tool_name", "status", "duration_ms", "redacted_summary"),
    "lane.reasoning.available": ("lane_id", "available", "updated_at"),
    "lane.approval.request": ("lane_id", "approval_id", "kind", "redacted_summary"),
    "lane.error": ("lane_id", "code", "redacted_summary"),
}

_COCKPIT_RECENT_EVENTS_MAX = 100
_COCKPIT_SUBSCRIBER_MAXSIZE = 100
_COCKPIT_SUBSCRIBERS: list[queue.Queue] = []
_COCKPIT_RECENT_EVENTS = deque(maxlen=_COCKPIT_RECENT_EVENTS_MAX)
_COCKPIT_LOCK = threading.Lock()

_SIGNAL_CATEGORIES = (
    "now",
    "needs_marco",
    "stuck_failed",
    "waiting",
    "recently_completed",
    "active_role_work",
    "recent_context",
    "archive",
)
_ACTIVE_STATUSES = ("active", "running", "in_progress", "working", "busy")
_FAILED_STATUSES = ("failed", "error", "stuck", "blocked", "offline")
_WAITING_STATUSES = ("waiting", "pending", "queued", "idle")
_COMPLETED_STATUSES = ("done", "complete", "completed", "success", "succeeded")
_MARCO_ATTENTION_TERMS = ("needs_marco", "approval", "approve", "human", "operator", "blocked", "waiting on human", "question")
_EXPLICIT_ATTENTION_TERMS = (
    "needs_marco",
    "approval",
    "approve",
    "blocked",
    "failed",
    "waiting on human",
    "waiting-on-human",
    "human approval",
    "operator approval",
    "needs operator",
    "needs human",
)
_ACTIVE_ROLES = ("forge", "vex", "quill", "ranger", "biff")
_ROLE_LABELS = {"biff": "Biff", "forge": "Forge", "vex": "Vex", "quill": "Quill", "ranger": "Ranger"}
_OPERATIONAL_FRESH_SECONDS = 30 * 60
_RECENT_CONTEXT_SECONDS = 24 * 60 * 60


@dataclass(frozen=True)
class CockpitLogicalLane:
    """Read-only process-local definition for a stable cockpit lane.

    This is display/projection metadata only.  It is intentionally not a
    SessionDB row, not a gateway routing target, and not a control alias.
    """

    alias: str
    title: str
    key: str


class CockpitLocalStore:
    """Minimal read-only local cockpit projection store.

    The store contains stable logical lanes that can span physical SessionDB
    rotation/backfill.  It has no persistence and exposes no write/control/send
    methods; callers pass observed sessions in and receive a projection out.
    """

    def __init__(self, logical_lanes: Optional[Iterable[CockpitLogicalLane]] = None):
        default_lanes = (CockpitLogicalLane(OPERATOR_CURRENT_ALIAS, OPERATOR_CURRENT_TITLE, "operator/current"),)
        self._logical_lanes = tuple(logical_lanes or default_lanes)

    def logical_lane_ids(self) -> tuple[str, ...]:
        """Return stable display lane ids for known logical lanes.

        This is a read-only seed/index for bounded discovery scans.  It is not a
        synthetic lane observation: callers must still find a mapped physical
        session, message, or event before emitting any signal for the id.
        """

        return tuple(build_lane_id("cockpit.logical", lane.key) for lane in self._logical_lanes)

    def logical_lane_for_session(self, session: Mapping[str, Any]) -> Optional[CockpitLogicalLane]:
        """Return the logical lane definition for a SessionDB observation, if any."""

        explicit = _first_text(
            session.get("cockpit_lane_alias"),
            session.get("logical_lane_alias"),
            session.get("lane_alias"),
            session.get("logical_lane"),
        )
        if explicit:
            for lane in self._logical_lanes:
                if explicit == lane.alias:
                    return lane

        chat_type = str(session.get("chat_type") or session.get("type") or "").strip().lower()
        role = _canonical_agent_role(session) or str(session.get("role") or "").strip().lower()
        title = str(session.get("title") or session.get("name") or "").strip().lower()
        platform = str(session.get("platform") or session.get("source") or "").strip().lower()

        if chat_type == "operator" and (role == "biff" or title == "biff" or title.startswith("biff ")):
            return self._operator_current_lane()
        if platform == "cli" and role == "biff" and not chat_type:
            return self._operator_current_lane()
        return None

    def _operator_current_lane(self) -> CockpitLogicalLane:
        for lane in self._logical_lanes:
            if lane.alias == OPERATOR_CURRENT_ALIAS:
                return lane
        return CockpitLogicalLane(OPERATOR_CURRENT_ALIAS, OPERATOR_CURRENT_TITLE, "operator/current")


_DEFAULT_COCKPIT_LOCAL_STORE = CockpitLocalStore()


def default_logical_lane_ids() -> tuple[str, ...]:
    """Return read-only logical lane ids that should be considered for scans."""

    return _DEFAULT_COCKPIT_LOCAL_STORE.logical_lane_ids()


def build_lane_id(source: Any, canonical_id: Any) -> str:
    """Return a deterministic, URL-safe, opaque lane identifier.

    The input values may contain raw chat/user/thread/session identifiers; they
    are only used as hash material and are never embedded in the returned ID.
    """

    material = f"{source or ''}\0{canonical_id or ''}".encode("utf-8", "surrogatepass")
    digest = hashlib.sha256(material).digest()[:18]
    token = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return f"lane_{token}"


def infer_lane_alias(platform: Any, chat_type: Any, title: Any, role: Optional[Any] = None) -> dict[str, Any]:
    """Infer a display-only lane alias.

    Aliases are labels for humans, not commands and not actionable references.
    Native mentions and prose are therefore preserved/sanitized only as display
    text and never converted into routing metadata.
    """

    alias = _display_alias(platform, chat_type, title, role)
    return {
        "alias": alias,
        "display_only": True,
        "ambiguous": False,
        "canonical_ref": None,
    }


def build_lane_snapshot_from_session(session: dict, *, local_store: Optional[CockpitLocalStore] = None) -> dict:
    """Build a display-safe, read-only lane snapshot from a session mapping."""

    store = local_store or _DEFAULT_COCKPIT_LOCAL_STORE
    logical_lane = store.logical_lane_for_session(session)
    source = session.get("source") or session.get("platform") or "unknown"
    agent_role = _canonical_agent_role(session)
    canonical_id = session.get("canonical_id") or session.get("session_id") or session.get("id") or ""
    if logical_lane is not None:
        lane_id = build_lane_id("cockpit.logical", logical_lane.key)
        alias = {
            "alias": logical_lane.alias,
            "display_only": True,
            "ambiguous": False,
            "canonical_ref": logical_lane.alias,
        }
        title = logical_lane.title
    else:
        lane_id = build_lane_id(source, canonical_id)
        alias = infer_lane_alias(
            session.get("platform") or source,
            session.get("chat_type") or session.get("type"),
            session.get("title") or session.get("name"),
            role=agent_role or session.get("role"),
        )
        if agent_role:
            title = _clean_scalar(session.get("goal") or session.get("title") or session.get("name") or f"{_ROLE_LABELS.get(agent_role, agent_role.title())} work")
        else:
            title = _clean_scalar(session.get("title") or session.get("name"))

    snapshot = {
        "schema_version": COCKPIT_SCHEMA_VERSION,
        "lane_id": lane_id,
        "alias": alias,
        "platform": _clean_scalar(session.get("platform") or source),
        "chat_type": _clean_scalar(session.get("chat_type") or session.get("type")),
        "title": title,
        "status": _clean_scalar(session.get("status") or _status_from_session(session)),
        "updated_at": _clean_scalar(session.get("updated_at") or session.get("last_active") or session.get("started_at")),
    }
    if agent_role:
        snapshot["agent_role"] = agent_role
        snapshot["role_label"] = _ROLE_LABELS.get(agent_role, agent_role.title())
    for key in ("delegate_kind", "issue_identifier", "goal"):
        value = session.get(key)
        if value not in (None, "") and not _looks_sensitive_value(value):
            snapshot[key] = _clean_scalar(value)
    subagent_id = session.get("subagent_id")
    if subagent_id:
        snapshot["subagent_ref"] = _opaque_ref(subagent_id, prefix="subagent")
    if session.get("ended_at") is not None:
        snapshot["completed_at"] = _clean_scalar(session.get("ended_at"))
    if logical_lane is not None:
        snapshot["logical"] = True
    return snapshot


def mark_ambiguous_aliases(snapshots: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return copies with duplicate display aliases marked ambiguous.

    Ambiguity is a property of a caller-provided batch/context. This helper is
    intentionally pure and does not mutate module globals or the input snapshots.
    """

    copied = [copy.deepcopy(snapshot) for snapshot in snapshots]
    aliases: dict[str, set[str]] = {}
    for snapshot in copied:
        alias = snapshot.get("alias")
        if not isinstance(alias, Mapping):
            continue
        alias_name = str(alias.get("alias") or "")
        if not alias_name:
            continue
        aliases.setdefault(alias_name, set()).add(str(snapshot.get("lane_id") or ""))

    ambiguous_names = {alias_name for alias_name, lane_ids in aliases.items() if len(lane_ids) > 1}
    for snapshot in copied:
        alias = snapshot.get("alias")
        if not isinstance(alias, MutableMapping):
            continue
        if str(alias.get("alias") or "") in ambiguous_names:
            alias["ambiguous"] = True
            alias["canonical_ref"] = None
    return copied


def coalesce_logical_lane_snapshots(snapshots: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Collapse physical SessionDB rotations into one stable logical lane row.

    `%operator/current` can span multiple Hermes sessions after compression or
    restart. The lane list should show one logical cockpit lane while retaining
    only display-safe aggregate metadata.
    """

    output: list[dict[str, Any]] = []
    logical_by_id: dict[str, dict[str, Any]] = {}
    for raw in snapshots:
        lane = copy.deepcopy(dict(raw))
        lane_id = str(lane.get("lane_id") or "")
        if not lane_id or lane.get("logical") is not True:
            output.append(lane)
            continue

        existing = logical_by_id.get(lane_id)
        if existing is None:
            lane["session_count"] = int(lane.get("session_count") or 1)
            logical_by_id[lane_id] = lane
            output.append(lane)
            continue

        existing["session_count"] = int(existing.get("session_count") or 1) + 1
        if _timestamp_sort_value(lane.get("updated_at")) >= _timestamp_sort_value(existing.get("updated_at")):
            existing["updated_at"] = lane.get("updated_at")
        if _status_priority(lane.get("status")) > _status_priority(existing.get("status")):
            existing["status"] = lane.get("status")

    return output


def normalize_api_run_event(event: dict) -> dict:
    """Normalize an API run event into the BIF-493 read-only lane grammar."""

    return _normalize_event(event)


def project_message_for_cockpit(message: Mapping[str, Any]) -> dict[str, Any]:
    """Project a stored SessionDB message into a display-safe cockpit row.

    The projection is intentionally lossy: raw tool inputs/outputs, provider
    response item blobs, reasoning, attachments, file content and identifiers are
    omitted or summarized before reaching dashboard clients.
    """

    projected: dict[str, Any] = {}
    role = _clean_scalar(message.get("role"))
    if role is not None:
        projected["role"] = role

    content = _project_message_content(message)
    if content is not None:
        projected["content"] = content

    for key in ("timestamp", "created_at", "status", "source"):
        value = message.get(key)
        if value is not None and not _looks_sensitive_value(value):
            projected[key] = _clean_scalar(value)

    return projected


def recent_cockpit_events() -> list[dict[str, Any]]:
    """Return a display-safe copy of process-local recent cockpit events."""

    with _COCKPIT_LOCK:
        return [copy.deepcopy(event) for event in _COCKPIT_RECENT_EVENTS]


def project_cockpit_signals(
    lanes: Iterable[Mapping[str, Any]],
    *,
    messages_by_lane: Optional[Mapping[str, Iterable[Mapping[str, Any]]]] = None,
    events: Optional[Iterable[Mapping[str, Any]]] = None,
    now: Optional[float] = None,
) -> dict[str, Any]:
    """Project real cockpit observations into ranked read-only signals."""

    enforce_freshness = now is not None
    observed_at = float(now if now is not None else time.time())
    lane_list = [copy.deepcopy(dict(lane)) for lane in lanes]
    lane_by_id = {str(lane.get("lane_id") or ""): lane for lane in lane_list if lane.get("lane_id")}
    safe_messages: dict[str, list[dict[str, Any]]] = {}
    for lane_id, raw_messages in (messages_by_lane or {}).items():
        safe_messages[str(lane_id)] = [project_message_for_cockpit(message) for message in raw_messages]
    safe_events = [copy.deepcopy(dict(event)) for event in (events or [])]

    categories: dict[str, dict[str, Any]] = {
        name: {
            "label": _signal_category_label(name),
            "empty_state": "No real signal observed for this category.",
            "signals": [],
        }
        for name in _SIGNAL_CATEGORIES
    }

    def add_signal(category: str, signal: dict[str, Any]) -> None:
        if category not in categories:
            return
        signal = copy.deepcopy(signal)
        freshness = _freshness_classification(signal.get("timestamp"), observed_at, enforce=enforce_freshness)
        target_category = category
        if category not in {"recent_context", "archive"} and freshness["bucket"] != "operational":
            target_category = freshness["bucket"]
        if target_category not in categories:
            return
        signal["category"] = category
        signal["schema_version"] = COCKPIT_SCHEMA_VERSION
        signal["recency_label"] = _recency_label(signal.get("timestamp"), observed_at)
        signal["freshness"] = freshness
        source = signal.get("source") if isinstance(signal.get("source"), MutableMapping) else {}
        source["freshness"] = freshness["bucket"]
        source["class"] = "operational" if freshness["bucket"] == "operational" else _stale_source_class(signal.get("provenance"))
        signal["source"] = source
        if target_category != category:
            signal["category"] = target_category
            signal["reason"] = _reason(f"demoted stale {category.replace('_', ' ')} signal to {target_category.replace('_', ' ')}", str(signal.get("reason") or ""))
        categories[target_category]["signals"].append(signal)

    for lane in lane_list:
        lane_id = str(lane.get("lane_id") or "")
        if not lane_id:
            continue
        messages = safe_messages.get(lane_id, [])
        latest_message = _latest_message(messages)
        status_text = _signal_text(lane.get("status"), latest_message.get("status") if latest_message else None)
        latest_timestamp = _max_timestamp(
            lane.get("updated_at"),
            latest_message.get("timestamp") if latest_message else None,
            latest_message.get("created_at") if latest_message else None,
        )
        base = _base_signal(
            lane,
            latest_message,
            timestamp=latest_timestamp,
            provenance="lane_message" if latest_message else "lane_snapshot",
        )

        if _contains_any(status_text, _ACTIVE_STATUSES) or latest_message:
            confidence = 0.82 if _contains_any(status_text, _ACTIVE_STATUSES) else 0.62
            add_signal("now", {**base, "confidence": confidence, "reason": _reason("recent lane activity", status_text)})
        if _contains_any(status_text, _FAILED_STATUSES):
            add_signal("stuck_failed", {**base, "confidence": 0.92, "reason": _reason("lane reports failed/stuck state", status_text)})
        if _contains_any(status_text, _WAITING_STATUSES):
            add_signal("waiting", {**base, "confidence": 0.72, "reason": _reason("lane reports waiting state", status_text)})
        if _contains_any(status_text, _COMPLETED_STATUSES):
            add_signal("recently_completed", {**base, "confidence": 0.78, "reason": _reason("lane reports completed state", status_text)})
        if _lane_has_active_role(lane) and not _contains_any(status_text, _FAILED_STATUSES) and (_contains_any(status_text, _ACTIVE_STATUSES) or latest_message):
            add_signal("active_role_work", {**base, "confidence": 0.86, "reason": _reason("active role work observed", status_text)})

        message_text = _signal_text(latest_message.get("content") if latest_message else None, status_text)
        if _needs_marco_attention(latest_message, status_text, message_text):
            add_signal("needs_marco", {**base, "confidence": 0.84, "reason": "message/status indicates Marco attention may be needed"})

    for event in safe_events:
        payload = event.get("payload") if isinstance(event.get("payload"), Mapping) else {}
        lane_id = str(payload.get("lane_id") or event.get("lane_id") or "")
        if not lane_id:
            continue
        lane = lane_by_id.get(lane_id, {"lane_id": lane_id, "alias": {"alias": "observed lane", "display_only": True}})
        event_type = str(event.get("type") or "")
        event_text = _signal_text(event_type, payload.get("status"), payload.get("kind"), payload.get("redacted_summary"))
        timestamp = _max_timestamp(payload.get("updated_at"), event.get("updated_at"), lane.get("updated_at"))
        base = _base_signal(lane, None, timestamp=timestamp, provenance="lane_event")
        if "approval" in event_text or _contains_any(event_text, _MARCO_ATTENTION_TERMS):
            add_signal("needs_marco", {**base, "confidence": 0.93, "reason": _reason("approval/attention event observed", event_text)})
        if _contains_any(event_text, _FAILED_STATUSES):
            add_signal("stuck_failed", {**base, "confidence": 0.9, "reason": _reason("failure event observed", event_text)})
        if _contains_any(event_text, _COMPLETED_STATUSES):
            add_signal("recently_completed", {**base, "confidence": 0.78, "reason": _reason("completion event observed", event_text)})

    total = 0
    for category in categories.values():
        signals = _dedupe_signals(category["signals"])
        signals.sort(key=lambda item: (_timestamp_sort_value(item.get("timestamp")), float(item.get("confidence") or 0)), reverse=True)
        category["signals"] = signals[:20]
        total += len(category["signals"])

    return {
        "schema_version": COCKPIT_SCHEMA_VERSION,
        "read_only": True,
        "generated_at": observed_at,
        "categories": categories,
        "total": total,
    }


def project_agent_activity(
    lanes: Iterable[Mapping[str, Any]],
    *,
    messages_by_lane: Optional[Mapping[str, Iterable[Mapping[str, Any]]]] = None,
    now: Optional[float] = None,
) -> dict[str, Any]:
    """Project named Biff OS agent lane observations into read-only activity rows.

    This is deliberately narrower than the lane/message APIs: it keeps only
    BIF-547 role metadata plus a short display-safe evidence line, never raw
    platform/channel identifiers and never full transcripts.
    """

    observed_at = float(now if now is not None else time.time())
    safe_messages: dict[str, list[dict[str, Any]]] = {}
    for lane_id, raw_messages in (messages_by_lane or {}).items():
        safe_messages[str(lane_id)] = [project_message_for_cockpit(message) for message in raw_messages]

    items: list[dict[str, Any]] = []
    for raw_lane in lanes:
        lane = copy.deepcopy(dict(raw_lane))
        role = _canonical_agent_role(lane)
        if role not in _ACTIVE_ROLES:
            continue
        lane_id = str(lane.get("lane_id") or "")
        if not lane_id:
            continue
        messages = safe_messages.get(lane_id, [])
        latest_message = _latest_message(messages)
        latest_timestamp = _max_timestamp(
            lane.get("updated_at"),
            lane.get("completed_at"),
            latest_message.get("timestamp") if latest_message else None,
            latest_message.get("created_at") if latest_message else None,
        )
        freshness = _freshness_classification(latest_timestamp, observed_at, enforce=True)
        status = _agent_activity_status(lane.get("status"), latest_message.get("status") if latest_message else None, freshness=freshness)
        evidence_source = latest_message.get("content") if latest_message else None
        evidence = _short_evidence(evidence_source or _reason("lane status", str(lane.get("status") or status)))
        item = {
            "schema_version": COCKPIT_SCHEMA_VERSION,
            "id": _signal_id(lane_id, "agent_activity", latest_timestamp, role),
            "lane_id": lane_id,
            "agent_role": role,
            "role_label": _ROLE_LABELS.get(role, role.title()),
            "status": status,
            "issue_identifier": _clean_scalar(lane.get("issue_identifier")),
            "goal": _clean_scalar(lane.get("goal") or lane.get("title")),
            "title": _clean_scalar(lane.get("title") or lane.get("goal") or f"{_ROLE_LABELS.get(role, role.title())} work"),
            "updated_at": _clean_scalar(lane.get("updated_at")),
            "completed_at": _clean_scalar(lane.get("completed_at")),
            "latest_evidence": evidence,
            "recency_label": _recency_label(latest_timestamp, observed_at),
            "freshness": freshness,
            "source": {
                "class": "operational" if freshness["bucket"] == "operational" else "sessiondb",
                "freshness": freshness["bucket"],
            },
        }
        if lane.get("delegate_kind") not in (None, ""):
            item["delegate_kind"] = _clean_scalar(lane.get("delegate_kind"))
        if latest_message and latest_message.get("role") not in (None, ""):
            item["latest_evidence_role"] = _clean_scalar(latest_message.get("role"))
        items.append(item)

    items.sort(key=lambda item: (_agent_activity_status_priority(item.get("status")), _timestamp_sort_value(item.get("updated_at"))), reverse=True)
    counts = {status: 0 for status in ("running", "completed", "failed", "blocked", "waiting", "stale")}
    for item in items:
        item_status = str(item.get("status") or "stale")
        counts[item_status] = counts.get(item_status, 0) + 1
    return {
        "schema_version": COCKPIT_SCHEMA_VERSION,
        "read_only": True,
        "actions_enabled": False,
        "mutation_enabled": False,
        "external_delivery_enabled": False,
        "generated_at": observed_at,
        "items": items,
        "total": len(items),
        "counts": counts,
        "empty_state": "No named-agent activity is visible yet.",
    }


def normalize_tui_event(event: dict) -> dict:
    """Normalize a TUI event into the BIF-493 read-only lane grammar."""

    return _normalize_event(event)


def normalize_gateway_observer_event(event: Mapping[str, Any]) -> dict:
    """Normalize a gateway lifecycle observation into lane grammar."""

    return _normalize_event(event)


def publish_cockpit_event(event: Mapping[str, Any]) -> dict:
    """Best-effort, non-blocking publish of a display-safe cockpit event.

    The observer bus is process-local and read-only: it does not persist to
    SessionDB, write files, enqueue gateway work, or affect routing.  Slow/full
    subscribers are dropped silently so gateway call sites never block.
    """

    normalized = normalize_gateway_observer_event(event)
    with _COCKPIT_LOCK:
        _COCKPIT_RECENT_EVENTS.append(copy.deepcopy(normalized))
        subscribers = list(_COCKPIT_SUBSCRIBERS)
    stale: list[queue.Queue] = []
    for subscriber in subscribers:
        try:
            subscriber.put_nowait(copy.deepcopy(normalized))
        except Exception:
            stale.append(subscriber)
    if stale:
        with _COCKPIT_LOCK:
            for subscriber in stale:
                try:
                    _COCKPIT_SUBSCRIBERS.remove(subscriber)
                except ValueError:
                    pass
    return normalized


def subscribe_cockpit_events(*, replay_recent: bool = True) -> queue.Queue:
    """Subscribe to process-local cockpit events with a bounded queue."""

    subscriber: queue.Queue = queue.Queue(maxsize=_COCKPIT_SUBSCRIBER_MAXSIZE)
    with _COCKPIT_LOCK:
        if replay_recent:
            # Replay newest observations first so short-lived clients asking for a
            # small event limit see the most recent real signal rather than stale
            # process-local history from prior requests/tests.
            for event in reversed(_COCKPIT_RECENT_EVENTS):
                try:
                    subscriber.put_nowait(copy.deepcopy(event))
                except Exception:
                    break
        _COCKPIT_SUBSCRIBERS.append(subscriber)
    return subscriber


def unsubscribe_cockpit_events(subscriber: queue.Queue) -> None:
    """Remove a cockpit event subscription, ignoring stale subscribers."""

    with _COCKPIT_LOCK:
        try:
            _COCKPIT_SUBSCRIBERS.remove(subscriber)
        except ValueError:
            pass


def _signal_category_label(name: str) -> str:
    return {
        "now": "Now",
        "needs_marco": "Needs Marco",
        "stuck_failed": "Stuck/Failed",
        "waiting": "Waiting",
        "recently_completed": "Recently Completed",
        "active_role_work": "Active Role Work",
        "recent_context": "Recent Context",
        "archive": "Archive",
    }.get(name, name.replace("_", " ").title())


def _freshness_classification(value: Any, now: float, *, enforce: bool) -> dict[str, Any]:
    timestamp = _timestamp_sort_value(value)
    if not enforce or timestamp == float("-inf"):
        return {"bucket": "operational", "age_seconds": None, "threshold_seconds": _OPERATIONAL_FRESH_SECONDS}
    age = max(0.0, now - timestamp)
    if age <= _OPERATIONAL_FRESH_SECONDS:
        bucket = "operational"
    elif age <= _RECENT_CONTEXT_SECONDS:
        bucket = "recent_context"
    else:
        bucket = "archive"
    return {
        "bucket": bucket,
        "age_seconds": age,
        "threshold_seconds": _OPERATIONAL_FRESH_SECONDS,
        "archive_threshold_seconds": _RECENT_CONTEXT_SECONDS,
    }


def _stale_source_class(provenance: Any) -> str:
    return "event" if str(provenance or "") == "lane_event" else "sessiondb"


def _needs_marco_attention(latest_message: Optional[Mapping[str, Any]], status_text: str, message_text: str) -> bool:
    if _contains_any(status_text, _EXPLICIT_ATTENTION_TERMS):
        return True
    if not latest_message:
        return False
    role = str(latest_message.get("role") or "").strip().lower()
    if role in {"user", "operator"}:
        return "?" in message_text or _contains_any(message_text, _EXPLICIT_ATTENTION_TERMS)
    return False


def _base_signal(
    lane: Mapping[str, Any],
    message: Optional[Mapping[str, Any]],
    *,
    timestamp: Any,
    provenance: str,
) -> dict[str, Any]:
    alias = lane.get("alias") if isinstance(lane.get("alias"), Mapping) else {}
    title = lane.get("title") or alias.get("alias") or lane.get("platform") or "Observed lane"
    source = {
        "lane_id": _clean_scalar(lane.get("lane_id")),
        "lane_title": _clean_scalar(title),
        "platform": _clean_scalar(lane.get("platform")),
        "status": _clean_scalar(lane.get("status")),
        "agent_role": _clean_scalar(lane.get("agent_role")),
        "role_label": _clean_scalar(lane.get("role_label")),
        "delegate_kind": _clean_scalar(lane.get("delegate_kind")),
        "issue_identifier": _clean_scalar(lane.get("issue_identifier")),
        "goal": _clean_scalar(lane.get("goal")),
    }
    if message:
        source["message_role"] = _clean_scalar(message.get("role"))
    return {
        "id": _signal_id(str(lane.get("lane_id") or ""), provenance, timestamp, title),
        "lane_id": str(lane.get("lane_id") or ""),
        "title": _clean_scalar(title),
        "source": source,
        "provenance": provenance,
        "timestamp": _clean_scalar(timestamp),
    }


def _signal_id(lane_id: str, provenance: str, timestamp: Any, title: Any) -> str:
    material = f"{lane_id}\0{provenance}\0{timestamp}\0{title}".encode("utf-8", "surrogatepass")
    token = base64.urlsafe_b64encode(hashlib.sha256(material).digest()[:12]).decode("ascii").rstrip("=")
    return f"signal_{token}"


def _latest_message(messages: Iterable[Mapping[str, Any]]) -> Optional[dict[str, Any]]:
    latest: Optional[dict[str, Any]] = None
    latest_value = float("-inf")
    for message in messages:
        value = _timestamp_sort_value(_max_timestamp(message.get("timestamp"), message.get("created_at")))
        if value >= latest_value:
            latest = dict(message)
            latest_value = value
    return latest


def _signal_text(*parts: Any) -> str:
    return " ".join(str(part or "") for part in parts if part is not None).lower()


def _contains_any(text: str, terms: Iterable[str]) -> bool:
    return any(term in text for term in terms)


def _lane_has_active_role(lane: Mapping[str, Any]) -> bool:
    role = _canonical_agent_role(lane)
    if role in _ACTIVE_ROLES:
        return True
    alias = lane.get("alias") if isinstance(lane.get("alias"), Mapping) else {}
    role_text = _signal_text(lane.get("role"), lane.get("title"), alias.get("alias"))
    return _contains_any(role_text, _ACTIVE_ROLES)


def _reason(prefix: str, detail: str) -> str:
    detail = detail.strip()
    return f"{prefix}: {detail}" if detail else prefix


def _dedupe_signals(signals: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for signal in signals:
        key = (str(signal.get("lane_id") or ""), str(signal.get("provenance") or ""))
        current = best.get(key)
        if current is None or (float(signal.get("confidence") or 0), _timestamp_sort_value(signal.get("timestamp"))) > (
            float(current.get("confidence") or 0),
            _timestamp_sort_value(current.get("timestamp")),
        ):
            best[key] = dict(signal)
    return list(best.values())


def _status_priority(status: Any) -> int:
    text = str(status or "").lower()
    if _contains_any(text, _FAILED_STATUSES):
        return 50
    if _contains_any(text, _ACTIVE_STATUSES):
        return 40
    if _contains_any(text, _WAITING_STATUSES):
        return 30
    if _contains_any(text, _COMPLETED_STATUSES):
        return 20
    return 10 if text else 0


def _max_timestamp(*values: Any) -> Any:
    best_value: Any = None
    best_sort = float("-inf")
    for value in values:
        sort_value = _timestamp_sort_value(value)
        if sort_value > best_sort:
            best_value = value
            best_sort = sort_value
    return best_value


def _agent_activity_status(*statuses: Any, freshness: Mapping[str, Any]) -> str:
    text = _signal_text(*statuses)
    if freshness.get("bucket") != "operational" and not _contains_any(text, _FAILED_STATUSES + _COMPLETED_STATUSES):
        return "stale"
    if _contains_any(text, ("blocked", "stuck", "needs_marco", "needs operator", "needs human", "approval")):
        return "blocked"
    if _contains_any(text, ("failed", "error", "offline")):
        return "failed"
    if _contains_any(text, _COMPLETED_STATUSES):
        return "completed"
    if _contains_any(text, _WAITING_STATUSES):
        return "waiting"
    if _contains_any(text, _ACTIVE_STATUSES):
        return "running"
    return "waiting"


def _agent_activity_status_priority(status: Any) -> int:
    return {"failed": 60, "blocked": 55, "running": 50, "waiting": 40, "stale": 30, "completed": 20}.get(str(status or ""), 0)


def _short_evidence(value: Any, limit: int = 180) -> str:
    text = str(_clean_scalar(value or "") or "").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return "No display-safe evidence in the bounded recent window."
    if len(text) <= limit:
        return text
    return f"{text[:limit].rstrip()}…"


def _timestamp_sort_value(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value:
        try:
            from datetime import datetime

            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except Exception:
            return float("-inf")
    return float("-inf")


def _recency_label(value: Any, now: float) -> str:
    timestamp = _timestamp_sort_value(value)
    if timestamp == float("-inf"):
        return "observed"
    age = max(0, now - timestamp)
    if age < 60:
        return "just now"
    if age < 3600:
        minutes = int(age // 60)
        return f"{minutes}m ago"
    if age < 86400:
        hours = int(age // 3600)
        return f"{hours}h ago"
    days = int(age // 86400)
    return f"{days}d ago"


def _normalize_event(event: Mapping[str, Any]) -> dict:
    raw_type = str(event.get("type") or "status")
    event_type = _EVENT_TYPE_MAP.get(raw_type, raw_type if raw_type.startswith("lane.") else f"lane.{raw_type}")
    allowed = _ALLOWED_PAYLOAD_FIELDS.get(event_type, ("lane_id", "status", "updated_at"))

    source = event.get("source") or event.get("platform") or "unknown"
    canonical_id = event.get("canonical_id") or event.get("session_id") or event.get("lane_key") or ""
    payload: dict[str, Any] = {"lane_id": build_lane_id(source, canonical_id)}

    for field in allowed:
        if field == "lane_id":
            continue
        if field in event:
            payload[field] = _clean_scalar(event[field])
        elif field == "redacted_summary":
            payload[field] = _safe_summary(event)

    return {
        "type": event_type,
        "schema_version": COCKPIT_SCHEMA_VERSION,
        "payload": payload,
    }


def _display_alias(platform: Any, chat_type: Any, title: Any, role: Optional[Any]) -> str:
    role_text = str(role or "").strip().lower()
    chat_text = str(chat_type or "").strip().lower()
    title_text = _clean_scalar(title)

    if chat_text == "operator" or role_text == "operator":
        return "%operator"
    if role_text in {"biff", "forge"}:
        return f"@{role_text}"
    if chat_text in {"channel", "group", "guild", "room"} and title_text:
        cleaned = str(title_text).strip().lstrip("#")
        return f"#{cleaned.lower()}"
    if title_text:
        return str(title_text)
    if role_text:
        return f"@{role_text}"
    platform_text = str(platform or "lane").strip().lower() or "lane"
    return platform_text


def _first_text(*values: Any) -> Optional[str]:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None



def _canonical_agent_role(session: Mapping[str, Any]) -> Optional[str]:
    for key in ("agent_role", "role", "subagent_role"):
        value = str(session.get(key) or "").strip().lower()
        if value in _ACTIVE_ROLES:
            return value
    title = str(session.get("title") or session.get("name") or "").strip().lower()
    for role in _ACTIVE_ROLES:
        if title == role or title.startswith(f"{role} ") or title.startswith(f"[{role}]"):
            return role
    return None


def _status_from_session(session: Mapping[str, Any]) -> str:
    explicit = str(session.get("status") or "").strip()
    if explicit:
        return explicit
    if session.get("ended_at") is not None:
        reason = str(session.get("end_reason") or "").lower()
        if any(token in reason for token in ("error", "fail", "timeout")):
            return "failed"
        return "completed"
    return "running" if session.get("parent_session_id") else "observing"


def _opaque_ref(value: Any, *, prefix: str) -> str:
    digest = hashlib.sha256(str(value).encode("utf-8", "surrogatepass")).hexdigest()[:12]
    return f"{prefix}_{digest}"

def _safe_summary(event: Mapping[str, Any]) -> str:
    summary = event.get("redacted_summary") or event.get("summary") or event.get("message") or ""
    return str(_clean_scalar(summary or "redacted"))


def _project_message_content(message: Mapping[str, Any]) -> Optional[str]:
    role = str(message.get("role") or "").strip().lower()
    if role == "tool":
        if message.get("content") is not None or message.get("output") is not None:
            return "[tool output redacted]"
        return None

    raw = message.get("content") if "content" in message else message.get("text")
    text = _extract_display_text(raw)
    if text is None:
        return None
    if _is_internal_prompt_text(text):
        return "[internal prompt hidden]"
    return str(_clean_scalar(text))


def _is_internal_prompt_text(text: str) -> bool:
    stripped = text.lstrip()
    return any(stripped.startswith(prefix) for prefix in _INTERNAL_PROMPT_PREFIXES)


def _extract_display_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, (list, tuple)):
        parts = [_extract_display_text(item) for item in value]
        return "\n".join(part for part in parts if part) or None
    if isinstance(value, Mapping):
        value_type = str(value.get("type") or "").lower()
        if value_type and value_type not in {"text", "input_text", "output_text"}:
            return None
        if isinstance(value.get("text"), str):
            return value.get("text")
        if isinstance(value.get("content"), str):
            return value.get("content")
        if "parts" in value:
            return _extract_display_text(value.get("parts"))
    return None


def _clean_scalar(value: Any) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        return _redact_text(value)
    if isinstance(value, (list, tuple, set)):
        return [_clean_scalar(item) for item in value if not _looks_sensitive_value(item)]
    if isinstance(value, Mapping):
        clean: dict[str, Any] = {}
        for key, nested in value.items():
            if _SECRET_KEY_RE.search(str(key)):
                continue
            if _looks_sensitive_value(nested):
                continue
            clean[str(key)] = _clean_scalar(nested)
        return clean
    return _redact_text(str(value))


def _looks_sensitive_value(value: Any) -> bool:
    if isinstance(value, str):
        return bool(_SECRET_VALUE_RE.search(value) or _SECRET_KEY_RE.search(value))
    if isinstance(value, Mapping):
        return any(_SECRET_KEY_RE.search(str(key)) or _looks_sensitive_value(nested) for key, nested in value.items())
    if isinstance(value, (list, tuple, set)):
        return any(_looks_sensitive_value(item) for item in value)
    return False


def _redact_text(text: str) -> str:
    text = _URL_RE.sub("[endpoint]", text)
    text = _HOSTPORT_RE.sub("[endpoint]", text)
    text = _BARE_DOMAIN_RE.sub("[endpoint]", text)
    text = _ENDPOINT_PATH_RE.sub("[endpoint]", text)
    text = _GATEWAY_SESSION_PHRASE_RE.sub("[id]", text)
    text = _SECRET_ASSIGNMENT_RE.sub("[redacted]", text)
    text = _PLATFORM_ID_RE.sub("[id]", text)
    text = _SECRET_VALUE_RE.sub("[redacted]", text)
    text = _LONG_NUMERIC_ID_RE.sub("[id]", text)
    return text


__all__ = [
    "COCKPIT_SCHEMA_VERSION",
    "OPERATOR_CURRENT_ALIAS",
    "OPERATOR_CURRENT_TITLE",
    "CockpitLogicalLane",
    "CockpitLocalStore",
    "default_logical_lane_ids",
    "build_lane_id",
    "infer_lane_alias",
    "build_lane_snapshot_from_session",
    "coalesce_logical_lane_snapshots",
    "mark_ambiguous_aliases",
    "normalize_api_run_event",
    "normalize_tui_event",
    "normalize_gateway_observer_event",
    "publish_cockpit_event",
    "subscribe_cockpit_events",
    "unsubscribe_cockpit_events",
    "recent_cockpit_events",
    "project_cockpit_signals",
    "project_agent_activity",
    "project_message_for_cockpit",
]
