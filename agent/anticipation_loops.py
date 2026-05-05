"""Anticipatory loop implementations.

V0 is intentionally heuristic-only. No LLM calls, no delivery, no scheduling.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Mapping

from agent.anticipation import AnticipationPermission
from agent.anticipation_policy import AnticipationCandidate

UNRESOLVED_PATTERNS: tuple[tuple[re.Pattern[str], float], ...] = (
    (re.compile(r"\bTODO\b", re.IGNORECASE), 0.12),
    (re.compile(r"\bnext step\b", re.IGNORECASE), 0.10),
    (re.compile(r"\bnext I can\b", re.IGNORECASE), 0.10),
    (re.compile(r"\bif you want\b", re.IGNORECASE), 0.08),
    (re.compile(r"\bfollow up\b", re.IGNORECASE), 0.08),
    (re.compile(r"\bshall I proceed\b", re.IGNORECASE), 0.08),
    (re.compile(r"\bwe should\b", re.IGNORECASE), 0.06),
    (re.compile(r"\blater\b", re.IGNORECASE), 0.04),
    (re.compile(r"\bunresolved\b", re.IGNORECASE), 0.04),
)

_COMPLETION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(done|completed|complete|finished|handled|resolved)\b", re.IGNORECASE),
    re.compile(r"\b(already did|we did|that part is done)\b", re.IGNORECASE),
    re.compile(r"\b(no|not now|skip it|nevermind|never mind)\b", re.IGNORECASE),
)


def build_stale_task_candidates(
    db: Any,
    *,
    now: datetime | None = None,
    lookback_days: int = 14,
    limit: int = 5,
    current_session_id: str | None = None,
) -> list[AnticipationCandidate]:
    """Build heuristic stale-task candidates from recent session history.

    The returned candidates are suggestions only. Callers must pass them through
    the anticipation policy gate before displaying, logging, or delivering.
    """

    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=max(1, lookback_days))
    sessions = db.list_sessions_rich(limit=max(limit * 10, 20), exclude_sources=["tool"])

    candidates: list[tuple[AnticipationCandidate, float]] = []
    for session in sessions:
        session_id = str(session.get("id") or "")
        if not session_id or session_id == current_session_id:
            continue
        if session.get("parent_session_id"):
            continue
        last_active = _coerce_datetime(session.get("last_active") or session.get("started_at"))
        if last_active and last_active < cutoff:
            continue

        messages = _conversation_messages(db, session_id)
        signal = _find_latest_unresolved_signal(messages)
        if not signal:
            continue
        signal_text, signal_score = signal
        confidence = min(0.95, 0.62 + signal_score + _recency_bonus(last_active, now))
        candidate = AnticipationCandidate(
            loop_id="stale_task_resurfacer",
            title=_candidate_title(session),
            body=_candidate_body(session, signal_text),
            confidence=round(confidence, 3),
            proposed_permission=AnticipationPermission.SUGGEST,
            dedupe_key=_dedupe_key(session_id, signal_text),
            created_at=now,
        )
        candidates.append((candidate, last_active.timestamp() if last_active else 0.0))

    candidates.sort(key=lambda item: (item[0].confidence, item[1]), reverse=True)
    return [candidate for candidate, _ in candidates[: max(0, limit)]]


def build_router_monitor_candidates(
    snapshot: Mapping[str, Any],
    *,
    now: datetime | None = None,
    stale_minutes: int = 30,
    limit: int = 5,
) -> list[AnticipationCandidate]:
    """Build router-monitor anticipation candidates from a sanitized snapshot.

    The snapshot is intentionally plain data. This loop does not SSH into the
    router, mutate router state, send Telegram messages, or schedule itself.
    """

    now = now or datetime.now(timezone.utc)
    candidates: list[AnticipationCandidate] = []

    health = _router_health_candidate(snapshot, now=now, stale_minutes=stale_minutes)
    if health:
        candidates.append(health)

    for device in _unknown_devices(snapshot):
        candidate = _router_unknown_device_candidate(device, now=now)
        if candidate:
            candidates.append(candidate)

    candidates.sort(
        key=lambda candidate: (
            _permission_sort_rank(candidate.proposed_permission),
            candidate.confidence,
        ),
        reverse=True,
    )
    return candidates[: max(0, limit)]


def _router_health_candidate(
    snapshot: Mapping[str, Any],
    *,
    now: datetime,
    stale_minutes: int,
) -> AnticipationCandidate | None:
    monitoring = snapshot.get("monitoring", {})
    if not isinstance(monitoring, Mapping):
        return None

    findings: list[str] = []
    cron_entries = [str(entry) for entry in monitoring.get("cron_entries", []) if str(entry).strip()]
    if monitoring.get("quarantine_dir_exists") is False:
        findings.append("/root/quarantine directory is missing")
    if not any("quarantine-v3.sh run" in entry for entry in cron_entries):
        findings.append("minutely quarantine cron job is missing")
    if not any("gl-ngx-watchdog.sh" in entry for entry in cron_entries):
        findings.append("GL.iNet watchdog cron job is missing")
    if not any("cron-guard.sh" in entry for entry in cron_entries):
        findings.append("cron guard job is missing")

    cron_log_last_at = _coerce_datetime(monitoring.get("cron_log_last_at"))
    if cron_log_last_at is None:
        findings.append("cron log timestamp is unavailable")
    elif now - cron_log_last_at > timedelta(minutes=max(1, stale_minutes)):
        age = int((now - cron_log_last_at).total_seconds() // 60)
        findings.append(f"cron log is stale by {age} minutes")

    segfault_count = _safe_int(monitoring.get("segfault_count_7d"), 0)
    if segfault_count > 0:
        findings.append(f"cron log contains {segfault_count} segmentation fault lines in the last 7 days")

    if not findings:
        return None

    confidence = 0.88
    if monitoring.get("quarantine_dir_exists") is False or not cron_entries:
        confidence = 0.94
    elif cron_log_last_at and now - cron_log_last_at > timedelta(hours=1):
        confidence = 0.91

    body = "\n".join(
        [
            "Router monitoring may be degraded.",
            "",
            "Findings:",
            *[f"- {finding}" for finding in findings],
            "",
            "Suggested next action: run a router status check, inspect cron/quarantine logs, and repair crond/watchdog only after confirmation.",
            "I did not change router state.",
        ]
    )
    return AnticipationCandidate(
        loop_id="router_monitor",
        title="Router monitoring may be degraded",
        body=body,
        confidence=confidence,
        proposed_permission=AnticipationPermission.ASK_TO_EXECUTE,
        dedupe_key="router_monitor:health:" + _hash_key("|".join(findings)),
        created_at=now,
    )


def _router_unknown_device_candidate(
    device: Mapping[str, Any],
    *,
    now: datetime,
) -> AnticipationCandidate | None:
    mac = _clean_mac(device.get("mac"))
    if not mac:
        return None
    active = _safe_bool(device.get("active", False), False)
    locally_administered = _safe_bool(device.get("locally_administered", False), False)
    seen_count = _safe_int(device.get("seen_count_72h"), 1)
    hostnames = _string_list(device.get("hostnames"))
    ips = _string_list(device.get("ips"))
    traffic_summary = _clean_router_text(device.get("traffic_summary"), default="no traffic summary available", max_length=160)

    if active:
        body = "\n".join(
            [
                "Router anticipation: active unknown device worth checking.",
                "",
                f"- MAC: {mac}",
                f"- IPs: {', '.join(ips) if ips else 'unknown'}",
                f"- Hostnames: {', '.join(hostnames) if hostnames else 'none reported'}",
                f"- Seen count in 72h: {seen_count}",
                f"- Traffic: {traffic_summary}",
                "",
                "Suggested next action: review router status before approving or blocking anything.",
                "I did not block anything or change router state.",
            ]
        )
        return AnticipationCandidate(
            loop_id="router_monitor",
            title=f"Router active unknown device: {mac[:8]}",
            body=body,
            confidence=0.89 if not locally_administered else 0.84,
            proposed_permission=AnticipationPermission.ASK_TO_EXECUTE,
            dedupe_key=f"router_monitor:active_unknown:{mac}",
            created_at=now,
        )

    if locally_administered and seen_count >= 3:
        body = "\n".join(
            [
                "Router anticipation: recurring randomized/private MAC.",
                "",
                f"- MAC: {mac}",
                f"- Hostnames: {', '.join(hostnames) if hostnames else 'none reported'}",
                f"- Seen count in 72h: {seen_count}",
                f"- Traffic: {traffic_summary}",
                "",
                "Suggested next action: correlate it against guests, Deco logs, or a known phone using randomized Wi-Fi MACs.",
                "I did not approve, block, or message anyone automatically.",
            ]
        )
        return AnticipationCandidate(
            loop_id="router_monitor",
            title=f"Router recurring randomized unknown: {mac[:8]}",
            body=body,
            confidence=0.80,
            proposed_permission=AnticipationPermission.SUGGEST,
            dedupe_key=f"router_monitor:recurring_randomized:{mac}",
            created_at=now,
        )

    if seen_count >= 3:
        body = "\n".join(
            [
                "Router anticipation: recurring unknown device.",
                "",
                f"- MAC: {mac}",
                f"- Hostnames: {', '.join(hostnames) if hostnames else 'none reported'}",
                f"- Seen count in 72h: {seen_count}",
                f"- Traffic: {traffic_summary}",
                "",
                "Suggested next action: correlate it against guests, Deco logs, or known household devices before approving or blocking.",
                "I did not approve, block, or message anyone automatically.",
            ]
        )
        return AnticipationCandidate(
            loop_id="router_monitor",
            title=f"Router recurring unknown: {mac[:8]}",
            body=body,
            confidence=0.78,
            proposed_permission=AnticipationPermission.SUGGEST,
            dedupe_key=f"router_monitor:recurring_unknown:{mac}",
            created_at=now,
        )

    if locally_administered:
        body = "\n".join(
            [
                "Router anticipation: one-time randomized/private MAC observed.",
                "",
                f"- MAC: {mac}",
                f"- Seen count in 72h: {seen_count}",
                f"- Traffic: {traffic_summary}",
                "",
                "Decision: silent log only. No Telegram-worthy interruption.",
            ]
        )
        return AnticipationCandidate(
            loop_id="router_monitor",
            title=f"Router one-time randomized unknown: {mac[:8]}",
            body=body,
            confidence=0.72,
            proposed_permission=AnticipationPermission.SILENT_LOG,
            dedupe_key=f"router_monitor:one_time_randomized:{mac}",
            created_at=now,
        )

    return None


def _unknown_devices(snapshot: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = snapshot.get("unknown_devices", [])
    if not isinstance(raw, list):
        return []
    return [device for device in raw if isinstance(device, Mapping)]


def _clean_mac(value: object) -> str:
    text = str(value or "").strip().lower()
    if re.fullmatch(r"([0-9a-f]{2}:){5}[0-9a-f]{2}", text):
        return text
    return ""


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned: list[str] = []
    for item in value[:10]:
        text = _clean_router_text(item, max_length=80)
        if text:
            cleaned.append(text)
    return cleaned


def _clean_router_text(value: object, *, default: str = "", max_length: int = 120) -> str:
    text = str(value or "")
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
    text = "".join(char if char.isprintable() else " " for char in text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return default
    if len(text) > max_length:
        return text[: max(0, max_length - 1)].rstrip() + "…"
    return text


def _safe_bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off", ""}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _safe_int(value: object, default: int) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _hash_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _permission_sort_rank(permission: AnticipationPermission) -> int:
    return {
        AnticipationPermission.SILENT_LOG: 0,
        AnticipationPermission.SUGGEST: 1,
        AnticipationPermission.DRAFT: 2,
        AnticipationPermission.ASK_TO_EXECUTE: 3,
        AnticipationPermission.EXECUTE_SAFE: 4,
    }[permission]


def _conversation_messages(db: Any, session_id: str) -> list[dict[str, Any]]:
    if hasattr(db, "get_messages_as_conversation"):
        messages = db.get_messages_as_conversation(session_id)
    else:
        messages = db.get_messages(session_id)
    return [message for message in messages if isinstance(message, dict)]


def _find_latest_unresolved_signal(messages: Iterable[dict[str, Any]]) -> tuple[str, float] | None:
    ordered = list(messages)
    best: tuple[int, str, float] | None = None
    for index, message in enumerate(ordered):
        if message.get("role") not in {"assistant", "user"}:
            continue
        content = str(message.get("content") or "").strip()
        if not content:
            continue
        score = _unresolved_score(content)
        if score <= 0:
            continue
        if _later_user_message_completes_thread(ordered[index + 1 :]):
            continue
        if best is None or index > best[0]:
            best = (index, content, score)
    if best is None:
        return None
    return best[1], best[2]


def _unresolved_score(content: str) -> float:
    return sum(weight for pattern, weight in UNRESOLVED_PATTERNS if pattern.search(content))


def _later_user_message_completes_thread(messages: Iterable[dict[str, Any]]) -> bool:
    for message in messages:
        if message.get("role") != "user":
            continue
        content = str(message.get("content") or "")
        if any(pattern.search(content) for pattern in _COMPLETION_PATTERNS):
            return True
    return False


def _candidate_title(session: dict[str, Any]) -> str:
    title = str(session.get("title") or session.get("preview") or session.get("id") or "Untitled session")
    return f"Stale thread: {title[:80]}"


def _candidate_body(session: dict[str, Any], signal_text: str) -> str:
    title = str(session.get("title") or session.get("preview") or "previous conversation")
    snippet = _single_line(signal_text)[:240]
    return (
        f"A previous Hermes session may have an unresolved next step.\n\n"
        f"Session: {title}\n"
        f"Signal: {snippet}\n\n"
        f"I did not take action; this is only a resurfacing suggestion."
    )


def _dedupe_key(session_id: str, signal_text: str) -> str:
    digest = hashlib.sha256(_single_line(signal_text).encode("utf-8")).hexdigest()[:16]
    return f"stale_task_resurfacer:{session_id}:{digest}"


def _single_line(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _recency_bonus(last_active: datetime | None, now: datetime) -> float:
    if not last_active:
        return 0.0
    age_days = max(0.0, (now - last_active).total_seconds() / 86400)
    if age_days <= 2:
        return 0.05
    if age_days <= 7:
        return 0.03
    return 0.0


def _coerce_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (TypeError, ValueError, OSError, OverflowError):
            if not isinstance(value, str):
                return None
            try:
                dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
