"""Agent-level runtime state for gateway sessions.

Sessions remain transcript windows. This module adds a small actor layer above
them: per-platform identity, append-only events, active directives, structured
state packets, and outbound policy checks.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

from gateway.session_context import get_session_env


_PUBLIC_PLATFORM_NAMES = {
    "discord",
    "slack",
    "telegram",
    "matrix",
    "mattermost",
    "feishu",
    "dingtalk",
    "wecom",
    "wecom_callback",
    "weixin",
    "qqbot",
}

_AUTONOMOUS_SOURCE_NAMES = {
    "cron",
    "hub",
    "webhook",
    "homeassistant",
    "api_server",
}

_STOP_WORD_RE = re.compile(
    r"\b(stop|turn\s+this\s+off|turn\s+off|disable|pause|shut\s+off|kill)\b",
    re.IGNORECASE,
)
_PUBLIC_BROADCAST_RE = re.compile(
    r"\b(post(?:ing)?|broadcast(?:ing)?|send(?:ing)?|digest|market|alpha|cron|general|channel|public)\b",
    re.IGNORECASE,
)


@dataclass
class GateDecision:
    allowed: bool
    reason: str = ""
    policy: str = ""
    event_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "policy": self.policy,
            "event_id": self.event_id,
        }


def _platform_value(value: Any) -> str:
    if hasattr(value, "value"):
        return str(value.value)
    return str(value or "")


def _source_to_payload(source: Any) -> Dict[str, Any]:
    try:
        return source.to_dict()
    except Exception:
        return {}


def _preview(text: Any, limit: int = 500) -> str:
    s = str(text or "").replace("\r", " ").replace("\n", " ").strip()
    if len(s) <= limit:
        return s
    return s[: limit - 3].rstrip() + "..."


def _load_db(db=None):
    if db is not None:
        return db, False
    from hermes_state import SessionDB

    return SessionDB(), True


def _split_identifier_csv(raw: str) -> set[str]:
    return {part.strip() for part in str(raw or "").split(",") if part.strip()}


def owner_user_ids_for_platform(platform: str) -> set[str]:
    """Return owner user IDs for a trusted platform.

    The provisioner-owned identity should eventually be emitted as env/config.
    Until then, SOUL.md is the generated local source available on agent VMs.
    Keep this narrowly scoped: only the explicit "Your owner" block is parsed.
    """
    platform = str(platform or "").strip().lower()
    ids: set[str] = set()
    ids.update(_split_identifier_csv(os.getenv("GATEWAY_OWNER_USER_IDS", "")))
    if platform:
        prefix = platform.upper()
        ids.update(_split_identifier_csv(os.getenv(f"{prefix}_OWNER_USER_ID", "")))
        ids.update(_split_identifier_csv(os.getenv(f"{prefix}_OWNER_USER_IDS", "")))
    if ids or platform != "discord":
        return ids

    try:
        from hermes_constants import get_hermes_home

        soul = get_hermes_home() / "SOUL.md"
        text = soul.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ids

    owner_match = re.search(r"(?ms)^## Your owner\b(?P<body>.*?)(?:^## |\Z)", text)
    if not owner_match:
        return ids
    body = owner_match.group("body")
    for pattern in (
        r"Discord:\s*`?@?[^`\n]*`?\s*\(user_id\s*`?(\d{5,})`?\)",
        r"\buser_id\s*`?(\d{5,})`?",
    ):
        ids.update(re.findall(pattern, body))
    return ids


def infer_platform_authority(source: Any) -> str:
    """Best-effort v1 authority without cross-platform person unification."""
    platform = _platform_value(getattr(source, "platform", ""))
    user_id = str(getattr(source, "user_id", "") or "")
    chat_type = str(getattr(source, "chat_type", "") or "")
    if not platform or not user_id:
        return "system"
    check_ids = {user_id}
    if "@" in user_id:
        check_ids.add(user_id.split("@", 1)[0])
    owner_ids = owner_user_ids_for_platform(platform)

    platform_allowlist_env = {
        "telegram": "TELEGRAM_ALLOWED_USERS",
        "discord": "DISCORD_ALLOWED_USERS",
        "whatsapp": "WHATSAPP_ALLOWED_USERS",
        "slack": "SLACK_ALLOWED_USERS",
        "signal": "SIGNAL_ALLOWED_USERS",
        "email": "EMAIL_ALLOWED_USERS",
        "sms": "SMS_ALLOWED_USERS",
        "mattermost": "MATTERMOST_ALLOWED_USERS",
        "matrix": "MATRIX_ALLOWED_USERS",
        "dingtalk": "DINGTALK_ALLOWED_USERS",
        "feishu": "FEISHU_ALLOWED_USERS",
        "wecom": "WECOM_ALLOWED_USERS",
        "wecom_callback": "WECOM_CALLBACK_ALLOWED_USERS",
        "weixin": "WEIXIN_ALLOWED_USERS",
        "bluebubbles": "BLUEBUBBLES_ALLOWED_USERS",
        "qqbot": "QQ_ALLOWED_USERS",
        "hub": "HUB_ALLOWED_USERS",
    }
    if owner_ids and check_ids & owner_ids:
        return "owner"
    platform_allow_all_env = f"{platform.upper()}_ALLOW_ALL_USERS"
    if os.getenv(platform_allow_all_env, "").strip().lower() in {"1", "true", "yes"}:
        return "user"

    allowed_raw = ",".join(
        part
        for part in (
            os.getenv(platform_allowlist_env.get(platform, ""), ""),
            os.getenv("GATEWAY_ALLOWED_USERS", ""),
        )
        if part
    )
    allowed = {p.strip() for p in allowed_raw.split(",") if p.strip()}
    if "*" in allowed:
        return "trusted"
    if allowed and check_ids & allowed:
        return "trusted"
    if chat_type == "dm":
        return "known"
    return "user"


def resolve_identity(db, source: Any, authority: str = "") -> str:
    platform = _platform_value(getattr(source, "platform", ""))
    user_id = str(getattr(source, "user_id", "") or "")
    if not platform or not user_id:
        return ""
    authority = authority or infer_platform_authority(source)
    return db.upsert_agent_identity(
        platform=platform,
        platform_user_id=user_id,
        display_name=str(getattr(source, "user_name", "") or ""),
        authority=authority,
        payload={
            "chat_id": str(getattr(source, "chat_id", "") or ""),
            "chat_type": str(getattr(source, "chat_type", "") or ""),
            "thread_id": str(getattr(source, "thread_id", "") or ""),
        },
    )


def record_inbound_event(
    db,
    *,
    source: Any,
    session_id: str,
    session_key: str,
    text: str,
    message_id: str = "",
    platform_update_id: str = "",
    authority: str = "",
) -> tuple[str, str]:
    """Record one inbound MessageEvent and return (event_id, person_id)."""
    person_id = resolve_identity(db, source, authority=authority)
    platform = _platform_value(getattr(source, "platform", ""))
    chat_type = str(getattr(source, "chat_type", "") or "")
    event_id = db.append_agent_event(
        event_type="inbound",
        event_subtype="message",
        status="received",
        session_id=session_id,
        session_key=session_key,
        actor_id="main",
        actor_kind="user" if person_id else "system",
        source=platform,
        person_id=person_id,
        sender_user_id=str(getattr(source, "user_id", "") or ""),
        sender_name=str(getattr(source, "user_name", "") or ""),
        chat_type=chat_type,
        audience_type="private" if chat_type == "dm" else "shared",
        platform=platform,
        platform_chat_id=str(getattr(source, "chat_id", "") or ""),
        platform_thread_id=str(getattr(source, "thread_id", "") or ""),
        platform_message_id=str(message_id or ""),
        platform_update_id=str(platform_update_id or ""),
        content=_preview(text, 1000),
        payload={
            "source": _source_to_payload(source),
            "authority": authority or infer_platform_authority(source),
        },
    )
    return event_id, person_id


def detect_public_broadcast_stop_directive(text: str) -> Optional[Dict[str, Any]]:
    """Conservative v1 detector for owner/trusted stop-broadcast directives."""
    if not text:
        return None
    if not _STOP_WORD_RE.search(text):
        return None
    if not _PUBLIC_BROADCAST_RE.search(text):
        return None
    return {
        "text": _preview(text, 1000),
        "target": "public_broadcasts",
        "behavior": "suppress",
        "reason": "trusted sender asked the agent to stop/disable public broadcast behavior",
    }


def maybe_record_directive_from_inbound(
    db,
    *,
    source: Any,
    session_id: str,
    session_key: str,
    inbound_event_id: str,
    person_id: str,
    text: str,
    authority: str,
) -> Optional[str]:
    """Persist a durable directive extracted from an inbound trusted message."""
    if authority not in {"trusted", "owner"}:
        return None
    directive = detect_public_broadcast_stop_directive(text)
    if not directive:
        return None
    platform = _platform_value(getattr(source, "platform", ""))
    return db.create_or_replace_agent_directive(
        directive_scope="actor",
        directive_key="public-broadcast-suppression",
        directive_type="suppress_public_broadcasts",
        payload={**directive, "source_event_id": inbound_event_id},
        session_id=session_id,
        session_key="",
        actor_id="main",
        issuer_person_id=person_id,
        issuer_platform=platform,
        issuer_user_id=str(getattr(source, "user_id", "") or ""),
        priority=100,
    )


def _format_directive_line(directive: Dict[str, Any]) -> str:
    payload = directive.get("payload") or {}
    text = payload.get("text") or directive.get("directive_type") or directive.get("directive_key")
    return f"- {directive.get('directive_type')}: {_preview(text, 220)}"


def _format_runtime_event_line(event: Dict[str, Any]) -> str:
    who = event.get("person_id") or event.get("sender_name") or event.get("sender_user_id") or "unknown"
    target = event.get("platform_chat_id") or "unknown"
    session = event.get("session_key") or "unknown-session"
    return (
        f"- {event.get('event_type') or '?'}:{event.get('event_subtype') or '?'} "
        f"{event.get('status') or 'unknown'} platform={event.get('platform') or event.get('source') or '?'} "
        f"chat={target} chat_type={event.get('chat_type') or '?'} "
        f"person={who} session={session} content={_preview(event.get('content'), 180)!r}"
    )


def _visible_recent_events(
    db,
    *,
    authority: str,
    session_key: str,
    person_id: str,
    platform: str = "",
    platform_chat_id: str = "",
) -> list[Dict[str, Any]]:
    if authority in {"owner", "trusted"}:
        return db.list_recent_agent_events(limit=8)
    events = db.list_recent_agent_events(session_key=session_key, limit=5)
    if person_id:
        by_person = db.list_recent_agent_events(person_id=person_id, limit=5)
        events = list({event["event_id"]: event for event in events + by_person}.values())
    # Cross-session outbound activity targeting the current audience. Without
    # this, the agent cannot see its own attempts (delivered or blocked) to
    # send to this same chat from autonomous/cron sessions, and confabulates
    # "must be a different session" when challenged. Blocked events are
    # explicitly included — they are the strongest signal of "I have been
    # trying to push content into this audience."
    if platform and platform_chat_id:
        by_audience = db.list_recent_agent_events(
            event_type="outbound",
            platform=platform,
            platform_chat_id=platform_chat_id,
            limit=5,
        )
        events = list({event["event_id"]: event for event in events + by_audience}.values())
    events.sort(key=lambda e: e.get("seq") or 0, reverse=True)
    return events[:10]


def build_state_packet(
    db,
    *,
    source: Any,
    session_id: str,
    session_key: str,
    inbound_event_id: str = "",
    person_id: str = "",
    authority: str = "",
) -> str:
    """Build an ephemeral structured system prefix for this exact event."""
    platform = _platform_value(getattr(source, "platform", ""))
    chat_id = str(getattr(source, "chat_id", "") or "")
    chat_type = str(getattr(source, "chat_type", "") or "")
    user_id = str(getattr(source, "user_id", "") or "")
    user_name = str(getattr(source, "user_name", "") or "")
    authority = authority or infer_platform_authority(source)
    person_id = person_id or (f"{platform}:{user_id}" if platform and user_id else "")

    try:
        directives = db.list_active_agent_directives(actor_id="main", limit=8)
    except Exception:
        directives = []
    try:
        recent_events = _visible_recent_events(
            db,
            authority=authority,
            session_key=session_key,
            person_id=person_id,
            platform=platform,
            platform_chat_id=chat_id,
        )
    except Exception:
        recent_events = []

    lines = [
        "## Agent Runtime State",
        "",
        "This packet is ephemeral runtime state, not conversation history.",
        "Use it for behavior and policy. Do not reveal private cross-session details unless the current sender is authorized and disclosure is needed.",
        "",
        "**Current Sender:**",
        f"- person_id: {person_id or 'unknown'}",
        f"- platform_user_id: {user_id or 'unknown'}",
        f"- display_name: {user_name or 'unknown'}",
        f"- authority: {authority}",
        "",
        "**Current Audience:**",
        f"- platform: {platform or 'unknown'}",
        f"- chat_id: {chat_id or 'unknown'}",
        f"- chat_type: {chat_type or 'unknown'}",
        f"- session_key: {session_key}",
        f"- inbound_event_id: {inbound_event_id or 'unknown'}",
    ]

    if directives:
        lines.extend(["", "**Active Agent Directives:**"])
        lines.extend(_format_directive_line(d) for d in directives)
    else:
        lines.extend(["", "**Active Agent Directives:**", "- none"])

    if recent_events:
        lines.extend(["", "**Recent Runtime Events:**"])
        lines.extend(_format_runtime_event_line(event) for event in recent_events)
    else:
        lines.extend(["", "**Recent Runtime Events:**", "- none"])

    lines.extend([
        "",
        "**Broadcast Policy Reminder:**",
        "- A cron, Hub, webhook, or other autonomous inbound must not be rebroadcast to a public channel unless the current event contains explicit trusted authorization.",
        "- If an action is blocked by policy, explain the causal source instead of trying another public target.",
    ])
    return "\n".join(lines)


def _iter_directive_payloads(db) -> Iterable[Dict[str, Any]]:
    try:
        directives = db.list_active_agent_directives(
            actor_id="main",
            directive_type="suppress_public_broadcasts",
            limit=20,
        )
    except Exception:
        directives = []
    for directive in directives:
        yield directive.get("payload") or {}


def _event_content_for_current_context(db) -> str:
    event_id = get_session_env("HERMES_AGENT_EVENT_ID", "")
    if not event_id:
        return ""
    try:
        event = db.get_agent_event(event_id)
    except Exception:
        event = None
    return str((event or {}).get("content") or "")


def _is_publicish_cross_session_target(
    *,
    target_platform: str,
    target_chat_id: str,
    source_platform: str,
    source_chat_id: str,
) -> bool:
    if target_platform not in _PUBLIC_PLATFORM_NAMES:
        return False
    if not source_platform or target_platform != source_platform:
        return True
    return bool(target_chat_id and source_chat_id and target_chat_id != source_chat_id)


def evaluate_send_message_policy(
    *,
    target_platform: str,
    target_chat_id: str,
    target_thread_id: str = "",
    message: str = "",
    db=None,
) -> GateDecision:
    """Gate model/tool initiated cross-channel sends."""
    db, should_close = _load_db(db)
    try:
        source_platform = get_session_env("HERMES_SESSION_PLATFORM", "").strip().lower()
        source_chat_id = get_session_env("HERMES_SESSION_CHAT_ID", "").strip()
        source_session_key = get_session_env("HERMES_SESSION_KEY", "").strip()
        target_platform = str(target_platform or "").strip().lower()
        target_chat_id = str(target_chat_id or "").strip()
        target_thread_id = str(target_thread_id or "").strip()
        message = str(message or "")
        risky_public_target = _is_publicish_cross_session_target(
            target_platform=target_platform,
            target_chat_id=target_chat_id,
            source_platform=source_platform,
            source_chat_id=source_chat_id,
        )

        active_stop = any(_iter_directive_payloads(db))
        if active_stop and risky_public_target:
            event_id = db.append_agent_event(
                event_type="outbound",
                event_subtype="send_message",
                status="blocked",
                session_key=source_session_key,
                actor_id="main",
                actor_kind="tool",
                source=source_platform or "tool",
                person_id=get_session_env("HERMES_AGENT_PERSON_ID", ""),
                sender_user_id=get_session_env("HERMES_SESSION_USER_ID", ""),
                sender_name=get_session_env("HERMES_SESSION_USER_NAME", ""),
                parent_event_id=get_session_env("HERMES_AGENT_EVENT_ID", ""),
                platform=target_platform,
                platform_chat_id=target_chat_id,
                platform_thread_id=target_thread_id,
                tool_name="send_message",
                content=_preview(message, 1000),
                payload={"decision": "deny", "source_platform": source_platform},
            )
            return GateDecision(
                allowed=False,
                reason="Active directive suppresses public/cross-session broadcasts.",
                policy="suppress_public_broadcasts",
                event_id=event_id,
            )

        inbound_content = _event_content_for_current_context(db)
        cron_like = inbound_content.lower().startswith("cronjob response:")
        autonomous_source = source_platform in _AUTONOMOUS_SOURCE_NAMES
        if risky_public_target and (cron_like or autonomous_source):
            event_id = db.append_agent_event(
                event_type="outbound",
                event_subtype="send_message",
                status="blocked",
                session_key=source_session_key,
                actor_id="main",
                actor_kind="tool",
                source=source_platform or "tool",
                person_id=get_session_env("HERMES_AGENT_PERSON_ID", ""),
                sender_user_id=get_session_env("HERMES_SESSION_USER_ID", ""),
                sender_name=get_session_env("HERMES_SESSION_USER_NAME", ""),
                parent_event_id=get_session_env("HERMES_AGENT_EVENT_ID", ""),
                platform=target_platform,
                platform_chat_id=target_chat_id,
                platform_thread_id=target_thread_id,
                tool_name="send_message",
                content=_preview(message, 1000),
                payload={
                    "decision": "deny",
                    "source_platform": source_platform,
                    "cron_like": cron_like,
                    "autonomous_source": autonomous_source,
                },
            )
            return GateDecision(
                allowed=False,
                reason="Autonomous cron/Hub/webhook context cannot rebroadcast to public/cross-session targets without explicit trusted authorization.",
                policy="autonomous_public_rebroadcast_guard",
                event_id=event_id,
            )

        return GateDecision(allowed=True)
    finally:
        if should_close:
            db.close()


def record_send_message_outbound(
    *,
    target_platform: str,
    target_chat_id: str,
    target_thread_id: str = "",
    message: str = "",
    status: str = "succeeded",
    result: Optional[Dict[str, Any]] = None,
) -> None:
    """Best-effort event log write after an allowed send attempt."""
    db = None
    try:
        from hermes_state import SessionDB

        db = SessionDB()
        db.append_agent_event(
            event_type="outbound",
            event_subtype="send_message",
            status=status,
            session_key=get_session_env("HERMES_SESSION_KEY", ""),
            actor_id="main",
            actor_kind="tool",
            source=get_session_env("HERMES_SESSION_PLATFORM", "tool"),
            person_id=get_session_env("HERMES_AGENT_PERSON_ID", ""),
            sender_user_id=get_session_env("HERMES_SESSION_USER_ID", ""),
            sender_name=get_session_env("HERMES_SESSION_USER_NAME", ""),
            parent_event_id=get_session_env("HERMES_AGENT_EVENT_ID", ""),
            platform=target_platform,
            platform_chat_id=target_chat_id,
            platform_thread_id=target_thread_id,
            tool_name="send_message",
            content=_preview(message, 1000),
            payload={"result": result or {}},
        )
    except Exception:
        pass
    finally:
        if db is not None:
            db.close()


def blocked_tool_result(decision: GateDecision) -> str:
    return json.dumps({
        "success": False,
        "blocked": True,
        "policy": decision.policy,
        "reason": decision.reason,
        "event_id": decision.event_id,
    }, ensure_ascii=False)
