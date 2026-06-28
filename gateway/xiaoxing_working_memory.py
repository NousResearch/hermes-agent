"""Short-lived cross-channel working memory for XiaoXing.

This is intentionally not a general long-term memory system.  It records recent
interaction events across all platforms and interlocutors so XiaoXing can
answer follow-up questions like "what were we just talking about?" without
mixing full channel transcripts.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from hermes_constants import get_hermes_home


DEFAULT_TTL_SECONDS = 3 * 24 * 60 * 60
MAX_TEXT_CHARS = 500
MAX_SUMMARY_CHARS = 220

DEFAULT_PERSON_ALIASES = {
    ("milky", "490008192"): ("dad", "爸爸"),
    ("napcat", "490008192"): ("dad", "爸爸"),
    ("weixin", "o9cq80x2iVOtDRClhdviY7rnS2kk@im.wechat"): ("dad", "爸爸"),
}


def _enabled() -> bool:
    return os.getenv("XIAOXING_WORKING_MEMORY", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _store_path() -> Path:
    configured = os.getenv("XIAOXING_WORKING_MEMORY_PATH", "").strip()
    if configured:
        return Path(configured).expanduser()
    return get_hermes_home() / "autonomy" / "working_memory" / "recent_events.jsonl"


def _now() -> float:
    return time.time()


def _clip(text: Any, limit: int = MAX_TEXT_CHARS) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "…"


def _summarize(text: str) -> str:
    value = " ".join(str(text or "").split())
    return _clip(value, MAX_SUMMARY_CHARS)


def _configured_aliases() -> Dict[Tuple[str, str], Tuple[str, str]]:
    aliases = dict(DEFAULT_PERSON_ALIASES)
    raw = os.getenv("XIAOXING_PERSON_ALIASES", "").strip()
    if not raw:
        return aliases
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return aliases
    if not isinstance(data, list):
        return aliases
    for item in data:
        if not isinstance(item, dict):
            continue
        platform = str(item.get("platform") or "").strip().lower()
        external_id = str(item.get("external_id") or item.get("id") or "").strip()
        person_id = str(item.get("person_id") or "").strip()
        person_name = str(item.get("person_name") or item.get("name") or person_id).strip()
        if platform and external_id and person_id:
            aliases[(platform, external_id)] = (person_id, person_name)
    return aliases


def resolve_person(
    *,
    platform: str,
    chat_id: str = "",
    user_id: str = "",
    interlocutor_id: str = "",
    interlocutor_name: str = "",
    user_name: str = "",
) -> Tuple[str, str]:
    """Resolve platform-specific IDs to a stable person identity."""
    platform_key = str(platform or "").strip().lower()
    aliases = _configured_aliases()
    for candidate in (interlocutor_id, user_id, chat_id):
        candidate = str(candidate or "").strip()
        if not candidate:
            continue
        hit = aliases.get((platform_key, candidate))
        if hit:
            return hit
    fallback_id = str(interlocutor_id or user_id or chat_id or "").strip()
    fallback_name = str(interlocutor_name or user_name or fallback_id or "unknown").strip()
    if fallback_id:
        return f"{platform_key}:{fallback_id}", fallback_name
    return "unknown", fallback_name or "unknown"


def _load_events(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    path = path or _store_path()
    if not path.exists():
        return []
    events: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    events.append(item)
    except OSError:
        return []
    return events


def _write_events(events: Iterable[Dict[str, Any]], path: Optional[Path] = None) -> None:
    path = path or _store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(path)


def prune_expired(path: Optional[Path] = None, *, now: Optional[float] = None) -> None:
    if not _enabled():
        return
    now = _now() if now is None else now
    events = [
        event
        for event in _load_events(path)
        if float(event.get("expires_at") or 0) > now
    ]
    _write_events(events[-200:], path)


def record_event(
    *,
    direction: str,
    platform: str,
    chat_id: str,
    text: str,
    session_id: str = "",
    session_key: str = "",
    user_id: str = "",
    user_name: str = "",
    interlocutor_id: str = "",
    interlocutor_name: str = "",
    trigger: str = "",
    response_text: str = "",
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Append one recent interaction event."""
    if not _enabled():
        return
    text = _clip(text)
    response_text = _clip(response_text)
    if not text and not response_text:
        return

    now = _now()
    actor_id = str(interlocutor_id or user_id or chat_id or "")
    actor_name = str(interlocutor_name or user_name or actor_id or "")
    person_id, person_name = resolve_person(
        platform=platform,
        chat_id=chat_id,
        user_id=user_id,
        user_name=user_name,
        interlocutor_id=actor_id,
        interlocutor_name=actor_name,
    )
    event = {
        "event_id": uuid.uuid4().hex,
        "agent_id": "xiaoxing",
        "timestamp": now,
        "expires_at": now + max(int(ttl_seconds), 60),
        "direction": direction,
        "platform": platform or "",
        "chat_id": str(chat_id or ""),
        "user_id": str(user_id or ""),
        "user_name": str(user_name or ""),
        "interlocutor_id": actor_id,
        "interlocutor_name": actor_name,
        "person_id": person_id,
        "person_name": person_name,
        "session_id": session_id or "",
        "session_key": session_key or "",
        "trigger": trigger or "",
        "text": text,
        "response_text": response_text,
        "summary": _summarize(response_text or text),
    }
    if metadata:
        event["metadata"] = metadata

    path = _store_path()
    events = [
        item
        for item in _load_events(path)
        if float(item.get("expires_at") or 0) > now
    ]
    events.append(event)
    _write_events(events[-200:], path)


def recent_events(
    *,
    limit: int = 8,
    platforms: Optional[set[str]] = None,
    now: Optional[float] = None,
) -> List[Dict[str, Any]]:
    if not _enabled():
        return []
    now = _now() if now is None else now
    selected: List[Dict[str, Any]] = []
    for event in _load_events():
        if float(event.get("expires_at") or 0) <= now:
            continue
        if platforms and str(event.get("platform") or "") not in platforms:
            continue
        selected.append(event)
    selected.sort(key=lambda item: float(item.get("timestamp") or 0), reverse=True)
    return selected[: max(limit, 0)]


def _dedupe_events(events: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[str] = set()
    deduped: List[Dict[str, Any]] = []
    for event in events:
        event_id = str(event.get("event_id") or "")
        if event_id and event_id in seen:
            continue
        if event_id:
            seen.add(event_id)
        deduped.append(event)
    return deduped


def _select_prompt_events(
    *,
    current_platform: str = "",
    current_chat_id: str = "",
    current_user_id: str = "",
    current_user_name: str = "",
    limit: int = 12,
) -> Tuple[List[Dict[str, Any]], Tuple[str, str]]:
    """Pick compact prompt events without losing the current person thread."""
    all_events = recent_events(limit=200)
    if not all_events:
        return [], ("", "")

    current_person_id = ""
    current_person_name = ""
    if current_platform or current_chat_id or current_user_id:
        current_person_id, current_person_name = resolve_person(
            platform=current_platform,
            chat_id=current_chat_id,
            user_id=current_user_id,
            interlocutor_id=current_user_id,
            interlocutor_name=current_user_name,
            user_name=current_user_name,
        )

    person_events: List[Dict[str, Any]] = []
    if current_person_id and current_person_id != "unknown":
        person_events = [
            event
            for event in all_events
            if str(event.get("person_id") or "") == current_person_id
        ]

    channel_events = [
        event
        for event in all_events
        if str(event.get("platform") or "") == str(current_platform or "")
        and str(event.get("chat_id") or "") == str(current_chat_id or "")
    ]

    # Keep current-person/channel continuity first, then fill with the latest
    # overall events. This prevents active Dad messages from losing the outbound
    # message they are replying to when a long Codex mentor run adds many events.
    selected = _dedupe_events([*person_events[:6], *channel_events[:4], *all_events])
    return selected[: max(limit, 0)], (current_person_id, current_person_name)


def build_prompt_block(
    *,
    current_platform: str = "",
    current_chat_id: str = "",
    current_user_id: str = "",
    current_user_name: str = "",
) -> str:
    """Return a compact system-prompt block with recent cross-channel events."""
    events, current_person = _select_prompt_events(
        current_platform=current_platform,
        current_chat_id=current_chat_id,
        current_user_id=current_user_id,
        current_user_name=current_user_name,
        limit=12,
    )
    if not events:
        return ""

    lines = [
        "## XiaoXing Recent Cross-Channel Working Memory",
        "",
        "These are short-lived interaction events across platforms, sessions, and people. They are not permanent facts.",
        "Use them to maintain continuity across QQ, Weixin, Codex mentor, cron, CLI, and future channels.",
        "They may involve Dad, Codex, XiaoXing himself, cron jobs, or other people. Do not assume every event is about Dad.",
        "If someone asks where a recent message came from, prefer these events over unrelated long-term memories.",
        "",
    ]
    current_person_id, current_person_name = current_person
    if current_person_id and current_person_id != "unknown":
        lines.append(
            f"Current channel resolves to: {current_person_name}<{current_person_id}>."
        )
        lines.append(
            "Treat messages from this channel as that person unless explicit metadata says otherwise."
        )
        lines.append("")
    for event in events:
        ts = time.strftime(
            "%Y-%m-%d %H:%M",
            time.localtime(float(event.get("timestamp") or 0)),
        )
        direction = event.get("direction") or "event"
        platform = event.get("platform") or "unknown"
        actor = event.get("person_name") or event.get("interlocutor_name") or event.get("user_name") or event.get("person_id") or event.get("interlocutor_id") or event.get("user_id") or "unknown"
        summary = event.get("summary") or event.get("text") or event.get("response_text") or ""
        person_id = event.get("person_id") or ""
        if person_id and person_id not in {"unknown", actor}:
            actor = f"{actor}<{person_id}>"
        line = f"- {ts} [{platform}/{direction}; with {actor}] {_clip(summary, 180)}"
        trigger = event.get("trigger") or ""
        if trigger:
            line += f" (source: {trigger})"
        lines.append(line)

    lines.extend([
        "",
        "Continuity rules:",
        "- Current channel history controls immediate tone and reply target.",
        "- Recent working memory controls what XiaoXing has just experienced across channels and interlocutors.",
        "- If the current person replies vaguely (for example 'show me'), first check recent outbound events to that person.",
        "- Keep identities separate: Dad, Codex, other contacts, and automated jobs can all appear here.",
        "- If a recent event is relevant, answer from it briefly. If none fits, say you did not connect that record instead of inventing a source.",
    ])
    return "\n".join(lines)
