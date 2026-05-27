"""Helpers for Telegram smart mention routing."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable


DEFAULT_SMART_MENTION_SYSTEM_PROMPT = """You are Hermes's Telegram smart mention router.

Decide whether the current Telegram group message is addressed to Hermes or asks Hermes to do something.

Return only JSON with this shape:
{"should_respond": true|false, "confidence": 0.0-1.0, "reason": "short reason"}

Return true when the message asks Hermes, the bot, or an assistant to help, answer, summarize, decide, run a task, or otherwise take action. Recent context is only supporting evidence; the current message is the one being classified.

Return false for ordinary human-to-human conversation, status chatter, jokes, messages addressed to someone else, or messages that only mention a topic without asking Hermes to act.

Do not answer the user. Only classify routing."""


@dataclass(frozen=True)
class SmartMentionConfig:
    enabled: bool = False
    system_prompt: str = DEFAULT_SMART_MENTION_SYSTEM_PROMPT
    include_recent_context: bool = True
    recent_context_messages: int = 5
    recent_context_max_chars: int = 2000
    pass_recent_context_to_agent: bool = False
    min_confidence: float = 0.6
    max_tokens: int = 128
    log_decisions: bool = True
    log_message_text: bool = False
    on_error: str = "ignore"


@dataclass(frozen=True)
class SmartMentionClassification:
    should_respond: bool
    confidence: float = 0.0
    reason: str = ""
    raw: str = ""


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "on"}
    return bool(value)


def _coerce_int(value: Any, default: int, *, min_value: int, max_value: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(min_value, min(max_value, parsed))


def _coerce_float(value: Any, default: float, *, min_value: float, max_value: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(min_value, min(max_value, parsed))


def normalize_smart_mention_config(raw: Any) -> SmartMentionConfig:
    if isinstance(raw, bool):
        return SmartMentionConfig(enabled=raw)
    if not isinstance(raw, dict):
        return SmartMentionConfig()

    prompt = raw.get("system_prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        prompt = DEFAULT_SMART_MENTION_SYSTEM_PROMPT
    else:
        prompt = prompt.strip()

    return SmartMentionConfig(
        enabled=_coerce_bool(raw.get("enabled"), False),
        system_prompt=prompt,
        include_recent_context=_coerce_bool(raw.get("include_recent_context"), True),
        recent_context_messages=_coerce_int(raw.get("recent_context_messages"), 5, min_value=0, max_value=50),
        recent_context_max_chars=_coerce_int(raw.get("recent_context_max_chars"), 2000, min_value=0, max_value=20000),
        pass_recent_context_to_agent=_coerce_bool(raw.get("pass_recent_context_to_agent"), False),
        min_confidence=_coerce_float(raw.get("min_confidence"), 0.6, min_value=0.0, max_value=1.0),
        max_tokens=_coerce_int(raw.get("max_tokens"), 128, min_value=16, max_value=1024),
        log_decisions=_coerce_bool(raw.get("log_decisions"), True),
        log_message_text=_coerce_bool(raw.get("log_message_text"), False),
        on_error=str(raw.get("on_error") or "ignore").strip().lower() or "ignore",
    )


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def build_smart_mention_messages(
    *,
    config: SmartMentionConfig,
    current_text: str,
    recent_context: Iterable[dict[str, Any]] = (),
    media_metadata: dict[str, Any] | None = None,
    bot_username: str = "",
    chat_id: str = "",
    thread_id: str = "",
) -> list[dict[str, str]]:
    context_lines: list[str] = []
    if config.include_recent_context and config.recent_context_messages > 0:
        recent_items = []
        for item in recent_context:
            sender = str(item.get("sender") or "user").strip() or "user"
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            media = str(item.get("media") or "").strip()
            recent_items.append((sender, text, media))
        for sender, text, media in recent_items[-config.recent_context_messages:]:
            suffix = f" [{media}]" if media else ""
            context_lines.append(f"- {sender}{suffix}: {text}")

    context_block = "\n".join(context_lines)
    if config.recent_context_max_chars > 0:
        context_block = truncate_text(context_block, config.recent_context_max_chars)

    metadata_lines = [
        f"bot_username: @{bot_username}" if bot_username else "bot_username: unknown",
        f"chat_id: {chat_id}" if chat_id else "chat_id: unknown",
        f"thread_id: {thread_id}" if thread_id else "thread_id: general",
    ]
    if media_metadata:
        metadata_lines.append("media:")
        for key, value in sorted(media_metadata.items()):
            if value is None or value == "":
                continue
            metadata_lines.append(f"  {key}: {value}")

    user_parts = [
        "Telegram group routing decision.",
        "",
        "Metadata:",
        "\n".join(metadata_lines),
        "",
        "Recent context:",
        context_block or "(none)",
        "",
        "Current message:",
        current_text.strip() or "(no text)",
    ]
    return [
        {"role": "system", "content": config.system_prompt},
        {"role": "user", "content": "\n".join(user_parts)},
    ]


def parse_smart_mention_response(raw: Any) -> SmartMentionClassification:
    text = str(raw or "").strip()
    if not text:
        return SmartMentionClassification(False, raw=text)

    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    candidate = match.group(1).strip() if match else text
    if not candidate.startswith("{"):
        brace = re.search(r"\{.*\}", candidate, re.DOTALL)
        if brace:
            candidate = brace.group(0)

    try:
        payload = json.loads(candidate)
    except Exception:
        return SmartMentionClassification(False, raw=text)

    value = payload.get("should_respond", payload.get("respond", payload.get("process", False)))
    should_respond = _coerce_bool(value, False)
    confidence = _coerce_float(payload.get("confidence"), 1.0 if should_respond else 0.0, min_value=0.0, max_value=1.0)
    reason = str(payload.get("reason") or "").strip()
    return SmartMentionClassification(should_respond, confidence=confidence, reason=reason, raw=text)


def format_recent_context_for_agent(recent_context: Iterable[dict[str, Any]], max_chars: int) -> str:
    lines: list[str] = []
    for item in recent_context:
        sender = str(item.get("sender") or "user").strip() or "user"
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        media = str(item.get("media") or "").strip()
        suffix = f" [{media}]" if media else ""
        lines.append(f"- {sender}{suffix}: {text}")
    body = truncate_text("\n".join(lines), max_chars)
    if not body:
        return ""
    return (
        "Telegram recent group context before the current smart-routed message:\n"
        f"{body}\n\n"
        "Use this only as ephemeral context. The current message is the user request."
    )
