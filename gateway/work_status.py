"""Live gateway work-status helpers.

This module owns the platform-independent part of Hermes' in-progress status
card.  A Telegram pinned message is one renderer; the model here is deliberately
called *work status* so other adapters can render the same lifecycle as an
edited message, card, status bar, etc.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

_VALID_CLEANUP_POLICIES = {"delete", "unpin_keep", "keep", "edit_done_then_delete"}


@dataclass(slots=True)
class WorkStatusConfig:
    """Resolved display.work_status configuration for one platform/chat."""

    enabled: bool = False
    mode: str = "auto"
    delay_seconds: float = 8.0
    ai_summary: bool = False
    update_progress: bool = False
    update_min_interval_seconds: float = 10.0
    max_updates_per_turn: int = 6
    cleanup_on_success: str = "edit_done_then_delete"
    cleanup_on_failure: str = "unpin_keep"
    cleanup_on_interrupt: str = "unpin_keep"
    delete_delay_seconds: float = 5.0
    pin_disable_notification: bool = True


@dataclass(slots=True)
class WorkStatusHandle:
    """Renderer-owned handle for a visible in-progress status."""

    status_id: str
    platform: str
    chat_id: str
    thread_id: Optional[str]
    session_key: str
    run_generation: Optional[int]
    message_id: Optional[str]
    mode: str
    pinned: bool = False
    state: str = "posted"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    update_count: int = 0
    cleanup_policy: str = "edit_done_then_delete"


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "enabled"}:
        return True
    if text in {"0", "false", "no", "off", "disabled"}:
        return False
    return default


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _cleanup_policy(value: Any, default: str) -> str:
    text = str(value or "").strip().lower()
    return text if text in _VALID_CLEANUP_POLICIES else default


def _normalise_chat_set(value: Any) -> Optional[set[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        if not value.strip():
            return None
        items = re.split(r"[\s,]+", value)
    elif isinstance(value, (list, tuple, set)):
        items = value
    else:
        return None
    result = {str(item).strip() for item in items if str(item).strip()}
    return result or None


def _nested_get(mapping: dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = mapping
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _platform_cfg(display_cfg: dict[str, Any], platform_key: str) -> dict[str, Any]:
    platforms = display_cfg.get("platforms") or {}
    value = platforms.get(platform_key)
    return value if isinstance(value, dict) else {}


def resolve_work_status_config(
    user_config: dict[str, Any],
    platform_key: str,
    *,
    chat_id: str | None = None,
) -> WorkStatusConfig:
    """Resolve global/per-platform Live Work Status config.

    Supports both the new structured ``display.work_status`` surface and the
    older Atom prototype keys ``display.pinned_work_summary*`` for migration.
    """
    cfg = user_config if isinstance(user_config, dict) else {}
    display = cfg.get("display") or {}
    if not isinstance(display, dict):
        display = {}
    p_cfg = _platform_cfg(display, platform_key)

    ws = display.get("work_status") or {}
    if not isinstance(ws, dict):
        ws = {"enabled": ws}
    ws_platforms = ws.get("platforms") or {}
    ws_p = ws_platforms.get(platform_key) if isinstance(ws_platforms, dict) else None
    if not isinstance(ws_p, dict):
        ws_p = {}

    def pick(key: str, default: Any = None, *, legacy: str | None = None) -> Any:
        if key in ws_p:
            return ws_p[key]
        if key in ws:
            return ws[key]
        if legacy and legacy in p_cfg:
            return p_cfg[legacy]
        if legacy and legacy in display:
            return display[legacy]
        return default

    enabled = _coerce_bool(pick("enabled", False, legacy="pinned_work_summary"), False)

    allow = _normalise_chat_set(pick("chats.allow", None))
    deny = _normalise_chat_set(pick("chats.deny", None))
    # YAML-friendly nested form: work_status: {chats: {allow: [...], deny: [...]}}
    chats_cfg = ws_p.get("chats") if isinstance(ws_p.get("chats"), dict) else ws.get("chats")
    if isinstance(chats_cfg, dict):
        allow = _normalise_chat_set(chats_cfg.get("allow")) or allow
        deny = _normalise_chat_set(chats_cfg.get("deny")) or deny
    # Legacy allowlist: display.pinned_work_summary_chats or per-platform key.
    legacy_chats = p_cfg.get("pinned_work_summary_chats", display.get("pinned_work_summary_chats"))
    if allow is None:
        allow = _normalise_chat_set(legacy_chats)

    if chat_id:
        chat = str(chat_id)
        if deny and ("*" in deny or chat in deny):
            enabled = False
        if allow and "*" not in allow and chat not in allow:
            enabled = False

    mode = str(pick("mode", "auto") or "auto").strip().lower()
    if mode not in {"auto", "pinned", "message", "off"}:
        mode = "auto"
    if mode == "off":
        enabled = False

    cleanup_cfg = pick("cleanup", {})
    if not isinstance(cleanup_cfg, dict):
        cleanup_cfg = {}

    return WorkStatusConfig(
        enabled=enabled,
        mode=mode,
        delay_seconds=max(0.0, _coerce_float(pick("delay_seconds", 8, legacy="pinned_work_summary_delay_seconds"), 8.0)),
        ai_summary=_coerce_bool(pick("ai_summary", False, legacy="pinned_work_summary_ai"), False),
        update_progress=_coerce_bool(pick("update_progress", False), False),
        update_min_interval_seconds=max(0.0, _coerce_float(pick("update_min_interval_seconds", 10), 10.0)),
        max_updates_per_turn=max(0, _coerce_int(pick("max_updates_per_turn", 6), 6)),
        cleanup_on_success=_cleanup_policy(cleanup_cfg.get("on_success"), "edit_done_then_delete"),
        cleanup_on_failure=_cleanup_policy(cleanup_cfg.get("on_failure"), "unpin_keep"),
        cleanup_on_interrupt=_cleanup_policy(cleanup_cfg.get("on_interrupt"), "unpin_keep"),
        delete_delay_seconds=max(0.0, _coerce_float(cleanup_cfg.get("delete_delay_seconds"), 5.0)),
        pin_disable_notification=_coerce_bool(_nested_get(ws_p, "telegram.disable_notification", _nested_get(ws, "telegram.disable_notification", True)), True),
    )


def compact_source_text(event: Any) -> str:
    """Return the best available text for a status label."""
    explicit = getattr(event, "work_status_text", None) or getattr(event, "pinned_summary_text", None)
    if isinstance(explicit, str) and explicit.strip():
        return " ".join(explicit.split())

    transcripts = [
        " ".join(str(t).split())
        for t in (getattr(event, "voice_transcripts", None) or [])
        if str(t).strip()
    ]
    if transcripts:
        return "\n".join(transcripts)

    current = " ".join((getattr(event, "text", None) or "").split())
    reply_to_text = " ".join(str(getattr(event, "reply_to_text", "") or "").split())
    if reply_to_text and (not current or len(current) <= 24):
        if current:
            return f"Replied-to message: {reply_to_text}\nFollow-up: {current}"
        return reply_to_text
    return current


def infer_status_mode(source: str, event: Any | None = None) -> str:
    """Infer a compact human-facing work mode for the live status card."""
    message_type = getattr(getattr(event, "message_type", None), "value", "") if event is not None else ""
    text = " ".join(str(source or "").lower().split())
    if message_type and message_type not in {"text", "message"}:
        text = f"{message_type} {text}"
    if not text:
        return "Ask"
    if "?" in text or re.search(r"\b(what|why|how|which|should i|can you explain|do you know|is it|are we)\b", text):
        return "Ask"
    if re.search(r"\b(debug|bug|fix|broken|error|fail(?:ed|ing)?|traceback|crash|issue|stuck|not working|dumb|wrong|regression|troubleshoot)\b", text):
        return "Debug"
    if re.search(r"\b(plan|strategy|design|sop|architecture|outline|roadmap|proposal|approach)\b", text):
        return "Plan"
    if re.search(r"\b(verify|test|check|confirm|status|inspect|audit|review)\b", text):
        return "Verify"
    if re.search(r"\b(research|search|find|look up|investigate|compare|analyze|summari[sz]e)\b", text):
        return "Research"
    if re.search(r"\b(build|create|add|implement|expand|update|change|modify|write|patch|replace|install|configure|enable)\b", text):
        return "Build"
    return "Task"


def _clean_request_interpretation(source: str, *, max_len: int = 92) -> str:
    """Normalize user/reply text into a short request-interpretation summary."""
    text = re.sub(r"\s+", " ", str(source or "")).strip(" \t\n\r`*_>")
    text = re.sub(r"^(please|pls|can you|could you|would you)\s+", "", text, flags=re.I)
    text = re.sub(r"\b(i think|maybe|probably)\b", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip(" .:-")
    if len(text) > max_len:
        text = text[: max_len - 3].rstrip(" ,;:-") + "..."
    return text


def interpret_status_request(source: str, mode: str) -> str:
    """Convert raw user wording into a short action-oriented status summary."""
    cleaned = _clean_request_interpretation(source, max_len=180)
    lowered = cleaned.lower()

    # Common complaint/fix phrasing should become an investigation target, not
    # a quote of the user's frustration. Example: "pinned message is dumb" ->
    # "Investigate pinned message code".
    if mode == "Debug":
        if re.search(r"\bpinned (message|status|work[- ]?status|summary)\b", lowered):
            return "Investigate pinned message code"
        if re.search(r"\b(work[- ]?status|status card|live status)\b", lowered):
            return "Investigate work-status code"
        if re.search(r"\b(gateway|telegram)\b", lowered):
            return "Investigate Telegram gateway behavior"

    if mode == "Plan":
        target = re.sub(r"\b(plan|strategy|design|outline|roadmap|proposal|approach)\b", "", cleaned, flags=re.I)
        target = _clean_request_interpretation(target, max_len=72)
        return f"Plan {target}" if target else "Plan implementation approach"

    if mode == "Ask":
        if re.search(r"\b(how did|how'd)\b.*\b(implementation|change|fix|patch|deploy|work)\b", lowered):
            return "Summarize implementation outcome"
        if re.search(r"\b(what'?s left|what is left|what remains|remaining|todo|next steps)\b", lowered):
            return "Summarize remaining work"
        if re.search(r"\b(did you|have you|all done|finished|complete)\b", lowered):
            return "Report completion status"
        if re.search(r"\b(what happened|what did you do|what changed)\b", lowered):
            return "Summarize completed changes"
        if re.search(r"\b(why|how)\b", lowered):
            return "Explain request context"
        return "Answer user question"

    if mode == "Verify":
        if lowered in {"test", "tests"}:
            return "Run tests"
        target = re.sub(r"\b(verify|test|check|confirm|status|inspect|audit|review)\b", "", cleaned, flags=re.I)
        target = _clean_request_interpretation(target, max_len=72)
        return f"Verify {target}" if target else "Verify current state"

    if mode == "Research":
        target = re.sub(r"\b(research|search|find|look up|investigate|compare|analyze|summari[sz]e)\b", "", cleaned, flags=re.I)
        target = _clean_request_interpretation(target, max_len=72)
        return f"Research {target}" if target else "Research request context"

    if mode == "Build":
        target = re.sub(r"\b(build|create|add|implement|expand|update|change|modify|write|patch|replace|install|configure|enable)\b", "", cleaned, flags=re.I)
        target = _clean_request_interpretation(target, max_len=72)
        verb = "Update" if re.search(r"\b(update|change|modify|patch|replace|expand)\b", lowered) else "Build"
        return f"{verb} {target}" if target else f"{verb} requested change"

    return cleaned


def format_status_text(mode: str, summary: str, *, from_reply: bool = False) -> str:
    """Render the status text users see in pins/messages."""
    mode = re.sub(r"[^A-Za-z0-9 /_-]", "", str(mode or "Task")).strip() or "Task"
    summary = _clean_request_interpretation(summary) or "review incoming request"
    prefix = f"📌 [{mode}]"
    if from_reply:
        return f"{prefix} From reply: {summary}"
    return f"{prefix} {summary}"


def fallback_status_text(event: Any) -> str:
    """Deterministic non-AI status text."""
    current = " ".join((getattr(event, "text", None) or "").split())
    reply_to_text = " ".join(str(getattr(event, "reply_to_text", "") or "").split())
    source = compact_source_text(event)
    if not source:
        message_type = getattr(getattr(event, "message_type", None), "value", None) or "message"
        return format_status_text("Ask", f"review incoming {message_type} request")

    from_reply = bool(reply_to_text and (not current or len(current) <= 24))
    mode = infer_status_mode(source, event)
    summary = interpret_status_request(source, mode)
    return format_status_text(mode, summary, from_reply=from_reply)


def sanitize_status_label(label: str, *, max_len: int = 110) -> str:
    """Clamp and de-risk LLM/user-derived status text."""
    text = " ".join(str(label or "").split()).strip('"` ')
    # Avoid accidental mentions/pings and raw Markdown injection in compact pins.
    text = re.sub(r"@(?=everyone|here|[A-Za-z0-9_])", "@", text)
    text = text.replace("\u0000", "")
    if len(text) > max_len:
        text = text[: max_len - 3].rstrip(" ,;:-") + "..."
    return text


async def maybe_ai_status_text(config: WorkStatusConfig, event: Any) -> str:
    """Return AI-interpreted status text when enabled, else fallback."""
    fallback = fallback_status_text(event)
    if not config.ai_summary:
        return fallback
    source = compact_source_text(event)
    if not source:
        return fallback
    try:
        from agent.auxiliary_client import async_call_llm, extract_content_or_reasoning

        response = await async_call_llm(
            "work_status",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Write a concise request-interpretation summary for a temporary live status card. "
                        "Describe what the assistant should do, not the user's literal wording. "
                        "Return only the summary text: no mode tag, no markdown, max 92 characters."
                    ),
                },
                {"role": "user", "content": f"Current request:\n{source[:1400]}"},
            ],
            temperature=0,
            max_tokens=80,
            timeout=12.0,
        )
        label = sanitize_status_label(extract_content_or_reasoning(response), max_len=92)
        if not label:
            return fallback
        mode = infer_status_mode(source, event)
        interpreted = interpret_status_request(source, mode)
        normal_label = re.sub(r"\W+", "", label).lower()
        normal_source = re.sub(r"\W+", "", source).lower()
        normal_interpreted = re.sub(r"\W+", "", interpreted).lower()
        label_quotes_source = bool(normal_label and (normal_label in normal_source or normal_source in normal_label))
        label_is_answer_echo = bool(re.match(r"^(answer|respond|reply)\b\s*:?", label, flags=re.I))
        if label_quotes_source or label_is_answer_echo:
            return format_status_text(mode, interpreted)
        if normal_label == normal_interpreted:
            return format_status_text(mode, interpreted)
        return format_status_text(mode, label)
    except Exception:
        logger.debug("AI work-status summary failed; using fallback", exc_info=True)
        return fallback


async def maybe_await_callback(callback: Callable[..., Any] | None, *args: Any) -> None:
    if not callback:
        return
    result = callback(*args)
    if inspect.isawaitable(result):
        await result
