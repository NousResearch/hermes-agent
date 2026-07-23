"""Cheap-LLM rewriting for user-facing gateway status updates.

Raw tool progress and lifecycle diagnostics can contain commands, provider details,
or implementation noise.  This module turns each event into one short status line,
or suppresses it when there is no useful user-facing update.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping

from agent.redact import redact_sensitive_text

logger = logging.getLogger(__name__)

_SKIP_TOKENS = {"skip", "silent", "no_reply", "no reply", "[silent]"}
_SYSTEM_PROMPT = """You rewrite internal AI-agent activity into a user-friendly status update.
Return exactly one short plain-English line, maximum 12 words.
Describe useful progress, not implementation details.
Never mention tool names, commands, APIs, models, providers, tokens, context windows,
compression, retries, stack traces, file paths, IDs, or secrets.
If the event is internal noise or gives the user no useful progress, return exactly SKIP.
Do not use markdown, bullets, labels, quotation marks, or explanations."""


def _bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "enabled", "llm"}
    return bool(value)


@dataclass(frozen=True)
class UserFriendlyStatusConfig:
    enabled: bool = False
    timeout: float = 2.0
    max_chars: int = 120
    fallback: str = "suppress"


class LatestStatusGeneration:
    """Track the newest in-flight rewrite for a shared status surface."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._generation = 0

    def next(self) -> int:
        with self._lock:
            self._generation += 1
            return self._generation

    def is_current(self, generation: int) -> bool:
        with self._lock:
            return self._generation == generation


def resolve_user_friendly_status_config(
    user_config: Mapping[str, Any] | None,
    platform_key: str | None = None,
) -> UserFriendlyStatusConfig:
    """Resolve global settings with an optional per-platform override."""
    display = (user_config or {}).get("display", {})
    if not isinstance(display, Mapping):
        display = {}
    merged: dict[str, Any] = {}
    global_section = display.get("user_friendly_status")
    if isinstance(global_section, Mapping):
        merged.update(global_section)
    elif global_section is not None:
        merged["enabled"] = global_section

    platforms = display.get("platforms")
    if platform_key and isinstance(platforms, Mapping):
        platform = platforms.get(platform_key)
        if isinstance(platform, Mapping):
            section = platform.get("user_friendly_status")
            if isinstance(section, Mapping):
                merged.update(section)
            elif section is not None:
                merged["enabled"] = section

    try:
        timeout = max(0.25, min(float(merged.get("timeout", 2.0)), 10.0))
    except (TypeError, ValueError):
        timeout = 2.0
    try:
        max_chars = max(40, min(int(merged.get("max_chars", 120)), 240))
    except (TypeError, ValueError):
        max_chars = 120
    fallback = str(merged.get("fallback", "suppress")).strip().lower()
    if fallback not in {"suppress", "raw"}:
        fallback = "suppress"
    return UserFriendlyStatusConfig(
        enabled=_bool(merged.get("enabled"), False),
        timeout=timeout,
        max_chars=max_chars,
        fallback=fallback,
    )


def _response_text(response: Any) -> str:
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError):
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, Mapping) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(getattr(item, "text", None), str):
                parts.append(item.text)
        return " ".join(parts)
    return str(content or "")


def _one_line(text: str, max_chars: int) -> str | None:
    first = next((line.strip() for line in text.splitlines() if line.strip()), "")
    first = first.strip("`*_#>- \t\"'")
    if not first or first.lower() in _SKIP_TOKENS:
        return None
    words = first.split()
    if len(words) > 12:
        first = " ".join(words[:12])
    if len(first) > max_chars:
        first = first[: max_chars - 1].rstrip() + "…"
    return first


class UserFriendlyStatusFilter:
    def __init__(
        self,
        *,
        enabled: bool = False,
        timeout: float = 2.0,
        max_chars: int = 120,
        fallback: str = "suppress",
        llm_call: Callable[..., Awaitable[Any]] | None = None,
    ) -> None:
        self.enabled = enabled
        self.timeout = timeout
        self.max_chars = max_chars
        self.fallback = fallback
        self._llm_call = llm_call

    @classmethod
    def from_config(
        cls,
        user_config: Mapping[str, Any] | None,
        platform_key: str | None = None,
    ) -> "UserFriendlyStatusFilter":
        cfg = resolve_user_friendly_status_config(user_config, platform_key)
        return cls(
            enabled=cfg.enabled,
            timeout=cfg.timeout,
            max_chars=cfg.max_chars,
            fallback=cfg.fallback,
        )

    async def rewrite(self, *, kind: str, text: str) -> str | None:
        raw = str(text or "").strip()
        if not raw:
            return None
        if not self.enabled:
            return raw

        safe = redact_sensitive_text(raw, force=True)[:600]
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Event type: {str(kind)[:40]}\nInternal event: {safe}",
            },
        ]
        try:
            if self._llm_call is None:
                from agent.auxiliary_client import async_call_llm

                llm_call = async_call_llm
            else:
                llm_call = self._llm_call
            response = await llm_call(
                task="status_filter",
                messages=messages,
                temperature=0,
                max_tokens=48,
                timeout=self.timeout,
            )
            rendered = _one_line(_response_text(response), self.max_chars)
            return redact_sensitive_text(rendered, force=True) if rendered else None
        except Exception as exc:  # status rendering must never break the turn
            logger.debug("user-friendly status rewrite failed: %s", exc)
            # Never expose the original diagnostic when filtering is enabled.
            # The legacy "raw" setting now means a safe generic heartbeat.
            return "Still working." if self.fallback == "raw" else None
