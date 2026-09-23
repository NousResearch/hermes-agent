"""Semantic Telegram forum-topic icon selection.

Telegram forum topics carry an icon drawn from a fixed catalog of ~110 custom-emoji stickers
(``getForumTopicIconStickers``); arbitrary emoji are rejected. When Hermes auto-titles a topic
lane it can also pick the catalog entry that best matches the title, so the sidebar bubble shows
a meaningful glyph instead of the first letter of the name. One small auxiliary call, thinking
off, constrained to the catalog; any failure degrades to "no icon", never to a broken rename.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from typing import Any, Dict, Iterable, List, Optional

from agent.auxiliary_client import call_llm

logger = logging.getLogger(__name__)

# The catalog only changes when Telegram ships new stickers; one refresh per gateway day is plenty.
CATALOG_TTL_SECONDS = 24 * 3600
ICON_MAX_TOKENS = 128
ICON_TIMEOUT_SECONDS = 20.0
# Variation selectors / ZWJ sequences: the model echoes "⚡" for the catalog's "⚡️"; normalise both sides.
_EMOJI_NOISE_RE = re.compile("[\ufe0e\ufe0f\u200d]")

_ICON_PROMPT_TEMPLATE = (
    "You pick an icon for a chat topic. Given the topic title, choose the ONE emoji from the "
    "allowed list that best represents the subject matter.\n\n"
    "Rules:\n"
    "- Only emoji from the allowed list are valid; anything else is rejected.\n"
    "- Prefer the subject (what the topic is about) over the action or the mood.\n"
    "- If nothing fits well, pick 💬.\n\n"
    "Allowed: __ALLOWED__\n\n"
    'Reply with JSON only: {"emoji": "..."}'
)

_ICON_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "topic_icon", "strict": True, "schema": {
        "type": "object", "properties": {"emoji": {"type": "string"}}, "required": ["emoji"], "additionalProperties": False}},
}


def _normalize_emoji(value: str) -> str:
    return _EMOJI_NOISE_RE.sub("", str(value or "")).strip()


class TopicIconCatalog:
    """Emoji -> ``custom_emoji_id`` map from ``getForumTopicIconStickers``, refreshed lazily.

    ``fetch`` is any awaitable returning the sticker list (adapter-injected so the catalog stays
    transport-agnostic and testable without a bot). Entries are keyed by normalised emoji.
    """

    def __init__(self, ttl_seconds: float = CATALOG_TTL_SECONDS) -> None:
        self._ttl = ttl_seconds
        self._loaded_at: float = 0.0
        self._by_emoji: Dict[str, str] = {}
        self._lock = threading.Lock()

    @property
    def emojis(self) -> List[str]:
        return list(self._by_emoji)

    def lookup(self, emoji: str) -> Optional[str]:
        return self._by_emoji.get(_normalize_emoji(emoji))

    def is_fresh(self) -> bool:
        return bool(self._by_emoji) and (time.monotonic() - self._loaded_at) < self._ttl

    def load(self, stickers: Iterable[Any]) -> None:
        """Replace the catalog from sticker objects (PTB ``Sticker`` or raw dicts)."""
        mapping: Dict[str, str] = {}
        for sticker in stickers or ():
            emoji = getattr(sticker, "emoji", None) if not isinstance(sticker, dict) else sticker.get("emoji")
            custom_id = (getattr(sticker, "custom_emoji_id", None) if not isinstance(sticker, dict)
                         else sticker.get("custom_emoji_id"))
            if emoji and custom_id:
                mapping.setdefault(_normalize_emoji(emoji), str(custom_id))
        with self._lock:
            self._by_emoji = mapping
            self._loaded_at = time.monotonic()

    async def ensure_loaded(self, fetch) -> bool:
        """Refresh through ``fetch()`` when stale; True when the catalog has entries afterwards."""
        if self.is_fresh():
            return True
        try:
            self.load(await fetch())
        except Exception:
            logger.debug("Forum topic icon catalog refresh failed", exc_info=True)
        return bool(self._by_emoji)


def _extract_emoji(raw: str) -> str:
    """``{"emoji": ...}`` payload, tolerant of fences and prose around it."""
    text = (raw or "").strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and isinstance(parsed.get("emoji"), str):
            return parsed["emoji"]
    except (ValueError, TypeError):
        pass
    match = re.search(r'"emoji"\s*:\s*"([^"]+)"', text)
    return match.group(1) if match else text.strip("`{} \n")


def choose_topic_icon(title: str, catalog: TopicIconCatalog, *, timeout: float = ICON_TIMEOUT_SECONDS) -> Optional[str]:
    """``custom_emoji_id`` for the catalog emoji best matching ``title``; None on any miss or failure.

    Synchronous (one aux call) — callers run it off-loop via ``asyncio.to_thread``.
    """
    allowed = catalog.emojis
    if not title or not allowed:
        return None
    prompt = _ICON_PROMPT_TEMPLATE.replace("__ALLOWED__", " ".join(allowed))
    try:
        response = call_llm(
            task="title_generation",  # same cheap tier and pinning as the title call it rides behind
            messages=[{"role": "system", "content": prompt}, {"role": "user", "content": title}],
            max_tokens=ICON_MAX_TOKENS, temperature=None, timeout=timeout,
            extra_body={"response_format": _ICON_RESPONSE_FORMAT},
            reasoning_config={"enabled": False},
        )
        emoji = _extract_emoji(response.choices[0].message.content or "")
    except Exception as e:
        logger.debug("Topic icon selection failed: %s", e, exc_info=True)
        return None
    icon_id = catalog.lookup(emoji)
    if icon_id is None:
        logger.debug("Topic icon %r not in catalog; leaving icon unset", emoji)
    return icon_id
