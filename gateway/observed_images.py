"""Bounded replay of cached images from passive Telegram group context."""

import logging
from pathlib import Path

logger = logging.getLogger("gateway.run")
_MAX_OBSERVED_IMAGE_ATTACHMENTS = 4


def observed_image_content(text: str, urls: list[str], types: list[str]) -> str | list[dict]:
    """Keep local image references in content, which survives SQLite transcript persistence."""
    from agent.image_routing import _IMAGE_EXTS

    parts = [{"type": "text", "text": text}]
    for i, path in enumerate(urls):
        mime = types[i] if i < len(types) else ""
        if (mime.startswith("image/") if mime else str(path).lower().endswith(_IMAGE_EXTS)):
            parts.append({"type": "image_url", "image_url": {"url": str(path)}})
    return parts if len(parts) > 1 else text


def observed_text(content):
    """Observed pixels are attached only to the next turn, never historical replay."""
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "") for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return content


def _observed_tail(history):
    tail = []
    for message in reversed(history or []):
        if not isinstance(message, dict) or message.get("role") in {"session_meta", "system"}:
            continue
        if not message.get("observed"):
            break
        tail.append(message)
    return list(reversed(tail))


def _cached_image_path(raw) -> str | None:
    """Only files in this profile's image cache may be promoted into an API attachment."""
    if not isinstance(raw, str) or not Path(raw).is_absolute():
        return None
    from gateway.platforms.base import get_image_cache_dir
    from hermes_constants import get_hermes_home

    try:
        path = Path(raw).resolve()
        roots = (get_image_cache_dir(), Path(get_hermes_home()) / "image_cache")
        if path.is_file() and any(path.is_relative_to(root.resolve()) for root in roots):
            return str(path)
    except (OSError, ValueError):
        logger.debug("Invalid observed image cache path", exc_info=True)
    return None


def collect_observed_image_paths(history) -> list[str]:
    """Most recent unique cached images in trailing observed rows, oldest first."""
    paths = []
    for message in _observed_tail(history):
        content = message.get("content")
        if message.get("role") != "user" or not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "image_url":
                continue
            ref = part.get("image_url")
            path = _cached_image_path(ref.get("url")) if isinstance(ref, dict) else None
            if path:
                if path in paths:
                    paths.remove(path)
                paths.append(path)
    return paths[-_MAX_OBSERVED_IMAGE_ATTACHMENTS:]


async def history_with_observed_tail(runner, history, source):
    """Agent results omit observed rows; reload photos that arrived during the active turn."""
    from gateway.config import Platform

    if source is None or source.platform != Platform.TELEGRAM or source.chat_type != "group":
        return history
    try:
        store = runner.async_session_store
        entry = await store.get_or_create_session(source)
        persisted = await store.load_transcript(entry.session_id)
    except Exception:
        logger.debug("Observed-tail reload failed; using agent history", exc_info=True)
        return history
    return list(history) + _observed_tail(persisted)


_TELEGRAM_OBSERVED_CONTEXT_PROMPT_MARKER = "observed Telegram group context"
def uses_telegram_observed_group_context(channel_prompt: str | None) -> bool:
    """Return True for Telegram group turns that may include observed chatter.

    Observed rows must not replay as ordinary user turns, or a weak wake word makes old chatter look like work.
    """
    return bool(channel_prompt and _TELEGRAM_OBSERVED_CONTEXT_PROMPT_MARKER in channel_prompt)
