"""Local thread-summary context injection helpers for gateway sessions."""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

DEFAULT_THREAD_CONTEXT_MAX_CHARS = 12_000
_SAFE_SEGMENT_TOKEN_RE = re.compile(r"[0-9A-Za-z_-]+|[\u3400-\u9FFF]+")


@dataclass
class ThreadContextConfig:
    """Configuration for injecting local per-thread summary files."""

    enabled: bool = False
    root: Path | None = None
    max_chars: int = DEFAULT_THREAD_CONTEXT_MAX_CHARS


def _default_root() -> Path:
    return get_hermes_home() / "workspace" / "topics"


def _context_root(config: ThreadContextConfig) -> Path:
    return Path(config.root) if config.root is not None else _default_root()


def _platform_name(platform: Any) -> str | None:
    value = getattr(platform, "value", platform)
    if value is None:
        return None
    return str(value).strip().lower() or None


def _safe_path_segment(value: Any) -> str | None:
    """Normalize untrusted platform/chat/thread IDs into one path segment."""

    if value is None:
        return None
    text = unicodedata.normalize("NFKC", str(value)).strip()
    if not text:
        return None
    tokens = _SAFE_SEGMENT_TOKEN_RE.findall(text)
    if not tokens:
        return None
    segment = "-".join(tokens)
    if not segment.strip("-_"):
        return None
    return segment


def resolve_thread_context_path(
    *,
    platform: Any,
    chat_id: str | None,
    thread_id: str | None,
    config: ThreadContextConfig,
    chat_name: str | None = None,
) -> Path | None:
    """Return the stable local summary path for a gateway thread.

    ``chat_name`` is accepted for call-site compatibility but deliberately not
    used in the path: display names are mutable and may contain private text.
    """

    del chat_name
    if not config.enabled or not chat_id or not thread_id:
        return None

    platform_segment = _safe_path_segment(_platform_name(platform))
    chat_segment = _safe_path_segment(chat_id)
    thread_segment = _safe_path_segment(thread_id)
    if not platform_segment or not chat_segment or not thread_segment:
        return None

    root = _context_root(config)
    path = root / platform_segment / chat_segment / f"thread-{thread_segment}.md"

    try:
        # Defense in depth: path is constructed from sanitized segments, but keep
        # the invariant explicit so future changes cannot accidentally traverse.
        if not path.resolve().is_relative_to(root.resolve()):
            logger.warning("Resolved thread context path escaped root: %s", path)
            return None
    except OSError:
        return None

    return path


def build_thread_context_block(
    *,
    platform: Any,
    chat_id: str | None,
    thread_id: str | None,
    config: ThreadContextConfig,
    chat_name: str | None = None,
) -> str | None:
    """Load and format a local thread summary as a prompt context block."""

    path = resolve_thread_context_path(
        platform=platform,
        chat_id=chat_id,
        thread_id=thread_id,
        chat_name=chat_name,
        config=config,
    )
    if path is None or not path.is_file():
        return None

    try:
        content = path.read_text(encoding="utf-8").strip()
    except OSError:
        logger.debug("Failed to read thread context file: %s", path, exc_info=True)
        return None

    if not content:
        return None

    truncated = False
    if config.max_chars > 0 and len(content) > config.max_chars:
        content = content[: config.max_chars].rstrip()
        truncated = True

    lines = [
        "## Thread Context",
        "",
        "Source: local thread summary",
        "",
        content,
    ]
    if truncated:
        lines.extend(
            [
                "",
                f"[Thread context truncated to {config.max_chars} characters.]",
            ]
        )
    return "\n".join(lines)
