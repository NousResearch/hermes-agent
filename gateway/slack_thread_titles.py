"""Persistent Slack thread preview titles.

Slack exposes different preview surfaces (notifications, sidebars, thread
lists) that do not all choose snippets the same way. Keep one stable title per
thread, make the visible body title placement configurable, and provide a
top-level fallback string for Slack clients that use notification fallbacks.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from pathlib import Path
from typing import Any, Optional

from hermes_constants import get_default_hermes_root
from utils import atomic_json_write

logger = logging.getLogger(__name__)

_STORE_LOCK = threading.Lock()
_STORE_RELATIVE_PATH = Path("gateway") / "slack_thread_titles.json"
_MAX_ENTRIES = 5000
_FALLBACK_TITLE = "Slack Thread Conversation With Hermes"
_RETITLE_RE = re.compile(
    r"^\s*(?:retitle|rename)\s+(?:this\s+)?thread\s*:\s*(?P<title>.+?)\s*$",
    re.IGNORECASE | re.DOTALL,
)
_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’+/#.-]*")
_MRKDWN_PREFIX_RE = re.compile(r"^\s*\*{1,2}(?P<title>[^*\n]{1,160}):\*{1,2}")
_PLAIN_PREFIX_RE = re.compile(r"^\s*(?P<title>[^\n:*`]{1,160}):")

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "i",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "this",
    "to",
    "with",
    "you",
    "your",
}
_FILLER_WORDS = ("Thread", "Discussion", "Context", "Update", "Summary")


def store_path() -> Path:
    """Return the shared Slack thread-title store path.

    Use the Hermes root rather than the active profile home so Slack-facing
    profiles in the same install agree on the same title for the same Slack
    thread. The payload contains no message bodies or secrets — only
    channel/thread IDs and short user-visible titles.
    """

    return get_default_hermes_root() / _STORE_RELATIVE_PATH


def make_thread_key(channel_id: str, thread_ts: str) -> str:
    """Return a stable key for a Slack thread title."""

    return f"{str(channel_id or '').strip()}:{str(thread_ts or '').strip()}"


def _load_store_unlocked() -> dict[str, Any]:
    path = store_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"version": 1, "threads": {}}
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Slack thread-title store unreadable (%s); starting empty", exc)
        return {"version": 1, "threads": {}}
    if not isinstance(raw, dict):
        return {"version": 1, "threads": {}}
    threads = raw.get("threads")
    if not isinstance(threads, dict):
        raw["threads"] = {}
    raw.setdefault("version", 1)
    return raw


def _save_store_unlocked(data: dict[str, Any]) -> None:
    try:
        atomic_json_write(store_path(), data, indent=2, sort_keys=True)
    except TypeError:
        # Older tests sometimes monkeypatch atomic_json_write without accepting
        # json.dump kwargs. Keep the production path sorted; degrade cleanly.
        atomic_json_write(store_path(), data, indent=2)


def normalize_title(title: str, *, fallback: str = _FALLBACK_TITLE) -> str:
    """Sanitize a human or generated title for use as a Slack marker."""

    text = str(title or "").strip()
    text = re.sub(r"^\*{1,2}|\*{1,2}$", "", text).strip()
    text = text.strip("`*_~ \t\r\n:-")
    text = re.sub(r"<@[^>]+>", "", text)
    text = re.sub(r"<([^|>]+)\|([^>]+)>", r"\2", text)
    text = re.sub(r"https?://\S+", "", text)
    words = _WORD_RE.findall(text)
    if not words:
        words = _WORD_RE.findall(fallback)
    # Preserve explicit acronym-ish casing but title-case ordinary lowercase.
    normalized: list[str] = []
    for word in words[:10]:
        if any(ch.isupper() for ch in word[1:]) or word.isupper():
            normalized.append(word)
        else:
            normalized.append(word[:1].upper() + word[1:])
    for filler in _FILLER_WORDS:
        if len(normalized) >= 5:
            break
        normalized.append(filler)
    return " ".join(normalized[:10]).strip() or fallback


def generate_title(seed_text: str) -> str:
    """Generate a deterministic 5-10 word title from thread text."""

    text = str(seed_text or "")
    text = re.sub(r"```[\s\S]*?```", " ", text)
    text = re.sub(r"`[^`]*`", " ", text)
    text = re.sub(r"<@[^>]+>", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\[[^\]]+\]:", " ", text)
    candidates = _WORD_RE.findall(text)
    if not candidates:
        return _FALLBACK_TITLE

    words: list[str] = []
    for candidate in candidates:
        clean = candidate.strip("._-/#")
        if not clean:
            continue
        if not words and clean.lower() in _STOPWORDS:
            continue
        words.append(clean)
        if len(words) >= 8:
            break

    if not words:
        words = candidates[:5]
    for filler in _FILLER_WORDS:
        if len(words) >= 5:
            break
        words.append(filler)
    return normalize_title(" ".join(words[:10]))


def get_thread_title(channel_id: str, thread_ts: str) -> Optional[str]:
    """Return a persisted title for a Slack thread, if any."""

    key = make_thread_key(channel_id, thread_ts)
    if key == ":":
        return None
    with _STORE_LOCK:
        data = _load_store_unlocked()
        entry = data.get("threads", {}).get(key)
    if isinstance(entry, dict):
        title = entry.get("title")
    else:
        title = entry
    if not isinstance(title, str) or not title.strip():
        return None
    return normalize_title(title)


def set_thread_title(channel_id: str, thread_ts: str, title: str) -> str:
    """Persist and return the normalized title for a Slack thread."""

    key = make_thread_key(channel_id, thread_ts)
    normalized = normalize_title(title)
    if key == ":":
        return normalized
    with _STORE_LOCK:
        data = _load_store_unlocked()
        threads = data.setdefault("threads", {})
        if not isinstance(threads, dict):
            threads = {}
            data["threads"] = threads
        threads[key] = {"title": normalized}
        if len(threads) > _MAX_ENTRIES:
            for old_key in list(threads)[: len(threads) - _MAX_ENTRIES]:
                threads.pop(old_key, None)
        _save_store_unlocked(data)
    return normalized


def get_or_create_thread_title(channel_id: str, thread_ts: str, seed_text: str) -> str:
    """Return an existing title or persist a deterministic title for the thread."""

    existing = get_thread_title(channel_id, thread_ts)
    if existing:
        return existing
    return set_thread_title(channel_id, thread_ts, generate_title(seed_text))


def extract_retitle_request(text: str) -> Optional[str]:
    """Return an explicit retitle request payload from a Slack message."""

    match = _RETITLE_RE.match(str(text or ""))
    if not match:
        return None
    title = normalize_title(match.group("title"))
    return title or None


def normalize_placement(placement: str | None) -> str:
    """Normalize visible title marker placement config."""

    value = str(placement or "both").strip().lower().replace("_", "-")
    aliases = {
        "prepend": "first",
        "prefix": "first",
        "start": "first",
        "append": "last",
        "suffix": "last",
        "end": "last",
        "edge": "both",
        "edges": "both",
    }
    value = aliases.get(value, value)
    if value not in {"first", "last", "both", "none"}:
        return "both"
    return value


def title_marker(title: str) -> str:
    """Return the canonical visible Markdown title marker."""

    return f"**{normalize_title(title)}:**"


def build_thread_title_prompt(title: str, *, placement: str = "both") -> str:
    """Build the ephemeral system instruction injected for a Slack thread."""

    normalized = normalize_title(title)
    where = normalize_placement(placement)
    if where == "first":
        instruction = (
            f"Begin every user-visible reply in this thread with exactly "
            f"`**{normalized}:**`."
        )
    elif where == "last":
        instruction = (
            f"End every user-visible reply in this thread with exactly "
            f"`**{normalized}:**` as the final paragraph."
        )
    elif where == "both":
        instruction = (
            f"Begin every user-visible reply in this thread with exactly "
            f"`**{normalized}:**` and end it with exactly `**{normalized}:**` "
            "as the final paragraph."
        )
    else:
        instruction = (
            "Slack delivery will attach the preview title fallback; do not add "
            "a visible title marker unless the user asks."
        )
    return (
        "[Slack thread title]\n"
        f"This Slack thread's fixed preview title is: {normalized}. "
        f"{instruction} Reuse this exact title until the user explicitly "
        "says `retitle this thread: <new title>`."
    )


def _title_from_markdown_line(line: str) -> Optional[str]:
    for pattern in (_MRKDWN_PREFIX_RE, _PLAIN_PREFIX_RE):
        match = pattern.match(line)
        if match:
            return normalize_title(match.group("title"))
    return None


def _title_from_content_edge(content: str, *, first: bool) -> Optional[str]:
    lines = [line.strip() for line in str(content or "").splitlines() if line.strip()]
    if not lines:
        return None
    return _title_from_markdown_line(lines[0] if first else lines[-1])


def content_has_title_marker(content: str, title: str) -> bool:
    """Return True when content already begins or ends with the requested title."""

    normalized = normalize_title(title).casefold()
    return any(
        existing and existing.casefold() == normalized
        for existing in (
            _title_from_content_edge(content, first=True),
            _title_from_content_edge(content, first=False),
        )
    )


def _content_edge_has_title_marker(content: str, title: str, *, first: bool) -> bool:
    normalized = normalize_title(title).casefold()
    existing = _title_from_content_edge(content, first=first)
    return bool(existing and existing.casefold() == normalized)


def apply_title_marker(content: str, title: str, *, placement: str = "both") -> str:
    """Apply the visible Slack thread title marker at the requested edge(s)."""

    body = str(content or "")
    if not body.strip():
        return body
    normalized = normalize_title(title)
    marker = title_marker(normalized)
    where = normalize_placement(placement)
    if where == "none":
        return body

    result = body.strip()
    if where in {"first", "both"} and not _content_edge_has_title_marker(
        result, normalized, first=True
    ):
        result = f"{marker}\n\n{result.lstrip()}"
    if where in {"last", "both"} and not _content_edge_has_title_marker(
        result, normalized, first=False
    ):
        result = f"{result.rstrip()}\n\n{marker}"
    return result


def _strip_title_marker_lines(content: str, title: str) -> str:
    normalized = normalize_title(title).casefold()
    kept: list[str] = []
    for line in str(content or "").splitlines():
        existing = _title_from_markdown_line(line.strip())
        if existing and existing.casefold() == normalized:
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def _plain_preview_text(content: str) -> str:
    text = str(content or "")
    text = re.sub(r"```[\s\S]*?```", " ", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", text)
    text = re.sub(r"_([^_]+)_", r"\1", text)
    text = re.sub(r"<([^|>]+)\|([^>]+)>", r"\2", text)
    text = re.sub(r"<([^>]+)>", r"\1", text)
    text = re.sub(r"\[[^\]]+\]\(([^)]+)\)", r"\1", text)
    text = re.sub(r"^[>\-#*\s]+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_preview_fallback(content: str, title: str, *, max_length: int = 240) -> str:
    """Return Slack top-level text that starts with the stable thread title.

    Slack documents top-level ``text`` as the notification fallback when a
    message uses blocks. Include a compact body excerpt after the title so the
    fallback remains useful for notifications and screen readers without
    sacrificing the preview prefix.
    """

    normalized = normalize_title(title)
    prefix = f"{normalized}:"
    body = _plain_preview_text(_strip_title_marker_lines(content, normalized))
    if not body:
        return prefix
    fallback = f"{prefix} {body}"
    if len(fallback) <= max_length:
        return fallback
    return fallback[: max_length - 1].rstrip() + "…"


def apply_title_prefix(content: str, title: str) -> str:
    """Compatibility wrapper: apply title markers at both visible edges."""

    return apply_title_marker(content, title, placement="both")
