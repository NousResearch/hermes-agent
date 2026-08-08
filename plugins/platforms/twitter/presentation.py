import re
import unicodedata
from importlib import import_module

from gateway.platforms.helpers import strip_markdown

X_WEIGHTED_LIMIT = 280
X_BOT_WEIGHTED_LIMIT = 2_800
X_MAX_FALLBACK_PARTS = 10
X_OVER_LIMIT_NOTICE = "Response was too long. Please ask for a narrower answer."
TWITTER_TEXT_INSTALL_HINT = (
    "Run `hermes gateway setup` and choose Twitter / X to install its plugin dependency."
)
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def weighted_parser_available() -> bool:
    try:
        return callable(getattr(import_module("twitter_text"), "parse_tweet", None))
    except ImportError:
        return False


def markdown_links_to_text(markdown: str) -> str:
    return _MARKDOWN_LINK_RE.sub(
        lambda match: f"{match.group(1)} ({match.group(2).strip()})", markdown
    )


def reject_disallowed_controls(text: str) -> None:
    if any(unicodedata.category(char) == "Cc" and char not in "\n\t" for char in text):
        raise ValueError("Twitter content contains a disallowed control character")


def validate_x_weighted_length(text: str) -> None:
    parsed = _parse_x_text(text)
    if parsed.weightedLength > X_WEIGHTED_LIMIT:
        raise ValueError(
            f"Twitter content exceeds the X weighted limit of {X_WEIGHTED_LIMIT}"
        )
    if not parsed.valid:
        raise ValueError("Twitter content is empty or contains unsupported characters")


def _parse_x_text(text: str):
    try:
        from twitter_text import parse_tweet
    except ImportError as exc:
        raise RuntimeError(
            f"Twitter weighted-text parser is not installed; run: {TWITTER_TEXT_INSTALL_HINT}"
        ) from exc
    return parse_tweet(text)


def x_weighted_length(text: str) -> int:
    return _parse_x_text(text).weightedLength


def format_public_message(markdown: str) -> str:
    text = markdown_links_to_text(markdown)
    text = strip_markdown(text)
    text = unicodedata.normalize("NFC", text)
    reject_disallowed_controls(text)
    if not text.strip():
        raise ValueError("Twitter content is empty or contains unsupported characters")
    return text


def _utf16_prefix_index(text: str, units: int) -> int:
    consumed = 0
    for index, char in enumerate(text):
        consumed += 2 if ord(char) > 0xFFFF else 1
        if consumed >= units:
            return index + 1
    return len(text)


def format_message(markdown: str) -> str:
    text = format_public_message(markdown)
    validate_x_weighted_length(text)
    return text


def format_thread_messages(markdown: str) -> list[str]:
    remaining = format_public_message(markdown)
    parts: list[str] = []
    while remaining:
        parsed = _parse_x_text(remaining)
        if parsed.weightedLength <= X_WEIGHTED_LIMIT:
            validate_x_weighted_length(remaining)
            parts.append(remaining)
            break

        prefix_end = _utf16_prefix_index(remaining, parsed.validRangeEnd + 1)
        candidate = remaining[:prefix_end]
        split_at = max(candidate.rfind(" "), candidate.rfind("\n"), candidate.rfind("\t"))
        if split_at <= 0:
            split_at = prefix_end
        part = remaining[:split_at].rstrip()
        if not part:
            raise ValueError("Twitter content is empty or contains unsupported characters")
        validate_x_weighted_length(part)
        parts.append(part)
        remaining = remaining[split_at:].lstrip()

    if not parts:
        validate_x_weighted_length(remaining)
    return parts
