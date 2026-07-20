import re
import unicodedata
from importlib import import_module

from gateway.platforms.helpers import strip_markdown

X_WEIGHTED_LIMIT = 280
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
    try:
        from twitter_text import parse_tweet
    except ImportError as exc:
        raise RuntimeError(
            f"Twitter weighted-text parser is not installed; run: {TWITTER_TEXT_INSTALL_HINT}"
        ) from exc

    parsed = parse_tweet(text)
    if parsed.weightedLength > X_WEIGHTED_LIMIT:
        raise ValueError(
            f"Twitter content exceeds the X weighted limit of {X_WEIGHTED_LIMIT}"
        )
    if not parsed.valid:
        raise ValueError("Twitter content is empty or contains unsupported characters")


def format_message(markdown: str) -> str:
    text = markdown_links_to_text(markdown)
    text = strip_markdown(text)
    text = unicodedata.normalize("NFC", text)
    reject_disallowed_controls(text)
    validate_x_weighted_length(text)
    return text
