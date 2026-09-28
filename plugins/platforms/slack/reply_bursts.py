"""Paragraph bubbles for completed conversational replies, never arbitrary sends."""
import re

from gateway.platforms.helpers import split_markdown_atoms


def reply_chunks(content: str) -> list[str]:
    # Loose lists, reference links, indented code and non-backtick fences can span
    # paragraphs. Keep that reply intact rather than detach their continuation.
    if re.search(r"(?m)^(?:[ \t]*(?:[-+*]|\d+[.)])\s|[ \t]*>|[ \t]*\[[^\]]+\]:| {4}|\t|[ \t]+```|`{4,}|[ \t]*~~~)", content):
        return [content]
    return split_markdown_atoms(content) or [content]
