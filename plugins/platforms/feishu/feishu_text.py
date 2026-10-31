"""Inbound Feishu text normalisation: mention placeholders and whitespace."""

import re
from typing import Any, Mapping, Optional

_MULTISPACE_RE = re.compile(r"[ \t]{2,}")
_MENTION_PLACEHOLDER_RE = re.compile(r"@_user_\d+")
_MENTION_WITH_SPACING_RE = re.compile(r"[ \t]*(@_user_\d+)[ \t]*")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_feishu_text(
    text: str, mentions_map: Optional[Mapping[str, Any]] = None, *, keep_layout: bool = False,
) -> str:
    """Resolve mention placeholders and tidy whitespace. ``keep_layout`` (typed text, code blocks)
    removes only the bot's own mention with its spacing, keeping indentation and blank lines;
    otherwise each line is collapsed (labels, card and forward summaries)."""
    refs = mentions_map or {}

    def _sub(match: re.Match[str]) -> str:
        ref = refs.get(match.group(0))
        return " " if ref is None else f"@{ref.name or ref.open_id or 'user'}"

    if keep_layout:
        text = _MENTION_WITH_SPACING_RE.sub(lambda m: m.group(0) if m.group(1) in refs else " ", text or "")
    cleaned = _MENTION_PLACEHOLDER_RE.sub(_sub, text or "")
    cleaned = cleaned.replace("@_all", "@all")
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    if keep_layout:
        return "\n".join(line.rstrip() for line in cleaned.split("\n")).strip()
    cleaned = "\n".join(_WHITESPACE_RE.sub(" ", line).strip() for line in cleaned.split("\n"))
    cleaned = "\n".join(line for line in cleaned.split("\n") if line)
    cleaned = _MULTISPACE_RE.sub(" ", cleaned)
    return cleaned.strip()

