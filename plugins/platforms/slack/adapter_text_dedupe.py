"""Comparison-only normalization for Slack's flat and structured authored text."""

import re


_SLACK_MRKDWN_LINK_RE = re.compile(r"<([a-zA-Z][a-zA-Z0-9+.\-]*:[^>|]+)(?:\|([^>]+))?>")
_SLACK_ENTITY_LABEL_RE = re.compile(r"<([@#!][^>|]*)\|[^>]*>")
FENCED_CODE_RE = re.compile(r"(?<!`)\n*```[ \t]*\n?(.*?)\n?[ \t]*```\n*(?!`)", re.DOTALL)
_SLACK_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
_SLACK_DATE_RE = re.compile(r"<!date\^([^>|]*)(?:\|([^>]*))?>")
# Only the permalink tail can be rebuilt from channel_id/message_ts-only blocks.
_SLACK_PERMALINK_RE = re.compile(r"https?://[^\s/]+/(archives/[A-Za-z0-9]+/p\d+)(?:\?[^\s)]*)?")
_SLACK_INLINE_STYLE_RE = re.compile(r"([*_~])([^\n]+?)\1")
_SLACK_HTML_ENTITY_RE = re.compile(r"&(amp|lt|gt);")
_SLACK_HTML_ENTITIES = {"amp": "&", "lt": "<", "gt": ">"}


def unescape_slack_entities(text: str) -> str:
    """Undo Slack's limited escaping; structured blocks already carry raw text."""
    return _SLACK_HTML_ENTITY_RE.sub(lambda match: _SLACK_HTML_ENTITIES[match.group(1)], text or "")


def normalize_slack_text_for_dedupe(text: str, bot_uid: str = "") -> str:
    """Canonical comparison that retains destinations, including arbitrary labels."""

    def _link(match: re.Match) -> str:
        url, label = match.group(1), match.group(2)
        return f"{label} ({url})" if label and label != url else url

    def _date(match: re.Match) -> str:
        fallback = match.group(2)
        if fallback:
            return fallback
        parts = match.group(1).split("^")
        return parts[2] if len(parts) > 2 else ""

    canonical = unescape_slack_entities(text or "")
    canonical = _SLACK_MRKDWN_LINK_RE.sub(_link, canonical)
    canonical = _SLACK_DATE_RE.sub(_date, canonical)
    canonical = _SLACK_PERMALINK_RE.sub(r"\1", canonical)
    canonical = _SLACK_ENTITY_LABEL_RE.sub(r"<\1>", canonical)
    if bot_uid:
        canonical = canonical.replace(f"<@{bot_uid}>", "")
    canonical = FENCED_CODE_RE.sub(r"\1", canonical)
    canonical = _SLACK_INLINE_CODE_RE.sub(r"\1", canonical)
    while True:
        unstyled = _SLACK_INLINE_STYLE_RE.sub(r"\2", canonical)
        if unstyled == canonical:
            break
        canonical = unstyled
    return re.sub(r"\s+", " ", canonical).strip()


def plain_autolink_mirror(element: dict, primary_text: str, bot_uid: str = "") -> str | None:
    """Alternative comparison ONLY for ordinary text/user sections, never hrefs or code.

    Slack can auto-link a typed domain in flat text while leaving its block leaf plain.
    The original flat text remains authoritative; this projection never reaches the model.
    """
    if element.get("type") != "rich_text_section" or "`" in primary_text:
        return None
    leaves = element.get("elements", [])
    if not all(
        leaf.get("type") in {"text", "user"}
        and isinstance(leaf.get("style") or {}, dict)
        and not (leaf.get("style") or {}).get("code")
        and isinstance(leaf.get("text", ""), str)
        and "`" not in leaf.get("text", "")
        for leaf in leaves
    ):
        return None

    def _label(match: re.Match) -> str:
        url, label = match.group(1), match.group(2)
        if label and url.startswith(("http://", "https://")):
            elided = url.split("://", 1)[1]
            # Only a root slash may be omitted, not a path's trailing slash.
            root_slash = elided == label + "/" and not any(c in label for c in "/?#")
            if label == elided or root_slash:
                return label
        return match.group(0)

    # Leave entities escaped here: canonical normalization must unescape exactly once.
    projected = _SLACK_MRKDWN_LINK_RE.sub(_label, primary_text)
    return normalize_slack_text_for_dedupe(projected, bot_uid)
