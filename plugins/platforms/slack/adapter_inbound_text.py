"""Slack's flat message ``text`` back to what the user typed.

Slack escapes ``&``, ``<`` and ``>`` as ``&amp;``, ``&lt;`` and ``&gt;`` in a message's flat
``text`` and wraps links as ``<url>`` or ``<url|label>``; Block Kit ``blocks`` stay raw. The agent
reads the text as typed, and text-vs-blocks dedupe compares both in one form.
"""
from __future__ import annotations

import re

#: Any ``<scheme:target|label>`` autolink (Slack is not limited to https/mailto).
_SLACK_MRKDWN_LINK_RE = re.compile(r"<([a-zA-Z][a-zA-Z0-9+.\-]*:[^>|]+)(?:\|([^>]+))?>")
_SLACK_HTML_ENTITY_RE = re.compile(r"&(amp|lt|gt);")
_SLACK_HTML_ENTITIES = {"amp": "&", "lt": "<", "gt": ">"}


def _unescape_slack_entities(text: str) -> str:
    """Undo Slack's ``&``/``<``/``>`` escaping in flat ``text``.
    ``blocks`` are raw, so text-vs-blocks comparison needs a common form (every "Copy link"
    permalink carries ``?thread_ts=…&cid=…``)."""
    return _SLACK_HTML_ENTITY_RE.sub(lambda match: _SLACK_HTML_ENTITIES[match.group(1)], text or "")


def _slack_link_text(match: re.Match) -> str:
    """``<url|label>`` as rendered rich text shows it: ``label (url)``, or the bare URL."""
    url, label = match.group(1), match.group(2)
    return f"{label} ({url})" if label and label != url else url


def _slack_text_for_agent(text: str) -> str:
    """Flat ``text`` as the user wrote it: links unwrapped, then ``&``/``<``/``>`` unescaped.
    Links go first so a literal ``<https://…>`` the user typed (sent escaped) stays as typed.
    Mentions (``<@U…>``, ``<#C…>``, ``<!…>``) are kept for routing and name resolution."""
    return _unescape_slack_entities(_SLACK_MRKDWN_LINK_RE.sub(_slack_link_text, text or ""))
