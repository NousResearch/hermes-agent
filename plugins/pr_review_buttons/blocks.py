"""Slack Block Kit builders for the PR-review digest and its posted state.

Two builders:

* :func:`build_digest_blocks` — one message carrying every unposted PR review,
  each with a header, the verbatim review body (split across the 3000-char
  section cap so nothing is ever dropped), and an actions row with three
  buttons (Approve / Request changes / Comment). The button ``value`` is only
  the store key — the body lives in the store, well under Slack's 2000-char
  ``value`` cap.
* :func:`build_posted_blocks` — the same header + body but with the buttons
  replaced by a context line recording who posted what. Used to rewrite the
  message after a click so it can't be double-actioned.

Pure dict builders — no Slack client, no I/O — so they unit-test directly.
"""

from __future__ import annotations

from typing import Any, Dict, List

MAX_BLOCKS = 50
MAX_SECTION_TEXT = 3000
MAX_HEADER_TEXT = 150

_EVENT_LABEL = {
    "approve": "✅ Approved",
    "request-changes": "🔴 Changes requested",
    "comment": "💬 Commented",
}

Block = Dict[str, Any]


def _split(text: str, limit: int = MAX_SECTION_TEXT) -> List[str]:
    """Split ``text`` into <=limit-char chunks on newlines, then hard cuts."""
    text = text or ""
    if len(text) <= limit:
        return [text]
    out: List[str] = []
    remaining = text
    while len(remaining) > limit:
        cut = remaining.rfind("\n", 0, limit)
        if cut <= 0:
            cut = limit
        out.append(remaining[:cut])
        remaining = remaining[cut:].lstrip("\n")
    if remaining:
        out.append(remaining)
    return out


def _header(title: str) -> Block:
    clean = (title or "PR review").strip()
    if len(clean) > MAX_HEADER_TEXT:
        clean = clean[: MAX_HEADER_TEXT - 1] + "…"
    return {"type": "header", "text": {"type": "plain_text", "text": clean, "emoji": True}}


def _section(text: str) -> Block:
    return {"type": "section", "text": {"type": "mrkdwn", "text": text or " "}}


def _context(text: str) -> Block:
    return {"type": "context", "elements": [{"type": "mrkdwn", "text": text}]}


def _title_line(entry: Dict[str, Any]) -> str:
    """The ``repo#N — title — verdict`` line with a link, as mrkdwn."""
    repo = entry.get("repo", "")
    num = entry.get("number", "")
    url = entry.get("url", "")
    title = entry.get("title", "")
    verdict = entry.get("verdict", "")
    label = f"{repo}#{num}"
    linked = f"<{url}|{label}>" if url else label
    parts = [f"*{linked}*"]
    if title:
        parts.append(title)
    if verdict:
        parts.append(f"_{verdict}_")
    return " — ".join(parts)


def _body_sections(entry: Dict[str, Any]) -> List[Block]:
    return [_section(chunk) for chunk in _split(entry.get("body", ""))]


def _action_row(key: str) -> Block:
    return {
        "type": "actions",
        "block_id": f"prreview::{key}",
        "elements": [
            {
                "type": "button",
                "text": {"type": "plain_text", "text": "Approve"},
                "style": "primary",
                "action_id": "prreview_approve",
                "value": key,
            },
            {
                "type": "button",
                "text": {"type": "plain_text", "text": "Request changes"},
                "style": "danger",
                "action_id": "prreview_request_changes",
                "value": key,
            },
            {
                "type": "button",
                "text": {"type": "plain_text", "text": "Comment"},
                "action_id": "prreview_comment",
                "value": key,
            },
        ],
    }


def build_digest_blocks(entries: List[Dict[str, Any]]) -> List[Block]:
    """Build the buttoned digest for ``entries`` (each must carry ``_key``).

    Guarantees the result stays within Slack's 50-block ceiling: if the full
    batch would overflow, PR bodies are progressively collapsed to just the
    title line (the operator can still act on the buttons; the full body is a
    click away in the store / on GitHub). Buttons are never dropped.
    """
    # First try: full bodies.
    blocks = _assemble(entries, include_body=True)
    if len(blocks) <= MAX_BLOCKS:
        return blocks
    # Fallback: title-only, so buttons for every PR still fit.
    return _assemble(entries, include_body=False)


def _assemble(entries: List[Dict[str, Any]], *, include_body: bool) -> List[Block]:
    blocks: List[Block] = []
    for i, entry in enumerate(entries):
        if i:
            blocks.append({"type": "divider"})
        blocks.append(_header(f"{entry.get('repo','')}#{entry.get('number','')}"))
        blocks.append(_section(_title_line(entry)))
        if include_body:
            blocks.extend(_body_sections(entry))
        blocks.append(_action_row(entry["_key"]))
    return blocks


def build_posted_blocks(
    entry: Dict[str, Any],
    *,
    event: str,
    user: str,
    detail: str = "",
) -> List[Block]:
    """Rebuild a PR block after a click: same context, buttons → decision line."""
    label = _EVENT_LABEL.get(event, event)
    decision = f"{label} by {user}"
    if detail and detail != "posted":
        decision += f" — {detail}"
    blocks: List[Block] = [
        _header(f"{entry.get('repo','')}#{entry.get('number','')}"),
        _section(_title_line(entry)),
    ]
    blocks.extend(_body_sections(entry))
    blocks.append(_context(decision))
    return blocks
