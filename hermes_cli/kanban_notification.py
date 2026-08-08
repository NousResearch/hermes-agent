"""Concise, link-safe Kanban completion notification formatting."""

from __future__ import annotations

import re
from collections.abc import Iterable

COMPLETION_NOTIFICATION_MAX_CHARS = 500
COMPLETION_NOTIFICATION_MAX_LINES = 4
_COMPLETION_HEADER_MAX_CHARS = 180
_DETAILS_NOTICE = "… Full details: Kanban card."
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^\s)]+)\)")
_BARE_URL_RE = re.compile(r"https?://[^\s<>]+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+")


def extract_complete_urls(text: str | None) -> list[str]:
    """Return complete, deduplicated HTTP(S) URLs without trailing prose punctuation."""
    value = str(text or "")
    urls: list[str] = []
    seen: set[str] = set()

    def _add(raw: str) -> None:
        url = raw.rstrip(".,;:!?)]}")
        if url and url not in seen:
            seen.add(url)
            urls.append(url)

    for match in _MARKDOWN_LINK_RE.finditer(value):
        _add(match.group(2))
    for match in _BARE_URL_RE.finditer(value):
        _add(match.group(0))
    return urls


def _link_priority(url: str) -> int:
    lowered = url.lower()
    if "github.com/" in lowered and "/pull/" in lowered:
        return 0
    if "/merge_requests/" in lowered or "/artifacts/" in lowered:
        return 1
    return 2


def _plain_summary(text: str, urls: Iterable[str]) -> str:
    value = _MARKDOWN_LINK_RE.sub(lambda match: match.group(1), text)
    for url in urls:
        value = value.replace(url, "")
    value = _BARE_URL_RE.sub("", value)
    value = value.replace("`", "")
    value = re.sub(r"[*_~]+", "", value)
    return " ".join(value.split()).strip(" -–—:;,.")


def _shorten_at_boundary(text: str, budget: int) -> tuple[str, bool]:
    value = " ".join(text.split())
    if len(value) <= budget:
        return value, False
    if budget <= 1:
        return "…"[:budget], True
    prefix = value[: budget - 1].rstrip()
    boundary = max(prefix.rfind(" "), prefix.rfind("/"), prefix.rfind("-"))
    if boundary >= max(1, (budget - 1) // 2):
        prefix = prefix[:boundary].rstrip()
    return prefix.rstrip(".,;:–—-") + "…", True


def _summary_lines(text: str, slots: int, per_line_budget: int) -> tuple[list[str], bool]:
    if not text:
        return ["Completed."], False
    chunks = [chunk.strip() for chunk in _SENTENCE_SPLIT_RE.split(text) if chunk.strip()]
    if not chunks:
        chunks = [text]
    lines: list[str] = []
    shortened = len(chunks) > slots
    for chunk in chunks[:slots]:
        line, clipped = _shorten_at_boundary(chunk, per_line_budget)
        if line:
            lines.append(line)
        shortened = shortened or clipped
    return lines or ["Completed."], shortened


def format_completion_notification(
    *,
    prefix: str,
    title: str,
    summary: str | None,
    explicit_urls: Iterable[str] = (),
) -> str:
    """Build a 2–4 line completion notice without slicing URLs or Markdown tokens."""
    title_budget = max(24, _COMPLETION_HEADER_MAX_CHARS - len(prefix) - 3)
    short_title, title_shortened = _shorten_at_boundary(title, title_budget)
    header = f"{prefix} — {short_title}" if short_title else prefix

    source = str(summary or "").strip()
    discovered = list(explicit_urls) + extract_complete_urls(source)
    unique_urls = list(dict.fromkeys(url for url in discovered if url))
    unique_urls.sort(key=_link_priority)

    # Reserve at least one summary line. Two complete links still fit the
    # 2–4-line contract; additional links remain on the Kanban card.
    link_lines: list[str] = []
    for url in unique_urls:
        if len(link_lines) >= 2:
            break
        prospective = [header, "Completed.", *link_lines, url]
        if len("\n".join(prospective)) <= COMPLETION_NOTIFICATION_MAX_CHARS:
            link_lines.append(url)

    summary_slots = max(1, COMPLETION_NOTIFICATION_MAX_LINES - 1 - len(link_lines))
    fixed_chars = len(header) + sum(len(url) for url in link_lines)
    fixed_newlines = len(link_lines) + summary_slots
    available = max(48, COMPLETION_NOTIFICATION_MAX_CHARS - fixed_chars - fixed_newlines)
    per_line_budget = max(48, available // summary_slots)
    plain = _plain_summary(source, unique_urls)
    body_lines, summary_shortened = _summary_lines(plain, summary_slots, per_line_budget)

    omitted_links = len(link_lines) < len(unique_urls)
    shortened = title_shortened or summary_shortened or omitted_links
    if shortened:
        last = body_lines[-1]
        notice_budget = per_line_budget - len(_DETAILS_NOTICE) - 1
        last, _ = _shorten_at_boundary(last, max(16, notice_budget))
        body_lines[-1] = f"{last} {_DETAILS_NOTICE}".strip()

    lines = [header, *body_lines, *link_lines]
    message = "\n".join(lines[:COMPLETION_NOTIFICATION_MAX_LINES])
    if len(message) > COMPLETION_NOTIFICATION_MAX_CHARS:
        # URLs are indivisible. Only shrink prose/header if the final accounting
        # is tight due to punctuation or a long task title.
        overflow = len(message) - COMPLETION_NOTIFICATION_MAX_CHARS
        target = max(24, len(body_lines[-1]) - overflow)
        body_lines[-1], _ = _shorten_at_boundary(body_lines[-1], target)
        lines = [header, *body_lines, *link_lines]
        message = "\n".join(lines[:COMPLETION_NOTIFICATION_MAX_LINES])
    return message
