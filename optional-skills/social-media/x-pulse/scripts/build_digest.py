#!/usr/bin/env python3
"""Assemble an X pulse digest from one or more ``x_search`` results.

The native ``x_search`` tool answers ONE query at a time and returns JSON with
an ``answer`` string, a ``citations`` list (``[{"url", "title"}]``), and a
``degraded`` flag that is ``True`` when the answer came from the model's own
knowledge rather than the live X index. This helper merges several per-topic
results into a single deduplicated markdown digest suitable for delivery via
``send_message`` or a scheduled ``cronjob``.

Input (``--input FILE`` or stdin): a JSON list of entries, one per topic::

    [{"topic": "AI research", "result": {<x_search JSON>}}, ...]

Output (stdout): a markdown digest. Pure stdlib; no network.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Tuple


def _norm_url(url: str) -> str:
    """Normalize a URL for dedup: drop fragment, trailing slash, lowercase."""
    u = (url or "").strip()
    u = u.split("#", 1)[0]
    if u.endswith("/"):
        u = u[:-1]
    return u.lower()


def dedupe_citations(citations: Any) -> List[Dict[str, str]]:
    """Deduplicate citations by normalized URL, preserving first-seen order."""
    seen: set = set()
    out: List[Dict[str, str]] = []
    for c in citations or []:
        if not isinstance(c, dict):
            continue
        url = str(c.get("url") or "").strip()
        if not url:
            continue
        key = _norm_url(url)
        if key in seen:
            continue
        seen.add(key)
        out.append({"url": url, "title": str(c.get("title") or "").strip()})
    return out


def _citation_lines(citations: List[Dict[str, str]]) -> List[str]:
    lines = []
    for c in citations:
        label = c["title"] or c["url"]
        lines.append(f"- [{label}]({c['url']})")
    return lines


def _topic_section(topic: str, result: Any) -> Tuple[str, List[Dict[str, str]]]:
    """Render one topic's section and return (markdown, its deduped citations)."""
    if not isinstance(result, dict) or not result.get("success"):
        err = result.get("error") if isinstance(result, dict) else None
        note = f": {err}" if err else ""
        return f"### {topic}\n\n_(no result{note})_\n", []

    answer = str(result.get("answer") or "").strip()
    citations = dedupe_citations(result.get("citations"))
    lines = [f"### {topic}", ""]
    if result.get("degraded"):
        lines.append(
            "> ⚠️ Unsourced — the answer came from the model's own knowledge, "
            "not the live X index. Treat with caution."
        )
        lines.append("")
    lines.append(answer or "_(empty answer)_")
    if citations:
        lines.append("")
        lines.extend(_citation_lines(citations))
    lines.append("")
    return "\n".join(lines), citations


def build_digest(entries: Any, title: str = "X Pulse", date_str: str = "") -> str:
    """Build a markdown digest from a list of ``{topic, result}`` entries."""
    header = f"# {title}"
    if date_str:
        header += f" — {date_str}"
    parts: List[str] = [header, ""]

    if not entries:
        parts.append("_No topics provided._")
        return "\n".join(parts) + "\n"

    all_citations: List[Dict[str, str]] = []
    for entry in entries:
        entry = entry or {}
        topic = str(entry.get("topic") or "").strip() or "(untitled)"
        section, cites = _topic_section(topic, entry.get("result"))
        parts.append(section)
        all_citations.extend(cites)

    merged = dedupe_citations(all_citations)
    if merged:
        parts.extend(["---", "", f"**Sources ({len(merged)})**", ""])
        parts.extend(_citation_lines(merged))

    return "\n".join(parts).rstrip() + "\n"


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Build an X pulse digest from x_search results."
    )
    ap.add_argument("--input", help="JSON file with per-topic results (default: stdin).")
    ap.add_argument("--title", default="X Pulse")
    ap.add_argument("--date", default="", dest="date_str")
    args = ap.parse_args(argv)

    raw = (
        open(args.input, encoding="utf-8").read()
        if args.input
        else sys.stdin.read()
    )
    try:
        entries = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"error: invalid JSON input: {exc}", file=sys.stderr)
        return 2
    if not isinstance(entries, list):
        print(
            "error: input must be a JSON list of {topic, result} entries",
            file=sys.stderr,
        )
        return 2

    sys.stdout.write(build_digest(entries, title=args.title, date_str=args.date_str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
