#!/usr/bin/env python3
"""Classify candidate items by urgency/importance and emit only the urgent ones.

The proactive-monitor pattern: a fetch step (watcher script, inbox dump, feed) produces a JSON list
of candidate items (stdin or --input-file); one call to the auxiliary ``monitor`` model scores the
whole batch and ONLY items at/above --threshold are printed. Empty stdout -> the cron job's
[SILENT]/empty-stdout path suppresses delivery, so quiet intervals never spam. A classifier failure
exits non-zero (never silently swallowed). Items are opaque objects; a title/subject/summary/text
field helps, and id/guid/message_id/url is echoed back for upstream dedup.

Usage: cat items.json | python classify_items.py --threshold 7 --criteria "Urgent if ..."
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

_ID_KEYS = ("id", "guid", "message_id", "url", "link")
_VIEW_KEYS = ("title", "subject", "summary", "text", "body", "from", "sender", "url")


def _eprint(*args: Any) -> None:
    print(*args, file=sys.stderr)


def _load_items(input_file: Optional[str]) -> List[Dict[str, Any]]:
    if input_file:
        with open(input_file, encoding="utf-8") as f:
            raw = f.read()
    else:
        raw = sys.stdin.read()
    raw = raw.strip()
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        _eprint(f"classify_items: input is not valid JSON: {e}")
        sys.exit(2)
    if isinstance(data, dict):
        # Allow {"items": [...]} or a single object.
        if isinstance(data.get("items"), list):
            return data["items"]
        return [data]
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    _eprint("classify_items: expected a JSON list or {items: [...]}")
    sys.exit(2)


def _item_id(item: Dict[str, Any], index: int) -> str:
    return next((str(item[key]) for key in _ID_KEYS if item.get(key)), f"item-{index}")


def _build_prompt(items: List[Dict[str, Any]], criteria: str) -> str:
    lines = [f"USER IMPORTANCE CRITERIA:\n{criteria}\n", "ITEMS:"]
    for i, item in enumerate(items):
        # Compact view of the salient fields; the whole object when none are present.
        view = {k: item[k] for k in _VIEW_KEYS if k in item} or item
        lines.append(f"[{i}] {json.dumps(view, ensure_ascii=False)[:1200]}")
    lines.append("\nReturn the JSON array of scores now (one object per item, same order).")
    return "\n".join(lines)


def _parse_scores(content: str, n_items: int) -> Dict[int, Dict[str, Any]]:
    text = (content or "").strip()
    # Tolerate accidental markdown fences.
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            text = text.split("\n", 1)[1]
    try:
        arr = json.loads(text)
    except json.JSONDecodeError:
        # Last-ditch: find the first [...] block.
        start = text.find("[")
        end = text.rfind("]")
        if not (start >= 0 and end > start):
            _eprint("classify_items: classifier returned no JSON array")
            return {}
        try:
            arr = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            _eprint("classify_items: could not parse classifier output")
            return {}
    if not isinstance(arr, list):
        return {}
    return {
        obj["index"]: obj
        for obj in arr
        if isinstance(obj, dict)
        and isinstance(obj.get("index"), int)
        and 0 <= obj["index"] < n_items
    }


def _render_text(surfaced: list) -> str:
    blocks = []
    for i, item, s in surfaced:
        title = item.get("title") or item.get("subject") or item.get("summary") or _item_id(item, i)
        block = f"## [{s.get('score')}/10] {title}"
        if url := item.get("url") or item.get("link") or "":
            block += f"\n{url}"
        if reason := s.get("reason", ""):
            block += f"\n_{reason}_"
        blocks.append(block)
    return "\n\n".join(blocks)


def main() -> int:
    parser = argparse.ArgumentParser(description="Classify items by urgency; emit only urgent ones.")
    parser.add_argument("--criteria", required=True, help="Plain-language importance criteria.")
    parser.add_argument("--threshold", type=int, default=7, help="Minimum score (0-10) to surface. Default 7.")
    parser.add_argument("--input-file", default=None, help="Read items JSON from this file instead of stdin.")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format for surfaced items.")
    args = parser.parse_args()

    items = _load_items(args.input_file)
    if not items:
        return 0  # nothing to classify -> silent (the common quiet-interval case)

    # Import here so --help works without the package importable.
    try:
        from agent.auxiliary_client import call_llm
    except Exception as e:  # pragma: no cover - import guard
        _eprint(f"classify_items: cannot import auxiliary client: {e}")
        return 3

    prompt = _build_prompt(items, args.criteria)
    try:
        resp = call_llm(
            task="monitor", messages=[{"role": "user", "content": prompt}], max_tokens=1024,
            temperature=0,
        )
        content = resp.choices[0].message.content
        if not isinstance(content, str):
            content = str(content) if content else ""
    except Exception as e:
        # A broken monitor must not quietly swallow important items: non-zero exit -> cron alerts.
        _eprint(f"classify_items: classifier call failed: {e}")
        return 4

    scores = _parse_scores(content, len(items))
    surfaced = []
    for i, item in enumerate(items):
        s = scores.get(i)
        score = s.get("score") if isinstance(s, dict) else None
        if isinstance(score, int) and score >= args.threshold:
            surfaced.append((i, item, s))

    if not surfaced:
        return 0  # below threshold -> silent; empty stdout suppresses delivery

    if args.format == "json":
        out = [
            {
                "id": _item_id(item, i), "score": s.get("score"),
                "reason": s.get("reason", ""), "item": item,
            }
            for (i, item, s) in surfaced
        ]
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print(_render_text(surfaced))
    return 0


if __name__ == "__main__":
    sys.exit(main())
