#!/usr/bin/env python3
"""Classify candidate items by urgency/importance and emit only the urgent ones.

The proactive-monitor pattern: a fetch step (a watcher script, an inbox dump, a
feed) produces a list of candidate items; this script scores each with a cheap
LLM and prints ONLY the items at or above a threshold. Below-threshold runs
print nothing, so a cron job wrapping this stays silent unless something
actually matters -- the classic urgency-monitor pattern (fetch -> classify
urgency -> surface only what's above the bar).

Design choices:
  * Uses Hermes' auxiliary client with task="monitor", so the classifier model
    is configured once in config.yaml (auxiliary.monitor.{provider,model}) and
    can be a cheap fast model independent of the main chat model.
  * Reads items as JSON (a list of objects) from stdin or --input-file.
  * One LLM call scores the whole batch (cheap, single round-trip) and returns
    structured scores; we filter locally.
  * Empty result -> empty stdout -> the cron job's [SILENT]/empty-stdout path
    suppresses delivery. No spam on quiet intervals.

Usage (standalone):
  cat items.json | python classify_items.py --threshold 7 \
    --criteria "Urgent if it needs a reply today or is from my manager/family"

Usage (wired to a watcher via cron, agent mode):
  Ask the agent: "Every 10 minutes, run watch_http_json.py for my inbox feed,
  pipe its JSON into classify_items.py with my urgency criteria, and deliver
  whatever it prints. Stay silent if it prints nothing."

Item schema (flexible): each item is an object; the classifier sees the whole
object. A "title"/"subject"/"summary"/"text" field helps it judge. An "id"
field (any of id/guid/message_id/url) is echoed back so duplicates can be
deduped upstream.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional


def _eprint(*args: Any) -> None:
    print(*args, file=sys.stderr)


def _load_items(input_file: Optional[str]) -> List[Dict[str, Any]]:
    raw = ""
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
    for key in ("id", "guid", "message_id", "url", "link"):
        val = item.get(key)
        if val:
            return str(val)
    return f"item-{index}"


_CLASSIFY_INSTRUCTIONS = (
    "You are an urgency classifier for a proactive assistant. You will be given "
    "a numbered list of items and the user's importance criteria. Score EACH "
    "item from 0 (ignore entirely) to 10 (interrupt the user now). Return ONLY a "
    "JSON array, one object per item, in the same order: "
    '[{"index": <int>, "score": <int 0-10>, "reason": "<short>"}]. '
    "No prose, no markdown fences. Be conservative: most items should score low. "
    "Only score high when the item clearly meets the user's criteria."
)


def _build_prompt(items: List[Dict[str, Any]], criteria: str) -> str:
    lines = [f"USER IMPORTANCE CRITERIA:\n{criteria}\n", "ITEMS:"]
    for i, item in enumerate(items):
        # Show a compact view; the model sees the salient fields.
        view = {
            k: item[k]
            for k in ("title", "subject", "summary", "text", "body", "from", "sender", "url")
            if k in item
        }
        if not view:
            view = item  # fall back to the whole object
        lines.append(f"[{i}] {json.dumps(view, ensure_ascii=False)[:1200]}")
    lines.append(
        "\nReturn the JSON array of scores now (one object per item, same order)."
    )
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
        if start >= 0 and end > start:
            try:
                arr = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                _eprint("classify_items: could not parse classifier output")
                return {}
        else:
            _eprint("classify_items: classifier returned no JSON array")
            return {}
    out: Dict[int, Dict[str, Any]] = {}
    if isinstance(arr, list):
        for position, obj in enumerate(arr):
            if not isinstance(obj, dict):
                continue
            idx = obj.get("index")
            if idx is None:
                if 0 <= position < n_items:
                    out[position] = obj
                continue
            if isinstance(idx, int) and 0 <= idx < n_items:
                out[idx] = obj
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Classify items by urgency; emit only urgent ones.")
    parser.add_argument("--criteria", required=True, help="Plain-language importance criteria.")
    parser.add_argument("--threshold", type=int, default=7, help="Minimum score (0-10) to surface. Default 7.")
    parser.add_argument("--input-file", default=None, help="Read items JSON from this file instead of stdin.")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format for surfaced items.")
    args = parser.parse_args()

    items = _load_items(args.input_file)
    if not items:
        # Nothing to classify -> silent. This is the common quiet-interval case.
        return 0

    # Import here so --help works without the package importable.
    try:
        from agent.auxiliary_client import call_llm
    except Exception as e:  # pragma: no cover - import guard
        _eprint(f"classify_items: cannot import auxiliary client: {e}")
        return 3

    prompt = _build_prompt(items, args.criteria)
    try:
        resp = call_llm(
            task="monitor",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0,
        )
        content = resp.choices[0].message.content
        if not isinstance(content, str):
            content = str(content) if content else ""
    except Exception as e:
        # Classification failure is NOT silent -- surface it so a broken monitor
        # doesn't quietly swallow important items. Non-zero exit -> cron alerts.
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
        # Below threshold -> silent. Empty stdout; cron suppresses delivery.
        return 0

    if args.format == "json":
        out = [
            {
                "id": _item_id(item, i),
                "score": s.get("score"),
                "reason": s.get("reason", ""),
                "item": item,
            }
            for (i, item, s) in surfaced
        ]
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        blocks = []
        for (i, item, s) in surfaced:
            title = (
                item.get("title")
                or item.get("subject")
                or item.get("summary")
                or _item_id(item, i)
            )
            url = item.get("url") or item.get("link") or ""
            reason = s.get("reason", "")
            block = f"## [{s.get('score')}/10] {title}"
            if url:
                block += f"\n{url}"
            if reason:
                block += f"\n_{reason}_"
            blocks.append(block)
        print("\n\n".join(blocks))
    return 0


if __name__ == "__main__":
    sys.exit(main())
