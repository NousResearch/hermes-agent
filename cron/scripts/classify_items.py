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
import datetime
import json
from agent.jev_runtime import is_jev_enabled
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_ID_KEYS = ("id", "guid", "message_id", "url", "link")
_VIEW_KEYS = ("title", "subject", "summary", "text", "body", "from", "sender", "url")

# Shadow-mode comparison log: `jev` (TypeSafe CLI) scores the same batch alongside the
# production LLM classifier below, purely for later diffing. See _jev_shadow_classify.
_JEV_SHADOW_LOG = Path("jev-shadow-classify.jsonl")


def _eprint(*args: Any) -> None:
    print(*args, file=sys.stderr)


def _load_items(input_file: Optional[str]) -> list[dict[str, Any]]:
    if input_file:
        with open(input_file, encoding="utf-8-sig") as f:
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


def _item_id(item: dict[str, Any], index: int) -> str:
    return next((str(item[key]) for key in _ID_KEYS if item.get(key)), f"item-{index}")


def _build_prompt(items: list[dict[str, Any]], criteria: str) -> str:
    lines = [f"USER IMPORTANCE CRITERIA:\n{criteria}\n", "ITEMS:"]
    for i, item in enumerate(items):
        # Compact view of the salient fields; the whole object when none are present.
        view = {k: item[k] for k in _VIEW_KEYS if k in item} or item
        lines.append(f"[{i}] {json.dumps(view, ensure_ascii=False)[:1200]}")
    lines.append("\nReturn the JSON array of scores now (one object per item, same order).")
    return "\n".join(lines)


def _parse_scores(content: str, n_items: int) -> dict[int, dict[str, Any]]:
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


def _jev_view_text(item: dict[str, Any]) -> str:
    view = {k: item[k] for k in _VIEW_KEYS if k in item} or item
    return json.dumps(view, ensure_ascii=False)[:1200]


def _append_jev_shadow_log(records: list[dict[str, Any]]) -> None:
    """Append JSON lines to the shadow log. Never raises: a log-write failure must not affect
    the caller, which is itself already inside a best-effort shadow path."""
    try:
        from hermes_constants import get_hermes_home
        log_path = get_hermes_home() / "logs" / _JEV_SHADOW_LOG
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.writelines(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    # health: allow BLE001 -- Shadow log writes must never abort classification; exception text may contain protected item data.
    except Exception as e:  # pragma: no cover - best-effort logging only
        _eprint(f"classify_items: jev shadow log write failed (non-fatal): {type(e).__name__}")


def _jev_shadow_classify(
    items: list[dict[str, Any]], criteria: str, llm_scores: dict[int, dict[str, Any]], threshold: int,
) -> None:
    """Best-effort: score the same batch with the `jev` CLI (TypeSafe) alongside the LLM
    classifier above, and log both side by side for later comparison. Shadow mode only -- this
    NEVER influences which items get surfaced, and any failure here (no `jev` on PATH, no
    credentials, timeout, malformed output) is logged and swallowed, never raised.
    """
    if not is_jev_enabled("classify_items"):
        return
    now = datetime.datetime.now(datetime.UTC).isoformat()
    try:
        rows = "\n".join(
            json.dumps({"id": _item_id(item, i), "text": _jev_view_text(item)}, ensure_ascii=False)
            for i, item in enumerate(items)
        )
        instructions = f"Score urgency 1-10 given this criteria: {criteria}"
        cmd = [
            "jev", "batch", "ask", "-i", "-", "--format", "json",
            "--", "--score", f"urgency={instructions}|1,2,3,4,5,6,7,8,9,10",
        ]
        from tools.environments.local import served_profile_child_env
        proc = subprocess.run(cmd, input=rows, capture_output=True, text=True, encoding="utf-8", errors="replace", env=served_profile_child_env(inherit_credentials=True), timeout=60, check=False)
        # `jev batch` exits 1 when any individual row errored; still parse whatever it printed.
        rows_out = json.loads(proc.stdout) if proc.stdout.strip() else []
        if not isinstance(rows_out, list):
            raise ValueError(f"unexpected jev batch output shape: {type(rows_out).__name__}")
    # health: allow BLE001 -- Optional subprocess judgments must not affect delivery; tracebacks may contain protected input/output.
    except Exception as e:
        _append_jev_shadow_log([{
            "timestamp": now, "batch_size": len(items), "jev_call_failed": True, "error": type(e).__name__,
        }])
        return

    jev_by_id: dict[str, dict[str, Any]] = {}
    for row in rows_out:
        if not isinstance(row, dict):
            continue
        rid = row.get("id")
        if not isinstance(rid, str):
            continue
        if not row.get("ok"):
            jev_by_id[rid] = {"error": "jev_row_failed"}
            continue
        try:
            urgency = row["result"]["answers"]["urgency"]
            legend = urgency["legend"]
            idx = min(max(round(urgency["score"]), 0), len(legend) - 1)
            jev_by_id[rid] = {"score": int(legend[str(idx)]), "confidence": urgency.get("confidence")}
        except (KeyError, TypeError, ValueError, OverflowError):
            jev_by_id[rid] = {"error": "unexpected jev response shape"}

    records = []
    for i, item in enumerate(items):
        item_id = _item_id(item, i)
        llm = llm_scores.get(i)
        llm_score = llm.get("score") if isinstance(llm, dict) else None
        llm_surfaced = isinstance(llm_score, int) and llm_score >= threshold
        jev = jev_by_id.get(item_id, {"error": "no jev result for this item"})
        record: dict[str, Any] = {
            "timestamp": now, "item_id": item_id, "llm_score": llm_score, "llm_surfaced": llm_surfaced,
            "jev": jev,
        }
        if "score" in jev:
            jev_surfaced = jev["score"] >= threshold
            record["jev_surfaced"] = jev_surfaced
            record["agree"] = llm_surfaced == jev_surfaced
        records.append(record)
    _append_jev_shadow_log(records)


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

    # Shadow mode (see module docstring / _jev_shadow_classify): compute jev's own urgency
    # scoring alongside the LLM and log it next to the LLM's, purely for comparison. Best-effort and
    # non-blocking to the production decision below -- double-guarded against ever raising.
    try:
        _jev_shadow_classify(items, args.criteria, scores, args.threshold)
    # health: allow BLE001 -- Last-resort shadow boundary preserves production classification; tracebacks may expose protected items.
    except Exception as e:  # pragma: no cover - _jev_shadow_classify already swallows its own errors
        _eprint(f"classify_items: jev shadow classify raised unexpectedly (non-fatal): {type(e).__name__}")

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
