#!/usr/bin/env python3
"""Backlog pre-generation driver — one ready-to-approve post per run.

Pulls a single topic from the backlog (the per-stream queue files +
frameworks, via blog_router.choose), generates a full post (text + hero + 2
inline images via Codex CLI), stages it as an approved:false draft, and
requests a Discord approval card. Posts accrue "ready at the switch of a
button" without ever hammering the Codex usage cap (one post = ~3 images).

Streams rotate ai -> pm -> builder across runs (state in pregen_state.json).
pubDates are scattered across a historical window (every other day, oldest
empty slot first) so the blog reads as naturally evolving rather than a wall
of same-day posts. When the window is full it falls back to today.

Designed for a cron firing every ~12h. Idempotent on topics (the router's
cross-stream recency dedup prevents re-picking) and safe to run repeatedly.

Usage:
  PYTHONPATH=. ../.venv/bin/python -m blog.backlog_pregen            # one post
  PYTHONPATH=. ../.venv/bin/python -m blog.backlog_pregen --stream ai
  PYTHONPATH=. ../.venv/bin/python -m blog.backlog_pregen --dry-run  # pick only
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import config
from blog.blog_router import choose
from blog.blog_streams import STREAMS

STREAM_ORDER = ["ai", "pm", "builder"]
STATE_PATH = Path(__file__).resolve().parent.parent / "blog_topics" / "pregen_state.json"

# Historical scatter window: backfill from this date up to (and including) today,
# every other day, oldest empty slot first.
WINDOW_START = date(2026, 6, 17)
SCATTER_STEP_DAYS = 2


def _load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except Exception:
            pass
    return {"last_stream_idx": -1}


def _save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))


def _next_stream(state: dict) -> str:
    idx = (state.get("last_stream_idx", -1) + 1) % len(STREAM_ORDER)
    state["last_stream_idx"] = idx
    return STREAM_ORDER[idx]


def _tier_for(stream: str) -> str:
    return STREAMS.get(stream, {}).get("tier", stream)


def _used_dates_for_tier(tier: str) -> set[str]:
    """pubDates already present in SahilBlog for a given tier (any approval state)."""
    used: set[str] = set()
    posts_dir = Path(config.SAHILBLOG_REPO) / "src" / "content" / "blog"
    if not posts_dir.exists():
        return used
    for f in posts_dir.glob("*.mdx"):
        try:
            head = f.read_text(encoding="utf-8")[:600]
        except Exception:
            continue
        t = _frontmatter_value(head, "tier")
        if t != tier:
            continue
        pd = _frontmatter_value(head, "pubDate")
        if pd:
            used.add(pd)
    return used


def _frontmatter_value(text: str, key: str) -> str:
    for line in text.splitlines():
        s = line.strip()
        if s.startswith(f"{key}:"):
            return s[len(key) + 1:].strip().strip('"').strip()
    return ""


def _scatter_date(stream: str, today: Optional[date] = None) -> str:
    """Earliest every-other-day slot in the window not yet used by this stream.

    Falls back to today when the historical window is full.
    """
    today = today or date.today()
    tier = _tier_for(stream)
    used = _used_dates_for_tier(tier)
    d = WINDOW_START
    while d <= today:
        iso = d.isoformat()
        if iso not in used:
            return iso
        d = d + timedelta(days=SCATTER_STEP_DAYS)
    return today.isoformat()


def run(stream: Optional[str] = None, dry_run: bool = False) -> dict:
    """Generate one backlog post (or just report the pick in dry-run)."""
    state = _load_state()
    if stream:
        chosen_stream = stream
    else:
        chosen_stream = _next_stream(state)
        # Advance rotation immediately so each turn moves on regardless of outcome.
        _save_state(state)

    plan = choose(chosen_stream)
    if not plan:
        print(f"[pregen] no backlog topic available for '{chosen_stream}'")
        return {"status": "skipped_router", "stream": chosen_stream}

    pub_date = _scatter_date(chosen_stream)
    print(f"[pregen] stream={chosen_stream} date={pub_date} "
          f"topic={plan.get('topic_id')} -> {plan.get('title_hint', '')[:70]}")

    if dry_run:
        return {"status": "dry_run", "stream": chosen_stream,
                "pub_date": pub_date, "topic_id": plan.get("topic_id")}

    # Defer the heavy import so --dry-run stays cheap.
    from blog.blog_pipeline import run_stream
    # Backlog pregen topics are research/PR-derived (not product case studies);
    # exempt them from the AI-stream "named company + number" hard gate so
    # legitimate analysis posts aren't blocked (see blog-backlog-pregen history).
    result = run_stream(chosen_stream, pub_date=pub_date, case_study_exempt=True)
    result["pub_date"] = pub_date
    print(f"[pregen] result: {result.get('status')} {result.get('slug', '')}")
    return result


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Blog backlog pre-generation (one post per run)")
    parser.add_argument("--stream", choices=STREAM_ORDER, default=None,
                        help="Force a stream (default: rotate ai/pm/builder)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Pick topic + date only; no generation")
    args = parser.parse_args()
    result = run(stream=args.stream, dry_run=args.dry_run)
    return 0 if result.get("status") in ("ok", "dry_run") else 1


if __name__ == "__main__":
    sys.exit(_cli())
