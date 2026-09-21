#!/usr/bin/env python3
"""Backfill blog posts into historical date gaps (18-27 JUN 2026).

Picks topics from frameworks queue → AI queue → PM queue, one per stream per
date, cycling through all available topics without overlap. PubDates are set
to fill the gap days chronologically.

Usage:
  set -a && . ~/.hermes/.env && set +a && PYTHONPATH=. python3 scripts/backfill.py
"""
import sys, os, json
from datetime import date, timedelta
from pathlib import Path

# Ensure we can import blog modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "content_engine"))

from blog.blog_router import choose, record
from blog.blog_generator import write_with_gate
from blog.blog_assembler import assemble

GAP_START = date(2026, 6, 18)
GAP_END = date(2026, 6, 27)
SAHILBLOG = "/home/kensei/repos/SahilBlog"

stream_order = ["ai", "pm", "builder"]
results = {"ok": 0, "failed": 0, "skipped": 0}

d = GAP_START
while d <= GAP_END:
    ds = d.isoformat()
    print(f"\n{'='*60}")
    print(f"DATE: {ds}")
    print(f"{'='*60}")

    for stream in stream_order:
        plan = choose(stream)
        if not plan:
            print(f"  [{stream}] No topic available, skipping")
            results["skipped"] += 1
            continue

        topic = plan.get("title_hint", "?")
        print(f"  [{stream}] Topic: {topic}")

        draft = write_with_gate(plan, stream=stream, strict_review=False)
        if not draft:
            print(f"  [{stream}] Generator failed")
            results["failed"] += 1
            continue

        body = draft.get("body_md", "") or ""
        if len(body.strip()) < 100:
            print(f"  [{stream}] Body too short ({len(body.strip())} chars)")
            results["failed"] += 1
            continue

        try:
            # Clear the used topics so the next stream picks something different
            mdx = assemble(draft, images={}, repo=SAHILBLOG, pub_date=ds)
            print(f"  [{stream}] Written: {mdx.name} ({len(body)} chars)")
            results["ok"] += 1
        except ValueError as e:
            print(f"  [{stream}] Assembly failed: {e}")
            results["failed"] += 1
            continue

    d += timedelta(days=1)

print(f"\n{'='*60}")
print(f"BACKFILL COMPLETE")
print(f"  OK:      {results['ok']}")
print(f"  Failed:  {results['failed']}")
print(f"  Skipped: {results['skipped']}")
print(f"{'='*60}")
