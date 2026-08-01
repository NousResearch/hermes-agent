#!/usr/bin/env python3
"""Mashup pipeline retention: archive stale proposals + blog queue entries.

- Proposals: move mashup-*.html older than PROPOSAL_DAYS (default 30) to
  <proposals>/archive/ (keep state files).
- Blog queue: for each stream in BLOG_STREAMS, split entries older than
  BLOG_DAYS (default 90) into <stream>.archive.jsonl and rewrite the active
  file with fresh entries only.
- Idempotent, safe: only moves/rewrites, never deletes. Exit 0 always.

Usage:
  python3 mashup_retention.py [--proposal-days 30] [--blog-days 90]
"""
import argparse
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

HERMES_HOME = Path(__import__("os").environ.get("HERMES_HOME", str(Path.home() / ".hermes")))
PROPOSALS_DIR = HERMES_HOME / "runbooks" / "proposals"
BLOG_DIR = Path(__import__("os").environ.get(
    "MASHUP_BLOG_DIR",
    "/home/kensei/repos/KenseiAgent/content_engine/blog_topics",
))
BLOG_STREAMS = ["ai.jsonl", "builder.jsonl", "pm.jsonl", "frameworks.jsonl"]


def parse_ts(entry: dict) -> datetime | None:
    """Best-effort timestamp from a blog queue entry."""
    raw = entry.get("created_at") or entry.get("timestamp") or entry.get("date")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def archive_proposals(cutoff: datetime) -> int:
    if not PROPOSALS_DIR.exists():
        return 0
    archive = PROPOSALS_DIR / "archive"
    archive.mkdir(parents=True, exist_ok=True)
    moved = 0
    for f in PROPOSALS_DIR.glob("mashup-*.html"):
        try:
            mtime = datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc)
        except OSError:
            continue
        if mtime < cutoff:
            f.replace(archive / f.name)
            moved += 1
    return moved


def archive_blog_stream(path: Path, cutoff: datetime) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    # Operate on the resolved real file; never replace a symlink with a file.
    path = path.resolve()
    archive_path = path.with_name(path.name.replace(".jsonl", ".archive.jsonl"))
    original = path.read_text(encoding="utf-8", errors="replace")
    fresh, stale = [], []
    for line in original.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            stale.append(line)  # malformed -> archive, never lose
            continue
        ts = parse_ts(entry)
        if ts is not None and ts < cutoff:
            stale.append(line)
        else:
            fresh.append(line)
    if stale:
        with archive_path.open("a", encoding="utf-8") as f:
            f.write("\n".join(stale) + "\n")
    if fresh != original.splitlines():
        # rewrite active file with fresh lines only (atomic-ish: tmp + replace)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text("\n".join(fresh) + ("\n" if fresh else ""), encoding="utf-8")
        tmp.replace(path)
    return len(fresh), len(stale)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--proposal-days", type=int, default=30)
    parser.add_argument("--blog-days", type=int, default=90)
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    prop_cutoff = now - timedelta(days=args.proposal_days)
    blog_cutoff = now - timedelta(days=args.blog_days)

    moved_props = archive_proposals(prop_cutoff)
    print(f"[retention] proposals archived: {moved_props}")

    total_fresh = total_stale = 0
    for stream in BLOG_STREAMS:
        fresh, stale = archive_blog_stream(BLOG_DIR / stream, blog_cutoff)
        total_fresh += fresh
        total_stale += stale
        print(f"[retention] {stream}: {fresh} fresh, {stale} archived")
    print(f"[retention] done — {total_fresh} fresh, {total_stale} archived across {len(BLOG_STREAMS)} streams")
    return 0


if __name__ == "__main__":
    sys.exit(main())
