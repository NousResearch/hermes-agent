#!/usr/bin/env python3
"""approval-requester script — no-agent version that posts compact batch
summary with MEDIA: attachments to #blog-management for each pending draft.

Reads blog_topics/pending_approvals.jsonl, groups pending drafts into a single
Discord delivery, and marks each entry's discord_message_id after delivery.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path("/home/kensei/repos/KenseiAgent/content_engine")
TRACKER = ROOT / "blog_topics" / "pending_approvals.jsonl"

CHANNEL = "discord:#blog-management"


def load_pending():
    if not TRACKER.exists():
        return []
    out = []
    for line in TRACKER.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("status") == "pending":
            out.append(e)
    return out


def build_message(entries):
    """Build a single batch summary with MEDIA: for each preview.

    Format:
      - One header line: "Blog Approval Batch (N pending):"
      - One block per entry: title, slug, tier, MEDIA:path
    """
    if not entries:
        return "[SILENT]"
    lines = [f"**Blog Approval Batch — {len(entries)} pending post(s)**", ""]
    for e in entries:
        title = e.get("title", "Untitled")
        slug = e.get("slug", "?")
        stream = e.get("stream", "?")
        tier = e.get("tier", "?")
        preview = e.get("preview_path", "")
        mdx = e.get("mdx_path", "")
        lines.append(f"**{title}**")
        lines.append(f"  stream: `{stream}` | tier: `{tier}` | slug: `{slug}`")
        lines.append(f"  file: `{mdx}`")
        if preview and Path(preview).exists():
            size_kb = Path(preview).stat().st_size // 1024
            if size_kb < 8 * 1024:  # Discord attachment cap
                lines.append(f"MEDIA:{preview}")
            else:
                lines.append(f"  (preview too large: {size_kb}KB)")
        else:
            lines.append(f"  (preview missing: {preview})")
        lines.append("  → reply: `!approve <slug>` | `!reject <slug> [reason]` | `!amend <slug> [notes]`")
        lines.append("")
    return "\n".join(lines).rstrip()


def main():
    pending = load_pending()
    msg = build_message(pending)
    print(msg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
