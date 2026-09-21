"""Manually run the approval-requester logic and post results.

Same logic as the cron, but outputs the full message we want to send.
"""
import json
import sys
from pathlib import Path

TRACKER = Path("/home/kensei/repos/KenseiAgent/content_engine/blog_topics/pending_approvals.jsonl")
CHANNEL = "discord:#blog-management"


def main():
    if not TRACKER.exists():
        print("[SILENT]")
        return
    pending = []
    for line in TRACKER.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("status") == "pending":
            pending.append(e)
    if not pending:
        print("[SILENT]")
        return
    # Print one message per entry (the cron emits one big batch; the
    # gateway's `extract_media` should still parse MEDIA: lines in a
    # single multi-paragraph response)
    out = [f"**Blog Approval Batch — {len(pending)} pending post(s)**", ""]
    for e in pending:
        out.append(f"**{e['title']}**")
        out.append(f"  stream: `{e.get('stream','?')}` | tier: `{e.get('tier','?')}` | slug: `{e.get('slug','?')}`")
        out.append(f"  file: `{e.get('mdx_path','?')}`")
        preview = e.get("preview_path", "")
        if preview and Path(preview).exists():
            out.append(f"MEDIA:{preview}")
        out.append(f"  → `!approve {e['slug']}` | `!reject {e['slug']} [reason]` | `!amend {e['slug']} [notes]`")
        out.append("")
    print("\n".join(out).rstrip())


if __name__ == "__main__":
    main()
