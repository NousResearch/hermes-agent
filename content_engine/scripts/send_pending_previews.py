"""Send pending blog previews directly to #blog-management via Discord API.

Bypasses the cron for one-time immediate delivery while the cron prompt is
being tuned. Reads pending_approvals.jsonl, posts one Discord message per
pending post with the HTML preview as a file attachment.
"""
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

CHANNEL_ID = "1521172577031028888"  # #blog-management
TRACKER = Path("/home/kensei/repos/KenseiAgent/content_engine/blog_topics/pending_approvals.jsonl")
ENV_PATH = Path("/home/kensei/.hermes/.env")


def get_token() -> str:
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            line = line.strip()
            if line.startswith("DISCORD_BOT_TOKEN="):
                return line[len("DISCORD_BOT_TOKEN="):].strip().strip('"').strip("'")
    return ""


def discord_send(channel_id: str, content: str, attachment_path: str) -> tuple[bool, str]:
    """Send a message with a file attachment to a Discord channel."""
    import base64
    token = get_token()
    if not token:
        return False, "no token"
    url = f"https://discord.com/api/v10/channels/{channel_id}/messages"
    body = {
        "content": content,
        "attachments": [{
            "id": "0",
            "filename": Path(attachment_path).name,
        }],
    }
    boundary = "----hermesboundary"
    parts = []
    parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"payload_json\"\r\n\r\n")
    parts.append(json.dumps(body))
    parts.append(f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"files[0]\"; filename=\"{Path(attachment_path).name}\"\r\nContent-Type: text/html\r\n\r\n")
    parts.append(Path(attachment_path).read_bytes().decode("utf-8", errors="replace"))
    parts.append(f"\r\n--{boundary}--\r\n")
    data = "".join(parts).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bot {token}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return True, f"HTTP {r.status}"
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}: {e.read().decode('utf-8', errors='replace')[:200]}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def main():
    if not TRACKER.exists():
        print("No tracker file")
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
        print("[SILENT] - no pending")
        return
    print(f"Sending {len(pending)} posts to channel {CHANNEL_ID}")
    sent = 0
    for e in pending:
        title = e.get("title", "Untitled")
        slug = e.get("slug", "?")
        stream = e.get("stream", "?")
        tier = e.get("tier", "?")
        mdx = e.get("mdx_path", "?")
        preview = e.get("preview_path", "")
        if not preview or not Path(preview).exists():
            print(f"  SKIP {slug} - no preview")
            continue
        content = (
            f"**{title}**\n"
            f"  stream: `{stream}` | tier: `{tier}` | slug: `{slug}`\n"
            f"  file: `{mdx}`\n"
            f"  → `!approve {slug}` | `!reject {slug} [reason]` | `!amend {slug} [notes]`"
        )
        ok, msg = discord_send(CHANNEL_ID, content, preview)
        status = "✓" if ok else "✗"
        print(f"  {status} {slug} - {msg}")
        if ok:
            sent += 1
    print(f"\nSent {sent}/{len(pending)}")


if __name__ == "__main__":
    main()
