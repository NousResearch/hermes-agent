#!/usr/bin/env python3
"""Pull the Nous Discord archive, reindex incrementally, and emit a rich digest
of new messages since the last processed commit.

State: ~/.hermes/nous-archive/last_sha  (last processed upstream commit)
Output to stdout (bounded, for the cron agent):
  - RUN_TIME, NEW_MESSAGES, CHANNELS
  - per-message entries WITH channel/thread attribution and Discord links
  - NEW_THREADS: brand-new forum threads (community-projects-showcase / plugins-skills-and-skins)
Exit 0 even on empty (delivers nothing), non-zero on real failure.
"""
import os
import re
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

REPO = os.path.expanduser("~/repos/nous-discord-archive")
STATE = os.path.expanduser("~/.hermes/nous-archive/last_sha")
DIFF_OUT = os.path.expanduser("~/.hermes/nous-archive/last_diff.txt")
MSG_HEADER_RE = re.compile(r"^\[([^\]]+)\] (.+) \(id=(\d+)\)$")
THREAD_FILENAME_RE = re.compile(r"^(\d+)-(.+)\.txt$")
MAX_PRINT_CHARS = 20000  # bounded digest for agent context

GUILD = "1506021204363051249"
# Top-level channel files -> discord channel id
CHANNEL_IDS = {
    "hermes-agent.txt": "1476316988992258300",
    "developers.txt": "1491258766648283238",
}
# Forum dirs -> forum id (thread ids come from filenames)
FORUM_IDS = {
    "plugins-skills-and-skins": "1485392832154832906",
    "community-projects-showcase": "1316137596535177246",
}


def sh(cmd, **kw):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        raise RuntimeError(f"cmd failed ({r.returncode}): {cmd}\n{r.stderr[-2000:]}")
    return r.stdout


def uk_now():
    return datetime.now(ZoneInfo("Europe/London")).strftime("%d/%m/%Y %H:%M")


def link_for(rel_path, msg_id):
    """Build a clickable Discord link for a message in a given archive file."""
    if rel_path.startswith("archives/"):
        rel_path = rel_path[len("archives/"):]
    parts = rel_path.split("/")
    if len(parts) == 1:  # top-level channel file
        ch = CHANNEL_IDS.get(parts[0])
        if not ch:
            return None
        return f"https://discord.com/channels/{GUILD}/{ch}/{msg_id}"
    # forum thread: <dir>/<threadid>-<title>.txt
    dirname = parts[0]
    m = THREAD_FILENAME_RE.match(parts[1])
    if m and dirname in FORUM_IDS:
        thread_id = m.group(1)
        return f"https://discord.com/channels/{GUILD}/{thread_id}/{msg_id}"
    return None


def thread_title_from_path(rel_path):
    if rel_path.startswith("archives/"):
        rel_path = rel_path[len("archives/"):]
    parts = rel_path.split("/")
    if len(parts) == 2:
        m = THREAD_FILENAME_RE.match(parts[1])
        if m:
            return parts[0], m.group(1), m.group(2).replace("-", " ")
    return None, None, None


def channel_group(path):
    """Map an archive path to a friendly channel group name."""
    if path.startswith("archives/"):
        path = path[len("archives/"):]
    parts = path.split("/")
    if len(parts) == 1:
        return parts[0][:-4]  # strip .txt
    return parts[0]  # forum dir name


def main():
    if not os.path.isdir(REPO):
        print("FATAL: repo missing. Run: git clone https://github.com/teknium1/nous-discord-archive " + REPO)
        sys.exit(1)

    sh(f"git -C {REPO} pull --quiet")

    last = ""
    if os.path.exists(STATE):
        last = open(STATE).read().strip()

    head = sh(f"git -C {REPO} rev-parse HEAD").strip()
    if last == head:
        print("NO_NEW_MESSAGES (upstream unchanged since last run)")
        return 0
    if not last:
        print(f"FIRST_RUN baseline at {head[:12]} (no diff yet — index built from full history)")
        open(STATE, "w").write(head)
        sh(f"python3 {os.path.expanduser('~/.hermes/nous-archive/index.py')}")
        return 0

    sh(f"python3 {os.path.expanduser('~/.hermes/nous-archive/index.py')}")

    diff_text = sh(f"git -C {REPO} diff {last}..{head} -- archives/ -- '*.txt'")

    messages = []
    new_threads = []  # (forum, thread_id, title)
    channel_counts = {}
    cur = None
    cur_path = None
    in_new_file = False
    for raw in diff_text.splitlines():
        if raw.startswith("diff --git"):
            mm = re.match(r"diff --git a/(\S+) b/(\S+)", raw)
            if mm:
                cur_path = mm.group(2)
            in_new_file = False
            continue
        if raw.startswith("new file mode"):
            in_new_file = True
            continue
        if in_new_file and raw.startswith("+") and not raw.startswith("+++"):
            line = raw[1:]
            m = MSG_HEADER_RE.match(line)
            if m and cur_path:
                forum, tid, title = thread_title_from_path(cur_path)
                if forum and tid:
                    new_threads.append((forum, tid, title))
                    in_new_file = False  # one thread entry per new file
        if raw.startswith("+") and not raw.startswith("+++"):
            line = raw[1:]
            m = MSG_HEADER_RE.match(line)
            if m:
                cur = {
                    "ts": m.group(1), "author": m.group(2), "id": m.group(3),
                    "lines": [], "path": cur_path,
                }
                messages.append(cur)
                chan_key = channel_group(cur_path) if cur_path else "?"
                channel_counts[chan_key] = channel_counts.get(chan_key, 0) + 1
            elif cur is not None and line.strip():
                cur["lines"].append(line[4:] if line.startswith("    ") else line)

    full = []
    for msg in messages:
        text = "\n".join(msg["lines"]).strip()
        if text:
            link = link_for(msg["path"], msg["id"]) if msg["path"] else None
            link_suffix = f" [link={link}]" if link else ""
            full.append(f"[{msg['ts']}] {msg['author']} (id={msg['id']}){link_suffix}\n{text}")
    with open(DIFF_OUT, "w") as f:
        f.write("\n\n".join(full))

    open(STATE, "w").write(head)

    if not messages and not new_threads:
        print("NO_NEW_MESSAGES (diff empty)")
        return 0

    # Per-channel counts (friendly: channel group, not thread filename)
    chan_summary = ", ".join(f"{k}: {v}" for k, v in sorted(channel_counts.items()))
    print(f"RUN_TIME: {uk_now()}")
    print(f"NEW_MESSAGES: {len(messages)} since {last[:12]} -> {head[:12]}")
    print(f"CHANNELS: {chan_summary}")
    print(f"FULL_DIFF_AT: {DIFF_OUT}")
    print(f"GUILD_ID: {GUILD}")

    # NEW THREADS section — brand-new forum threads (showcase + plugins)
    if new_threads:
        print("NEW_THREADS:")
        for forum, tid, title in new_threads:
            thread_link = f"https://discord.com/channels/{GUILD}/{tid}"
            print(f"  [{forum}] {title} (thread_id={tid}) link={thread_link}")

    print("--- digest (oldest first, capped) ---")
    body = "\n\n".join(full)
    if len(body) > MAX_PRINT_CHARS:
        body = body[:MAX_PRINT_CHARS] + "\n...[truncated, see FULL_DIFF_AT]"
    print(body)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
