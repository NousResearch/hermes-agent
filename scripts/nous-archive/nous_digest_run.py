#!/usr/bin/env python3
"""Nous Discord digest — no_agent cron entrypoint.

Chain: pull_and_diff.py (pull + index + diff, advances last_sha)
     -> parse its stdout into a JSON sidecar
     -> build_digest.py (dark HTML + clean Discord message on stdout)

stdout contract (no_agent cron):
  - empty stdout            -> [SILENT] (nothing new / first run)
  - non-empty stdout        -> delivered verbatim to the target channel
  - last line is MEDIA:<abs html path> for the native attachment

On pull failure a ONE-LINE alert is emitted (delivered, so failures are
visible) and the run exits 0 — the cron framework treats non-zero as an
error alert, which would double-post.
"""
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PULL = os.path.join(HERE, "pull_and_diff.py")
BUILDER = os.path.join(HERE, "build_digest.py")

MSG_HEADER = re.compile(r"^\[(?P<ts>[^\]]+)\] (?P<author>[^(]+?) \(id=(?P<mid>\d+)\)(?P<rest>.*)$", re.S)
THREAD_LINE = re.compile(r"^\s+\[(?P<forum>[^\]]+)\] (?P<title>.+?) \(thread_id=(?P<tid>\d+)\) link=(?P<url>\S+)$")


def fail_alert(reason):
    sys.stdout.write("⚠️ Nous digest: " + reason[:300].replace("\n", " "))
    sys.stdout.flush()


def parse_pull_output(out):
    """Parse pull_and_diff.py stdout into the builder's JSON sidecar shape."""
    lines = out.splitlines()
    run_time, new_msgs, since_head = "", "", ""
    channels = {}
    threads = []
    in_threads = False
    digest_start = None
    for i, ln in enumerate(lines):
        if ln.startswith("RUN_TIME: "):
            run_time = ln[len("RUN_TIME: "):].strip()
        elif ln.startswith("NEW_MESSAGES: "):
            m = re.match(r"NEW_MESSAGES: (\d+) since (\S+) -> (\S+)", ln)
            if m:
                new_msgs, since_head = m.group(1), m.group(2) + " -> " + m.group(3)
        elif ln.startswith("CHANNELS: "):
            for part in ln[len("CHANNELS: "):].split(", "):
                if ":" in part:
                    k, v = part.rsplit(": ", 1)
                    try:
                        channels[k.strip()] = int(v)
                    except ValueError:
                        pass
        elif ln.startswith("NEW_THREADS:"):
            in_threads = True
            continue
        elif in_threads:
            tm = THREAD_LINE.match(ln)
            if tm:
                threads.append({"forum": tm.group("forum"),
                                "thread_id": tm.group("tid"),
                                "title": re.sub(r"\s+", " ", tm.group("title")).strip()})
            elif ln.strip() == "":
                in_threads = False
        if ln.startswith("--- digest (oldest first"):
            digest_start = i + 1
            break

    raw_blocks = []
    if digest_start is not None:
        body = "\n".join(lines[digest_start:])
        if body.endswith("\n...[truncated, see FULL_DIFF_AT]"):
            body = body[: -len("\n...[truncated, see FULL_DIFF_AT]")]
        raw_blocks = [b.strip() for b in body.split("\n\n") if b.strip()]

    messages = []
    for block in raw_blocks:
        m = MSG_HEADER.match(block)
        if not m:
            continue
        rest = m.group("rest") or ""
        link_m = re.match(r"\s*\[link=(https?://\S+?)\]", rest)
        link = link_m.group(1) if link_m else None
        body_text = rest[link_m.end():].strip() if link_m else rest.strip()
        if not body_text:
            continue
        messages.append("[{}] {} (id={}){}\n{}".format(
            m.group("ts"), m.group("author").strip(), m.group("mid"),
            (" [link=%s]" % link) if link else "", body_text))

    return {
        "run_time": run_time,
        "new_messages": int(new_msgs) if new_msgs.isdigit() else len(messages),
        "since": since_head,
        "channels": channels,
        "new_threads": threads,
        "messages": messages,
    }


def main():
    try:
        r = subprocess.run(
            [sys.executable, PULL],
            capture_output=True, text=True, timeout=900,
        )
    except Exception as e:  # noqa: BLE001
        fail_alert("pull crashed: %s" % e)
        return 0
    out = (r.stdout or "").strip()
    err = (r.stderr or "").strip()

    if r.returncode != 0 or out.startswith("ERROR"):
        fail_alert("pull_and_diff failed (rc=%s): %s" % (r.returncode, (err or out)[:300]))
        return 0
    if out.startswith("NO_NEW_MESSAGES") or out.startswith("FIRST_RUN"):
        return 0  # silent

    sidecar = os.path.join(HERE, "last_digest_input.json")
    data = parse_pull_output(out)
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    b = subprocess.run([sys.executable, BUILDER, sidecar],
                       capture_output=True, text=True, timeout=120)
    if b.returncode != 0:
        fail_alert("builder failed: %s" % (b.stderr.strip()[:300] or "unknown"))
        return 0
    # Pass through the clean message verbatim (its last line is MEDIA:...).
    sys.stdout.write(b.stdout)
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
