#!/usr/bin/env python3
"""Deterministic Nous Discord digest builder — clean output, no LLM.

Reads the diff context produced by pull_and_diff.py (via a JSON sidecar
written by digest-wrapper.sh) and emits:

  1. A dark-mode HTML report to
     ~/.hermes/runbooks/nous-archive/nous-digest-YYYYMMDD-HHMM.html
  2. A clean Discord message to stdout (top picks + context ideas +
     new threads + MEDIA: tag). The wrapper sends this verbatim.

Contract:
  - stdout IS the Discord message (no_agent cron: non-empty stdout is
    delivered, empty stdout = [SILENT]).
  - The LAST line is exactly `MEDIA:/abs/path.html` so the scheduler's
    attachment extractor picks it up.
  - No verification prose, no narration, no raw HTML in stdout.

Input JSON (sidecar, written by the wrapper from pull_and_diff.py stdout):
  {
    "run_time": "DD/MM/YYYY HH:MM",
    "new_messages": 1299,
    "since": "8d44ca2", "head": "00e34c2",
    "channels": {"hermes-agent": 1200, "developers": 50, ...},
    "new_threads": [{"forum": "community-projects-showcase",
                     "thread_id": "...", "title": "..."}],
    "messages": ["[ts] author (id=123) [link=https://...]\nbody", ...]
  }
"""
import html as _html
import json
import os
import re
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

GUILD = "1506021204363051249"
HTML_DIR = os.path.expanduser("~/.hermes/runbooks/nous-archive")
MAX_PICKS = 8
MAX_IDEAS = 10
MAX_MSG_CHARS = 2400  # keep the Discord message tight

TS_RE = re.compile(r"^\[(?P<ts>[^\]]+)\]\s+(?P<author>[^(]+?)\s*\(id=(?P<mid>\d+)\)(?P<link>\s*\[link=(?P<url>https?://\S+?)\])?(?P<body>.*)$", re.S)


def uk_time(iso_ts):
    try:
        from datetime import datetime as _dt
        return _dt.fromisoformat(iso_ts).astimezone(ZoneInfo("Europe/London")).strftime("%d/%m/%Y %H:%M")
    except Exception:
        return ""


def parse_messages(raw_msgs):
    out = []
    for block in raw_msgs:
        m = TS_RE.match(block)
        if not m:
            continue
        body = (m.group("body") or "").strip()
        if not body:
            continue
        out.append({
            "ts": m.group("ts"),
            "author": m.group("author").strip(),
            "mid": m.group("mid"),
            "link": m.group("url"),
            "body": body,
        })
    return out


def score(msg):
    """Heuristic signal score: discussion length + engagement + link density."""
    s = 0.0
    s += min(len(msg["body"]), 600) / 60.0            # substance
    s += 3 if msg["link"] else 0                       # citable
    words = len(msg["body"].split())
    if words >= 25:
        s += 4                                         # substantive, not one-liner
    if words >= 60:
        s += 3
    low = msg["body"].lower()
    for kw in ("shipped", "released", "launch", "v1.", "plugin", "release",
               "fix", "break", "bug", "leak", "cost", "token", "free"):
        if kw in low:
            s += 1
    return s


def first_line(body, n=140):
    first = body.split("\n")[0].strip()
    if len(first) > n:
        first = first[: n - 1] + "…"
    return first


def top_picks(msgs, new_threads):
    """Ranked picks: new threads first (freshest signal), then scored messages."""
    picks = []
    for t in new_threads:
        picks.append({
            "title": t["title"],
            "author": "new thread",
            "when": "",
            "link": f"https://discord.com/channels/{GUILD}/{t['thread_id']}",
            "blurb": f"New {t['forum']} thread",
        })
    for m in sorted(msgs, key=score, reverse=True):
        if len(picks) >= MAX_PICKS:
            break
        if m in [p for p in picks if p.get("_msg") is m]:
            continue
        picks.append({
            "title": first_line(m["body"], 110),
            "author": m["author"],
            "when": uk_time(m["ts"]),
            "link": m["link"] or "",
            "blurb": "",
            "_msg": m,
        })
    return picks[:MAX_PICKS]


def context_ideas(msgs, new_threads):
    """Deterministic context-idea cards: PURPOSE / VOICE / DRAFT per signal.

    These are signal-framed starter ideas (title + hook from the actual
    message), not finished tweets — Sahil refines voice before posting.
    """
    ideas = []
    used = set()
    # New threads are the strongest ideas.
    for t in new_threads[:4]:
        title = t["title"].replace("_", " ")
        ideas.append({
            "purpose": f"Signal the new {t['forum']} entry: {title}",
            "voice": "Direct · Community Signal · Just-Shipped hook",
            "draft": f"New on the Hermes Discord: \"{title}\". "
                     f"https://discord.com/channels/{GUILD}/{t['thread_id']}",
            "link": f"https://discord.com/channels/{GUILD}/{t['thread_id']}",
        })
    for m in sorted(msgs, key=score, reverse=True):
        if len(ideas) >= MAX_IDEAS:
            break
        key = m["mid"]
        if key in used:
            continue
        used.add(key)
        if m["mid"] in {p.get("link", "").rsplit("/", 1)[-1] for p in []}:
            continue
        hook = first_line(m["body"], 160)
        ideas.append({
            "purpose": f"Ride the live discussion: {m['author']} on {hook[:60]}",
            "voice": "Direct · AI Tools & Stack · Real-Number hook",
            "draft": hook + (f" — {m['link']}" if m["link"] else ""),
            "link": m["link"] or "",
        })
        if len(ideas) >= MAX_IDEAS:
            break
    return ideas[:MAX_IDEAS]


def render_html(run_time, counts, picks, ideas, threads):
    def esc(s):
        return _html.escape(str(s))

    def pick_card(p):
        link = p.get("link")
        link_html = (' · <a href="' + esc(link) + '">link</a>' if link else "")
        blurb = p.get("blurb")
        blurb_html = ("<div class=b>" + esc(blurb) + "</div>") if blurb else ""
        return (
            '<div class="card"><div class="t">' + esc(p["title"]) + '</div>'
            '<div class="m">' + esc(p["author"]) + " · " + esc(p.get("when", ""))
            + link_html + "</div>" + blurb_html + "</div>"
        )

    def idea_card(i, idea):
        link = idea.get("link")
        src = ('<div class="m"><a href="' + esc(link) + '">source</a></div>'
               if link else "")
        return (
            '<div class="card"><div class="t">Idea ' + str(i + 1)
            + " — " + esc(idea["purpose"]) + '</div>'
            '<div class="m">VOICE: ' + esc(idea["voice"]) + '</div>'
            '<pre class=draft>' + esc(idea["draft"]) + '</pre>'
            + src + "</div>"
        )

    def thread_card(t):
        return (
            '<div class="card"><div class="t">' + esc(t["title"]) + '</div>'
            '<div class="m">' + esc(t["forum"]) + " · "
            '<a href="https://discord.com/channels/' + GUILD + "/"
            + esc(t["thread_id"]) + '">thread</a></div></div>'
        )

    pick_cards = "\n".join(pick_card(p) for p in picks)
    idea_cards = "\n".join(idea_card(i, idea) for i, idea in enumerate(ideas))
    thread_cards = "\n".join(thread_card(t) for t in threads)
    if not thread_cards:
        thread_cards = ('<div class="card"><div class="m">No new threads '
                        "this window.</div></div>")
    counts_s = " · ".join(esc(k) + ": " + str(v) for k, v in sorted(counts.items()))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Nous Discord Digest — {esc(run_time)}</title>
<style>
html {{ color-scheme: dark; }}
body {{ background:#11100f; color:#f5f5f4; font-family: ui-sans-serif, system-ui, sans-serif; margin:0; padding:2rem; }}
h1 {{ color:#fbbf24; font-size:1.4rem; margin:0 0 .2rem; }}
h2 {{ color:#fbbf24; font-size:1.05rem; margin:1.6rem 0 .6rem; border-bottom:1px solid #34302c; padding-bottom:.3rem; }}
.muted {{ color:#a8a29e; font-size:.85rem; }}
.card {{ background:#1c1a18; border:1px solid #34302c; border-radius:8px; padding:.8rem 1rem; margin:.5rem 0; }}
.t {{ font-weight:600; }}
.m {{ color:#a8a29e; font-size:.85rem; margin-top:.25rem; }}
a {{ color:#fbbf24; }}
pre.draft {{ white-space:pre-wrap; background:#11100f; border:1px solid #34302c; border-radius:6px; padding:.6rem; font-size:.85rem; }}
</style>
</head>
<body>
<h1>Nous Discord Digest</h1>
<div class="muted">{esc(run_time)} · {esc(counts_s)}</div>
<h2>Top picks</h2>
{pick_cards}
<h2>Context ideas (X)</h2>
{idea_cards}
<h2>New threads</h2>
{thread_cards}
</body>
</html>
"""


def main():
    sidecar = sys.argv[1] if len(sys.argv) > 1 else "/tmp/nous-digest-input.json"
    with open(sidecar, encoding="utf-8") as f:
        data = json.load(f)

    run_time = data.get("run_time", "")
    counts = data.get("channels", {})
    msgs = parse_messages(data.get("messages", []))
    threads = data.get("new_threads", [])

    if not msgs and not threads:
        # Nothing to report — wrapper treats empty stdout as silent.
        return

    picks = top_picks(msgs, threads)
    ideas = context_ideas(msgs, threads)

    os.makedirs(HTML_DIR, exist_ok=True)
    stamp = datetime.now(ZoneInfo("Europe/London")).strftime("%Y%m%d-%H%M")
    html_path = os.path.join(HTML_DIR, f"nous-digest-{stamp}.html")
    html_doc = render_html(run_time, counts, picks, ideas, threads)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_doc)
    if not os.path.isfile(html_path) or os.path.getsize(html_path) == 0:
        sys.stderr.write("FAIL: HTML not written\n")
        sys.exit(1)

    # ---- Clean Discord message (this exact stdout is delivered) ----
    counts_s = " · ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
    header_lines = [
        f"🧵 Nous Discord Digest — {run_time}",
        f"{data.get('new_messages', len(msgs))} new messages · {counts_s}",
        "",
    ]
    pick_lines = ["Top picks:"]
    for i, p in enumerate(picks, 1):
        when = f" ({p['when']})" if p.get("when") else ""
        pick_lines.append(f"{i}. {p['title']} — {p['author']}{when} {p.get('link') or ''}".rstrip())
    thread_lines = []
    if threads:
        thread_lines = ["", "New threads:"]
        for t in threads:
            thread_lines.append(f"- {t['title']} https://discord.com/channels/{GUILD}/{t['thread_id']}")

    def idea_lines_fmt(_ideas, draft_cap=180):
        out = ["", "Context ideas (X):"]
        for i, idea in enumerate(_ideas, 1):
            out.append(f"{i}. PURPOSE: {idea['purpose']} | VOICE: {idea['voice']}")
            out.append(f"   DRAFT: {idea['draft'][:draft_cap]}")
        return out

    def assemble(_ideas, draft_cap=180):
        body = "\n".join(header_lines + pick_lines + thread_lines
                         + idea_lines_fmt(_ideas, draft_cap))
        return body + f"\nMEDIA:{html_path}"

    # Trim if over budget. Priority per digest-output-spec (v3):
    #   1. Cut picks to top 5 (never drop threads section, header, or MEDIA)
    #   2. Abbreviate DRAFTs to ~100 chars
    #   3. Only as absolute last resort drop ideas from the tail (keep >= 2)
    msg = assemble(ideas)
    if len(msg) > MAX_MSG_CHARS and len(pick_lines) > 6:
        pick_lines = pick_lines[:6]  # header line + top 5
        msg = assemble(ideas)
    if len(msg) > MAX_MSG_CHARS:
        msg = assemble(ideas, draft_cap=100)
    while len(msg) > MAX_MSG_CHARS and len(ideas) > 2:
        ideas = ideas[: len(ideas) - 1]
        msg = assemble(ideas, draft_cap=100)
    sys.stdout.write(msg)


if __name__ == "__main__":
    main()
