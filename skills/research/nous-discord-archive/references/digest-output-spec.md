# Nous Digest — Output Specification (v3)

The nous-archive-digest cron (id 53921a891657, every 6h at 04/10/16/22 UK) delivers to
Discord #hermes-discord (channel 1536199395006095420). This spec defines EXACTLY what
the digest must look like. Follow it precisely.

## Input (injected stdout from pull_and_diff.py)

- `NO_NEW_MESSAGES` / `FIRST_RUN` → reply `[SILENT]`, no message, no file.
- Otherwise: RUN_TIME, NEW_MESSAGES, CHANNELS (grouped counts), FULL_DIFF_AT,
  GUILD_ID, NEW_THREADS block (forum threads with links), bounded digest of new
  messages oldest-first, each with `[link=...]` Discord URL.

## Discord message format (plain text, NO raw HTML)

```
🧵 Nous Discord Digest — <RUN_TIME from input>
N new messages · <channel counts>

Top picks:
1. <title> — <author>, <channel> (<DD/MM/YYYY HH:MM>) <link>
2. ...

Content ideas (X) — copy-paste ready:
1. PURPOSE: <one line: what this post does / why it lands>
   VOICE: <register · pillar · hook style>
   DRAFT: <the actual tweet text, ready to paste — near-max chars for single tweets>

New in community-projects-showcase:
- <title> <link> — <1-line>

New in plugins-skills-and-skins:
- <title> <link> — <1-line>

MEDIA:/home/kensei/.hermes/runbooks/nous-archive/nous-digest-YYYYMMDD-HHMM.html
```

Use EXACTLY the directory `~/.hermes/runbooks/nous-archive/` — it already exists.
Do NOT invent a new directory (e.g. nous-discord-digest/).

Rules:
- Use RUN_TIME from input for the header timestamp. Never invent one.
- The visible Discord message MUST carry the content inline — it is NOT just a
  teaser. Always include: top picks (3-8, one line each with link), ALL content
  ideas (each with PURPOSE line + VOICE tag + DRAFT), and the new-threads
  sections. The HTML attachment is the formatted version of the same content,
  not the only place the content lives.
- Compact format for the message: idea = "PURPOSE: ... | VOICE: ... | DRAFT: ..."
  on 2-3 lines each. If the message would exceed ~2500 chars, cut to top 5
  picks + all ideas still visible (DRAFT may be abbreviated to first ~100 chars
  with "…" if absolutely needed). Do NOT drop the ideas from the message.
- No raw HTML, no process narration, no memory/prompt leakage in the message.
- Write the HTML file FIRST (verify with `test -f`), then output the MEDIA tag.

## Content ideas — copy-paste ready (REQUIRED)

Generate up to 10 X/Twitter content ideas, primarily from hermes-agent traffic
(use plugins/skills/showcase only if hermes-agent is thin). Each idea MUST have
three parts, exactly:

1. **PURPOSE** — one line explaining the intent of the post (e.g. "position the
   Hermes plugin ecosystem as the reason to run a local agent", "practical
   anti-hype take on memory plugins backed by community reports").
2. **VOICE** — which of Sahil's X registers + pillar + hook style, e.g.
   `Direct · AI Tools & Stack · Just-Shipped hook` or `Wry · Wry Observation ·
   Triple-Punch`. See the sahil-twitter-voice skill for the full taxonomy.
3. **DRAFT** — the actual copy-paste tweet text. Single tweet = near-max chars
   (240-259). Short threads (3-5 tweets) only for genuinely complex topics, max
   2 threads per digest. No hashtags beyond 1-2. Never engagement bait, never
   "thread incoming" bait, no AI-slop template structure.

Every idea needs the source link too (the Discord [link=...] it came from), so
Sahil can verify the signal.

Voice rules (from sahil-twitter-voice):
- ~60-65% Direct register, ~20% Wry, ~15-20% Honest Debrief. Never two Honest
  Debrief in a row.
- Banned: hustle-bro, thread bait, engagement bait, generic AI hype
  ("AI will change everything"), tech-bro absolutism ("X is dead"), lad bantz.
- No @-mentions unless directly relevant. Max 2 hashtags.
- Drafts must pass the "banger" quality bar: specific, grounded in the actual
  community signal, with real numbers or real mechanics. No invented stats.

## HTML report (dark mode, mandatory)

Write to `~/.hermes/runbooks/nous-archive/nous-digest-YYYYMMDD-HHMM.html`.
- `color-scheme: dark`; body `#11100f`; cards `#1c1a18`; text `#f5f5f4`;
  muted `#a8a29e`; accent `#fbbf24`; borders `#34302c`. No light backgrounds.
- Sections: Highlights (title, author, channel, UK time, link) · Content ideas
  (up to 10, each with PURPOSE/VOICE/DRAFT + source link) · New in
  community-projects-showcase · New in plugins-skills-and-skins.
- NO full message-by-message log section anywhere.
- Timestamps UK format DD/MM/YYYY HH:MM (archive is ISO UTC; Europe/London).

## Forbidden

- No verification scripts appended to output.
- No `[SILENT]` followed by content.
- No light-mode HTML, no raw HTML tags in the Discord message.
- No full message dump in the HTML.
- No kanban task creation. Output-only job.
