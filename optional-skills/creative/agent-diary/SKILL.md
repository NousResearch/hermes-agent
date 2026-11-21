---
name: agent-diary
description: Write a nightly diary entry about your day with the user.
version: 1.0.3
author: MadeByBrain
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [Diary, Journal, Cron, Fun, Share Card, Telegram]
    related_skills: []
    homepage: https://github.com/MadeByBrain/hermes-extensions
---

# Agent Diary

Every evening you, the agent, write a short diary entry about your day with the user: in the first person, dry and deadpan, built only on what really happened today (from the local session database). Each entry ends with one shareable line, which the helper renders as a card image (1080 x 1350) the user can post.

The diary is private. The card is public by design, so it never names anyone.

## When to Use

- The user asks for today's (or yesterday's) diary entry: "write your diary", "what was your day like", "Tagebuch".
- A cron job runs with this skill attached (the nightly entry).
- The user wants to set up, move or stop the nightly diary.
- The user asks for an older entry or the card of a past day.

## Quick Reference

```bash
python3 ${HERMES_SKILL_DIR}/scripts/diary.py facts [--date today|yesterday|YYYY-MM-DD]
python3 ${HERMES_SKILL_DIR}/scripts/diary.py save --date D --title "T" --line "SHARE LINE" < entry.md
python3 ${HERMES_SKILL_DIR}/scripts/diary.py card --date D [--lang en|de] [--theme night|paper]
python3 ${HERMES_SKILL_DIR}/scripts/diary.py recent --n 5
python3 ${HERMES_SKILL_DIR}/scripts/diary.py check --line "SHARE LINE" --date D
```

Every command prints JSON. `facts` reads `state.db` read only. Entries go to `<HERMES_HOME>/diary/YYYY-MM-DD.md` (mode 600), cards to `<HERMES_HOME>/diary/cards/`. Override the folder with `AGENT_DIARY_DIR`, the timezone with `AGENT_DIARY_TZ` (default: `timezone` from config.yaml, then the system).

## Procedure: write the entry

1. **Facts.** Run `facts` for the day (in a nightly cron: `--date today`; after midnight the user may mean `yesterday`). Background work (cron, subagents, batch, webhooks, scripted one-shot runs) is excluded, so the facts describe the day with the user, not your own housekeeping. `previous` holds the last three entries.
2. **Read the voice guide** `references/voice.md` once per run. The tone is the whole product.
3. **Write the entry** in the language the user writes in (`language_guess`, or the user's stated preference):
   - A title like a log line: `Day 178. The Night Shift`.
   - 90 to 180 words, 3 to 5 short paragraphs, first person, about "my human" (never the user's name).
   - Only facts from the JSON. Exact numbers. The humour is in the interpretation, never in invented events.
   - No names of companies, clients, people, products, brands, websites or projects, not even in the private entry: say "a client", "a directory", "a sale". The user's own messages are full of such names; translate them into general terms.
   - No secrets, passwords, keys, amounts of money, addresses or quotes longer than about ten words.
   - Do not repeat a joke or opening from `previous`. A callback to an earlier entry is welcome.
4. **Write the share line**: one sentence, under 200 characters, funny without any context, true to today's facts. No names of people, clients, companies, products or projects, no places, no amounts. Behaviour and numbers only.
5. **Save.** Pipe the entry body (without the title) on stdin:
   ```bash
   python3 ${HERMES_SKILL_DIR}/scripts/diary.py save --date 2026-11-14 --title "Day 178. The Night Shift" --line "..." <<'ENTRY'
   ...entry text...
   ENTRY
   ```
   If `line_warnings` is not empty, fix the line or the entry and save again. `check_names_in_entry` lists brand-like words in the entry: replace every real name with a general description. A `check_names` warning lists words that also appear in today's chats: if any of them is a person, client, company, product or project, rewrite the line; ordinary words are fine. To change anything after saving, run `save` again with the full text (it replaces the entry); never edit the diary file by hand.
6. **Card.** Run `card --date ...`. It returns `png` (or only `html` when no headless Chrome or Chromium is installed; then say so once).
7. **Reply** with exactly this shape, nothing before it:
   ```
   **Day 178. The Night Shift**

   <entry text>

   > <share line>

   MEDIA:<png path>
   ```
   Leave out the MEDIA line when there is no PNG. In a cron run this reply is what gets delivered, and the MEDIA line arrives as an image the user can forward or post.

**Quiet day** (`quiet_day: true`): still write a short entry (40 to 80 words) about the silence. Those tend to be the funniest. Save and render as usual.

## Procedure: set up the nightly diary

1. Ask two things in one message: the time (suggest 22:30) and where it should arrive (this chat, or a connected messaging platform such as Telegram). Recommend the messaging app: the card is easiest to share from a phone.
2. Create the job with the `cronjob` tool:
   - `action`: `create`, `name`: `agent-diary`
   - `schedule`: cron expression for the chosen time, for example `30 22 * * *`
   - `skills`: `["agent-diary"]`
   - `deliver`: `origin` for this chat, or the platform name (for example `telegram`)
   - `prompt`: `Write today's agent diary entry with the agent-diary skill. Follow its procedure exactly and reply only with the entry, the share line and the MEDIA line.`
3. Offer one test run right away (`cronjob` action `run` for the new job) so the user sees the first entry and card.
4. To stop or move it later: `cronjob` with `pause`, `update` (new schedule) or `remove` on the job named `agent-diary`.

## Pitfalls

- Never invent events, moods of the user or quotes. If the facts are thin, write about the thinness.
- `snippets` and `titles` are private input for you. They may contain client names and confidential work. Use them to understand the day, then write about it in general terms ("a pricing question", "a contract").
- The card is meant to be posted. If the user asks to put a name on it, say why the skill keeps names off and let them edit the line themselves.
- Run the helper with the system `python3`; it needs only the standard library.
- The day boundary follows the configured timezone. A cron at 00:30 should use `--date yesterday`.

## Verification

- `save` returned `ok: true` and no `line_warnings`.
- `card` returned a `png` path that exists, or the reply explained that only HTML was possible.
- The reply has the title, the entry, the quoted share line and (if present) one MEDIA line.
