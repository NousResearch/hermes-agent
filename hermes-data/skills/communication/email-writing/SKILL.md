---
name: email-writing
description: |
  THE single authority for how every email body is written and formatted before it
  becomes a Gmail draft — new mail, threaded reply, forward, escalation, cron/job emails,
  Telegram-dictated emails. Enforces rich formatting (headings, bold, bullets, numbered
  lists, tables) in an HTML part and forbids hard-wrapped "poem format" plain text.
  Build with scripts/email_draft.py (markdown -> multipart draft). Load this BEFORE
  email-drafter / messaging-drafts / email whenever the task is "draft an email".
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [email, drafting, formatting, gmail, html, communication]
    related_skills: [email-drafter, messaging-drafts, email]
---

# Email Writing — one standard for every email

**This skill governs the WORDS and the FORMAT of every email. It is not optional and it
applies equally in Telegram sessions, cron jobs, and api_server runs.** Other email skills
(`email-drafter`, `messaging-drafts`, `email`) own Gmail mechanics — threading, recipients,
attachments, API pitfalls. If they disagree with this file on presentation, this file wins.

## 0. The failure this exists to kill

NDR, verbatim: *"the drafting of the email is terrible… it keeps wrapping after every fourth
or fifth word. It looks horrible, it's not presentable, not readable."* And earlier: *"one
continuous running text… not very visually easy to read."*

Two distinct bugs, both banned:

1. **Hard-wrapped plain text** — a `\n` every 55–80 characters so a paragraph renders as a
   tall narrow ribbon that uses a third of the window width. Cause: the body was written as a
   pre-formatted string (terminal habit) instead of flowing text. **A paragraph is ONE line in
   the source. The client wraps it.**
2. **Structureless prose** — everything in paragraphs, no headings, no bullets, no numbers, so
   a five-point email has to be read twice to be understood.

## 1. The formatting contract (hard rules)

| Rule | Detail |
|---|---|
| **Never hard-wrap** | No `\n` inside a paragraph, ever. Blank line = new paragraph only. |
| **Always send both parts** | `multipart/alternative` with `text/plain` (unwrapped) + `text/html` (formatted). Default, even when NDR never said "HTML". |
| **Structure every multi-point email** | 3+ parallel items → bullets. 3+ sequenced asks → numbered list. 2+ distinct topics → `##` section headings. Figures/dates/owners → table. |
| **Bold carries the scan** | Bold the actor + the ask (`**Hero film** — v4 not approved`), not whole sentences. |
| **No decoration** | No colours, no logos, no emoji, no marketing banners unless he asks for a teaser/brochure (`dra-brand-html-teaser` covers that separately). |
| **Length** | Internal/colleague mail: fit on one screen. External: 150–250 words. Never pad to look thorough. |
| **Never send** | Drafts only. Always. |

### Which visual mode

| Situation | Mode |
|---|---|
| NDR says "HTML", "formatted", "nicely presented", "make it presentable" | `--style full` (branded header bar + body card) |
| **Default — no formatting instruction at all** | `--style standard` (clean business HTML: headings, bullets, tables, 15px/1.62) |
| NDR explicitly says "plain", "no HTML", "simple text" | `--no-html` — still NO hard wraps, still `•` bullets |
| Escalation / legal / regulator | `--style standard`, tighter prose, numbered para references |

**A plain-text email is never the lazy default.** The default is a formatted HTML part. Only
`--no-html` produces a text-only mail, and only on his explicit instruction.

## 2. Build path — mandatory

Write the body as **Markdown**, then let the renderer produce both parts. Do not hand-write
HTML, and never hand-write a `text/plain` body with `\n` breaks.

```bash
# 1. body.md  (markdown: ##, **bold**, - bullets, 1. numbers, | tables |)
# 2. build the draft (creates, verifies, prints JSON)
/opt/hermes/.venv/bin/python \
  /data/hermes/skills/communication/email-writing/scripts/email_draft.py \
  --md /tmp/body.md \
  --to "Dharmesh Ranka <dharmesh.ranka1@gmail.com>" \
  --cc "Manish Ranka <mdr@draas.com>" \
  --subject "Re: Ranka Amber - landing page" \
  --style standard \
  --thread 1a19c7f7c2c0d3ce \
  --in-reply-to "<CAF...@mail.gmail.com>" \
  --references "<CAF...@mail.gmail.com> <CAF...2@mail.gmail.com>"
```

Flags: `--style standard|full`, `--no-html`, `--update DRAFT_ID` (revise in place, keeps the
draft id), `--attach /path/file.pdf` (repeatable), `--dry-run` (writes `/tmp/emailbuild/body.{html,txt}`
without touching Gmail), `--service google-draas`, `--state /tmp/draft_id.txt`.

The script prints `{draft_id, message_id, thread_id, parts, line_stats, warnings}` and exits
non-zero if the stored structure or line lengths look wrong. **`line_stats.suspect` must be 0** —
that is the hard-wrap detector.

### Threading
Reply ⇒ `--thread <original threadId>` **and** `--in-reply-to <last inbound Message-ID>`
**and** `--references` = full chain (root first). Both headers are required for Gmail to nest
it. Missing `References` is what orphans a reply into a new conversation. Chain-building rules,
the "reply where the last message is my own" trap, and cross-mailbox threads live in
`email-drafter` — read `references/threaded-reply-all-routing.md` and
`references/gmail-draft-update-in-place.md` there.

## 3. Writing shape that works for NDR

Open with the conclusion or the ask in one line — no "I hope this finds you well", no
throat-clearing. Then sections. Then the specific asks with owner + date. Then sign-off.

```
<Name>,

<one line: why this mail, and the decision/action wanted>

## Where it stands
prose paragraph (one line per paragraph)

## What I need
1. **X** — detail
2. **Y** — detail

| Item | Owner | Due |
|---|---|---|

<consequence / deadline sentence>

Regards,
Nishant
```

Rules of thumb:
- Colleague coordination, 3+ points → **always** headings + bullets. This is the single most
  common failure.
- Escalation → numbered list of *what was agreed / what slipped / what is needed*, in that
  order; one bold line at the top with the consequence.
- Short acknowledgements ("noted, will revert by Friday") → 2 lines, no structure needed.
- Never dump a scope list as bullets if it is one sentence of prose; never keep itemised data
  (figures, dates, deliverables) as prose.

## 4. Verify before reporting back (non-optional)

The script already reads the stored draft back. Still confirm from its JSON:
`parts == ["text/plain","text/html"]`, `line_stats.suspect == 0`, `line_stats.max` well over
150 (proves no fixed-width wrap), `thread_id` is the intended thread, and `subject`/`to` match.
If you revised an existing draft, re-read it — `draft_id` is stable across `--update`, the
`message_id` is not.

## 5. When something already went out plain

If NDR points at a **sent** email and says it looks bad, do not re-argue it: reply-all on the
same thread with the same content reformatted (`--thread` + `--in-reply-to` + `--references`),
delete any earlier draft on that thread, and say in your report that you resent it formatted.

## 6. Related

- `email-drafter` — Gmail API mechanics, threading edge cases, recipient resolution, attachments.
- `messaging-drafts` — WhatsApp / cross-channel drafting; email sections defer to this file.
- `dra-brand-html-teaser` — actual designed marketing collateral (not correspondence).
- Public prior art checked Nov 2026: Resend/Anthropic `email-best-practices` (deliverability,
  SPF/DKIM, compliance) and `react-email` (React components) — neither covers business
  correspondence layout, so this skill is the local standard.
