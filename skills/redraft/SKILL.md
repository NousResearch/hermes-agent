---
triggers: ["redraft this", "make this readable", "clean this up", "format this properly", "restructure this text", "tidy up this message"]
name: redraft
description: "Redraft text into a clean, structured, scannable message."
version: 1.0.0
author: Nishant Ranka (nranka79, ndr@draas.com)
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [redraft, format, structure, readability, editing, writing, telegram]
    category: communication
    related_skills: [humanizer]
---

# Redraft Skill

Restructures any messy text (paste, voice transcription, quoted message, or file) into a clean, easy-to-read message with proper paragraph separation, bullet points, numbered lists, and bold highlights on key tasks, facts, figures, and asks. It restructures — it does NOT rewrite content.

## When to Use

Load this skill when the user:
- says `/redraft` (with text inline, or replying to a message)
- says "redraft this", "make this readable", "clean this up", "format this properly"
- pastes a wall of unformatted text (voice-transcribed notes, rough drafts, forwarded messages)

## Prerequisites

None. This is a pure text-transformation skill.

## How to Run

The `/redraft` slash command is registered automatically when this skill is installed — any user on any bot can call it:

1. **Inline:** `/redraft <paste the text>`
2. **Reply:** reply to a message with `/redraft` — redraft the replied-to message's text
3. **File:** `/redraft <file path>` — load it with `read_file` first
4. **Ask:** `/redraft` plus an extra request, e.g. "also shorten it to 5 lines"

## Quick Reference

| Input | Output |
| --- | --- |
| Wall of text | One short context line + separated paragraphs |
| Parallel items (facts, options, features) | Bullet points |
| Sequence (steps, timeline, priorities) | Numbered list |
| Task, ask, deadline, figure | **Bold** highlight |
| Filler and repetition | Removed (only when nothing factual is lost) |

## Procedure

1. **Get the input.** Use the inline text, the replied-to message's text, or `read_file` for a file path. If the input is a pasted document or attachment, ask the user to paste it or name the file.
2. **Read it fully** before restructuring. Identify: the main point, supporting facts, tasks/asks, deadlines, figures, and who is responsible for what.
3. **Open with one short context line** stating the main point — only if the original text implies one; never invent one.
4. **Separate into paragraphs** — one idea per paragraph, blank line between.
5. **Convert parallel items to bullets.** Use a numbered list when the order matters (steps, sequence, ranked asks, timeline).
6. **Bold the load-bearing items** (`**text**`):
   - tasks and asks ("**Please confirm by Wednesday**")
   - deadlines and dates
   - figures, amounts, numbers, percentages
   - names of people/entities when they are the subject of an ask
7. **Keep every fact verbatim.** Copy numbers, units, dates, names, and addresses exactly as written. A redraft must never change, add, or drop information.
8. **Match the tone** of the original (formal, casual, urgent). Do not soften or amplify it.
9. **Output the redrafted message directly** — no preamble, no commentary, no code block. Only add a change-note if the user asked for one.
10. **Length rule:** redraft ≠ summarize. Keep everything unless the user explicitly says "shorten" — then summarize and state what was cut.

## Pitfalls

- **Never invent content.** Structure must not fabricate facts, figures, or asks that are not in the source text.
- **Never drop numbers** — a missing figure or unit silently changes meaning. If the input is ambiguous, keep the original phrasing.
- **Do not turn an ask into a statement** — if the text contains a question or request, keep it a question/request and highlight it.
- **Don't bullet narrative prose.** If it reads as a story or explanation, keep paragraphs; over-bulleting destroys readability.
- **Don't over-bold.** Highlight only key tasks, facts, figures, and asks — at most one bold per sentence, never whole sentences.
- **Don't wrap the output in a code block** — that breaks readability on Telegram and other chat platforms.
- **Don't apply humanizer-style rewriting unless asked** — redraft is structure, not de-AI-ing. (Use the `humanizer` skill for that.)

## Verification

Before sending, self-check:
- Every number, unit, date, and name in the input appears in the output (copy them over, don't retype from memory).
- Every ask/question in the input is still visibly an ask/question.
- Output uses paragraphs, bullets, and numbered lists where the content calls for them.
- Bold highlights point at tasks, facts, figures, and asks — nothing else.
- The message reads cleanly in plain text (no markdown symbols leaking except `**`).
