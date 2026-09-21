---
name: lesson-delivery
title: Lesson Delivery Guidelines
description: Standardised process and formatting for daily AI/ML lesson delivery via Discord, matching Sahil's style preferences.
created: 30/05/2026
updated: 30/05/2026
type: skill
tags: [teaching, lesson, delivery, discord, html, audio]
---

# Lesson Delivery Guidelines

## Purpose
Standardise how MrHermagi delivers daily AI/ML lessons to Sahil via Discord, ensuring consistency, clarity, and adherence to Sahil's preferences.

## Core Formatting Rules (Discord summary)
- **Title line** must follow the exact pattern:
  `Week {N}: {Theme} - {Topic} - Lesson {N} - {Lesson Name}`
- **Learning objective**: one sentence prefixed with `**Learning objective:**`.
- **The gist**: 3‑5 sentences summarising the analogy, technical core, and plain‑English restatement.
- **Active‑recall**: 2‑3 questions listed as a numbered list, encouraging reply in thread.
- **Attachments**: Two `MEDIA:` tags – one for the full HTML lesson, one for the audio MP3.
- **Character limits**: Discord message ≤ 1800 characters; HTML with dark‑mode styling as defined in the template.

## Style Preferences (embedded from user)
- British English spelling and phrasing.
- Direct, no‑fluff tone.
- Every jargon term defined inline.
- No Mermaid diagrams (Discord strips them).
- Analogy first, then technical explanation, then plain‑English restatement.

## File Templates (see `references/`)
- `lesson-summary-template.txt` – skeleton for the Discord summary.
- `lesson-html-template.html` – dark‑mode HTML skeleton with placeholders for title, objective, sections, and footer.
- `lesson-audio-command.sh` – script to generate audio via `edge-tts` using the transcript file.

## Workflow
1. Identify the day from `curriculum.yaml` (status `next`).
2. Generate HTML using the HTML template, filling in the lesson content.
3. Generate the plain‑text summary using the text template.
4. Create the audio file via the audio script.
5. Write both files to `~/hermes/runbooks/mrhermagi/{date}/`.
6. Output the Discord summary with `MEDIA:` tags pointing to the created files.

## Pitfalls & Checks
- Ensure the target directories exist before writing files.
- Verify the HTML file exists before emitting the `MEDIA:` tag (prevents Failure A from cron‑output‑contract).
- Confirm the audio file path is under an allowed media directory (`HERMES_MEDIA_ALLOW_DIRS` or `MEDIA_DELIVERY_SAFE_ROOTS`).
- Run `cron-output-lint.py` after changes to avoid format violations.

## References
- `references/lesson-summary-template.txt`
- `references/lesson-html-template.html`
- `references/lesson-audio-command.sh`

---

*This skill captures the agreed‑upon lesson delivery process and style preferences for future sessions.*