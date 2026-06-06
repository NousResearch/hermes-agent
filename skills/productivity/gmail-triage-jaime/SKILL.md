---
name: gmail-triage-jaime
description: Use when triaging Jaime's Gmail summaries, marking messages read, separating noise from actionable mail, and preserving account boundaries. Never deletes or trashes messages without explicit confirmation.
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [gmail, email, triage, unread, personal-ops]
    related_skills: [google-workspace]
---

# Gmail Triage for Jaime

## Overview

Handles the repeated Gmail loop: unread digests, "leído" replies, noise classification, and action extraction. Reduce cognitive load without losing important mail or mixing accounts.

## When to Use

- User replies "leído" to a Gmail digest.
- A cron reports new Gmail messages.
- User asks to silence recurring noise.
- A message may need follow-up, calendar, reminder, or FacOps handoff.

## Account Boundary Rule

Keep accounts separate: personal Gmail/calendar for normal Jaime ops; scoped invoice/task tracking account for FacOps only. Invoices/receipts for autónomo go to FacOps, not personal inbox logic.

## Workflow

1. Identify digest type: single alert, multi-message summary, noise warning, or actionable message.
2. Interpret reply: "Leído"/"Hecho" means acknowledge and mark only referenced messages read if tools are available; "silencia este" means propose filter/unsubscribe before durable suppression; "para Excel de autónomo" routes to FacOps.
3. Classify conservatively: important, normal, or noise.
4. Never delete/trash/archive/destructively filter without explicit confirmation.
5. Report briefly: simple "leído" gets "Hecho."; actionable mail gets next action.

## User Simulation Tests

- Patreon policy update + "Leído" → acknowledge, no lecture.
- Santander movement notice + "Leído" → acknowledge, do not delete.
- "Silencia este" to bank noise → propose narrow filter, ask before applying.
- Receipt + "para Excel" → route to FacOps.
- Mixed digest with security alert → do not mark everything noise.

## Common Pitfalls

1. Deleting obvious spam; Jaime forbids deletion without confirmation.
2. Mixing Gmail identities.
3. Over-talking simple acknowledgements.
4. Silencing finance/security too broadly.

## Verification Checklist

- [ ] Referenced messages only, not whole inbox.
- [ ] No destructive actions without confirmation.
- [ ] Account scope respected.
- [ ] Actionable messages surfaced.
- [ ] Reply is short unless there is a decision.
