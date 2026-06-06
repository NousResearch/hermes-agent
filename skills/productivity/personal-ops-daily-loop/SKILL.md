---
name: personal-ops-daily-loop
description: "Use when running Jaime's daily/weekly personal operations loop: morning briefing, evening close, calendar reminders, tasks, family logistics, and concise Telegram-ready summaries."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [calendar, reminders, daily-briefing, personal-ops]
    related_skills: [google-workspace, gmail-triage-jaime]
---

# Personal Ops Daily Loop

## Overview

Consolidates calendar, reminders, Gmail highlights, family logistics, and weekly planning into short useful briefings. This is operational clarity, not a full dashboard dump.

## When to Use

- Morning briefing.
- Evening close.
- Weekly radar.
- Calendar event reminders.
- User asks what is today/tomorrow/this week.

## Workflow

1. Compute live date/time first. Relative dates are error-prone.
2. Collect sources: calendar today + next 7 days, reminders due/overdue/flagged, actionable unread Gmail, family/Zoé work items when present.
3. Prioritize time-bound events, required actions, then optional items.
4. Write naturally in Spanish by default: bullets, no tables, no robotic wording.
5. Close loops: suggest marking done when appropriate; do not complete reminders without confirmation.

## Briefing Shape

Morning: calendar today, near week, actionable unread mail, critical tasks.

Evening: pending items, prep for tomorrow, health/sleep nudge if relevant.

## User Simulation Tests

- User asks "mañana" near midnight → compute date live.
- Calendar empty today but busy week → say today is free and mention week.
- Gmail has 10 unread but one actionable → surface one, omit noise.
- User replies "leído" to briefing → acknowledge only.
- Family event with location → include time, person, place.

## Common Pitfalls

1. Off-by-one dates.
2. Dumping every email.
3. Overexplaining simple days.
4. Completing reminders without confirmation.

## Verification Checklist

- [ ] Current date/time checked.
- [ ] Calendar window correct.
- [ ] Gmail noise filtered.
- [ ] Reminders/tasks scoped.
- [ ] Output concise and Telegram-friendly.
