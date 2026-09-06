---
title: "UPSTREAM-REMINDER-ONLY: manual update workflow"
status: OPEN
date: 2026-09-06
type: ticket
target_repo: hermes-agent
---

# Upstream updater reminder-only ticket

## Objective

Keep upstream update notifications concise and make the actual update a
deliberate manual code operation.

## Scope

- Send a daily reminder when an upstream update is available.
- Report only non-sensitive update status and change size.
- Direct the operator to the code workflow for review and application.
- Keep deploy, restart, rollback, and push outside Telegram reminders.

## Acceptance criteria

- [x] Reminder-only message is implemented and documented.
- [x] Reminder does not request an approval phrase or interactive update reply.
- [x] Automated tests cover update and no-update messages.
- [x] Manual code workflow remains the update path.
- [x] Reminder excludes repository implementation details.

## Out of scope

- Automatic upstream application.
- Telegram command execution for updates.
- Changes to the live Hermes release.

