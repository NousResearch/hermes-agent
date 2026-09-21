# Discord Notification Migration Notes

Use this reference when auditing or cleaning Hermes notifications after Telegram → Discord migration.

## What went wrong in the May 2026 migration

- Gateway lifecycle messages reached Discord, but most scheduled notifications still routed to Telegram because existing cron `deliver` fields and prompts/scripts retained old Telegram/topic assumptions.
- Some jobs were already sending to Discord but still produced Telegram-style HTML/tags, expandable blocks, or topic wording.
- Provider/session limits can look like notification failures. Keep them separate: routing/output fixes should not silently alter model/provider fallbacks.

## Audit table shape

Always give Sahil a table with at least:

| Notification / Job | Purpose | Schedule / Trigger | Provider | Topic / Channel | Routing field | Output mode | Notes |
|---|---|---|---|---|---|---|---|

Definitions:
- Provider: Telegram, Discord, local, all, origin, or explicit platform target.
- Topic / Channel: human channel name when known, otherwise raw target such as `telegram:<chat>:<topic>` or `discord:<channel_id>`.
- Routing field: where it is configured, usually cron `deliver`, gateway default, script send target, or hardcoded wrapper.
- Output mode: LLM summary, no_agent raw stdout, script-only watchdog, gateway/system event.

## Cleanup sequence

1. Inventory first. Do not start editing until all visible notification sources are listed.
2. Patch stale `deliver` targets from Telegram topics to Discord channels only after explicit scope is clear.
3. Rewrite prompts/scripts to the Discord output contract: concise summary + HTML attachment where detail is needed.
4. Ensure any attachment directory exists before the job prints `MEDIA:/absolute/path.html`.
5. Run representative jobs manually and verify delivery in Discord, not just local stdout.
6. Keep provider/model capacity issues separate. If a cron fails due to a 429/session limit, report it as capacity, not formatting or routing.

## Discord attachment pattern

Hermes `MEDIA:/absolute/path.html` works with Discord as a native file attachment, provided the file exists before send time.

Message pattern:

```text
✅ Job Name · DD/MM/YY HH:MM:SS
checked · 12 items · 2 need attention
- Bullet 1
- Bullet 2
MEDIA:/home/kensei/.hermes/runbooks/job/date/report.html
```

Do not paste full HTML into Discord. Do not use Telegram tags such as `<b>`, `<code>`, expandable blocks, or topic labels in user-facing Discord output.
