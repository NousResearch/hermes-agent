# Observations

Patterns this agent noticed but was not allowed to message about (Tier 2 of the
proactivity contract in `SOUL.md`). Append-only: one dated line per sighting, newest at
the bottom, evidence included. Nothing here reaches a chat on its own — only a trigger
register entry does.

Status ladder:

- `open` — seen once; wait for the second sighting before judging it.
- `confirmed` — seen twice or more, or verified by a second source; now either raise a
  trigger or keep watching.
- `actioned` — handed to a human decision, a kanban card, or a fix; link it.
- `dismissed` — explained, expected, or noise; say why so the next session doesn't
  re-investigate it.

Format: `YYYY-MM-DD HH:MM <tz> | <status> | <observation> | <occurrences> | <evidence>`

```
2026-01-04 09:12 UTC | open       | feed returns 200 with an empty body on Sundays | 1x | 2026-01-04 cron run log
2026-01-11 09:10 UTC | confirmed  | feed returns 200 with an empty body on Sundays | 2x | 2026-01-04, 2026-01-11 cron run logs
```

Review the log when a trigger fires, when the same observation reaches `confirmed`, and
before editing the trigger register — an observation that keeps recurring is a candidate
for Tier 3, and one that never matters is a candidate for deletion.
