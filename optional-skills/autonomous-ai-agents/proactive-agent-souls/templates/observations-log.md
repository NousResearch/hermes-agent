# Observations

Patterns this agent noticed but was not allowed to message about (Tier 2 of the
proactivity contract in `SOUL.md`). Append-only: one dated line per sighting, newest at
the bottom, evidence included. Nothing here reaches a chat on its own — only a rule in the
operator's rules file does.

Status ladder:

- `open` — seen once; wait for the second sighting before judging it.
- `confirmed` — seen twice or more, or verified by a second source; now either promote it to
  work (a task, a card, an issue), raise a rule, or keep watching.
- `actioned` — handled: a human decision recorded, a task or card opened, or a Tier 3 heal
  performed; link it.
- `dismissed` — explained, expected, or noise; say why so the next session doesn't
  re-investigate it.

Format: `YYYY-MM-DD HH:MM <tz> | <status> | <observation> | <occurrences> | <evidence>`

```
2026-01-04 09:12 UTC | open       | feed returns 200 with an empty body on Sundays | 1x | 2026-01-04 cron run log
2026-01-11 09:10 UTC | confirmed  | feed returns 200 with an empty body on Sundays | 2x | 2026-01-04, 2026-01-11 cron run logs
2026-01-11 09:40 UTC | actioned   | re-armed the job's retry after the empty-body sighting | 2x | Tier 3 heal: one retry, verified 09:38, no message sent
```

Review the log when a rule fires, when an observation reaches `confirmed`, and before editing
the rules file — an observation that keeps recurring is a candidate for a rule, and one that
never matters is a candidate for deletion.
