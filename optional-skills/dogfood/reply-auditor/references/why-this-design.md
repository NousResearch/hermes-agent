# Why pre-send audit vs. log-scraping audit

A note on design choice.

## Two places to look for lies

An agent can lie at any of:

1. **Draft time** — before the message is sent.
2. **Run time** — in the message, but accompanied by logs that
   contradict it.
3. **Retrospective** — after the message has been sent.

Each has a different tool:

| When | Tool | Skill |
|------|------|-------|
| Draft time | Mechanical regex on the draft | `reply-auditor` (this) |
| Run time | Cross-reference draft ↔ logs | `self-audit` (already in dogfood) |
| Retrospective | Read git log of past sessions | session_search |

This skill focuses on draft time because that's the cheapest fix —
reject the message *before* it goes out.

## Why not just check logs later?

Because the user reads the message first. By the time a log audit
finds a contradiction, the user has already been misled. The damage
is done: trust ticks down, even if the lie is later corrected.

Pre-send audit is a tradeoff: the auditor occasionally false-positives
on innocent language ("I haven't done X yet"), and that costs us
conversational friction. We accept that cost because the cost of
sending a confident-sounding claim without proof is much higher.

## When to disable the auditor

If you've audited once and want to send a follow-up that doesn't need
re-audit, just send a short reply (< 300 chars). The trigger for
loading the skill is response length + presence of claim words. A
short reply usually has neither.
