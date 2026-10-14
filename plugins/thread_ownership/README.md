# thread_ownership

Coordinate Slack thread ownership across multiple Hermes agents that share a
channel — prevents "reply storms" where every bot answers every message.

When a Hermes bot is @-mentioned in a thread, Hermes auto-follows that thread
(`plugins/platforms/slack/adapter.py` `_mentioned_threads`) and opens a turn on
every later message. With several bots in one channel they *all* open a turn on
every follow-up — and an LLM that opens a turn must emit *something*, so the bots
that weren't addressed blurt out "this isn't for me." This plugin gates that at
`pre_gateway_dispatch`, **before the model runs** — the non-owner skips the turn
entirely, no tokens burned.

## How it works

The rule is **"the most-recently @-mentioned bot owns the thread."** Every bot
following the thread sees every message and runs the same parse, so they all
converge on the same owner with **zero coordination** — no lock service, no
shared state, no roster. A bot only needs to know its *own* Slack user id.

| Situation | This bot… |
|---|---|
| Message @-mentions me (anywhere in the list) | replies; the **last** mention owns the thread going forward |
| Message @-mentions other bots, not me | stays silent (skips the turn) |
| Plain follow-up (no @), I'm the current owner | replies |
| Plain follow-up, I'm not the owner | stays silent |
| DM, or top-level message (not in a thread) | replies — Slack's own root @-gating handles those |

The hook returns `{"action": "skip", "reason": …}` for the silent cases, so the
non-owner never invokes the model.

> **Why parse `raw_message`?** Hermes strips the bot's own `<@id>` from
> `event.text` before the hook runs, which would make a direct `@me …` look like
> an un-addressed follow-up. The plugin reads the untouched `raw_message["text"]`
> so it sees its own mention and the full ordered mention list, falling back to
> `event.text` only if the raw text is unavailable.

## Setup

Add it to your service's `config.yaml`:

```yaml
plugins:
  enabled:
    - thread_ownership
```

That's the entire setup — **no operator knobs, no env vars to configure.** The
plugin learns its own Slack user id from the running Slack adapter, which already
authenticates every configured bot token and keeps a per-workspace
`team_id → bot_user_id` map (`_team_bot_user_ids`). It reads that map keyed by
each message's own workspace, so multi-workspace deployments (comma-separated
`SLACK_BOT_TOKEN`, or OAuth-added workspaces) resolve the right identity per
message rather than assuming a single token.

If it can't resolve the bot's own user id for a message's workspace, it **fails
open** — it never silences a bot it can't identify.

## State & restarts

Ownership is per-process, in-memory (`{channel:thread → am I owner}`). On restart
a bot goes quiet in a thread until it's @-mentioned again — identical to Hermes's
own in-memory auto-follow state.

## Compatibility

Pure Python (stdlib only), no platform-specific syscalls — runs anywhere Hermes
runs. Only acts on Slack messages; a no-op for every other platform.
