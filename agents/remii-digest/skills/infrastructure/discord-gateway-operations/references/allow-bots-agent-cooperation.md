# allow_bots: Agent-to-Agent Cooperation in Discord Channels

## Problem

In a multi-bot Discord deployment, bots silently ignore each other's messages because `DISCORD_ALLOW_BOTS` defaults to `none`. This means explicit handoffs like Kensei posting "Dezzy — please review this brief" in `#design-review` never reach Dezzy — the message is dropped at intake before any `require_mention` or `free_response_channels` check.

## Solution

Set `allow_bots: mentions` in the `discord.extra` block of every bot that should accept directed work from other bots.

```yaml
discord:
  extra:
    gateway_restart_notification: false
    allow_bots: mentions
```

## How it interacts with other Discord config

| Setting | Bot A sends | Bot B sees it? |
|---|---|---|
| `allow_bots: none` (default) | `@BotB check this` | ❌ Dropped at intake |
| `allow_bots: mentions` | `@BotB check this` | ✅ Only if BotB is @mentioned |
| `allow_bots: mentions` | `BotB — check this` (no @mention) | ❌ Dropped — plain text doesn't trigger |
| `free_response_channels` set | `BotB — check this` (no @mention) | ✅ Only if channel is in BotB's free_response list |

**Key caveat**: `allow_bots: mentions` requires an actual Discord @mention. Plain text "Dezzy — ..." without `@Dezzy` produces an empty `message.mentions` list and is dropped. For home channels, `free_response_channels` covers text-only references. For co-working channels, always use proper @mentions.

## How it works (gateway internals)

Three-layer filter in `adapter.py`:

1. **Intake filter** (line 797-812): `DISCORD_ALLOW_BOTS=none` → `return` immediately. `mentions` → only proceed if `self._client.user in message.mentions`. `all` → pass through.

2. **Multi-agent filter** (line 828-862): After intake, messages that mention OTHER bots but not THIS bot are dropped. Prevents cross-consumption in co-working channels (e.g. @Wesker message doesn't trigger Dezzy, and @Dezzy message doesn't trigger Wesker).

3. **History backfill** (line 3908-3957): `mentions` is treated as `all` here — context assembly is about information, not gating. Other bot messages are included in backfill context regardless of mention status.

## Per-bot, not global

`allow_bots` is per-gateway. Setting it on Dezzy means Dezzy processes messages FROM other bots. It does NOT affect whether other bots see Dezzy's messages — that's controlled by each bot's own setting. Set it on every bot that should be reachable for handoffs.

## Deployment (from 2026-05-24)

Set on all 9 profiles (Kensei root + 8 specialists):

```
KENSEI-root: allow_bots: mentions
dezzy:       allow_bots: mentions
misa-misa:   allow_bots: mentions
remii:       allow_bots: mentions
wesker:      allow_bots: mentions
gojo:        allow_bots: mentions
octacon:     allow_bots: mentions
ceecee:      allow_bots: mentions
mrhermagi:      allow_bots: mentions
```
