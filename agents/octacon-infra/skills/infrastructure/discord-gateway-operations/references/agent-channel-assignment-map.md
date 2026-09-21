# Agent-to-Channel Assignment Map (2026-05-24)

## Channel structure

All channels are in the "Kensei Camp" Discord server (ID: 1506021204363051249).

| Channel | ID | Category | Free-response bots | @mention-only bots |
|---|---|---|---|---|
| `#general` | 1506021205797507265 | general | Kensei | All others |
| `#cron-outputs` | 1506021598958719027 | ops | Kensei | All others |
| `#kanban` | 1506021665904001096 | ops | Kensei | All others |
| `#ops` | 1506022531025469623 | ops | Kensei, Wesker | All others |
| `#governance` | 1506022593122009280 | governance | Kensei | All others |
| `#approvals` | 1507448578778333267 | governance | Kensei | All others |
| `#decisions` | 1507448576832311406 | governance | Kensei | All others |
| `#war-room` | 1507448573783183432 | general | Kensei | All others |
| `#misa-misa` | 1506022800190607370 | intake | Misa-Misa | All others |
| `#research-digest` | 1506021736536215813 | research | Remii | All others |
| `#research-ops` | 1507448577784283367 | research | Remii, Wesker | All others |
| `#knowledge` | 1507448575796187278 | knowledge | — | All bots |
| `#mailbox__calendar` | 1506022733287391282 | admin | Gojo | All others |
| `#job-hunt` | 1506022690501169295 | admin | Gojo | All others |
| `#build-log` | 1507448396015734984 | build | Octacon | All others |
| `#build-review` | 1507448572784939151 | build | Octacon, Dezzy | All others |
| `#design-review` | 1507542104262443192 | build | Dezzy | All others |
| `#content` | 1506022640035303494 | content | CeeCee | All others |
| `#ai-learning-qa` | 1507397516642091149 | teaching | MrHermagi | All others |

## Per-bot free_response_channels

| Bot | free_response_channels (names) | IDs | allow_bots |
|---|---|---|---|
| Kensei (root) | #general, #cron-outputs, #kanban, #governance, #approvals, #decisions, #ops, #war-room | 1506021205797507265,1506021598958719027,1506021665904001096,1506022593122009280,1507448578778333267,1507448576832311406,1506022531025469623,1507448573783183432 | mentions |
| Dezzy | #design-review | 1507542104262443192 | mentions |
| Misa-Misa | #misa-misa | 1506022800190607370 | mentions |
| Remii | #research-digest, #research-ops | 1506021736536215813,1507448577784283367 | mentions |
| Wesker | #ops, #research-ops | 1506022531025469623,1507448577784283367 | mentions |
| Gojo | #mailbox__calendar, #job-hunt | 1506022733287391282,1506022690501169295 | mentions |
| Octacon | #build-log, #build-review | 1507448396015734984,1507448572784939151 | mentions |
| CeeCee | #content | 1506022640035303494 | mentions |
| MrHermagi | #ai-learning-qa | 1507397516642091149 | mentions |

## How messages flow through the stack

```
Message arrives in Discord
  → Gateway adapter checks message.author.bot
    → If bot: check DISCORD_ALLOW_BOTS
      → "none": silent drop
      → "mentions": only proceed if @mentioned
      → "all": always proceed
  → Multi-agent filter: if other bots @mentioned but not us, drop
  → Check free_response_channels (channel whitelist for no-mention)
    → If channel IS in free_response_channels: always process
    → If channel IS NOT: require @mention
      → Has @mention: process + backfill history (up to 50 messages)
      → No @mention: silent drop
  → Event assembled: text + channel_context + channel_prompt + skills
  → Session created (new or continued)
  → Agent processes, generates response
  → Response posted to channel
```

## Gateway lifecycle notification routing

`gateway_restart_notification` is per-profile in the `discord:` section's `extra:` block:

| Profile | Flag | Effect |
|---|---|---|
| Kensei-root | `true` | Restart notifications sent to Kensei's home channel (`#general`, ID 1506021205797507265) |
| All 8 specialists | `false` | Silent — no notifications anywhere |

This means shutdown/restart messages only appear in `#general`. No flood in domain channels. The flag has no per-channel routing option (it's all-or-nothing), so the split approach achieves single-channel notifications by giving Kensei the only `true` flag.

## Key principle: free_response is for domain channels, not for general chat

- A bot's HOME CHANNEL is where it should be free-response (e.g. `#design-review` for Dezzy)
- CO-WORKING channels (e.g. `#war-room`, `#research-ops`) have multiple bots that may be free-response or @mention-only
- GENERAL channels (`#general`, `#cron-outputs`) should be Kensei-only for free-response
- All other channels: require @mention to reduce unwanted chime-in

## History: how we got here

1. **2026-05-22**: 8-gateway deployment. Specialist configs created without `discord:` sections.
2. **2026-05-24 01:00**: Dezzy silent-drop discovered. `discord:` section added with `free_response_channels: #design-review`.
3. **2026-05-24 01:20**: Remaining 7 specialists also missing `discord:` sections. Added `free_response_channels` per domain.
4. **2026-05-24 01:30**: Duplicate `discord:` sections found on 6 profiles (old empty + new populated). Cleaned up.
5. **2026-05-24 01:40**: Root Kensei config had `#ai-learning-qa` as free_response with MrHermagi prompt — hijacked from MrHermagi. Migrated to MrHermagi's profile.
6. **2026-05-24 01:45**: `allow_bots: mentions` added to all 9 profiles for explicit bot-to-bot handoff with @mention.
