# Discord Multi-Bot Architecture Decision (2026-05-22)

## Problem

KENSEI has multiple personas (Remii, Wesker, Gojo, Octacon, CeeCee, Misa-Misa, MrHermagi) that each need:
- Independent conversational context and memory (not diluted through Kensei)
- Ability to co-work in the same Discord channel (Sahil talks to both Remii and Wesker in the same room)
- Profile isolation for different model/provider/config settings

## Decision

Use separate Discord bot + Hermes gateway per persona. NOT a single central bot with persona modes.

## Why not one bot?

| Problem | One bot | Separate bots |
|---|---|---|
| Context dilution | Remii's research context polluted by Wesker's ops alerts | Each has own session/state |
| Co-working | One bot can't be two conversational entities | Each bot responds independently |
| Permission isolation | One token, one surface area | Per-bot tokens, channel-specific access |
| Visual identity | One avatar | Distinct persona identity per bot |

## Final state (22/05/26 18:25)

8 gateways live on VPS:

| Bot | Service | Status | RSS | Purpose |
|---|---|---|---|---|
| Kensei | hermes-gateway | active | ~230MB | Main: crons, MCPs, API server, orchestration |
| Misa-Misa | hermes-gateway-misa-misa | active | ~130MB | Voice intake, STT/TTS |
| Remii | hermes-gateway-remii | active | ~130MB | Research, signals, market scanning |
| Wesker | hermes-gateway-wesker | active | ~130MB | Ops, security, infra, gateway health |
| Gojo | hermes-gateway-gojo | active | ~130MB | Admin, mailbox, calendar, job-hunt |
| Octacon | hermes-gateway-octacon | active | ~130MB | Coding, build, debugging, PRs |
| CeeCee | hermes-gateway-ceecee | active | ~130MB | Content drafting, brand, social |
| MrHermagi | hermes-gateway-mrhermagi | active | ~130MB | Daily AI/ML teaching |

Total: ~1.1GB across all 8 gateways.

## Channels by persona

All 20 text channels on the Kensei Camp server (ID: 1506021204363051249):

| Channel | ID | Purpose | Free-response bots | @mention-only bots | Category |
|---|---|---|---|---|---|
| #general | 1506021205797507265 | Human chat / landing | Kensei | All others | general |
| #cron-outputs | 1506021598958719027 | Automation output | Kensei | All others | ops |
| #kanban | 1506021665904001096 | Task boards | Kensei | All others | ops |
| #ops | 1506022531025469623 | Ops/security/infra | Kensei, Wesker | — | ops |
| #governance | 1506022593122009280 | Audit, triage, quality | Kensei | All others | governance |
| #approvals | 1507448578778333267 | Sign-offs | Kensei | All others | governance |
| #decisions | 1507448576832311406 | Decision records | Kensei | All others | governance |
| #war-room | 1507448573783183432 | All-hands co-working | Kensei | All others | general |
| #misa-misa | 1506022800190607370 | Voice intake text channel | Misa-Misa | All others | intake |
| #research-digest | 1506021736536215813 | Daily research & signals | Remii | All others | research |
| #research-ops | 1507448577784283367 | Remii + Wesker co-working | Remii, Wesker | All others | research |
| #knowledge | 1507448575796187278 | Knowledge base queries | — | All | knowledge |
| #mailbox__calendar | 1506022733287391282 | Mailbox + calendar | Gojo | All others | admin |
| #job-hunt | 1506022690501169295 | Job hunt | Gojo | All others | admin |
| #build-log | 1507448396015734984 | Build output | Octacon | All others | build |
| #build-review | 1507448572784939151 | Code review | Octacon | All others | build |
| #design-review | 1507542104262443192 | UX/design review | Dezzy | All others | build |
| #content | 1506022640035303494 | Content drafting | CeeCee | All others | content |
| #ai-learning-qa | 1507397516642091149 | Learning Q&A | MrHermagi | All others | teaching |

### How free_response_channels works

- **Free-response bots**: respond without needing @mention in that channel. Configured via `discord.free_response_channels` in the profile's `config.yaml`. Used for the bot's home channel where it should always be listening.
- **@mention-only bots**: require `@BotName` to respond. This prevents specialist bots from chiming in on channels outside their domain.

### Per-bot free_response_channels configuration (as of 2026-05-24)

Each specialist's `config.yaml` was updated to include a `discord:` section with the correct `free_response_channels`. See `references/silent-drop-diagnosis.md` for why this is critical.

| Bot | free_response_channels (names) | free_response_channels (IDs) | Config file |
|---|---|---|---|
| Kensei | #general, #cron-outputs, #kanban, #governance, #approvals, #decisions, #ops, #war-room | `1506021205797507265,1506021598958719027,1506021665904001096,1506022593122009280,1507448578778333267,1507448576832311406,1506022531025469623,1507448573783183432` | Root config.yaml |
| Dezzy | #design-review | `1507542104262443192` | profiles/dezzy/config.yaml |
| Misa-Misa | #misa-misa | `1506022800190607370` | profiles/misa-misa/config.yaml |
| Remii | #research-digest, #research-ops | `1506021736536215813,1507448577784283367` | profiles/remii/config.yaml |
| Wesker | #ops, #research-ops | `1506022531025469623,1507448577784283367` | profiles/wesker/config.yaml |
| Gojo | #mailbox__calendar, #job-hunt | `1506022733287391282,1506022690501169295` | profiles/gojo/config.yaml |
| Octacon | #build-log, #build-review | `1507448396015734984,1507448572784939151` | profiles/octacon/config.yaml |
| CeeCee | #content | `1506022640035303494` | profiles/ceecee/config.yaml |
| MrHermagi | #ai-learning-qa | `1507397516642091149` | profiles/mrhermagi/config.yaml (+ channel_prompt migrated from Kensei) |

### Important: channel_prompts are per-bot

`channel_prompts` entries in the `discord:` section override the bot's SOUL.md for that specific channel. This was the root cause of Kensei hijacking MrHermagi's identity — Kensei's root config.yaml had `channel_prompts.1507397516642091149` set to `"You are MrHermagi..."`. Moved to MrHermagi's config.yaml on 2026-05-24.

### Co-working channels

Multiple bots can coexist in the same channel. Each maintains its own session context. Bots respond independently when addressed by name.

| Channel | Bots present |
|---|---|
| #war-room | All 9 bots theoretically present (only Kensei has free-response; others need @mention) |
| #research-ops | Remii, Wesker (both free-response) |
| #build-review | Octacon (free-response), Dezzy, others by @mention |

## Setup script

`/home/kensei/.hermes/scripts/setup-specialist-bots.sh`

Backup: `/home/kensei/backups/specialist-bots-20260522-174807`

## Keys to success

1. **Intents must be enabled per bot** — every new Discord app needs MESSAGE CONTENT INTENT and SERVER MEMBERS INTENT in Developer Portal → Bot tab → Privileged Gateway Intents. Discord sometimes doesn't persist the setting on first save (toggle OFF→Save→ON→Save to force-flush).
2. **Private bots need manual invite URLs** — the OAuth2 URL Generator doesn't support them. Construct manually with `client_id` and `permissions` parameters.
3. **Custom Install URL must be cleared before toggling Public Bot off** — lives on the Installation tab, blocks the toggle if populated.
4. **Specialist gateways get their own systemd service** with `HERMES_HOME` pointing to their profile directory. No cron ticker, no MCP servers, no API server. Only Discord platform.
5. **Check gateway.log, not journalctl, for connection state** — journalctl shows old PID output merged with new, making it look like the failure persists.
6. **Memory is manageable** — 8 gateways at ~1.1GB total leaves plenty of headroom on an 8GB VPS.
