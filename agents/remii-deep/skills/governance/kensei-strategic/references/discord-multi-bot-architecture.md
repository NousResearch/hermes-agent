# Discord Multi-Bot Architecture for KENSEI Personas

Decision date: 2026-05-22
Context: Sahil wants context isolation per lead persona and co-working capability (multiple personas active in the same channel).

## Architecture Decision

**8 separate Discord bots**, each running its own Hermes gateway instance with `HERMES_HOME` pointed at the corresponding profile directory.

### Separate bots (8)

| Bot | Profile | Model | Purpose | Justification |
|---|---|---|---|---|
| Kensei | default | `openai-codex / gpt-5.5` | Orchestration, crons, MCPs, API server | Central control. Full tools. |
| Misa-Misa | misa-misa | `ollama-cloud / gemma4:31b` | Voice intake, STT/TTS | Different transport. Voice concurrency. |
| Remii | remii | `ollama-cloud / kimi-k2.6` | Research, signals, market scanning | Heavy research context. Co-works with Wesker. |
| Wesker | wesker | `ollama-cloud / kimi-k2.6` | Ops, security, infra | Technical context isolation. Co-works with Remii. |
| Gojo | gojo | `ollama-cloud / gemma4:31b` | Admin, mailbox, calendar, job-hunt | Personal/sensitive data. Privacy isolation. |
| Octacon | octacon | `ollama-cloud / kimi-k2.6` | Coding, build, debugging, PRs | Deep coding sessions. Co-works with Quan. |
| CeeCee | ceecee | `ollama-cloud / glm-5.1` | Content drafting, brand, social | Content review with Kensei. |
| MrHermagi | mrhermagi | `ollama-cloud / deepseek-v4-flash` | Daily AI/ML teaching | Daily lessons with dedicated teaching space. |

### Persona modes under Kensei (for now)

| Persona | Domain | Reason not separate yet |
|---|---|---|
| Light | Knowledge/wiki | Mostly passive reads/writes. |
| Denji | Governance/ledger | Periodic audits, not continuous chat. |
| Quan | QA/review | Review cycles triggered by Octacon. |

## Memory Verification

**Actual RSS as deployed (22/05/26):**

| Gateway | RSS |
|---|---|
| Kensei | ~234MB |
| Misa-Misa | ~140MB |
| Remii | ~144MB |
| Wesker | ~144MB |
| Gojo | ~143MB |
| Octacon | ~143MB |
| CeeCee | ~144MB |
| MrHermagi | ~133MB |
| **Total** | **~1.1GB** |

Actual memory is lower than the initial estimate (~130MB/specialist vs ~150-250MB expected). All 8 gateways fit comfortably.

## Intents Portal Quirk

Discord's Developer Portal does not always persist privileged intents on first save. The toggle shows ON but the bot still fails with `PrivilegedIntentsRequired`. Fix:
1. Verify you're editing the correct application (when 6+ bots were created in rapid succession, it's easy to toggle intents on the wrong app)
2. Toggle both intents OFF -> Save -> toggle ON -> Save
3. Restart the gateway
4. Check the profile's `logs/gateway.log` for actual current session state, not journalctl

## Co-Working Pattern

Multiple bots coexist in the same channel. Sahil addresses them by name. Each responds independently from its own session/memory/context.

## Specialist Gateway Configuration

Each specialist gateway:
- `HERMES_HOME=/home/kensei/.hermes/profiles/<name>`
- Discord platform only
- No cron ticker
- No MCP servers (MCPs live only on Kensei's main gateway)
- No API server
- Memory limits: 800MB high, 1GB max
- Profile-local cron jobs disabled

## Channel Structure

### Command Centre
| Channel | Bots | Purpose |
|---|---|---|
| `#general` | Kensei, Misa-Misa | Human chat, casual commands |
| `#ops` | Wesker, Kensei | Gateway, services, infra |
| `#governance` | Kensei, Denji (mode) | Audit, triage, ledger |
| `#approvals` | Kensei only | High-signal Sahil decisions |
| `#cron-outputs` | Kensei only | Scheduled automation output |

### Mission Tracks
| Channel | Bots | Purpose |
|---|---|---|
| `#job-hunt` | Gojo, Kensei | CV, inbox, recruiter comms |
| `#plenishd` | Octacon, CeeCee, Kensei | Product/build/content |
| `#property` | Gojo, Kensei | Property track |
| `#coachsense` | Octacon, Remii, Kensei | Build/research |
| `#matchdaymaestro` | CeeCee, Octacon, Kensei | Content/product |

### Research & Intelligence
| Channel | Bots | Purpose |
|---|---|---|
| `#research-digest` | Remii | Daily digest output |
| `#research-deep` | Remii | Long-form investigation |
| `#signals` | Remii | Raw watchlist signals |
| `#research-ops` | Remii, Wesker | Research + infra overlap |

### Build & QA
| Channel | Bots | Purpose |
|---|---|---|
| `#build-log` | Octacon | Build/deploy updates |
| `#build-review` | Octacon, Quan (mode) | Code review |
| `#security` | Wesker | Security findings |

### War Room
| Channel | Bots | Purpose |
|---|---|---|
| `#war-room` | Kensei, Remii, Wesker, Octacon | Multi-specialist co-working |

### Learning
| Channel | Bots | Purpose |
|---|---|---|
| `#mrhermagi-lessons` / `#ai-learning-qa` | MrHermagi | Daily lessons, follow-up Q&A |

### Voice
| Channel | Bot | Purpose |
|---|---|---|
| `Misa-Misa Intake` | Misa-Misa | Voice commands |
| `Strategy Room` | Kensei (via Misa-Misa) | Voice planning |

## Setup Script

`/home/kensei/.hermes/scripts/setup-specialist-bots.sh` handles 6 bots in one shot. For a 7th+ bot added after batch deploy, use the single-bot addition recipe in `discord-gateway-operations` skill.
