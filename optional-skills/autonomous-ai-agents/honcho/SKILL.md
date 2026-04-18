---
name: honcho
description: Configure and use Honcho memory with Hermes -- cross-session user modeling, multi-profile peer isolation, observation config, dialectic reasoning, session summaries, and context budget enforcement. Use when setting up Honcho, troubleshooting memory, managing profiles with Honcho peers, or tuning observation, recall, and dialectic settings.
version: 2.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [Honcho, Memory, Profiles, Observation, Dialectic, User-Modeling, Session-Summary]
    homepage: https://docs.honcho.dev
    related_skills: [hermes-agent]
prerequisites:
  pip: [honcho-ai]
---

# Honcho Memory for Hermes

Honcho provides AI-native cross-session user modeling. It learns who the user is across conversations and gives every Hermes profile its own peer identity while sharing a unified view of the user.

## When to Use

- Setting up Honcho (cloud or self-hosted)
- Troubleshooting memory not working / peers not syncing
- Creating multi-profile setups where each agent has its own Honcho peer
- Tuning observation, recall, dialectic depth, or write frequency settings
- Understanding what the 5 Honcho tools do and when to use them
- Configuring context budgets and session summary injection

## Setup

### Cloud (app.honcho.dev)

```bash
hermes honcho setup
# select "cloud", paste API key from https://app.honcho.dev
```

### Self-hosted

```bash
hermes honcho setup
# select "local", enter base URL (e.g. http://localhost:8000)
```

See: https://docs.honcho.dev/v3/guides/integrations/hermes#running-honcho-locally-with-hermes

### Verify

```bash
hermes honcho status    # shows resolved config, connection test, peer info
```

## Architecture

### Base Context Injection

When Honcho injects context into the system prompt (in `hybrid` or `context` recall modes), it assembles the base context block in this order:

1. **Session summary** -- a short digest of the current session so far (placed first so the model has immediate conversational continuity)
2. **User representation** -- Honcho's accumulated model of the user (preferences, facts, patterns)
3. **AI peer card** -- the identity card for this Hermes profile's AI peer

The session summary is generated automatically by Honcho at the start of each turn (when a prior session exists). It gives the model a warm start without replaying full history.

### Cold / Warm Prompt Selection

Honcho automatically selects between two prompt strategies:

| Condition | Strategy | What happens |
|-----------|----------|--------------|
| No prior session or empty representation | **Cold start** | Lightweight intro prompt; skips summary injection; encourages the model to learn about the user |
| Existing representation and/or session history | **Warm start** | Full base context injection (summary → representation → card); richer system prompt |

You do not need to configure this -- it is automatic based on session state.

### Peers

Honcho models conversations as interactions between **peers**. Hermes creates two peers per session:

- **User peer** (`peerName`): represents the human. Honcho builds a user representation from observed messages.
- **AI peer** (`aiPeer`): represents this Hermes instance. Each profile gets its own AI peer so agents develop independent views.

### Observation

Each peer has two observation toggles that control what Honcho learns from:

| Toggle | What it does |
|--------|-------------|
| `observeMe` | Peer's own messages are observed (builds self-representation) |
| `observeOthers` | Other peers' messages are observed (builds cross-peer understanding) |

Default: all four toggles **on** (full bidirectional observation).

Configure per-peer in `honcho.json`:

```json
{
  "observation": {
    "user": { "observeMe": true, "observeOthers": true },
    "ai":   { "observeMe": true, "observeOthers": true }
  }
}
```

Or use the shorthand presets:

| Preset | User | AI | Use case |
|--------|------|----|----------|
| `"directional"` (default) | me:on, others:on | me:on, others:on | Multi-agent, full memory |
| `"unified"` | me:on, others:off | me:off, others:on | Single agent, user-only modeling |

Settings changed in the [Honcho dashboard](https://app.honcho.dev) are synced back on session init -- server-side config wins over local defaults.

### Sessions

Honcho sessions scope where messages and observations land. Strategy options:

| Strategy | Behavior |
|----------|----------|
| `per-directory` (default) | One session per working directory |
| `per-repo` | One session per git repository root |
| `per-session` | New Honcho session each Hermes run |
| `global` | Single session across all directories |

Manual override: `hermes honcho map my-project-name`

### Recall Modes

How the agent accesses Honcho memory:

| Mode | Auto-inject context? | Tools available? | Use case |
|------|---------------------|-----------------|----------|
| `hybrid` (default) | Yes | Yes | Agent decides when to use tools vs auto context |
| `context` | Yes | No (hidden) | Minimal token cost, no tool calls |
| `tools` | No | Yes | Agent controls all memory access explicitly |

## Three Orthogonal Knobs

Honcho's dialectic behavior is controlled by three independent dimensions. Each can be tuned without affecting the others:

### Cadence (when)

Controls **how often** dialectic and context calls happen.

| Key | Default | Description |
|-----|---------|-------------|
| `contextCadence` | `1` | Min turns between context API calls |
| `dialecticCadence` | `3` | Min turns between dialectic API calls |
| `injectionFrequency` | `every-turn` | `every-turn` or `first-turn` for base context injection |

Higher cadence values reduce API calls and cost. `dialecticCadence: 3` (default) means the dialectic engine fires at most every 3rd turn.

### Depth (how many)

Controls **how many rounds** of dialectic reasoning Honcho performs per query.

| Key | Default | Range | Description |
|-----|---------|-------|-------------|
| `dialecticDepth` | `1` | 1-3 | Number of dialectic reasoning rounds per query |
| `dialecticDepthLevels` | -- | array | Optional per-depth-round level overrides (see below) |

`dialecticDepth: 2` means Honcho runs two rounds of dialectic synthesis. The first round produces an initial answer; the second refines it.

`dialecticDepthLevels` lets you set the reasoning level for each round independently:

```json
{
  "dialecticDepth": 3,
  "dialecticDepthLevels": ["low", "medium", "high"]
}
```

If `dialecticDepthLevels` is omitted, rounds use **proportional levels** derived from `dialecticReasoningLevel` (the base):

| Depth | Pass levels |
|-------|-------------|
| 1 | [base] |
| 2 | [minimal, base] |
| 3 | [minimal, base, low] |

This keeps earlier passes cheap while using full depth on the final synthesis.

### Level (how hard)

Controls the **intensity** of each dialectic reasoning round.

| Key | Default | Description |
|-----|---------|-------------|
| `dialecticReasoningLevel` | `low` | `minimal`, `low`, `medium`, `high`, `max` |
| `dialecticDynamic` | `true` | When `true`, the model can pass `reasoning_level` to `honcho_reasoning` to override the default per-call. `false` = always use `dialecticReasoningLevel`, model overrides ignored |

Higher levels produce richer synthesis but cost more tokens on Honcho's backend.

## Multi-Profile Setup

Each Hermes profile gets its own Honcho AI peer while sharing the same workspace (user context). This means:

- All profiles see the same user representation
- Each profile builds its own AI identity and observations
- Conclusions written by one profile are visible to others via the shared workspace

### Create a profile with Honcho peer

```bash
hermes profile create coder --clone
# creates host block hermes.coder, AI peer "coder", inherits config from default
```

What `--clone` does for Honcho:
1. Creates a `hermes.coder` host block in `honcho.json`
2. Sets `aiPeer: "coder"` (the profile name)
3. Inherits `workspace`, `peerName`, `writeFrequency`, `recallMode`, etc. from default
4. Eagerly creates the peer in Honcho so it exists before first message

### Backfill existing profiles

```bash
hermes honcho sync    # creates host blocks for all profiles that don't have one yet
```

### Per-profile config

Override any setting in the host block:

```json
{
  "hosts": {
    "hermes.coder": {
      "aiPeer": "coder",
      "recallMode": "tools",
      "dialecticDepth": 2,
      "observation": {
        "user": { "observeMe": true, "observeOthers": false },
        "ai": { "observeMe": true, "observeOthers": true }
      }
    }
  }
}
```

## Tools

The agent has 5 bidirectional Honcho tools (hidden in `context` recall mode):

| Tool | LLM call? | Cost | Use when |
|------|-----------|------|----------|
| `honcho_profile` | No | minimal | Quick factual snapshot at conversation start or for fast name/role/pref lookups |
| `honcho_search` | No | low | Fetch specific past facts to reason over yourself — raw excerpts, no synthesis |
| `honcho_context` | No | low | Full session context snapshot: summary, representation, card, recent messages |
| `honcho_reasoning` | Yes | medium–high | Natural language question synthesized by Honcho's dialectic engine |
| `honcho_conclude` | No | minimal | Write or delete a persistent fact; pass `peer: "ai"` for AI self-knowledge |

### `honcho_profile`
Read or update a peer card — curated key facts (name, role, preferences, communication style). Pass `card: [...]` to update; omit to read. No LLM call.

### `honcho_search`
Semantic search over stored context for a specific peer. Returns raw excerpts ranked by relevance, no synthesis. Default 800 tokens, max 2000. Good when you need specific past facts to reason over yourself rather than a synthesized answer.

### `honcho_context`
Full session context snapshot from Honcho — session summary, peer representation, peer card, and recent messages. No LLM call. Use when you want to see everything Honcho knows about the current session and peer in one shot.

### `honcho_reasoning`
Natural language question answered by Honcho's dialectic reasoning engine (LLM call on Honcho's backend). Higher cost, higher quality. Pass `reasoning_level` to control depth: `minimal` (fast/cheap) → `low` → `medium` → `high` → `max` (thorough). Omit to use the configured default (`low`). Use for synthesized understanding of the user's patterns, goals, or current state.

### `honcho_conclude`
Write or delete a persistent conclusion about a peer. Pass `conclusion: "..."` to create. Pass `delete_id: "..."` to remove a conclusion (for PII removal — Honcho self-heals incorrect conclusions over time, so deletion is only needed for PII). You MUST pass exactly one of the two.

### Bidirectional peer targeting

All 5 tools accept an optional `peer` parameter:
- `peer: "user"` (default) — operates on the user peer
- `peer: "ai"` — operates on this profile's AI peer
- `peer: "<explicit-id>"` — any peer ID in the workspace

Examples:
```
honcho_profile                        # read user's card
honcho_profile peer="ai"              # read AI peer's card
honcho_reasoning query="What does this user care about most?"
honcho_reasoning query="What are my interaction patterns?" peer="ai" reasoning_level="medium"
honcho_conclude conclusion="Prefers terse answers"
honcho_conclude conclusion="I tend to over-explain code" peer="ai"
honcho_conclude delete_id="abc123"    # PII removal
```

## Agent Usage Patterns

Guidelines for Hermes when Honcho memory is active.

### On conversation start

```
1. honcho_profile                  → fast warmup, no LLM cost
2. If context looks thin → honcho_context  (full snapshot, still no LLM)
3. If deep synthesis needed → honcho_reasoning  (LLM call, use sparingly)
```

Do NOT call `honcho_reasoning` on every turn. Auto-injection already handles ongoing context refresh. Use the reasoning tool only when you genuinely need synthesized insight the base context doesn't provide.

### When the user shares something to remember

```
honcho_conclude conclusion="<specific, actionable fact>"
```

Good conclusions: "Prefers code examples over prose explanations", "Working on a Rust async project through April 2026"
Bad conclusions: "User said something about Rust" (too vague), "User seems technical" (already in representation)

### When the user asks about past context / you need to recall specifics

```
honcho_search query="<topic>"       → fast, no LLM, good for specific facts
honcho_context                       → full snapshot with summary + messages
honcho_reasoning query="<question>"  → synthesized answer, use when search isn't enough
```

### When to use `peer: "ai"`

Use AI peer targeting to build and query the agent's own self-knowledge:
- `honcho_conclude conclusion="I tend to be verbose when explaining architecture" peer="ai"` — self-correction
- `honcho_reasoning query="How do I typically handle ambiguous requests?" peer="ai"` — self-audit
- `honcho_profile peer="ai"` — review own identity card

### When NOT to call tools

In `hybrid` and `context` modes, base context (user representation + card + session summary) is auto-injected before every turn. Do not re-fetch what was already injected. Call tools only when:
- You need something the injected context doesn't have
- The user explicitly asks you to recall or check memory
- You're writing a conclusion about something new

### Cadence awareness

`honcho_reasoning` on the tool side shares the same cost as auto-injection dialectic. After an explicit tool call, the auto-injection cadence resets — avoiding double-charging the same turn.

## Config Reference

Config file: `$HERMES_HOME/honcho.json` (profile-local) or `~/.honcho/config.json` (global).

### Key settings

| Key | Default | Description |
|-----|---------|-------------|
| `apiKey` | -- | API key ([get one](https://app.honcho.dev)) |
| `baseUrl` | -- | Base URL for self-hosted Honcho |
| `peerName` | -- | User peer identity |
| `aiPeer` | host key | AI peer identity |
| `workspace` | host key | Shared workspace ID |
| `recallMode` | `hybrid` | `hybrid`, `context`, or `tools` |
| `observation` | all on | Per-peer `observeMe`/`observeOthers` booleans |
| `writeFrequency` | `async` | `async`, `turn`, `session`, or integer N |
| `sessionStrategy` | `per-directory` | `per-directory`, `per-repo`, `per-session`, `global` |
| `messageMaxChars` | `25000` | Max chars per message (chunked if exceeded) |

### Dialectic settings

| Key | Default | Description |
|-----|---------|-------------|
| `dialecticReasoningLevel` | `low` | `minimal`, `low`, `medium`, `high`, `max` |
| `dialecticDynamic` | `true` | Auto-bump reasoning by query complexity. `false` = fixed level |
| `dialecticDepth` | `1` | Number of dialectic rounds per query (1-3) |
| `dialecticDepthLevels` | -- | Optional array of per-round levels, e.g. `["low", "high"]` |
| `dialecticMaxInputChars` | `10000` | Max chars for dialectic query input |

### Context budget and injection

| Key | Default | Description |
|-----|---------|-------------|
| `contextTokens` | uncapped | Max tokens for the combined base context injection (summary + representation + card). Opt-in cap — omit to leave uncapped, set to an integer to bound injection size. |
| `injectionFrequency` | `every-turn` | `every-turn` or `first-turn` |
| `contextCadence` | `1` | Min turns between context API calls |
| `dialecticCadence` | `3` | Min turns between dialectic LLM calls |

The `contextTokens` budget is enforced at injection time. If the session summary + representation + card exceed the budget, Honcho trims the summary first, then the representation, preserving the card. This prevents context blowup in long sessions.

### Memory-context sanitization

Honcho sanitizes the `memory-context` block before injection to prevent prompt injection and malformed content:

- Strips XML/HTML tags from user-authored conclusions
- Normalizes whitespace and control characters
- Truncates individual conclusions that exceed `messageMaxChars`
- Escapes delimiter sequences that could break the system prompt structure

This fix addresses edge cases where raw user conclusions containing markup or special characters could corrupt the injected context block.

## Troubleshooting

### "Honcho not configured"
Run `hermes honcho setup`. Ensure `memory.provider: honcho` is in `~/.hermes/config.yaml`.

### Memory not persisting across sessions
Check `hermes honcho status` -- verify `saveMessages: true` and `writeFrequency` isn't `session` (which only writes on exit).

### Profile not getting its own peer
Use `--clone` when creating: `hermes profile create <name> --clone`. For existing profiles: `hermes honcho sync`.

### Observation changes in dashboard not reflected
Observation config is synced from the server on each session init. Start a new session after changing settings in the Honcho UI.

### Messages truncated
Messages over `messageMaxChars` (default 25k) are automatically chunked with `[continued]` markers. If you're hitting this often, check if tool results or skill content is inflating message size.

### Context injection too large
If you see warnings about context budget exceeded, lower `contextTokens` or reduce `dialecticDepth`. The session summary is trimmed first when the budget is tight.

### Session summary missing
Session summary requires at least one prior turn in the current Honcho session. On cold start (new session, no history), the summary is omitted and Honcho uses the cold-start prompt strategy instead.

## CLI Commands

| Command | Description |
|---------|-------------|
| `hermes honcho setup` | Interactive setup wizard (cloud/local, identity, observation, recall, sessions) |
| `hermes honcho status` | Show resolved config, connection test, peer info for active profile |
| `hermes honcho enable` | Enable Honcho for the active profile (creates host block if needed) |
| `hermes honcho disable` | Disable Honcho for the active profile |
| `hermes honcho peer` | Show or update peer names (`--user <name>`, `--ai <name>`, `--reasoning <level>`) |
| `hermes honcho peers` | Show peer identities across all profiles |
| `hermes honcho mode` | Show or set recall mode (`hybrid`, `context`, `tools`) |
| `hermes honcho tokens` | Show or set token budgets (`--context <N>`, `--dialectic <N>`) |
| `hermes honcho sessions` | List known directory-to-session-name mappings |
| `hermes honcho map <name>` | Map current working directory to a Honcho session name |
| `hermes honcho identity` | Seed AI peer identity or show both peer representations |
| `hermes honcho sync` | Create host blocks for all Hermes profiles that don't have one yet |
| `hermes honcho migrate` | Step-by-step migration guide from OpenClaw native memory to Hermes + Honcho |
| `hermes memory setup` | Generic memory provider picker (selecting "honcho" runs the same wizard) |
| `hermes memory status` | Show active memory provider and config |
| `hermes memory off` | Disable external memory provider |
