# Honcho Memory Provider

AI-native cross-session user modeling with multi-pass dialectic reasoning, session summaries, bidirectional peer tools, and persistent conclusions.

> **Honcho docs:** <https://docs.honcho.dev/v3/guides/integrations/hermes>

## Requirements

- `pip install honcho-ai`
- Honcho API key from [app.honcho.dev](https://app.honcho.dev), or a self-hosted instance

## Setup

```bash
hermes honcho setup    # full interactive wizard (cloud or local)
hermes memory setup    # generic picker, also works
```

Or manually:
```bash
hermes config set memory.provider honcho
echo "HONCHO_API_KEY=***" >> ~/.hermes/.env
```

## Architecture Overview

### Two-Layer Context Injection

Context is injected into the **user message** at API-call time (not the system prompt) to preserve prompt caching. Only a static mode header goes in the system prompt. The injected block is wrapped in `<memory-context>` fences with a system note clarifying it's background data, not new user input.

Two independent layers, each on its own cadence:

**Layer 1 — Base context** (refreshed every `contextCadence` turns):
1. **SESSION SUMMARY** — from `session.context(summary=True)`, placed first
2. **User Representation** — Honcho's evolving model of the user
3. **User Peer Card** — key facts snapshot
4. **AI Self-Representation** — Honcho's model of the AI peer
5. **AI Identity Card** — AI peer facts

**Layer 2 — Dialectic supplement** (fired every `dialecticCadence` turns):
Multi-pass `.chat()` reasoning about the user, appended after base context.

Both layers are joined, then truncated to fit `contextTokens` budget via `_truncate_to_budget` (tokens × 4 chars, word-boundary safe).

### Cold Start vs Warm Session Prompts

Dialectic pass 0 automatically selects its prompt based on session state:

- **Cold** (no base context cached): "Who is this person? What are their preferences, goals, and working style? Focus on facts that would help an AI assistant be immediately useful."
- **Warm** (base context exists): "Given what's been discussed in this session so far, what context about this user is most relevant to the current conversation? Prioritize active context over biographical facts."

Not configurable — determined automatically.

### Dialectic Depth (Multi-Pass Reasoning)

`dialecticDepth` (1–3, clamped) controls how many `.chat()` calls fire per dialectic cycle:

| Depth | Passes | Behavior |
|-------|--------|----------|
| 1 | single `.chat()` | Base query only (cold or warm prompt) |
| 2 | audit + synthesis | Pass 0 result is self-audited; pass 1 does targeted synthesis. Conditional bail-out if pass 0 returns strong signal (>300 chars or structured with bullets/sections >100 chars) |
| 3 | audit + synthesis + reconciliation | Pass 2 reconciles contradictions across prior passes into a final synthesis |

### Proportional Reasoning Levels

When `dialecticDepthLevels` is not set, each pass uses a proportional level relative to `dialecticReasoningLevel` (the "base"):

| Depth | Pass levels |
|-------|-------------|
| 1 | [base] |
| 2 | [minimal, base] |
| 3 | [minimal, base, low] |

Override with `dialecticDepthLevels`: an explicit array of reasoning level strings per pass.

### Three Orthogonal Dialectic Knobs

| Knob | Controls | Type |
|------|----------|------|
| `dialecticCadence` | How often — minimum turns between dialectic firings | int |
| `dialecticDepth` | How many — passes per firing (1–3) | int |
| `dialecticReasoningLevel` | How hard — reasoning ceiling per `.chat()` call | string |

### Input Sanitization

`run_conversation` strips leaked `<memory-context>` blocks from user input before processing. When `saveMessages` persists a turn that included injected context, the block can reappear in subsequent turns via message history. The sanitizer removes `<memory-context>` blocks plus associated system notes.

## Tools

Five bidirectional tools. All accept an optional `peer` parameter (`"user"` or `"ai"`, default `"user"`).

| Tool | LLM call? | Description |
|------|-----------|-------------|
| `honcho_profile` | No | Peer card — key facts snapshot |
| `honcho_search` | No | Semantic search over stored context (800 tok default, 2000 max) |
| `honcho_context` | No | Full session context: summary, representation, card, messages |
| `honcho_reasoning` | Yes | LLM-synthesized answer via dialectic `.chat()` |
| `honcho_conclude` | No | Write a persistent fact/conclusion about the user |

Tool visibility depends on `recallMode`: hidden in `context` mode, always present in `tools` and `hybrid`.

## Config Resolution

Config is read from the first file that exists:

| Priority | Path | Scope |
|----------|------|-------|
| 1 | `$HERMES_HOME/honcho.json` | Profile-local (isolated Hermes instances) |
| 2 | `~/.hermes/honcho.json` | Default profile (shared host blocks) |
| 3 | `~/.honcho/config.json` | Global (cross-app interop) |

Host key is derived from the active Hermes profile: `hermes` (default) or `hermes.<profile>`.

For every key, resolution order is: **host block > root > env var > default**.

## Full Configuration Reference

### Identity & Connection

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `apiKey` | string | — | API key. Falls back to `HONCHO_API_KEY` env var |
| `baseUrl` | string | — | Base URL for self-hosted Honcho. Local URLs auto-skip API key auth |
| `environment` | string | `"production"` | SDK environment mapping |
| `enabled` | bool | auto | Master toggle. Auto-enables when `apiKey` or `baseUrl` present |
| `workspace` | string | host key | Honcho workspace ID. Shared environment — all profiles in the same workspace can see the same user identity and related memories |
| `peerName` | string | — | User peer identity |
| `aiPeer` | string | host key | AI peer identity |

### Memory & Recall

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `recallMode` | string | `"hybrid"` | `"hybrid"` (auto-inject + tools), `"context"` (auto-inject only, tools hidden), `"tools"` (tools only, no injection). Legacy `"auto"` → `"hybrid"` |
| `observationMode` | string | `"directional"` | Preset: `"directional"` (all on) or `"unified"` (shared pool). Use `observation` object for granular control |
| `observation` | object | — | Per-peer observation config (see Observation section) |

### Write Behavior

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `writeFrequency` | string/int | `"async"` | `"async"` (background), `"turn"` (sync per turn), `"session"` (batch on end), or integer N (every N turns) |
| `saveMessages` | bool | `true` | Persist messages to Honcho API |

### Session Resolution

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `sessionStrategy` | string | `"per-directory"` | `"per-directory"`, `"per-session"`, `"per-repo"` (git root), `"global"` |
| `sessionPeerPrefix` | bool | `false` | Prepend peer name to session keys |
| `sessions` | object | `{}` | Manual directory-to-session-name mappings |

#### Session Name Resolution

The Honcho session name determines which conversation bucket memory lands in. Resolution follows a priority chain — first match wins:

| Priority | Source | Example session name |
|----------|--------|---------------------|
| 1 | Manual map (`sessions` config) | `"myproject-main"` |
| 2 | `/title` command (mid-session rename) | `"refactor-auth"` |
| 3 | Gateway session key (Telegram, Discord, etc.) | `"agent-main-telegram-dm-8439114563"` |
| 4 | `per-session` strategy | Hermes session ID (`20260415_a3f2b1`) |
| 5 | `per-repo` strategy | Git root directory name (`hermes-agent`) |
| 6 | `per-directory` strategy | Current directory basename (`src`) |
| 7 | `global` strategy | Workspace name (`hermes`) |

Gateway platforms always resolve via priority 3 (per-chat isolation) regardless of `sessionStrategy`. The strategy setting only affects CLI sessions.

If `sessionPeerPrefix` is `true`, the peer name is prepended: `eri-hermes-agent`.

#### What each strategy produces

- **`per-directory`** — basename of `$PWD`. Opening hermes in `~/code/myapp` and `~/code/other` gives two separate sessions. Same directory = same session across runs.
- **`per-repo`** — git root directory name. All subdirectories within a repo share one session. Falls back to `per-directory` if not inside a git repo.
- **`per-session`** — Hermes session ID (timestamp + hex). Every `hermes` invocation starts a fresh Honcho session. Falls back to `per-directory` if no session ID is available.
- **`global`** — workspace name. One session for everything. Memory accumulates across all directories and runs.

### Multi-Profile Pattern

Multiple Hermes profiles can share one workspace while maintaining separate AI identities. Config resolution is **host block > root > env var > default** — host blocks inherit from root, so shared settings only need to be declared once:

```json
{
  "apiKey": "***",
  "workspace": "hermes",
  "peerName": "yourname",
  "hosts": {
    "hermes": {
      "aiPeer": "hermes",
      "recallMode": "hybrid",
      "sessionStrategy": "per-directory"
    },
    "hermes.coder": {
      "aiPeer": "coder",
      "recallMode": "tools",
      "sessionStrategy": "per-repo"
    }
  }
}
```

Both profiles see the same user (`yourname`) in the same shared environment (`hermes`), but each AI peer builds its own observations, conclusions, and behavior patterns. The coder's memory stays code-oriented; the main agent's stays broad.

Host key is derived from the active Hermes profile: `hermes` (default) or `hermes.<profile>` (e.g. `hermes -p coder` → host key `hermes.coder`).

### Dialectic & Reasoning

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `dialecticDepth` | int | `1` | Passes per dialectic cycle (1–3, clamped). 1=single query, 2=audit+synthesis, 3=audit+synthesis+reconciliation |
| `dialecticDepthLevels` | array | — | Optional array of reasoning level strings per pass. Overrides proportional defaults. Example: `["minimal", "low", "medium"]` |
| `dialecticReasoningLevel` | string | `"low"` | Base reasoning level for `.chat()`: `"minimal"`, `"low"`, `"medium"`, `"high"`, `"max"` |
| `dialecticDynamic` | bool | `true` | When `true`, model can override reasoning level per-call via `honcho_reasoning` tool. When `false`, always uses `dialecticReasoningLevel` |
| `dialecticMaxChars` | int | `600` | Max chars of dialectic result injected into system prompt |
| `dialecticMaxInputChars` | int | `10000` | Max chars for dialectic query input to `.chat()`. Honcho cloud limit: 10k |

### Token Budgets

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `contextTokens` | int | SDK default | Token budget for `context()` API calls. Also gates prefetch truncation (tokens × 4 chars) |
| `messageMaxChars` | int | `25000` | Max chars per message sent via `add_messages()`. Exceeding this triggers chunking with `[continued]` markers. Honcho cloud limit: 25k |

### Cadence (Cost Control)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `contextCadence` | int | `1` | Minimum turns between base context refreshes (session summary + representation + card) |
| `dialecticCadence` | int | `1` | Minimum turns between dialectic `.chat()` firings |
| `injectionFrequency` | string | `"every-turn"` | `"every-turn"` or `"first-turn"` (inject context on the first user message only, skip from turn 2 onward) |
| `reasoningLevelCap` | string | — | Hard cap on reasoning level: `"minimal"`, `"low"`, `"medium"`, `"high"` |

### Observation (Granular)

Maps 1:1 to Honcho's per-peer `SessionPeerConfig`. When present, overrides `observationMode` preset.

```json
"observation": {
  "user": { "observeMe": true, "observeOthers": true },
  "ai":   { "observeMe": true, "observeOthers": true }
}
```

| Field | Default | Description |
|-------|---------|-------------|
| `user.observeMe` | `true` | User peer self-observation (Honcho builds user representation) |
| `user.observeOthers` | `true` | User peer observes AI messages |
| `ai.observeMe` | `true` | AI peer self-observation (Honcho builds AI representation) |
| `ai.observeOthers` | `true` | AI peer observes user messages (enables cross-peer dialectic) |

Presets:
- `"directional"` (default): all four `true`
- `"unified"`: user `observeMe=true`, AI `observeOthers=true`, rest `false`

### Hardcoded Limits

| Limit | Value |
|-------|-------|
| Search tool max tokens | 2000 (hard cap), 800 (default) |
| Peer card fetch tokens | 200 |

## Environment Variables

| Variable | Fallback for |
|----------|-------------|
| `HONCHO_API_KEY` | `apiKey` |
| `HONCHO_BASE_URL` | `baseUrl` |
| `HONCHO_ENVIRONMENT` | `environment` |
| `HERMES_HONCHO_HOST` | Host key override |

## CLI Commands

| Command | Description |
|---------|-------------|
| `hermes honcho setup` | Full interactive setup wizard |
| `hermes honcho status` | Show resolved config for active profile |
| `hermes honcho enable` / `disable` | Toggle Honcho for active profile |
| `hermes honcho mode <mode>` | Change recall or observation mode |
| `hermes honcho peer --user <name>` | Update user peer name |
| `hermes honcho peer --ai <name>` | Update AI peer name |
| `hermes honcho tokens --context <N>` | Set context token budget |
| `hermes honcho tokens --dialectic <N>` | Set dialectic max chars |
| `hermes honcho map <name>` | Map current directory to a session name |
| `hermes honcho sync` | Create host blocks for all Hermes profiles |

## Example Config

```json
{
  "apiKey": "***",
  "workspace": "hermes",
  "peerName": "username",
  "contextCadence": 2,
  "dialecticCadence": 3,
  "dialecticDepth": 2,
  "hosts": {
    "hermes": {
      "enabled": true,
      "aiPeer": "hermes",
      "recallMode": "hybrid",
      "observation": {
        "user": { "observeMe": true, "observeOthers": true },
        "ai": { "observeMe": true, "observeOthers": true }
      },
      "writeFrequency": "async",
      "sessionStrategy": "per-directory",
      "dialecticReasoningLevel": "low",
      "dialecticDepth": 2,
      "dialecticMaxChars": 600,
      "saveMessages": true
    },
    "hermes.coder": {
      "enabled": true,
      "aiPeer": "coder",
      "sessionStrategy": "per-repo",
      "dialecticDepth": 1,
      "dialecticDepthLevels": ["low"],
      "observation": {
        "user": { "observeMe": true, "observeOthers": false },
        "ai": { "observeMe": true, "observeOthers": true }
      }
    }
  },
  "sessions": {
    "/home/user/myproject": "myproject-main"
  }
}
```
