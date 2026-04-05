# Honcho Memory Provider

AI-native cross-session user modeling with dialectic Q&A, semantic search, peer cards, and persistent conclusions.

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
echo "HONCHO_API_KEY=your-key" >> ~/.hermes/.env
```

## Config Resolution

Config is read from the first file that exists:

| Priority | Path | Scope |
|----------|------|-------|
| 1 | `$HERMES_HOME/honcho.json` | Profile-local (isolated Hermes instances) |
| 2 | `~/.hermes/honcho.json` | Default profile (shared host blocks) |
| 3 | `~/.honcho/config.json` | Global (cross-app interop) |

Host key is derived from the active Hermes profile: `hermes` (default) or `hermes.<profile>`.

## Tools

| Tool | LLM call? | Description |
|------|-----------|-------------|
| `honcho_profile` | No | User's peer card -- key facts snapshot |
| `honcho_search` | No | Semantic search over stored context (800 tok default, 2000 max) |
| `honcho_context` | Yes | LLM-synthesized answer via dialectic reasoning |
| `honcho_conclude` | No | Write a persistent fact about the user |

Tool availability depends on `recallMode`: hidden in `context` mode, always present in `tools` and `hybrid`.

## Full Configuration Reference

### Identity & Connection

| Key | Type | Default | Scope | Description |
|-----|------|---------|-------|-------------|
| `apiKey` | string | -- | root / host | API key. Falls back to `HONCHO_API_KEY` env var |
| `baseUrl` | string | -- | root | Base URL for self-hosted Honcho. Local URLs (`localhost`, `127.0.0.1`, `::1`) auto-skip API key auth |
| `environment` | string | `"production"` | root / host | SDK environment mapping |
| `enabled` | bool | auto | root / host | Master toggle. Auto-enables when `apiKey` or `baseUrl` present |
| `workspace` | string | host key | root / host | Honcho workspace ID |
| `peerName` | string | -- | root / host | User peer identity |
| `aiPeer` | string | host key | root / host | AI peer identity |

### Memory & Recall

| Key | Type | Default | Scope | Description |
|-----|------|---------|-------|-------------|
| `recallMode` | string | `"hybrid"` | root / host | `"hybrid"` (auto-inject + tools), `"context"` (auto-inject only, tools hidden), `"tools"` (tools only, no injection). Legacy `"auto"` normalizes to `"hybrid"` |
| `observationMode` | string | `"directional"` | root / host | Shorthand preset: `"directional"` (all on) or `"unified"` (shared pool). Use `observation` object for granular control |
| `observation` | object | -- | root / host | Per-peer observation config (see below) |

#### Observation (granular)

Maps 1:1 to Honcho's per-peer `SessionPeerConfig`. Set at root or per host block -- each profile can have different observation settings. When present, overrides `observationMode` preset.

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

Presets for `observationMode`:
- `"directional"` (default): all four booleans `true`
- `"unified"`: user `observeMe=true`, AI `observeOthers=true`, rest `false`

Per-profile example -- coder profile observes the user but user doesn't observe coder:

```json
"hosts": {
  "hermes.coder": {
    "observation": {
      "user": { "observeMe": true, "observeOthers": false },
      "ai":   { "observeMe": true, "observeOthers": true }
    }
  }
}
```

Settings changed in the [Honcho dashboard](https://app.honcho.dev) are synced back on session init.

### Write Behavior

| Key | Type | Default | Scope | Description |
|-----|------|---------|-------|-------------|
| `writeFrequency` | string or int | `"async"` | root / host | `"async"` (background thread), `"turn"` (sync per turn), `"session"` (batch on end), or integer N (every N turns) |
| `saveMessages` | bool | `true` | root / host | Whether to persist messages to Honcho API |

### Session Resolution

| Key | Type | Default | Scope | Description |
|-----|------|---------|-------|-------------|
| `sessionStrategy` | string | `"per-directory"` | root / host | `"per-directory"`, `"per-session"` (new each run), `"per-repo"` (git root name), `"global"` (single session) |
| `sessionPeerPrefix` | bool | `false` | root / host | Prepend peer name to session keys |
| `sessions` | object | `{}` | root | Manual directory-to-session-name mappings: `{"/path/to/project": "my-session"}` |

### Token Budgets & Dialectic

| Key | Type | Default | Scope | Description |
|-----|------|---------|-------|-------------|
| `contextTokens` | int | SDK default | root / host | Token budget for `context()` API calls. Also gates prefetch truncation (tokens x 4 chars) |
| `dialecticReasoningLevel` | string | `"low"` | root / host | Base reasoning level for `peer.chat()`: `"minimal"`, `"low"`, `"medium"`, `"high"`, `"max"` |
| `dialecticDynamic` | bool | `true` | root / host | Auto-bump reasoning based on query length: `<120` chars = base level, `120-400` = +1, `>400` = +2 (capped at `"high"`). Set `false` to always use `dialecticReasoningLevel` as-is |
| `dialecticMaxChars` | int | `600` | root / host | Max chars of dialectic result injected into system prompt |
| `dialecticMaxInputChars` | int | `10000` | root / host | Max chars for dialectic query input to `peer.chat()`. Honcho cloud limit: 10k |
| `messageMaxChars` | int | `25000` | root / host | Max chars per message sent via `add_messages()`. Messages exceeding this are chunked with `[continued]` markers. Honcho cloud limit: 25k |

### Cost Awareness (Advanced)

These are read from the root config object, not the host block. Must be set manually in `honcho.json`.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `injectionFrequency` | string | `"every-turn"` | `"every-turn"` or `"first-turn"` (inject context only on turn 0) |
| `contextCadence` | int | `1` | Minimum turns between `context()` API calls |
| `dialecticCadence` | int | `1` | Minimum turns between `peer.chat()` API calls |
| `reasoningLevelCap` | string | -- | Hard cap on auto-bumped reasoning: `"minimal"`, `"low"`, `"mid"`, `"high"` |

### Hardcoded Limits (Not Configurable)

| Limit | Value | Location |
|-------|-------|----------|
| Search tool max tokens | 2000 (hard cap), 800 (default) | `__init__.py` handle_tool_call |
| Peer card fetch tokens | 200 | `session.py` get_peer_card |

## Config Precedence

For every key, resolution order is: **host block > root > env var > default**.

Host key derivation: `HERMES_HONCHO_HOST` env > active profile (`hermes.<profile>`) > `"hermes"`.

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
  "apiKey": "your-key",
  "workspace": "hermes",
  "peerName": "eri",
  "hosts": {
    "hermes": {
      "enabled": true,
      "aiPeer": "hermes",
      "workspace": "hermes",
      "peerName": "eri",
      "recallMode": "hybrid",
      "observation": {
        "user": { "observeMe": true, "observeOthers": true },
        "ai": { "observeMe": true, "observeOthers": true }
      },
      "writeFrequency": "async",
      "sessionStrategy": "per-directory",
      "dialecticReasoningLevel": "low",
      "dialecticMaxChars": 600,
      "saveMessages": true
    },
    "hermes.coder": {
      "enabled": true,
      "aiPeer": "coder",
      "workspace": "hermes",
      "peerName": "eri",
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
