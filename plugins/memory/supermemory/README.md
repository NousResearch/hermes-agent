# Supermemory Memory Provider

Semantic long-term memory with profile recall, semantic search, explicit memory tools, and full-session conversation ingest (one ingest per session) for richer profiles.

## Requirements

- `pip install supermemory`
- Supermemory API key from [app.supermemory.ai/integrations?connect=hermes](http://app.supermemory.ai/integrations?connect=hermes)

## Setup

```bash
hermes memory setup    # select "supermemory"
```

Or manually:

```bash
hermes config set memory.provider supermemory
echo 'SUPERMEMORY_API_KEY=***' >> ~/.hermes/.env
```

## Config

Config file: `$HERMES_HOME/supermemory.json`

| Key | Default | Description |
|-----|---------|-------------|
| `container_tag` | `hermes` | Container tag used for search and writes. Supports `{identity}` template for profile-scoped tags (e.g. `hermes-{identity}` → `hermes-coder`). |
| `auto_recall` | `true` | Inject relevant memory context before turns |
| `auto_capture` | `true` | Store cleaned user-assistant turns after each response |
| `max_recall_results` | `10` | Max recalled items to format into context |
| `profile_frequency` | `50` | Include profile facts on first turn and every N turns |
| `capture_mode` | `all` | Skip tiny or trivial turns by default |
| `search_mode` | `hybrid` | Search mode: `hybrid` (profile + memories), `memories` (memories only), `documents` (documents only) |
| `entity_context` | built-in default | Extraction guidance passed to Supermemory |
| `api_timeout` | `5.0` | Timeout for SDK and ingest requests |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `SUPERMEMORY_API_KEY` | API key (required) |
| `SUPERMEMORY_CONTAINER_TAG` | Override container tag (takes priority over config file) |

## Tools

Kebab-case names are registered for the agent; snake_case aliases remain supported.

| Tool | Alias | Description |
|------|-------|-------------|
| `supermemory-save` | `supermemory_store` | Store an explicit memory |
| `supermemory-search` | `supermemory_search` | Search memories by semantic similarity |
| `supermemory-forget` | `supermemory_forget` | Forget a memory by ID or best-match query |
| `supermemory-profile` | `supermemory_profile` | Retrieve persistent profile and recent context |

## Source attribution

All Supermemory API calls send `x-sm-source: hermes`, and document writes stamp
`metadata.sm_source: hermes`. This is a **functional routing key, not telemetry**:
it groups Hermes-written memories into a dedicated "Hermes" Space in the
Supermemory app, so you can filter, browse, and bulk-manage them per source agent
(alongside Codex, Claude Code, etc.) from the Supermemory UI.

## Behavior

When enabled, Hermes can:

- prefetch relevant memory context before each turn
- buffer the full conversation and ingest it as **one session** at session end (or on `/reset`, branch, compression, or shutdown)
- ingest the full session to the conversations endpoint for richer profile/graph updates
- expose explicit tools for search, store, forget, and profile access

The session is written once via the conversations endpoint, which drives Supermemory's entity extraction and profile building while keeping a clean, retrievable full transcript.

## Memory Scope Isolation

By default, all conversations sharing the same Hermes profile read and write
the same memory (both built-in files and provider containers). The `memory.scope`
config option enables per-user, per-chat, or per-session isolation:

```yaml
memory:
  scope: conversation
```

| Scope | Behaviour | Persists across `/new`? |
|-------|-----------|------------------------|
| `identity` (default) | Profile-global — all chats share memory | Yes |
| `user` | One namespace per platform user | Yes |
| `conversation` | One namespace per gateway DM/group/channel/topic/thread; local CLI/Desktop uses the durable session | Yes for gateway conversations; local `/new` starts a new conversation |
| `session` | One namespace per durable Hermes session | No — rebinds on session rotation |

When a non-identity scope is active:

- **Built-in memory** (`MEMORY.md` / `USER.md`) is always scoped — stored under `memories/scopes/<scope-suffix>/` instead of the base `memories/` directory.
- **Supermemory** container gets the scope suffix appended (e.g. `hermes_default_a4c981…`). This is the only external provider that currently honours `memory.scope`.
- **Custom containers** (Supermemory multi-container mode) remain unchanged — they are deliberately cross-conversation knowledge spaces.

> **Provider coverage:** Built-in memory always honours scope. External providers must explicitly declare and implement each supported mode. Currently only Supermemory supports non-`identity` scopes; other providers are disabled with an error when a narrower scope is configured so they cannot silently access shared remote memory.

Scope keys are 16-character SHA256-derived pseudonymous hashes of stable identifiers. Raw chat IDs, phone numbers, and user IDs are not written into paths or container tags; low-entropy identifiers should not be treated as anonymized against offline enumeration.

Fallbacks: local CLI/Desktop `conversation` scope uses the durable session ID; local `user` scope has no platform user and therefore falls back to `identity`. If no identifier required by the selected scope is available, Hermes falls back to `identity` with a debug log.

## Profile-Scoped Containers

Use `{identity}` in the `container_tag` to scope memories per Hermes profile:

```json
{
  "container_tag": "hermes-{identity}"
}
```

For a profile named `coder`, this resolves to `hermes-coder`. The default profile resolves to `hermes-default`. Without `{identity}`, all profiles share the same container.

## Multi-Container Mode

For advanced setups (e.g. OpenClaw-style multi-workspace), you can enable custom container tags so the agent can read/write across multiple named containers:

```json
{
  "container_tag": "hermes",
  "enable_custom_container_tags": true,
  "custom_containers": ["project-alpha", "project-beta", "shared-knowledge"],
  "custom_container_instructions": "Use project-alpha for coding tasks, project-beta for research, and shared-knowledge for team-wide facts."
}
```

When enabled:
- `supermemory-search`, `supermemory-save`, `supermemory-forget`, and `supermemory-profile` accept an optional `container_tag` parameter
- The tag must be in the whitelist: primary container + `custom_containers`
- Automatic operations (turn sync, prefetch, memory write mirroring, session ingest) always use the **primary** container only
- Custom container instructions are injected into the system prompt

## Support

- [Supermemory Discord](https://supermemory.link/discord)
- [support@supermemory.com](mailto:support@supermemory.com)
- [supermemory.ai](https://supermemory.ai)
