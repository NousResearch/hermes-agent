# Supermemory Memory Provider

Semantic long-term memory with profile recall, semantic search, explicit memory tools, and session-end conversation ingest.

## Requirements

- `pip install supermemory`
- Supermemory API key from [supermemory.ai](https://supermemory.ai)

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

| Tool | Description |
|------|-------------|
| `supermemory_store` | Store an explicit memory |
| `supermemory_search` | Search memories by semantic similarity |
| `supermemory_forget` | Forget a memory by ID or best-match query |
| `supermemory_profile` | Retrieve persistent profile and recent context |

## Behavior

When enabled, Hermes can:

- prefetch relevant memory context before each turn
- store cleaned conversation turns after each completed response
- ingest the full session on session end for richer graph updates
- expose explicit tools for search, store, forget, and profile access

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
- `supermemory_search`, `supermemory_store`, `supermemory_forget`, and `supermemory_profile` accept an optional `container_tag` parameter
- The tag must be in the whitelist: primary container + `custom_containers`
- Automatic operations (turn sync, prefetch, memory write mirroring, session ingest) always use the **primary** container only
- Custom container instructions are injected into the system prompt

## Support

- [Supermemory Discord](https://supermemory.link/discord)
- [support@supermemory.com](mailto:support@supermemory.com)
- [supermemory.ai](https://supermemory.ai)
