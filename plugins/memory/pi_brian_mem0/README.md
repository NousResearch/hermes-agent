# pi-brian self-hosted Mem0 provider

Memory provider for Hermes that talks to the self-hosted Mem0 REST API already used by `pi-brian`.

## Purpose

- preserve long-tail semantic memory during migration
- keep Hermes hot memory (`USER.md`, `MEMORY.md`) small
- reuse existing `MEM0_BASE_URL` deployment instead of switching to Mem0 cloud

## Config

Environment variables:

- `MEM0_BASE_URL` - self-hosted Mem0 base URL, e.g. `http://127.0.0.1:8000`
- `MEM0_USER_ID` - default fallback user scope for CLI sessions
- `MEM0_AGENT_ID` - default agent identifier for write attribution
- `MEM0_BEARER_TOKEN` - optional bearer token if your self-hosted Mem0 is protected

Optional config file: `$HERMES_HOME/pi_brian_mem0.json`

```json
{
  "base_url": "http://127.0.0.1:8000",
  "user_id": "1176823362",
  "agent_id": "hermes-brian",
  "prefetch_limit": 5,
  "prefetch_chars": 1200,
  "sync_turns": true,
  "request_timeout_seconds": 10
}
```

## Activation

```yaml
memory:
  provider: pi_brian_mem0
```

## Tooling

- `mem0_profile`
- `mem0_search`
- `mem0_conclude`

## Notes

- built-in Hermes hot memory still matters; this provider is for long-tail recall
- gateway user IDs override `MEM0_USER_ID`, so Telegram/Discord users get separate memory scopes automatically
- built-in memory writes are mirrored back into self-hosted Mem0 via `on_memory_write`
