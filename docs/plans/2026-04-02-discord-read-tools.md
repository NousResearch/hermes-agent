# Discord Read Tools Plan

Date: 2026-04-02

## Scope

Add three read-only Discord tools:

- `discord_list_channels`
- `discord_read_history`
- `discord_search_messages`

## Design Constraints

- Use Discord HTTP API v10 only.
- Keep the feature read-only.
- Enforce explicit allowlists:
  - `DISCORD_READ_ALLOWED_GUILDS`
  - `DISCORD_READ_ALLOWED_CHANNELS`
  - `DISCORD_READ_INCLUDE_DMS`
- Auto-allow the current Discord session target so Hermes can inspect the conversation it is already in.
- Include thread discovery and thread-aware name resolution.
- Hard-cap history and search sizes.
- Return message permalinks when possible.

## Implementation Notes

- Guild allowlists expose readable text channels plus active threads in those guilds.
- Channel allowlists expose specific channels/threads directly; allowed parent text channels also expose active child threads.
- DM discovery is limited to channels Hermes already knows from prior sessions.
- Search scans a bounded recent-message window instead of pretending to be a full-server index.

## Verification Targets

- Tool registration and `get_tool_definitions()` exposure.
- Allowlist denial outside scope.
- Current-session access for guild and DM contexts.
- Thread discovery and deterministic qualified-name resolution.
- Config bridging from `config.yaml` into Discord read env vars.
