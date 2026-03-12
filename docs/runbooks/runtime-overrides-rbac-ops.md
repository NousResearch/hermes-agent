# Runtime Overrides + RBAC Ops Runbook

## Purpose
Operational guide for using and validating thread/session runtime controls and write-command governance from Discord.

## A) Per-query overrides (`/ask`)

### Use high reasoning for one query
```text
/ask reasoning=high <your prompt>
```
Example:
```text
/ask reasoning=high do one last cohesive audit and summarize blockers
```

### Combine provider/model + reasoning for one query
```text
/ask model=openai/gpt-5.3-codex reasoning=high <prompt>
```
or
```text
/ask --provider anthropic --model claude-sonnet-4.5 --reasoning low <prompt>
```

### Expected behavior
- The query runs as normal message content.
- A runtime receipt appears before response payload:
  - model
  - provider
  - reasoning
  - effective task class

## B) Thread/session runtime controls

- `/modelpin [provider:model|clear]` → pin/unpin thread model
- `/reasoning [xhigh|high|medium|low|minimal|none|default]` → thread-local reasoning override
- `/route [auto|command|vision|audio|document|code|analysis|chat]` → force task classification
- `/runtime` → inspect active runtime state

## C) Write-command RBAC

### Environment controls
- `GATEWAY_WRITE_COMMANDS`
  - defaults to: `sethome,set-home,model,update,reload-mcp,reload_mcp`
- `GATEWAY_WRITE_ALLOW_ALL=true|false`
- `GATEWAY_WRITE_ALLOWLIST=<comma-separated user IDs>`
- `<PLATFORM>_WRITE_ALLOWLIST=<comma-separated user IDs>`
  - example: `DISCORD_WRITE_ALLOWLIST=12345,67890`

### Audit trail
Write-command attempts append JSONL lines to:
```text
~/.hermes/logs/gateway_command_audit.jsonl
```
Fields include timestamp, platform, user, command, args, authorized, reason.

## D) Exec-owner status commands

- `/now` → active priorities + cross-app bridge suggestions
- `/blocked` → blocked items and reasons
- `/next` → recommended next actions

These commands are built from artifacts under:
- `~/.hermes/reports/github_sync_latest.json`
- `~/.hermes/kb/twitter_bookmarks_state.json`
- `~/.hermes/kb/twitter_vector_state.json`

## E) Incident checks (quick)

1. **Runtime override not applied**
   - check command format and reasoning value set.
   - valid: `xhigh|high|medium|low|minimal|none`.
2. **Write command denied unexpectedly**
   - verify user ID is in global or platform allowlist.
   - inspect latest line in `gateway_command_audit.jsonl`.
3. **/blocked shows stale snapshot**
   - verify sync jobs are running and artifact mtime is fresh (<15m).
4. **Document attachment not ingested**
   - confirm file size under `DISCORD_MAX_DOCUMENT_BYTES`.
   - check gateway logs for cache failure fallback.
