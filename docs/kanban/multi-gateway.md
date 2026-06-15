# Multi-gateway deployment

Hermes supports multiple gateway processes running concurrently — one per profile
(default, writer, admin, coder, researcher). Each gateway opens its own connection
to platform APIs and delivers messages for its profile's subscribers.

## Single-dispatcher posture

Only one gateway owns the kanban dispatcher. The owning gateway keeps
`kanban.dispatch_in_gateway: true` (the default); every other gateway sets it
to `false`.

**Why this matters:** a gateway with `dispatch_in_gateway: true` opens per-board
SQLite connections for both the dispatcher and the notifier watcher. Multiple
gateways doing this concurrently multiplies the open file descriptors on each
`kanban.db` and amplifies WAL `-shm` reader contention. Gating both paths on the
same flag means exactly one process touches the kanban DBs.

## Configuration

On the dispatch-owning gateway (typically the `default` profile), no change is
needed. On every other profile gateway, add to `~/.hermes/config.yaml`:

```yaml
kanban:
  dispatch_in_gateway: false
```

Or set the env var: `HERMES_KANBAN_DISPATCH_IN_GATEWAY=false`

## What each gateway does

| Gateway role | dispatch_in_gateway | Opens per-board DBs? | Runs dispatcher + notifier? |
|---|---|---|---|
| default (dispatch owner) | true (default) | yes | yes |
| writer, admin, coder, etc. | false | no | no |

Non-dispatch gateways still deliver messages for their own platform adapters
(Telegram, Discord, etc.) — they just don't poll kanban boards.
