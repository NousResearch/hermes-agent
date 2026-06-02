# ADR 001: Multi-Entry Adapter and Session Binding

**Date:** 2026-06-02  
**Status:** Accepted  
**Baseline:** v2.9.1  

---

## Context

Hermes currently uses Feishu Bot ("马尔蒂尼") as an important mobile entrypoint.
However, Feishu's single Bot chat surface causes context pollution and does not
provide a natural multi-session UX.

Discord may provide a better mobile session workspace because categories,
channels, and threads can map naturally to Hermes workspaces and sessions.

Hermes may later evolve into a Mac App, but the app should not become a
separate core. Feishu, Discord, Web Console, CLI, and future Mac App should
all be treated as entry adapters into the same Hermes Core.

Hermes v2.9.1 is the current clean baseline:

- backend tests: 267 passed, 0 failed
- frontend build: zero errors
- tag: v2.9.1
- working tree clean
- task taxonomy, ledger normalization, router explanation, policy dry-run,
  agent capability contracts all complete

---

## Decision

Introduce a **Multi-Entry Adapter** architecture.

All external entrypoints shall normalize inbound messages into a common
`EntryEvent` model before creating tasks or runs.

Supported entrypoints:

- `feishu`
- `discord`
- `web`
- `cli`
- `mac_app` (future)

---

## EntryEvent Model

```
EntryEvent:
  event_id
  entrypoint
  external_source_id
  external_channel_id
  external_thread_id
  external_user_id
  workspace_id
  session_id
  message
  intent
  created_at
  origin_entrypoint
  dedupe_key
```

---

## Session Binding

Hermes shall maintain source-to-session bindings:

| External Source | Maps To |
|----------------|---------|
| Discord category | workspace |
| Discord channel / thread | session |
| Feishu thread | session |
| Web route | session |
| CLI working directory | workspace/session |
| Mac App selected session | session |

---

## Feishu Role

Feishu shall be used primarily for:

- mobile notifications
- approvals
- quick commands
- status checks
- fallback lightweight input

Feishu shall **not** be treated as the primary multi-session UI unless
interactive cards are implemented.

Existing code: `hermes_cli/feishu_martini_bot/main.py` (马尔蒂尼 bot)

---

## Discord Role

Discord shall be evaluated as the primary mobile multi-session workspace.

Mapping:

- Discord guild/server → Hermes deployment
- Discord category → workspace
- Discord channel → session
- Discord thread → sub-session or task thread
- Discord message mentioning Hermes → EntryEvent

---

## Web Console Role

Web Console remains the local inspection and management interface.

It shall eventually expose:

- workspace list
- session list
- session timeline
- scoped task/run/ledger views
- adapter health

Existing entrypoints:

- `http://127.0.0.1:9119/operations`
- `http://127.0.0.1:9119/agents`
- `http://127.0.0.1:9119/runs`

---

## Future Mac App Role

The Mac App shall be a desktop shell over Hermes Core:

- start/stop backend
- show adapter health
- open Web Console
- display local notifications
- manage settings

It shall **not** fork Hermes Core.

---

## Non-Goals

- No automatic demotion
- No automatic agent switching
- No automatic routing weight mutation
- No v3.0 review gate
- No Mac App packaging (yet)
- No cross-posting every message between Feishu and Discord
- No cloud deployment
- No public exposure of local Hermes API

---

## Phased Plan

### Phase 1: Foundation

- Add Workspace and Session models
- Add EntryEvent model
- Add EntryAdapter interface
- Add session binding store
- Add workspace_id/session_id to task/run/ledger
- Preserve legacy compatibility

### Phase 2: Discord Adapter MVP

- Local Discord bot process
- Map category/channel/thread to workspace/session
- Create EntryEvents from messages
- Reply with resolved workspace/session
- No global fallback

### Phase 3: Feishu Adapter Refactor

- Convert Feishu messages into EntryEvents
- Add thread-to-session binding
- Use Feishu for notifications and approvals
- Add ambiguity guard

### Phase 4: Web Console Session UI

- Add `/sessions`
- Add `/sessions/:session_id`
- Add workspace/session filters
- Add adapter health panel

### Phase 5: Mac App Shell

- Wrap Web Console
- Manage backend process
- Show Feishu/Discord adapter status
- Provide local notifications

---

## Success Criteria

- Feishu and Discord can coexist without duplicating core logic
- Context pollution prevented by session binding
- Discord can provide channel-based mobile session UX
- Feishu remains useful for approval and notification
- Future Mac App can reuse the same entry/session model
- Backend tests remain green (267 passed, 0 failed)
- Frontend build remains green (zero errors)
