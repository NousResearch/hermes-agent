# Live Glass — Design Proposal

**Status:** Draft
**Branch:** `feat/live-glass-plugin`
**PR:** https://github.com/NousResearch/hermes-agent/pull/41542

## Summary

Live Glass is a plugin-first observability feature for `computer_use` that provides
a live window into desktop-control activity. It emits three event types — `frame`,
`log`, and `approval_request` — through an in-process event bus, and renders them
on the dashboard and messaging platforms without modifying Hermes core code.

## Motivation

`computer_use` is powerful but opaque. Users currently cannot see what the agent
is doing on the desktop in real time. Live Glass fills this gap: a dashboard tab
shows the live screenshot, an event log tracks every action, and approval prompts
render with inline controls on messaging platforms.

## Architecture

### Plugin-first design

The entire feature is a single bundled PluginManager plugin at
`plugins/observability/live_glass/`. Zero core edits were required:

```
plugins/observability/live_glass/
├── __init__.py              # Event bus + hook registration
├── plugin.yaml              # Plugin manifest
├── approval_bridge.py       # Bridges computer_use approval → PluginManager hooks
├── frame_poller.py          # Active desktop capture poller
├── adapters/
│   ├── telegram.py          # Telegram (photo + inline keyboard)
│   ├── discord_slack.py     # Discord (file + components) + Slack (Block Kit)
│   └── bluebubbles.py       # iMessage (image + reply-to-approve text)
└── dashboard/
    ├── manifest.json        # Hidden tab registration
    ├── plugin_api.py        # WebSocket event stream endpoint
    └── dist/index.js        # Live view UI (viewport + log + approval controls)
```

### Event bus

Three event types, all JSON-serializable:

| Event | Source | Payload |
|-------|--------|---------|
| `frame` | `post_tool_call` (computer_use captures) + frame poller | image_url, mime_type, mode, dimensions, summary |
| `log` | `post_tool_call` (all tools) + `post_approval_response` | tool_name, args, status, duration_ms, error |
| `approval_request` | `pre_approval_request` hook (via approval bridge) | command, description, surface, pattern_keys |

Public API: `publish()`, `subscribe(…, replay=True)`, `get_events(event_type, since_sequence, limit)`.

### Key integration seams

1. **PluginManager hooks** — `post_tool_call`, `pre_approval_request`, `post_approval_response`.
   These were already defined in `hermes_cli/plugins.py::VALID_HOOKS`. No changes needed.

2. **Approval bridge** — `computer_use` has its own approval callback path that did not
   previously fire the shared PluginManager approval hooks. The bridge wraps the
   callback at plugin registration time, firing `pre_approval_request` / `post_approval_response`
   without changing who makes the decision. The bridge is idempotent (won't double-wrap)
   and observer-only.

3. **Dashboard WebSocket** — the dashboard already streams agent events over WebSocket
   (`/api/pub` + `/api/events`). The plugin registers its own `@router.websocket("/events")`
   under `/api/plugins/live-glass/events` using the existing dashboard plugin API.
   No core dashboard edits.

4. **Platform adapters** — each adapter is a standalone class that subscribes to the
   event bus and calls a pluggable sender interface. The gateway wires the real
   Telegram/Discord/Slack/BlueBubbles client. No core gateway edits.

## Privacy and security

- **Session scoping:** events carry a `session_id`. Platform adapters only send to the
  chat/channel mapped to that session. Unmapped sessions are silently skipped.
- **Observer-only approvals:** the plugin observes approval prompts but never approves
  or denies. The approval decision stays with the existing `tools/approval.py` flow.
- **No extra desktop access:** the primary frame source extracts screenshots from
  already-produced `computer_use` multimodal tool results. The frame poller is optional
  and uses the existing `computer_use` backend with throttling.
- **Exception isolation:** all subscriber and sender exceptions are caught and logged.
  A failing adapter cannot break the agent loop.
- **Redaction:** event payloads are deep-copied and JSON-serialized. No raw tool output
  leaks through the event bus.

## Test coverage

86 tests across 8 test files covering:
- Event bus: publish, subscribe, replay, filtering, bounded retention
- Approval bridge: wrapping, idempotency, all verdicts, exception handling
- Dashboard WebSocket: connect, heartbeat, frame replay, concurrent clients, disconnect
- All 4 platform adapters: frame/log/approval rendering, unmapped sessions, send failures
- Frame poller: poll cycle, backend exceptions, start/stop idempotency, minimum interval
- Plugin discovery: manifest validation, PluginManager opt-in loading
- Packaging: plugin.yaml and __init__.py presence

## Questions for maintainers

1. **Gateway built-in hook slot:** `gateway/builtin_hooks/` has an empty
   `_register_builtin_hooks()` reserved for future always-on hooks. Should the
   live-glass event bus be promoted from a user-enabled plugin to a built-in hook
   after the prototype proves useful? This would let the gateway publish events
   without requiring users to opt in via `plugins.enabled`.

2. **Dashboard plugin API stability:** the current `dashboard/plugin_api.py` +
   `dashboard/manifest.json` pattern worked without issue. Is this API considered
   stable for third-party plugins, or should we document known limitations
   (e.g., project-plugin backend routes are intentionally skipped)?

3. **computer_use approval hook gap:** the approval bridge wraps the callback at
   the Python level, which works but is fragile (depends on module-level state).
   Would maintainers accept a small patch to `tools/computer_use/tool.py` that
   fires PluginManager hooks natively, making the bridge unnecessary?

## Next steps

- [ ] Maintainer feedback on this proposal
- [ ] If accepted: mark the dashboard tab as `hidden: false` (unhide for production)
- [ ] If accepted: add gateway wiring for platform adapters
- [ ] CI: add live-glass tests to the main test matrix
