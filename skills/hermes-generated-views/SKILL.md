---
name: hermes-generated-views
description: Create safe, persistent, movable desktop views for Hermes without a plugin.
version: 1.0.0
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [desktop, views, artifacts, ui]
    category: productivity
    related_skills: []
---

# Hermes Generated Views

Create a small interactive desktop surface that appears in **Artifacts → Views**
and opens as a normal draggable, splittable, tabbable Hermes pane. Generated
views are plain HTML documents, not plugins: they run behind explicit user
approval in an opaque-origin iframe with no network access.

## Use this when

- The user wants a dashboard, monitor, status card, report, checklist, small
  interactive tool, or durable visual answer inside Hermes Desktop.
- The view only needs its own UI, optional theme tokens, bounded per-view state,
  and one of the fixed read-only Hermes data bindings.

Use a desktop plugin instead only when the user explicitly needs trusted host
authority such as commands, arbitrary gateway methods, navigation, sockets, or
new contribution types.

## Location and document

Write one directory under the active profile's Hermes home:

```text
$HERMES_HOME/generated-views/<id>/
├── view.json
└── index.html
```

`<id>` must match `^[a-z0-9][a-z0-9_-]{0,63}$` and must equal `view.json.id`.
Resolve the active `$HERMES_HOME` from `hermes status` or the running profile;
do not assume `~/.hermes` when a named profile is active.

```json
{
  "version": 1,
  "id": "usage-monitor",
  "title": "Usage Monitor",
  "entry": "index.html",
  "capabilities": [],
  "bindings": []
}
```

Allowed capabilities:

- `theme:read` — request the current Hermes theme tokens.
- `state:persist` — read/write up to 64 KB of JSON state, host-bound to this view.

Allowed read-only bindings:

- `hermes:status` — sanitized gateway state, active-session count, platform
  states, version, and release date.
- `hermes:usage-30d` — 30-day totals, daily/model usage, skill summary, and
  tool counts.

Anything else is rejected. A view never receives filesystem, network, plugin
SDK, general REST/gateway, socket, model, or navigation authority.

## Procedure

1. Start from the smallest matching template in this skill's `templates/`:
   `single-file`, `themed-stateful`, or `usage-monitor`.
2. Copy both files into `$HERMES_HOME/generated-views/<id>/`, then update the id,
   title, content, capabilities, and bindings. Keep everything inline: CDN and
   sibling asset loads are blocked.
3. Make the UI responsive to pane resizing. Prefer CSS grid/flex, semantic HTML,
   and a useful empty/loading/error state.
4. Tell the user to open **Artifacts → Views**, inspect the declared authority,
   and click **Open view**. The first run shows an exact-content approval gate.
5. After edits, expect the live iframe to disappear. This is correct: source,
   capability, or binding changes invalidate the digest and require approval of
   the new bytes.

The desktop discovers new directories within a few seconds and watches existing
`view.json` and entry files. No plugin reload or app restart is required.

## Bridge

Child requests are `postMessage` envelopes with `{ v: 1, type, requestId }`:

- `hermes:getTheme` → `hermes:theme { tokens }`
- `hermes:getState` → `hermes:state { state, version }`
- `hermes:setState { state }` → `hermes:state { state, version }`
- `hermes:getData { bindingId }` → `hermes:data { bindingId, data }`

Errors answer as `hermes:error { requestId, code, message }`. Unknown or
malformed messages are silently dropped. In the child, accept replies only when
`event.source === window.parent`, `event.data.v === 1`, and the `requestId`
matches your pending request. See the themed and usage templates for a complete
helper.

## Verification

- The card appears in Artifacts → Views with the expected id, entry,
  capabilities, bindings, and digest prefix.
- No iframe exists until the user approves the exact digest.
- The approved view opens as `generated-view:<id>` and can be dragged, split,
  tabbed, closed, and reopened without losing its user-chosen layout position.
- Theme/state/data requests outside the manifest are denied.
- Saving either file removes the running frame and returns to approval.
- Removing the view directory removes its contribution and any stale layout-tree
  entry on the next discovery pass.
