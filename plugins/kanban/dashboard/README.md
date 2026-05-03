# Hermes Kanban dashboard plugin

The `/kanban` tab serves two stacked governance views:

1. **Top — Kanban (Class A autonomous tasks).** Backed by `~/.hermes/kanban.db`
   via this plugin's REST + WebSocket API at `/api/plugins/kanban/*`.
2. **Bottom — Mission Control (Class B human-paired strategic cards).** Backed
   by Convex at `https://uncommon-emu-480.convex.cloud`, polled every 30s
   directly from the browser.

## Mission Control data source

The Convex query is hard-coded in `dist/index.js`:

```js
const MC_CONVEX_URL = "https://uncommon-emu-480.convex.cloud";
const MC_QUERY_PATH = "tasks:list";
const MC_POLL_MS = 30000;
```

Read-only, no auth — the deployment exposes `tasks:list` as a public query and
the response includes `title`, `description`, `status`, `assignee`, `priority`,
`updatedAt`, `projectId`. The section keeps its own client-side filters
(status, assignee, search) and never writes back; status changes flow through
the existing `mc-complete` CLI / Discord channels.

### Updating when the Convex schema changes

If the Convex deployment renames the query or adds/removes fields:

1. Confirm the new shape with a probe — e.g.
   `curl -sS -X POST $MC_CONVEX_URL/api/query -H 'Content-Type: application/json' -d '{"path":"tasks:list","args":{},"format":"json"}' | jq '.value[0]'`.
2. If the path is renamed (e.g. `strategic:list`), edit `MC_QUERY_PATH` at the
   top of the `MissionControlSection` block in `dist/index.js`.
3. If new fields land that you want surfaced (e.g. `class`, `tier`, `tags`),
   extend the `<thead>` and the row rendering in `MissionControlSection`. Pill
   classes for new statuses can be added under
   `dist/style.css` → `.hermes-mc-pill-<status>`.
4. If the deployment URL changes, update `MC_CONVEX_URL`.

The plugin is plain IIFE JS — no build step. After editing `dist/index.js`
or `dist/style.css`, the running dashboard picks it up on browser refresh
(the gateway serves the files directly).

## Auth (Cloudflare Access)

The dashboard is fronted by Cloudflare Tunnel at `https://kanban.tilos.com` and
gated by a Cloudflare Access self-hosted application
(`Hermes Kanban Dashboard`, app id `95ca4f40-2fb4-44c5-9f1b-b8394a1e6bd2`,
allowed IdP: Google `51b301f3-f75a-4c1c-9e9d-8f3f1ba7d40e`, allow-list
policy `info@tilos.com`). Unauthenticated GETs return a 302 to
`https://tilos.cloudflareaccess.com/cdn-cgi/access/login/kanban.tilos.com`.
The Convex API is public read — no Access cookie is required for the MC
section to load once the user is past the gate.
