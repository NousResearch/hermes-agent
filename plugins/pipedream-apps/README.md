# Pipedream Apps

A Desktop **Capabilities → Apps** tab backed by Pipedream Connect, shipped as two
independent halves:

| Half | Path | Responsibility |
| --- | --- | --- |
| Desktop plugin | `desktop-plugins/pipedream-apps/plugin.js` | App catalog, search, categories, Connect dialogs, connected accounts |
| Backend | `plugins/pipedream-apps/dashboard/plugin_api.py` | Project OAuth token, Connect token minting, app/account proxy, ownership-checked delete |

The renderer never receives the Pipedream client secret or the server OAuth access
token. It only receives the short-lived Connect Link it opens in the browser.

## Configuration

Set these in the gateway process environment (for example `~/.hermes/.env`):

```sh
PIPEDREAM_CLIENT_ID=...
PIPEDREAM_CLIENT_SECRET=...
PIPEDREAM_PROJECT_ID=proj_...
PIPEDREAM_ENVIRONMENT=production        # or development
```

Optional: `PIPEDREAM_MCP_BASE` (or `PIPEDREAM_APP_URL` / `NEXT_PUBLIC_APP_URL`) — a
base app URL or a complete `.../api/pipedream/mcp` prefix. When present, a successful
connect ensures `pipedream_<slug>` exists and is enabled through the Dashboard MCP API
(`POST /api/mcp/servers`, `PUT /api/mcp/servers/{name}/enabled`). Disconnecting the
last account for an app disables that entry instead of deleting it. There is no live
reload: the change applies on the next session or gateway restart.

## Install

Run the gateway from this checkout (the backend is discovered as a bundled plugin), or
install both halves into the Hermes home:

```powershell
Copy-Item -Recurse plugins\pipedream-apps $HOME\.hermes\plugins\pipedream-apps
Copy-Item -Recurse desktop-plugins\pipedream-apps $HOME\.hermes\desktop-plugins\pipedream-apps
```

For a **user-source** backend, add `pipedream-apps` to `plugins.enabled` in
`~/.hermes/config.yaml`. The Desktop half is enabled by default; disable it in
**Capabilities → Plugins** if needed.

The Apps tab requires the Desktop core build that includes the
`capabilities.tabs` contribution area. Rebuild/relaunch Desktop after updating.

## Backend routes

Mounted under `/api/plugins/pipedream-apps/` and authenticated by the gateway:

- `GET /config` → `{ configured, environment, mcpBaseUrl }` (no secrets)
- `GET /apps?q=&limit=&after=&before=&category=` → one page of apps + cursors
- `GET /categories`
- `GET /apps/{slug}`
- `POST /connect` `{ externalUserId, appSlug }` → `{ token, connectLinkUrl, expiresAt }`
- `POST /mcp/enable` `{ externalUserId, appSlug }` → ensures `pipedream_<slug>` exists and is
  enabled via the Dashboard MCP API; replaces a stale plugin-owned URL, never touches a
  foreign same-name entry
- `GET /accounts?externalUserId=` → safe accounts (credentials stripped)
- `DELETE /accounts/{accountId}?externalUserId=` → ownership check, then delete; disables
  `pipedream_<slug>` when this was the last account for that app

## Identity

The Desktop plugin uses the signed-in Clerk user id as `external_user_id`, with a
persisted plugin-storage UUID fallback when Clerk is unavailable. Accounts are scoped
to that id, matching the existing Next.js implementation.

## Verification

```sh
scripts/run_tests.sh tests/plugins/test_pipedream_apps_api.py
pnpm --filter unitalk-desktop typecheck
pnpm --filter unitalk-desktop lint
pnpm --filter unitalk-desktop test:ui
```

Manual: configure the env vars, install both halves, restart the gateway/Desktop, open
**Capabilities → Apps**, search/connect an app, finish the browser sign-in, confirm the
account appears, and disconnect it.
