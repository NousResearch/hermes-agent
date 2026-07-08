# HT AI Agent — Web frontend (`ht-web`)

A custom, brand-owned web chat UI for HT AI Agent. It talks to the existing
agent gateway over the same JSON-RPC 2.0 WebSocket protocol the terminal UI,
desktop app, and dashboard use — so the Python core needs no changes.

Unlike `web/` (the management dashboard, which inherits the Nous Research
design system), `ht-web` owns its entire look: plain React + Tailwind, no
`@nous-research/ui` dependency. That independence is the point.

## Architecture

```
apps/ht-web (React 19 + Vite + Tailwind)
    │  JSON-RPC 2.0 over WebSocket (/api/ws)
    ▼
ht serve  →  tui_gateway  →  agent core
```

- Protocol client: `@hermes/shared` (`JsonRpcGatewayClient`, `buildHermesWebSocketUrl`).
- Protocol contract: [`docs/ht-web-gateway-protocol.md`](../../docs/ht-web-gateway-protocol.md) — the MVP method/event subset, extracted from `ui-tui/src/gatewayTypes.ts`.
- The correctness core is `src/gateway/chatReducer.ts` (pure, no React/DOM) — it turns the gateway event stream into render state and is exhaustively unit-tested.

## Layout

| Path | Role |
|---|---|
| `src/gateway/types.ts` | MVP protocol types |
| `src/gateway/chatReducer.ts` | Pure event → chat-state reducer (unit-tested) |
| `src/gateway/skin.ts` | Gateway skin payload → brand + CSS vars (server-driven branding) |
| `src/gateway/useGateway.ts` | React hook: connection, sessions, submit/interrupt/respond |
| `src/components/*` | Message list, composer, tool activity, approval/clarify dialogs, session sidebar |
| `src/App.tsx` | Shell wiring the hook to the components |

## Develop

```bash
# 1. Start the agent gateway (headless) in one terminal:
ht serve --host 127.0.0.1 --port 9119

# 2. Start the dev server (proxies /api → the gateway):
npm run dev --workspace apps/ht-web
#   Override the gateway URL with HT_GATEWAY_URL=http://host:port
```

## Verify

```bash
npm run typecheck --workspace apps/ht-web   # tsc, no emit
npm run test --workspace apps/ht-web        # vitest (reducer + skin + component)
npm run build --workspace apps/ht-web       # production build → dist/
```

## Chat (Phase 3, M1–M4)

Connect + skin handshake, streaming chat with client-side Markdown, tool-call
activity feed, approval/clarify dialogs, and session list/resume/new.

## Management pages (Phase 4)

The app is a shell: a left nav rail routes between **Chat** and the REST-backed
management pages below. The gateway WebSocket connection lives in
`GatewayContext` above the router, so chat state survives navigating away and
back. All 19 dashboard pages are ported except the embedded PTY terminal
(ht-web has its own native chat) and DocsPage (a static iframe).

| Path | Page | Endpoints |
|---|---|---|
| `/` | Chat | `/api/ws` (gateway) |
| `/sessions` | Sessions — list, rename, delete | `/api/sessions*` |
| `/models` | Models — current model + provider/model picker | `/api/model/{info,options,set}` |
| `/skills` | Skills + Toolsets — toggles, hub search/install | `/api/skills*`, `/api/tools/toolsets` |
| `/plugins` | Plugins — dashboard + agent-plugin hub | `/api/dashboard/plugins*` |
| `/mcp` | MCP — servers, test, catalog install | `/api/mcp/*` |
| `/cron` | Cron — jobs, pause/resume/trigger, create | `/api/cron/*` |
| `/channels` | Channels — messaging platform config + test | `/api/messaging/platforms*` |
| `/webhooks` | Webhooks — routes CRUD, enable | `/api/webhooks*` |
| `/pairing` | Pairing — approve/revoke devices | `/api/pairing*` |
| `/profiles` | Profiles — CRUD + SOUL.md editor | `/api/profiles*` |
| `/files` | Files — managed-file browser (upload/mkdir/delete) | `/api/files*` |
| `/config` | Config — raw `config.yaml` editor | `/api/config/raw` |
| `/env` | Env & Keys — CRUD + reveal | `/api/env*` |
| `/analytics` | Analytics — token/cost usage + per-model | `/api/analytics/*` |
| `/logs` | Logs — level/line filters | `/api/logs` |
| `/system` | System — gateway + host status | `/api/status`, `/api/system/stats` |

Deferred within ported pages: Telegram/WhatsApp guided onboarding wizards on
Channels (multi-step polling flows) and the profile-builder wizard — the core
CRUD for each is present.

### REST layer

- `src/api/client.ts` — base client (session-token header, base path, gated
  cookie path, 401 → login redirect). Unit-tested with a mocked `fetch`.
- `src/api/endpoints.ts` — typed wrappers + response types, mirroring
  `web/src/lib/api.ts` (source of truth) for the subset the pages use.

### Porting the next page (the pattern)

1. Add the response type + a typed wrapper in `src/api/endpoints.ts`.
2. Create `src/pages/<Name>Page.tsx` using `ManagementPage` + `useResource` +
   `ResourceView` from `src/components/PageScaffold.tsx` for the
   loading/error/empty lifecycle.
3. Register it in `src/app/nav.ts` (one `NavItem`) and add the lazy `<Route>`
   in `src/App.tsx`.

The client provides the shared building blocks pages reuse: `apiGet/Post/Put/
Delete`, `apiFetch`, `authedFetch` (raw response for blobs/FormData —
Files upload), and `getActionStatus`/`pollAction` for async ops (skills/plugin
hub installs stream progress through it).

## Out of scope

Voice, billing overlays, subagent spawn-tree view, the PTY-embedded terminal
(ht-web has its own native chat), and the dashboard plugin system.
