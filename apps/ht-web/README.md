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

## MVP scope (Phase 3, M1–M4)

Connect + skin handshake, streaming chat with client-side Markdown, tool-call
activity feed, approval/clarify dialogs, and session list/resume/new. Out of
scope for the MVP: the management pages (see `web/`), voice, billing overlays,
subagent spawn-tree view, and the PTY-embedded terminal.
