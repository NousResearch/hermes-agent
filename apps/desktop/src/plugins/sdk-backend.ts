/**
 * Desktop plugin SDK backend shim.
 *
 * The web dashboard is served BY the backend (same origin), so its plugin SDK
 * pulls `fetchJSON` / `authedFetch` / `buildWsUrl` straight out of
 * `web/src/lib/api.ts`, which `fetch`es same-origin and reads the session token
 * from `window.__HERMES_SESSION_TOKEN__`.
 *
 * The Electron renderer has NEITHER: it runs from `file://` (packaged) or the
 * Vite dev server (`http://127.0.0.1:5174`), while the backend is a SEPARATE
 * origin (`http://127.0.0.1:<port>`). The backend's absolute origin + session
 * token live behind `window.hermesDesktop.getConnection()` →
 * `HermesConnection { baseUrl, token, authMode, wsUrl }` (see global.d.ts).
 *
 * This module reimplements the three plugin-facing helpers against that
 * connection so the SAME unmodified plugin bundles route correctly:
 *  - `fetchJSON`   → delegates to `window.hermesDesktop.api` (handles token +
 *                    oauth cookie in one place, in Electron main).
 *  - `authedFetch` → raw cross-origin `fetch` to `${baseUrl}${path}` with the
 *                    `X-Hermes-Session-Token` header (uploads / blob downloads).
 *  - `buildWsUrl`  → builds a `ws(s)://` URL off `connection.wsUrl` /
 *                    `getGatewayWsUrl()` with the session token query param.
 */

import type { HermesConnection } from "@/global";

// Matches web/src/lib/api.ts (SESSION_HEADER) and apps/desktop/src/hermes.ts.
const SESSION_HEADER = "X-Hermes-Session-Token";

// Cache the resolved connection. The primary (window) backend descriptor is
// stable for the window's lifetime; kanban v1 targets the primary. A profile
// switch would relaunch the renderer (see global.d.ts: profile.set reloads the
// window), so this cache never goes stale within a window session.
let _connPromise: Promise<HermesConnection> | null = null;

function getConnection(): Promise<HermesConnection> {
  if (!_connPromise) {
    _connPromise = window.hermesDesktop.getConnection();
  }

  return _connPromise;
}

/** Absolute backend origin (e.g. `http://127.0.0.1:<port>`) for the primary backend. */
export async function getBackendBaseUrl(): Promise<string> {
  const conn = await getConnection();

  return conn.baseUrl;
}

/**
 * JSON `fetch` for dashboard `/api/...` endpoints. Routes through
 * `window.hermesDesktop.api`, which performs the request against
 * `${connection.baseUrl}${path}` with the session token (and oauth-cookie mode)
 * handled in Electron main. Throws `Error("<status>: <body>")` shapes via the
 * IPC layer, matching the web `fetchJSON` contract closely enough for plugins.
 */
export async function fetchJSON<T = unknown>(
  url: string,
  init?: RequestInit,
): Promise<T> {
  const method = (init?.method ?? "GET").toUpperCase();
  let body: unknown;

  if (init?.body != null) {
    if (typeof init.body === "string") {
      try {
        body = JSON.parse(init.body);
      } catch {
        body = init.body;
      }
    } else {
      body = init.body;
    }
  }

  return window.hermesDesktop.api<T>({ path: url, method, body });
}

/**
 * Authenticated raw `fetch` for NON-JSON endpoints (uploads via `FormData`,
 * binary/blob downloads). Mirrors web `authedFetch`: returns the raw `Response`,
 * does not parse, does not throw on non-2xx. Cross-origin to the renderer
 * (file:// or 5174) → backend (127.0.0.1:<port>); auth via the session-token
 * header (token mode). OAuth-cookie mode rides along via `credentials`.
 *
 * v1 SCOPE (loopback/token-first): in OAuth-remote mode the authenticated
 * HttpOnly cookies live in the dedicated `persist:hermes-remote-oauth` Electron
 * session, which this renderer-side `fetch` (default partition) cannot attach —
 * so plugin raw endpoints (e.g. kanban attachment upload/download) would 401
 * under an OAuth-remote gateway. v1 targets the loopback/token backend where
 * the header auth above is sufficient; an OAuth-partitioned main-process raw
 * fetch IPC is a documented follow-up. (review: codex P2)
 */
export async function authedFetch(
  url: string,
  init?: RequestInit,
): Promise<Response> {
  const conn = await getConnection();
  const headers = new Headers(init?.headers);

  if (conn.token && !headers.has(SESSION_HEADER)) {
    headers.set(SESSION_HEADER, conn.token);
  }

  return fetch(`${conn.baseUrl}${url}`, {
    ...init,
    headers,
    credentials: init?.credentials ?? "include",
  });
}

/**
 * Build an absolute `ws(s)://` URL for a dashboard WebSocket endpoint with the
 * session token query param. The desktop backend exposes its WS base via
 * `connection.wsUrl` / `getGatewayWsUrl()`. Loopback/token mode appends
 * `?token=<token>` (the same param the gateway accepts in non-gated mode).
 *
 * `path` is the dashboard-relative path (e.g. `/api/plugins/kanban/events`);
 * the host + scheme come from the backend connection, NOT the renderer origin.
 */
export async function buildWsUrl(
  path: string,
  params?: Record<string, string>,
): Promise<string> {
  const [authName, authValue] = await buildWsAuthParam();
  const conn = await getConnection();
  // Derive scheme + host from the absolute backend origin so the URL points at
  // the backend, never at the file:// / 5174 renderer origin.
  const httpBase = new URL(conn.baseUrl);
  const wsProto = httpBase.protocol === "https:" ? "wss:" : "ws:";
  const qs = new URLSearchParams(params ?? {});
  qs.set(authName, authValue);

  // Preserve any path prefix on the backend origin (e.g. a remote gateway
  // mounted at https://host/hermes) so the socket lands at /hermes/api/...,
  // mirroring how REST calls build `${baseUrl}${path}`. Without this, plugin
  // sockets behind a reverse proxy / prefixed deployment connect to the wrong
  // path. (review: codex P2)
  const prefix = httpBase.pathname.replace(/\/+$/, "");

  return `${wsProto}//${httpBase.host}${prefix}${path}?${qs}`;
}

/** Resolve the `[authParamName, authParamValue]` pair for a WS connect. */
export async function buildWsAuthParam(): Promise<[string, string]> {
  const conn = await getConnection();

  // v1 SCOPE (loopback/token-first): an OAuth-gated remote backend needs a
  // single-use ws-ticket (POST /api/auth/ws-ticket via the OAuth session)
  // rather than ?token=. The desktop's loopback backend uses token mode, so v1
  // targets that; plugin WebSockets under an OAuth-remote gateway are a
  // documented follow-up (route through the main-process OAuth session, as the
  // web SDK does). Returning the (possibly empty) token keeps loopback working.
  // (review: codex P2)
  return ["token", conn.token ?? ""];
}

/**
 * Minimal `api` object for the SDK surface. The web SDK exposes the full
 * `web/src/lib/api.ts` client; kanban only needs `getPlugins` (the loader fetch
 * goes through usePlugins directly), so this is intentionally small. Plugins
 * that need more call `fetchJSON` against their own `/api/plugins/<name>/...`.
 */
export const api = {
  getPlugins: () => fetchJSON<unknown[]>("/api/dashboard/plugins"),
};
