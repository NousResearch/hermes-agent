/**
 * gateway-ws-origin.ts
 *
 * Same-origin rewrite for the renderer's gateway WebSocket upgrades.
 *
 * The dashboard applies a DNS-rebinding Host/Origin guard to WebSocket
 * upgrades (web_server.py `_ws_host_origin_reason`): when a browser Origin
 * header is present, it must target the bound dashboard host, or the upgrade
 * is refused with 403 BEFORE `accept()`. Chromium stamps the window's web
 * origin on every WS handshake and the renderer cannot override it, so a
 * renderer served from anywhere other than the gateway host can never pass
 * that check:
 *
 *   - dev (`npm run dev`): the window is the Vite server, so the handshake
 *     carries `Origin: http://127.0.0.1:5174` — a remote gateway bound to
 *     another host closes the socket pre-accept and the app surfaces the
 *     opaque "Could not connect to Hermes gateway". Every REST call keeps
 *     working (those route through the main process's `net`, which sends no
 *     Origin), which makes the failure look like a gateway-side WS bug.
 *   - packaged: the window is `file://`, a non-web origin the server-side
 *     guard already exempts — which is why this only bites source checkouts.
 *
 * The desktop shell owns its gateway connection (the credential — the
 * single-use ticket minted over the authenticated cookie session — is the
 * real auth boundary), so presenting the gateway's own origin is the honest
 * native-client behavior and keeps the guard meaningful for real browsers.
 *
 * Kept electron-free (same pattern as connection-config.ts) so the rewrite
 * decision is unit-testable; main.ts wires it into
 * `session.defaultSession.webRequest.onBeforeSendHeaders`.
 */

// Gateway WS endpoints (web_server.py): the JSON-RPC sidecar plus the
// chat-tab broadcast pair. Everything else — including arbitrary ws:// URLs a
// session might embed — passes through untouched.
const GATEWAY_WS_PATH_RE = /\/api\/(ws|pub|events)$/

/**
 * The same-origin value for a gateway WS endpoint URL, or null when the URL
 * is not a gateway WS upgrade (wrong scheme or path) and must not be touched.
 */
function gatewayWsOrigin(rawUrl) {
  let parsed

  try {
    parsed = new URL(String(rawUrl || ''))
  } catch {
    return null
  }

  if (parsed.protocol !== 'ws:' && parsed.protocol !== 'wss:') {
    return null
  }

  // Strip trailing slashes so `/api/ws/` matches too; path prefixes
  // (`/hermes/api/ws`) match by design — the regex is anchored at the end.
  if (!GATEWAY_WS_PATH_RE.test(parsed.pathname.replace(/\/+$/, ''))) {
    return null
  }

  return `${parsed.protocol === 'wss:' ? 'https:' : 'http:'}//${parsed.host}`
}

/**
 * Rewrite `headers`' Origin for a gateway WS upgrade to `rawUrl`.
 *
 * Returns a NEW headers object with the Origin replaced (any casing of the
 * original key removed first — Chromium sends `Origin`, but don't rely on
 * it), or null when the request is not a gateway WS upgrade and the caller
 * should pass the original headers through unchanged.
 */
function applyGatewayWsOrigin(rawUrl, headers) {
  const origin = gatewayWsOrigin(rawUrl)

  if (!origin) {
    return null
  }

  const next = { ...(headers || {}) }

  for (const key of Object.keys(next)) {
    if (key.toLowerCase() === 'origin') {
      delete next[key]
    }
  }

  next.Origin = origin

  return next
}

export { applyGatewayWsOrigin, gatewayWsOrigin }
