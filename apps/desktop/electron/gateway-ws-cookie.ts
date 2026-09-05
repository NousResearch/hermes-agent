// Forwarding a forward-auth proxy's session onto the renderer's gateway
// WebSocket upgrade — and ONLY onto that upgrade.
//
// The main process does REST through `electron net` bound to the OAuth
// partition, but the RENDERER opens the gateway WebSocket and runs on
// `defaultSession`. Behind a forward-auth proxy the proxy's session cookie
// only ever lands in the OAuth partition, so the `/api/ws` upgrade is rejected
// before it reaches Hermes: REST works, the UI never connects.
//
// Copying the cookies into `defaultSession` is the wrong fix: it makes their
// authority ambient to every page that session loads, and only works once
// they are re-stamped `SameSite=None` (the renderer document is `file://`, so
// the upgrade counts as cross-site), which strips the proxy cookie's CSRF
// protection and persists it that way.
//
// So the jar is read in the main process and the value is attached as a
// `Cookie` header on one request: the exact, freshly-minted WebSocket URL the
// renderer is about to open. That URL carries a single-use ~30s ticket, which
// makes it a natural lifetime bound for the forwarded credential:
//
//   - keyed by the EXACT ws url, so ordinary HTTP(S) traffic to the gateway,
//     sibling paths, and unrelated sockets get nothing;
//   - one live url per OAuth partition — registering the next mint drops the
//     previous one, so a stale/pre-rotation ticket url carries no authority;
//   - additionally time-bounded, so an upgrade that never happens expires
//     instead of lingering for the process lifetime;
//   - dropped per partition on sign-out, since one jar backs several urls (the
//     portal and a Cloud agent share the legacy partition, so signing out of
//     the portal must drop the agent's entry too).
//
// This never mints or alters credentials: it forwards a session the user
// already obtained interactively, to the one request it was needed for.

export interface GatewayCookie {
  name: string
  value: string
}

export interface GatewayWsCookieStoreDependencies {
  // Cookies currently in `baseUrl`'s OAuth jar, or null when there is no jar
  // for it. Callers are responsible for warming a lazily-hydrating jar first.
  readCookies: (baseUrl: string) => Promise<GatewayCookie[] | null>
  // Which cookie jar backs a url. Entries sharing one are dropped together.
  resolvePartition: (baseUrl: string) => string
  now?: () => number
  // How long a registered upgrade stays authorized. Defaults to well over the
  // ticket's own ~30s TTL so a slow connect still succeeds, while an upgrade
  // that never happens cannot linger.
  ttlMs?: number
  onError?: (message: string) => void
}

export interface RemoteRequestDetails {
  url?: string
  requestHeaders?: Record<string, string>
  resourceType?: string
}

export interface RemoteRequestResponse {
  requestHeaders?: Record<string, string>
}

const DEFAULT_TTL_MS = 120_000

export function createGatewayWsCookieStore(dependencies: GatewayWsCookieStoreDependencies) {
  const entries = new Map<string, { expiresAt: number; header: string; partition: string }>()
  const now = () => (dependencies.now ? dependencies.now() : Date.now())
  const ttlMs = dependencies.ttlMs ?? DEFAULT_TTL_MS

  const dropPartition = (partition: string) => {
    for (const [wsUrl, entry] of entries) {
      if (entry.partition === partition) {
        entries.delete(wsUrl)
      }
    }
  }

  // Authorize exactly one upgrade: `wsUrl`, using `baseUrl`'s jar. Replaces any
  // url previously registered for the same partition.
  const register = async (wsUrl: string, baseUrl: string) => {
    if (!wsUrl || !baseUrl) {
      return
    }

    const partition = dependencies.resolvePartition(baseUrl)

    let cookies: GatewayCookie[] | null

    try {
      cookies = await dependencies.readCookies(baseUrl)
    } catch (error) {
      // Non-fatal: a gateway with no proxy in front connects without this.
      dropPartition(partition)
      dependencies.onError?.(error instanceof Error ? error.message : String(error))

      return
    }

    const header = (cookies || [])
      .filter(cookie => cookie?.name)
      .map(cookie => `${cookie.name}=${cookie.value}`)
      .join('; ')

    // Replace, never accumulate: only the newest ticket url stays authorized.
    dropPartition(partition)

    if (header) {
      entries.set(wsUrl, { expiresAt: now() + ttlMs, header, partition })
    }
  }

  // Drop every url authorized from `baseUrl`'s jar — all of them sharing its
  // partition, since sign-out empties the jar they were all read from.
  const forget = (baseUrl: string) => {
    if (!baseUrl) {
      return
    }

    dropPartition(dependencies.resolvePartition(baseUrl))
  }

  // The header for a request, or null. Exact url match AND, when Chromium
  // reports one, a `webSocket` resource type. An expired entry is dropped
  // rather than used.
  const headerFor = (details: RemoteRequestDetails) => {
    const url = details?.url

    if (!url) {
      return null
    }

    const entry = entries.get(url)

    if (!entry) {
      return null
    }

    if (entry.expiresAt <= now()) {
      entries.delete(url)

      return null
    }

    // `resourceType` is absent in some call shapes; the exact-url match is the
    // primary gate, so treat "not reported" as acceptable and only refuse a
    // type that is positively something else.
    if (details.resourceType && details.resourceType !== 'webSocket') {
      return null
    }

    return entry.header
  }

  // Merge the cookie into an outgoing request's headers when that request is
  // the authorized upgrade, leaving every other request untouched.
  //
  // Verified on Electron 40 (file:// renderer, `webSocket` resource type): a
  // `Cookie` set here reaches the upgrade intact. The hook runs BEFORE
  // Chromium attaches jar cookies, so ours suppresses whatever the jar would
  // have added for that origin — moot while defaultSession holds no gateway
  // cookies (the whole problem), and scoped to this one url regardless. The
  // append below is defensive: `requestHeaders` carries no `Cookie` today.
  const apply = (details: RemoteRequestDetails, response: RemoteRequestResponse) => {
    const header = headerFor(details)

    if (!header) {
      return response
    }

    const headers = { ...(response?.requestHeaders || details.requestHeaders || {}) }
    const existing = Object.keys(headers).find(name => name.toLowerCase() === 'cookie')

    headers[existing || 'Cookie'] = existing && headers[existing] ? `${headers[existing]}; ${header}` : header

    return { ...(response || {}), requestHeaders: headers }
  }

  return { apply, forget, register }
}
