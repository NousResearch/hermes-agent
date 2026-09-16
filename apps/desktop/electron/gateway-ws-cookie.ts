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
//   - selected for the upgrade's own path, not the gateway root: a proxy
//     cookie scoped to `Path=/api/`, or to a reverse-proxy prefix, does not
//     apply to the bare base url, so reading the base returned a snapshot
//     missing exactly the credential this exists to forward;
//   - one live url per CONSUMER (the gateway, plus the window and purpose of
//     the caller that re-mints for that socket) — its next mint drops its own
//     previous url, so a stale / pre-rotation ticket carries no authority.
//     Replacement stops there: two Cloud agents share the legacy partition, a
//     shared remote serves several profiles at one baseUrl, and one window's
//     chat, speech and secondary flows all mint against the same route, so no
//     other gateway, profile, window or purpose may cancel an in-flight
//     handshake nobody signed out;
//   - authorized only when the caller dials the url it minted. Speech rewrites
//     the path before dialing, so authorizing its minted url would park a
//     credential nothing can consume; that endpoint is out of scope here;
//   - consumed by the upgrade that uses it and additionally time-bounded, so
//     an upgrade that never happens expires instead of lingering for the
//     process lifetime, and bounded in count;
//   - dropped on sign-out by BOTH the partition and the base url. One jar
//     backs several urls (the portal and a Cloud agent share the legacy
//     partition, so signing out of the portal must drop the agent's entry
//     too), and the partition a url resolves to is read from the live registry
//     and can change while an entry is live, so the base url is what follows
//     the gateway. A sign-out with no base url clears the whole jar and so
//     revokes everything.
//
// Registration reads the jar asynchronously, so both of its exits are fenced
// against work that overtook them: a logout epoch per SCOPE — partition, base
// url, and the blanket scope a whole-jar clear bumps — since a read that
// resolves after sign-out must not republish the signed-out cookie (clearing
// the jar cannot retract this separate snapshot), and a per-owner generation,
// from a process-wide counter, since a slow read must neither replace nor
// revoke a newer registration's entry. Each scope is released independently:
// tying them together left one publishable at an epoch an in-window read had
// already captured, merely because a sibling scope was still busy.
//
// Sign-out empties the jar asynchronously, so `forget` also opens a window,
// closed by the callback it returns, during which no read may publish: one
// started mid-cleanup sees cookies that are on their way out.
//
// This never mints or alters credentials: it forwards a session the user
// already obtained interactively, to the one request it was needed for.

export interface GatewayCookie {
  name: string
  value: string
}

// Does a stored cookie apply to `host`? Standard cookie domain-matching: an
// exact host match, or a Domain cookie set on a PARENT of it.
//
// This is the counterpart of the forwarding below -- whatever we are willing
// to forward, sign-out has to be able to delete. A `{url}` filter cannot do
// it: it applies PATH matching, so it never sees the `Path=/api/` proxy cookie
// this feature exists to forward, and sign-out left it in the jar.
//
// Measured on the pinned Electron (40.10.2), against a jar holding a
// `.example.test` parent cookie, a `gw.example.test` host cookie, a
// `Path=/api/` cookie, a sibling and a subdomain:
//   get({url: 'https://gw.example.test'})  -> host, parent          (no /api/)
//   get({url: 'https://gw.example.test/api/ws'}) -> host, parent, /api/
//   get({domain: 'gw.example.test'})       -> host, parent, /api/   (no sibling/sub)
// So Electron's `{domain}` filter would also be correct here. Matching
// explicitly instead keeps what sign-out deletes pinned by unit tests that
// need no Electron, and legible at the call site: this host, a parent's Domain
// cookie, any path -- never a sibling or a subdomain, which in the shared
// legacy jar belong to other connections.
export function cookieAppliesToHost(cookie: { domain?: string } | null, host: string) {
  const domain = String(cookie?.domain || '')
    .replace(/^\./, '')
    .toLowerCase()
  const target = String(host || '').toLowerCase()

  if (!domain || !target) {
    return false
  }

  return target === domain || target.endsWith(`.${domain}`)
}

export interface GatewayWsCookieStoreDependencies {
  // Cookies applicable to `cookieUrl` in `baseUrl`'s OAuth jar, or null when
  // there is no jar for it. `baseUrl` identifies the jar; `cookieUrl` is the
  // http(s) equivalent of the upgrade being authorized, and selection must be
  // made for THAT url -- a proxy cookie scoped to `Path=/api/` or to a reverse
  // proxy prefix does not apply to the bare base url, so reading the base
  // silently returned a snapshot without it. Domain/Path/Secure filtering is
  // the cookie store's job; never widen this to every cookie on the host.
  // Callers are responsible for warming a lazily-hydrating jar first.
  readCookies: (cookieUrl: string, baseUrl: string) => Promise<GatewayCookie[] | null>
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
// Live urls are one per consumer (gateway x window x purpose x route) and
// short-lived; the cap is a backstop bounding how many can be live at once.
const MAX_ENTRIES = 32
// Owners are (gateway, consumer) pairs drawn from the user's own connections,
// so this only stops the generation ledger growing for a process lifetime.
const MAX_OWNERS = 256

// The upgrade's http(s) equivalent, which is what cookie selection matches on:
// same host and path, `ws:`/`wss:` mapped to `http:`/`https:`. Query and hash
// carry the single-use ticket and mean nothing to cookie matching, so they are
// dropped. An unparseable url falls back to the caller's base url.
function cookieUrlForUpgrade(wsUrl: string, baseUrl: string) {
  try {
    const url = new URL(wsUrl)

    url.protocol = url.protocol === 'ws:' ? 'http:' : url.protocol === 'wss:' ? 'https:' : url.protocol
    url.search = ''
    url.hash = ''

    return url.toString()
  } catch {
    return baseUrl
  }
}

interface GatewayWsCookieEntry {
  expiresAt: number
  header: string
  owner: string
  partition: string
}

export function createGatewayWsCookieStore(dependencies: GatewayWsCookieStoreDependencies) {
  const entries = new Map<string, GatewayWsCookieEntry>()
  // Latest generation issued per owner, and the logout state per SCOPE. A
  // registration belongs to two scopes -- its partition and its base url --
  // because the partition a url resolves to is read from the live registry and
  // can change under us: a sign-out that resolved a different partition than
  // the entry recorded would otherwise leave that entry authorized. Both are
  // read before the jar await and re-checked after it.
  // Generations come from one process-wide counter rather than a per-owner
  // one, so a forgotten owner can never reissue a number a pending read is
  // still holding: that read simply finds no generation and stands down.
  let sequence = 0
  const generations = new Map<string, number>()
  const epochs = new Map<string, number>()
  // Sign-outs still clearing their jar, per scope.
  const signOuts = new Map<string, number>()
  const partitionScope = (partition: string) => `partition\n${partition}`
  const baseUrlScope = (baseUrl: string) => `baseUrl\n${baseUrl}`
  // Every registration also belongs to this scope, so a sign-out with no base
  // url -- which clears the WHOLE jar -- can revoke and fence all of them.
  const EVERY_SCOPE = 'every'
  const now = () => (dependencies.now ? dependencies.now() : Date.now())
  const ttlMs = dependencies.ttlMs ?? DEFAULT_TTL_MS

  const dropWhere = (matches: (entry: GatewayWsCookieEntry) => boolean) => {
    for (const [wsUrl, entry] of entries) {
      if (matches(entry)) {
        entries.delete(wsUrl)
      }
    }
  }

  // Keep the map bounded: expired entries first, then the soonest to expire.
  const prune = () => {
    const cutoff = now()

    dropWhere(entry => entry.expiresAt <= cutoff)

    if (entries.size <= MAX_ENTRIES) {
      return
    }

    const oldest = [...entries.entries()]
      .sort((a, b) => a[1].expiresAt - b[1].expiresAt)
      .slice(0, entries.size - MAX_ENTRIES)

    for (const [wsUrl] of oldest) {
      entries.delete(wsUrl)
    }
  }

  // Authorize exactly one upgrade: `wsUrl`, using `baseUrl`'s jar. Replaces the
  // url previously registered by the SAME consumer of the same gateway, and
  // nothing else. `consumer` identifies the caller that will re-mint for this
  // socket; callers that omit it share one owner per gateway.
  const register = async (wsUrl: string, baseUrl: string, consumer: string) => {
    if (!wsUrl || !baseUrl) {
      return
    }

    // Namespaced by baseUrl so a consumer label can never span two gateways.
    // `consumer` is required: defaulting it would silently put two independent
    // sockets under one owner, and the second would retire the first.
    const owner = `${baseUrl}\n${consumer}`
    const partition = dependencies.resolvePartition(baseUrl)
    const generation = ++sequence
    // EVERY_SCOPE last: a sign-out with no base url clears the whole jar, so
    // every registration must be fenced against it too.
    const scopes = [partitionScope(partition), baseUrlScope(baseUrl), EVERY_SCOPE]
    const scopedEpochs = scopes.map(scope => [scope, epochs.get(scope) ?? 0] as const)

    // Re-inserted so the map stays in least-recently-registered order.
    generations.delete(owner)
    generations.set(owner, generation)

    while (generations.size > MAX_OWNERS) {
      const oldest = generations.keys().next().value

      if (oldest === undefined) {
        break
      }

      generations.delete(oldest)
    }

    // True only while this registration is still the newest one for its
    // consumer AND no sign-out has touched either scope it read from. Checked
    // on both exits: superseded or revoked work must neither publish nor
    // delete.
    const stillCurrent = () =>
      generations.get(owner) === generation &&
      scopedEpochs.every(([scope, epoch]) => (epochs.get(scope) ?? 0) === epoch && !signOuts.get(scope))

    let cookies: GatewayCookie[] | null

    try {
      cookies = await dependencies.readCookies(cookieUrlForUpgrade(wsUrl, baseUrl), baseUrl)
    } catch (error) {
      // Non-fatal: a gateway with no proxy in front connects without this.
      if (stillCurrent()) {
        dropWhere(entry => entry.owner === owner)
      }

      dependencies.onError?.(error instanceof Error ? error.message : String(error))

      return
    }

    if (!stillCurrent()) {
      return
    }

    const header = (cookies || [])
      .filter(cookie => cookie?.name)
      .map(cookie => `${cookie.name}=${cookie.value}`)
      .join('; ')

    // Replace, never accumulate: only this consumer's newest ticket url stays
    // authorized. Other consumers and gateways keep theirs.
    dropWhere(entry => entry.owner === owner)

    if (header) {
      entries.set(wsUrl, { expiresAt: now() + ttlMs, header, owner, partition })
      prune()
    }
  }

  // Drop every url authorized from `baseUrl`'s jar — all of them sharing its
  // partition, since sign-out empties the jar they were all read from.
  //
  // Call the returned callback once the jar itself has been cleared. Until
  // then the partition stays closed to new authority, because a registration
  // racing the cleanup would read cookies that are already being deleted.
  const forget = (baseUrl: string) => {
    // No base url means the caller is clearing the ENTIRE jar --
    // clearOauthSession passes its filter straight through -- so nothing may
    // outlive it. Returning a no-op left the widest clear as the only
    // unfenced one.
    if (!baseUrl) {
      const revokeEverything = () => {
        epochs.set(EVERY_SCOPE, (epochs.get(EVERY_SCOPE) ?? 0) + 1)
        entries.clear()
      }

      revokeEverything()
      signOuts.set(EVERY_SCOPE, (signOuts.get(EVERY_SCOPE) ?? 0) + 1)

      let allClosed = false

      return () => {
        if (allClosed) {
          return
        }

        allClosed = true

        const remaining = (signOuts.get(EVERY_SCOPE) ?? 1) - 1

        if (remaining > 0) {
          signOuts.set(EVERY_SCOPE, remaining)

          return
        }

        signOuts.delete(EVERY_SCOPE)
        revokeEverything()
      }
    }

    const partition = dependencies.resolvePartition(baseUrl)
    const ownerPrefix = `${baseUrl}\n`
    // Each scope is revoked and released INDEPENDENTLY. Tying them together
    // would let one scope stay publishable at an epoch an in-window read had
    // already captured, just because its sibling was still busy -- reachable
    // whenever the live registry moves a gateway between partitions mid-logout.
    const scopes = [
      { key: partitionScope(partition), matches: (entry: GatewayWsCookieEntry) => entry.partition === partition },
      { key: baseUrlScope(baseUrl), matches: (entry: GatewayWsCookieEntry) => entry.owner.startsWith(ownerPrefix) }
    ]

    // Bump the scope's epoch as well as dropping its live entries: a jar read
    // already in flight must not publish the signed-out cookie.
    const revokeScope = (scope: (typeof scopes)[number]) => {
      epochs.set(scope.key, (epochs.get(scope.key) ?? 0) + 1)
      dropWhere(scope.matches)
    }

    for (const scope of scopes) {
      revokeScope(scope)
      signOuts.set(scope.key, (signOuts.get(scope.key) ?? 0) + 1)
    }

    let closed = false

    return () => {
      if (closed) {
        return
      }

      closed = true

      for (const scope of scopes) {
        const remaining = (signOuts.get(scope.key) ?? 1) - 1

        if (remaining > 0) {
          signOuts.set(scope.key, remaining)

          continue
        }

        signOuts.delete(scope.key)
        // A read that began during this scope's window resolves against the
        // pre-logout jar, so retire that generation rather than let it land.
        revokeScope(scope)
      }
    }
  }

  // The header for a request, or null. Exact url match AND, when Chromium
  // reports one, a `webSocket` resource type. An expired entry is dropped
  // rather than used.
  //
  // CONSUMED on use: the authority lasts one upgrade, mirroring the single-use
  // ticket already in the url. Nothing re-attempts an upgrade with the same
  // url — every OAuth connect re-mints through freshGatewayWsUrl /
  // ws-url-for before dialing, and a ticket that has been presented is spent
  // anyway — so an unconsumed entry could only ever serve a request this
  // authorization was not granted for. A refusal (wrong resource type,
  // expired) does not consume.
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

    entries.delete(url)

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
