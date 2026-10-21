/**
 * Remote gateway plugins: a named transport registers a URL scheme and
 * turns a saved connection url into a live loopback endpoint the app talks
 * plain HTTP to. Downstream of `ensure()`, a remote-plugin-backed remote is a
 * plain remote descriptor (`baseUrl: http://127.0.0.1:<port>`); the
 * plugin owns its pipe end to end.
 *
 * Rules:
 *  - `http(s)` is not registrable — built-in remotes never touch this file.
 *  - Saved urls are opaque to core: the scheme selects the plugin, the
 *    plugin canonicalizes the body. Core never parses pairing material.
 *
 * Kept free of `import 'electron'` so it unit-tests with plain vitest
 * (same pattern as connection-registry.ts). main.ts builds the registry.
 */

/** A live local access point for a remote gateway, plus its teardown. */
export interface RemoteTarget {
  /** Loopback origin the app speaks plain HTTP/WS against. */
  baseUrl: string
  /** Close the access point (idempotent; errors are the plugin's to swallow). */
  teardown: () => Promise<void>
  /** Optional pill label for UI (e.g. `example:abcd1234`); the loopback baseUrl
   * is not meaningful to display. See `remotePluginHostLabel`. */
  hostLabel?: string
  /** Optional live status for surfaces that want it; must be cheap. */
  status?: () => RemotePluginStatus
}

export type RemotePluginState = 'idle' | 'connecting' | 'online' | 'error'

export interface RemotePluginStatus {
  state: RemotePluginState
  detail?: string
}

/** A transport implementation registered by id + scheme. */
export interface RemotePlugin {
  /** The plugin's manifest id (globally unique), e.g. 'example-plugin'. */
  id: string
  /** Dispatch scheme WITHOUT `://` (e.g. 'example'). Lowercase, RFC3986 shape. */
  scheme: string
  /** Add-Connection UI hint ("paste the pairing code"). */
  hint?: string
  /** Optional regex SOURCE for opaque pasted pairing material (bare tickets)
   * the plugin claims even without its scheme prefix. Core compiles and
   * tests it generically — it never hardcodes a pairing format. */
  pastePattern?: string
  /** Canonicalize a saved url for this scheme (called at save time). May be
   * async (a native ticket parse). Throw on input this plugin cannot dial. */
  normalize?: (url: string) => string | Promise<string>
  /** Claim a scheme-less pasted value (bare ticket). Only consulted by
   * `normalize`/`accepts` when no scheme matched. */
  claims?: (raw: string) => boolean
  /** Bring up a local access point for `savedUrl`. Idempotent per saved url. */
  connect: (url: string, log: (msg: string) => void) => Promise<RemoteTarget>
  /** Close every access point this plugin holds (app shutdown). */
  teardownAll?: () => Promise<void>
}

const BUILT_IN_SCHEMES = new Set(['http', 'https'])

/** Normalize a scheme token: lowercase. Returns '' for anything that is not
 * a valid RFC3986 scheme-shaped token. */
export function canonicalScheme(scheme: unknown): string {
  if (typeof scheme !== 'string') {
    return ''
  }

  const value = scheme.trim().toLowerCase()

  return /^[a-z][a-z0-9+.-]*$/.test(value) ? value : ''
}

/** `example://ticket` → `example`; `http://x` → `http`; anything without a
 * `<scheme>://` prefix → ''. Case-insensitive on the scheme. */
export function urlScheme(url: unknown): string {
  if (typeof url !== 'string') {
    return ''
  }

  const match = /^([a-z][a-z0-9+.-]*):\/\//i.exec(url.trim())

  return match ? canonicalScheme(match[1]) : ''
}

/** Generic pill label for a remote-plugin-backed remote: `scheme:<first 8 of
 * body>` (e.g. `example:abcd1234`). Consumers need no knowledge of any
 * scheme's pairing format to label it. */
export function remotePluginHostLabel(url: unknown): string {
  const scheme = urlScheme(url)

  if (!scheme) {
    return ''
  }

  const body = String(url).trim().slice(`${scheme}://`.length)

  return `${scheme}:${body.slice(0, 8)}`
}

/** Raised when a url names a scheme no installed plugin implements. */
export class UnsupportedSchemeError extends Error {
  readonly scheme: string

  constructor(scheme: string) {
    super(`No plugin installed for scheme '${scheme}://'`)
    this.name = 'UnsupportedSchemeError'
    this.scheme = scheme
  }
}

export interface RemotePluginSummary {
  id: string
  scheme: string
  hint?: string
  pastePattern?: string
}

export interface RemotePluginRegistry {
  /** UI-facing inventory (Add-Connection list, settings). */
  list(): RemotePluginSummary[]
  /** The plugin owning `url`'s scheme, or null for http(s)/scheme-less. */
  forUrl(url: unknown): RemotePlugin | null
  /** True iff core or a remote plugin can canonicalize this raw pasted value. */
  accepts(raw: unknown): Promise<boolean>
  /** Canonical saved form: the plugin's `normalize` for its scheme, the
   * core https normalizer otherwise. Rejects for unclaimed schemes. */
  normalize(raw: unknown): Promise<string>
  /** Bring up (idempotently) the access point for a saved url. Returns null
   * for plain https remotes — the caller then takes the core https path.
   * Rejects UnsupportedSchemeError / plugin errors. */
  ensure(url: unknown, log?: (msg: string) => void): Promise<RemoteTarget | null>
  /** Close everything (app shutdown). */
  teardownAll(): Promise<void>
  /** Hot-swap the plugin set (renderer enable/disable pushes); pipes of
   * surviving plugins are untouched. */
  reseed(next: RemotePlugin[]): Promise<void>
}

/**
 * Build the dispatch registry.
 *
 * @param plugins validated plugin instances (loader output; the loader
 * resolves duplicate schemes before they reach here).
 * @param normalizeHttp the core https normalizer (`normalizeRemoteBaseUrl`).
 */
export function createRemotePluginRegistry({
  plugins = [],
  normalizeHttp,
  log = () => {}
}: {
  plugins?: RemotePlugin[]
  normalizeHttp: (url: string) => string
  log?: (msg: string) => void
}): RemotePluginRegistry {
  const byScheme = new Map<string, RemotePlugin>()

  for (const plugin of plugins) {
    const scheme = canonicalScheme(plugin.scheme)

    if (!scheme || BUILT_IN_SCHEMES.has(scheme) || typeof plugin.connect !== 'function') {
      log(`[remote-plugins] rejected invalid plugin ${JSON.stringify(plugin?.id)} (scheme='${plugin?.scheme}')`)

      continue
    }

    if (byScheme.has(scheme)) {
      log(`[remote-plugins] scheme '${scheme}://' already claimed by '${byScheme.get(scheme)!.id}' — ignoring '${plugin.id}'`)

      continue
    }

    byScheme.set(scheme, plugin)
  }

  /** RemotePlugin claiming a scheme-less value via a declared pastePattern. */
  function claimerForBare(raw: string): RemotePlugin | null {
    for (const plugin of byScheme.values()) {
      try {
        if (plugin.claims?.(raw)) {
          return plugin
        }

        if (plugin.pastePattern && new RegExp(plugin.pastePattern, 'i').test(raw)) {
          return plugin
        }
      } catch (err) {
        log(`[remote-plugins] '${plugin.id}' claim probe failed: ${String((err as Error)?.message ?? err)}`)
      }
    }

    return null
  }

  function resolve(raw: string): { plugin: null | RemotePlugin; canonical: string } {
    const scheme = urlScheme(raw)

    if (!scheme || scheme === 'http' || scheme === 'https') {
      // A bare paste (no scheme) may be a remote plugin's pairing material:
      // claiming must happen before any http-prepending would mangle it.
      if (!scheme) {
        const claimer = claimerForBare(raw.trim())

        if (claimer) {
          return { plugin: claimer, canonical: `${claimer.scheme}://${raw.trim()}` }
        }
      }

      return { plugin: null, canonical: '' }
    }

    const plugin = byScheme.get(scheme)

    if (!plugin) {
      throw new UnsupportedSchemeError(scheme)
    }

    return { plugin, canonical: '' }
  }

  async function normalize(raw: unknown): Promise<string> {
    if (typeof raw !== 'string' || !raw.trim()) {
      throw new Error('Remote gateway URL is required.')
    }

    const value = raw.trim()
    const { plugin, canonical } = resolve(value)

    if (canonical) {
      // Claimed bare paste: add the scheme, then let the plugin canonicalize
      // the body (sanitization is the plugin's contract).
      return plugin!.normalize ? String(await plugin!.normalize(canonical)) : canonical
    }

    if (!plugin) {
      return normalizeHttp(value)
    }

    return plugin.normalize ? String(await plugin.normalize(value)) : value
  }

  async function accepts(raw: unknown): Promise<boolean> {
    try {
      await normalize(raw)

      return true
    } catch {
      return false
    }
  }

  // Live access points keyed by plugin id + canonical saved url: the same
  // saved url reuses its pipe (idempotent), a changed url gets a new one.
  const live = new Map<string, RemoteTarget>()

  // In-flight dials keyed like `live`: concurrent ensure() calls for the
  // same saved url share one connect(), never two pipes.
  const pending = new Map<string, Promise<RemoteTarget>>()

  /** Swap the registered plugin set (enable/disable hot-changes) WITHOUT
   * disturbing the pipes of plugins that remain. Dropped plugins get
   * their pipes torn down, then their own teardownAll. */
  async function reseed(next: RemotePlugin[]): Promise<void> {
    const nextByScheme = new Map<string, RemotePlugin>()
    const issues: string[] = []

    for (const plugin of next) {
      const scheme = canonicalScheme(plugin.scheme)

      if (!scheme || BUILT_IN_SCHEMES.has(scheme) || typeof plugin.connect !== 'function') {
        issues.push(`${plugin?.id}: invalid plugin`)

        continue
      }

      if (nextByScheme.has(scheme)) {
        issues.push(`${plugin.id}: duplicate scheme '${scheme}'`)

        continue
      }

      nextByScheme.set(scheme, plugin)
    }

    byScheme.clear()

    for (const [scheme, plugin] of nextByScheme) {
      byScheme.set(scheme, plugin)
    }

    const survivingIds = new Set([...nextByScheme.values()].map(c => c.id))
    const droppedKeys = [...live.keys()].filter(key => !survivingIds.has(key.split('\u0000')[0]))

    for (const key of droppedKeys) {
      const target = live.get(key)

      live.delete(key)

      try {
        await target?.teardown()
      } catch (err) {
        log(`[remote-plugins] teardown on reseed failed: ${String((err as Error)?.message ?? err)}`)
      }
    }

    for (const message of issues) {
      log(`[remote-plugins] reseed: ${message}`)
    }
  }

  async function ensure(url: unknown, callerLog?: (msg: string) => void): Promise<RemoteTarget | null> {
    const emit = callerLog ?? log
    const savedUrl = await normalize(url)
    const scheme = urlScheme(savedUrl)
    const plugin = scheme ? byScheme.get(scheme) ?? null : null

    if (!plugin) {
      return null
    }

    const key = `${plugin.id}\u0000${savedUrl}`
    const existing = live.get(key) ?? pending.get(key)

    if (existing) {
      return existing
    }

    emit(`[remote-plugins] ${plugin.id}: dialing ${savedUrl.slice(0, 48)}…`)

    const dial = (async (): Promise<RemoteTarget> => {
      const target = await plugin.connect(savedUrl, emit)

      if (!target || typeof target.baseUrl !== 'string' || typeof target.teardown !== 'function') {
        throw new Error(`RemotePlugin '${plugin.id}' returned an invalid remote target`)
      }

      emit(`[remote-plugins] ${plugin.id}: access point ${target.baseUrl}`)

      let closed = false

      const managed: RemoteTarget = {
        baseUrl: target.baseUrl,
        hostLabel: target.hostLabel,
        status: target.status,
        teardown: async () => {
          // Once-only at the registry level: a second teardown of the same
          // saved url (e.g. delete-connection racing app shutdown) must not
          // re-enter the plugin's close path.
          if (closed) {
            return
          }

          closed = true
          live.delete(key)

          try {
            await target.teardown()
          } catch (err) {
            emit(`[remote-plugins] ${plugin.id}: teardown failed: ${String((err as Error)?.message ?? err)}`)
          }
        }
      }

      // Store the managed wrapper, not the raw target: its teardown must
      // also release the slot.
      live.set(key, managed)

      return managed
    })()

    pending.set(key, dial)

    try {
      return await dial
    } finally {
      pending.delete(key)
    }
  }

  async function teardownAll(): Promise<void> {
    const targets = [...live.values()]

    live.clear()

    await Promise.all(
      targets.map(async target => {
        await target.teardown().catch(() => {})
      })
    )

    await Promise.all(
      [...byScheme.values()].map(async plugin => {
        await plugin.teardownAll?.().catch(err =>
          log(`[remote-plugins] '${plugin.id}' teardownAll failed: ${String((err as Error)?.message ?? err)}`)
        )
      })
    )
  }

  return {
    list: () =>
      [...byScheme.values()].map(({ id, scheme, hint, pastePattern }) => ({
        id,
        scheme,
        ...(hint ? { hint } : {}),
        ...(pastePattern ? { pastePattern } : {})
      })),
    forUrl: (url): RemotePlugin | null => {
      const scheme = urlScheme(url)

      return scheme ? byScheme.get(scheme) ?? null : null
    },
    accepts,
    normalize,
    ensure,
    teardownAll,
    reseed
  }
}
