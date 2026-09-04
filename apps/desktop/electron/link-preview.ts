import { toDataUrl } from './favicon'

/**
 * Link preview resolution — the click-to-expand unfurl (D7).
 *
 * A mentioned URL is never fetched. When the user clicks "Load link preview",
 * the desktop's main process fetches the page once, extracts the Open Graph /
 * Twitter-card meta plus <title>, and returns a small card payload. Main
 * process rather than renderer because cross-origin HTML is unreadable from a
 * web page (CORS) and because the fetch must carry no page context — no
 * cookies, no referrer, no origin — only a browser-shaped UA. The renderer
 * never talks to the network for this; it gets an IPC envelope.
 *
 * The policy lives here so there is one home for it (the favicon ladder and
 * the title fetcher each learned the same rules the hard way):
 *
 *  - Only public http(s) URLs. Localhost-shaped hostnames are refused by
 *    name, and the hostname is ALSO resolved before fetching — a DNS name
 *    that answers with a loopback/RFC1918/link-local address is the classic
 *    SSRF doorway, and "the name looked public" is not a defense.
 *  - Per-host rate limiting: at most one fetch every PREVIEW_HOST_SPACING_MS
 *    to the same host, and a process-wide cap on concurrent fetches. Twenty
 *    clicks on twenty same-host links must not look like a bot.
 *  - Bounded everything: byte budget and timeouts come from the injected I/O
 *    (the same tier-1 curl budget the title fetcher uses), field caps below,
 *    a 24h cache TTL, and a failed fetch is never cached.
 *
 * Everything that decides is pure; the network, clock, DNS, and persistence
 * are injected, so the whole policy is testable without Electron or a
 * network. main.ts supplies the production deps and the IPC surface.
 */

export interface LinkPreviewMeta {
  /** Normalized http(s) URL that was actually fetched. */
  url: string
  /** Page title or og:title — error/captcha pages scrubbed to ''. */
  title: string
  /** og:description, twitter:description, or meta description; ≤300 chars. */
  description: string
  /** og:site_name, or '' when the page does not declare one. */
  siteName: string
  /** Absolute og:image / twitter:image URL, or '' when absent or non-http. */
  imageUrl: string
  /**
   * Thumbnail as a validated data URL, or '' when the image could not be
   * fetched and sniffed under the guard. The renderer paints THIS — it never
   * GETs imageUrl itself (a renderer-side <img> fetch bypasses every SSRF
   * guard); data-URL bytes ride the IPC envelope with the rest of the meta.
   */
  image: string
  /** Epoch ms when the meta was fetched; drives the cache TTL. */
  fetchedAt: number
}

export type LinkPreviewFailureReason = 'private-url' | 'error'

export type LinkPreviewResult =
  | { ok: true; meta: LinkPreviewMeta }
  | { ok: false; reason: LinkPreviewFailureReason }

export const PREVIEW_DESCRIPTION_MAX = 300
export const PREVIEW_TITLE_MAX = 240
export const PREVIEW_CACHE_LIMIT = 400
export const PREVIEW_TTL_MS = 24 * 60 * 60 * 1000
/** Minimum spacing between fetches to the same host. */
export const PREVIEW_HOST_SPACING_MS = 10_000
/** Process-wide cap on concurrent preview fetches. */
export const PREVIEW_MAX_CONCURRENT = 4

const HTML_ENTITIES: Record<string, string> = {
  amp: '&',
  lt: '<',
  gt: '>',
  quot: '"',
  apos: "'",
  nbsp: ' ',
  '#39': "'"
}

/** Meta property/name keys, in the order we prefer them. */
const TITLE_KEYS = ['og:title', 'twitter:title']
const DESCRIPTION_KEYS = ['og:description', 'twitter:description', 'description']
const IMAGE_KEYS = ['og:image', 'og:image:url', 'og:image:secure_url', 'twitter:image', 'twitter:image:src']

// Errors/captcha walls that would poison the cache with junk titles. Mirrors
// the title fetcher's scrub list.
const PREVIEW_ERROR_RE =
  /\b(access denied|attention required|captcha|error|forbidden|just a moment|request blocked|too many requests)\b/i

export function canonicalPreviewKey(rawUrl: string): string {
  const value = String(rawUrl || '').trim()

  if (!value) {
    return ''
  }

  try {
    const url = new URL(value)
    const host = url.hostname.replace(/^www\./i, '').toLowerCase()
    const pathname = url.pathname === '/' ? '/' : url.pathname.replace(/\/+$/, '') || '/'

    return `${host}${pathname}${url.search || ''}`
  } catch {
    return value
  }
}

export function decodeHtmlEntities(value: string): string {
  return value
    .replace(/&(amp|lt|gt|quot|apos|nbsp|#39);/gi, (_, k: string) => HTML_ENTITIES[k.toLowerCase()] ?? '')
    .replace(/&#x([0-9a-f]+);/gi, (_, hex: string) => String.fromCodePoint(parseInt(hex, 16) || 32))
    .replace(/&#(\d+);/g, (_, dec: string) => String.fromCodePoint(parseInt(dec, 10) || 32))
}

const cleanField = (value: string, max: number): string =>
  decodeHtmlEntities(value)
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, max)

const absoluteHttp = (href: string, base: string): string => {
  try {
    const url = new URL(href.trim(), base)

    return url.protocol === 'http:' || url.protocol === 'https:' ? url.toString() : ''
  } catch {
    return ''
  }
}

/**
 * Pull og/twitter/description meta out of an HTML document.
 *
 * Tier-1 budgets cap the HTML we see (96 KB), which is head-region territory
 * for every sane page, so a linear tag scan is enough. Relative image URLs
 * resolve against the page URL; resolution failures and non-http schemes
 * drop the field rather than poisoning the card.
 */
export function parseLinkMeta(html: string, baseUrl: string): Omit<LinkPreviewMeta, 'fetchedAt'> {
  const meta = new Map<string, string>()

  for (const tag of String(html || '').matchAll(/<meta\s+[^>]*>/gi)) {
    const tagText = tag[0]

    const key =
      attrValue(tagText, 'property')?.toLowerCase() ||
      attrValue(tagText, 'name')?.toLowerCase() ||
      ''

    const content = attrValue(tagText, 'content')

    if (!key || content === null || content === undefined || meta.has(key)) {
      continue
    }

    meta.set(key, content)
  }

  const titleTag = String(html || '').match(/<title[^>]*>([\s\S]*?)<\/title>/i)?.[1] ?? ''

  const first = (keys: string[]): string => {
    for (const key of keys) {
      const value = meta.get(key)

      if (value && cleanField(value, Number.MAX_SAFE_INTEGER)) {
        return cleanField(value, key.includes('description') ? PREVIEW_DESCRIPTION_MAX : PREVIEW_TITLE_MAX)
      }
    }

    return ''
  }

  const title = first(TITLE_KEYS) || cleanField(titleTag, PREVIEW_TITLE_MAX)
  const imageUrlRaw = first(IMAGE_KEYS)
  const imageUrl = imageUrlRaw ? absoluteHttp(imageUrlRaw, baseUrl) : ''

  return {
    url: baseUrl,
    title: usableTitle(title),
    description: first(DESCRIPTION_KEYS),
    siteName: cleanField(meta.get('og:site_name') ?? '', PREVIEW_TITLE_MAX),
    imageUrl,
    image: ''
  }
}

function attrValue(tag: string, name: string): string | null {
  const match = tag.match(new RegExp(`\\b${name}\\s*=\\s*("([^"]*)"|'([^']*)'|([^\\s"'>]+))`, 'i'))

  if (!match) {
    return null
  }

  return match[2] ?? match[3] ?? match[4] ?? null
}

/** Strips known error/captcha titles so they never reach the card. */
export function usableTitle(value: string): string {
  return value && !PREVIEW_ERROR_RE.test(value) ? value : ''
}

/**
 * True when an address is network-internal. Hostname-shaped checks (the
 * favicon ladder's) run first; this is the resolved-IP half of the guard.
 */
export function isPrivateAddress(ip: string): boolean {
  const value = (ip || '').trim().toLowerCase()

  if (!value) {
    return true
  }

  if (value.includes(':')) {
    // IPv6: loopback, unspecified, link-local (fe80::/10), unique-local (fc00::/7),
    // and IPv4-mapped (::ffff:10.0.0.0-style) all count as private.
    const bare = value.replace(/^\[|\]$/g, '')

    return (
      bare === '::1' ||
      bare === '::' ||
      /^f[cd][0-9a-f]{2}:/.test(bare) ||
      // Site-local (fec0::/10) joins link-local (fe80::/10): the fec0 range
      // is deprecated but still routable in some estates, and a name that
      // answers with one is no more public than one answering with fe80 —
      // both refuse. (Classification parity with upstream #63171/65bfbeddaf.)
      /^fe[89ab][0-9a-f]:/.test(bare) ||
      /^fe[c-f][0-9a-f]?[0-9a-f]?:/.test(bare) ||
      /^::ffff:(?!172\.(?:3[01]|[12]\d|1[6-9])\.)(?:10\.|192\.168\.|169\.254\.|127\.)/.test(bare) ||
      /^::ffff:172\.(?:1[6-9]|2\d|3[01])\./.test(bare)
    )
  }

  const octets = value.split('.').map(Number)

  if (octets.length !== 4 || octets.some(n => !Number.isInteger(n) || n < 0 || n > 255)) {
    return true
  }

  const [a, b] = octets

  return (
    a === 0 ||
    a === 10 ||
    a === 127 ||
    (a === 100 && b >= 64 && b <= 127) ||
    (a === 169 && b === 254) ||
    (a === 172 && b >= 16 && b <= 31) ||
    (a === 192 && b === 168) ||
    (a === 192 && b === 0 && (octets[2] === 0 || octets[2] === 2)) ||
    (a === 198 && (b === 18 || b === 19)) ||
    a >= 224
  )
}

/** Hostname-level guard: names that only ever mean "this machine or close". */
export function isPrivateHostname(hostname: string): boolean {
  const host = (hostname || '').toLowerCase().replace(/\.$/, '')

  if (!host || host === 'localhost' || host === '::1') {
    return true
  }

  if (host.endsWith('.localhost') || host.endsWith('.local') || host.endsWith('.internal') || host.endsWith('.home.arpa')) {
    return true
  }

  // A single-label name ("intranet", "mymac") can only ever resolve inside
  // someone's search domain — refuse. Real public hosts carry a dot.
  if (!host.includes('.')) {
    return true
  }

  // A literal IP hostname is judged by the address rules; a name is public
  // here and gets its real verdict from the DNS resolution step, whose
  // answers are checked with isPrivateAddress (the SSRF half of the guard).
  return isPrivateAddress(host) && isDottedQuad(host)
}

const DOTTED_QUAD_RE = /^\d{1,3}(?:\.\d{1,3}){3}$/

function isDottedQuad(value: string): boolean {
  return DOTTED_QUAD_RE.test(value)
}

/** Redirects the guarded fetcher may follow — the old curl --max-redirs 3. */
export const PREVIEW_MAX_REDIRECTS = 3

/** One HTTP hop: a single response, with redirect following left to the caller. */
export interface HttpHopResponse<B = string> {
  /** Status code; 0 means transport failure (an empty final body). */
  status: number
  /** Raw Location header value — possibly relative — or ''. */
  location: string
  /** Response body with the header block stripped. */
  body: B
}

export interface GuardedRedirectIo<B = string> {
  /** Exactly one HTTP request; must never follow redirects on its own. */
  fetchOnce: (url: string, addresses: string[]) => Promise<HttpHopResponse<B>>
  /** Resolved addresses for a hostname; [] when resolution fails. */
  resolveHost: (hostname: string) => Promise<string[]>
}

export type GuardedRedirectResult<B = string> =
  | { ok: true; url: string; body: B }
  | { ok: false; reason: LinkPreviewFailureReason }

/**
 * One hop's full SSRF verdict: scheme, hostname guard, and where the name
 * actually resolves. The DNS half is the point — a redirect to a fresh
 * attacker-controlled name that answers with an RFC1918/loopback address is
 * exactly the doorway the initial-URL guard closes; without per-hop
 * re-validation it stays wide open. Returns the vetted addresses so the
 * caller can PIN the connection to one of them (curl --resolve): letting the
 * transport resolve the name again would reopen a rebinding window between
 * this verdict and the actual request. (Shape adopted from upstream #63171.)
 */
async function guardHop<B>(url: URL, io: GuardedRedirectIo<B>): Promise<{ refusal: LinkPreviewFailureReason | null; addresses: string[] }> {
  if (url.protocol !== 'http:' && url.protocol !== 'https:') {
    return { refusal: 'private-url', addresses: [] }
  }

  if (isPrivateHostname(url.hostname)) {
    return { refusal: 'private-url', addresses: [] }
  }

  let addresses: string[] = []

  try {
    addresses = await io.resolveHost(url.hostname)
  } catch {
    addresses = []
  }

  if (!addresses.length || addresses.some(isPrivateAddress)) {
    return { refusal: 'private-url', addresses: [] }
  }

  return { refusal: null, addresses }
}

/**
 * The tier-1 page fetch with the SSRF guard applied to EVERY hop.
 *
 * `curl --location` followed redirects with no re-validation, so a public URL
 * answering 302 → http://127.0.0.1/ (or 169.254.169.254, or any RFC1918 name)
 * was fetched and its title/description handed to the renderer — the guard on
 * the initial URL proved only that the FIRST hop was safe. This walks the
 * chain one hop at a time instead: each Location target must be http(s), pass
 * the hostname guard, and resolve to public addresses before it is requested,
 * with at most PREVIEW_MAX_REDIRECTS hops. Everything else about the fetch
 * (timeouts, byte budget, UA) stays in the injected fetchOnce.
 */
export async function fetchWithGuardedRedirects<B = string>(
  rawUrl: string,
  io: GuardedRedirectIo<B>,
  options: { maxRedirects?: number } = {}
): Promise<GuardedRedirectResult<B>> {
  const maxRedirects = options.maxRedirects ?? PREVIEW_MAX_REDIRECTS

  let current: URL

  try {
    current = new URL(String(rawUrl || '').trim())
  } catch {
    return { ok: false, reason: 'error' }
  }

  const firstHop = await guardHop(current, io)

  if (firstHop.refusal) {
    return { ok: false, reason: firstHop.refusal }
  }

  let addresses = firstHop.addresses
  let redirectsFollowed = 0

  for (;;) {
    let response: HttpHopResponse<B>

    try {
      response = await io.fetchOnce(current.toString(), addresses)
    } catch {
      // Transport failure ends the walk; the empty body is the caller's miss signal.
      return { ok: true, url: current.toString(), body: '' as B }
    }

    const location = (response.location || '').trim()
    const isRedirect = response.status >= 300 && response.status < 400 && location !== ''

    if (!isRedirect) {
      return { ok: true, url: current.toString(), body: response.body }
    }

    if (redirectsFollowed >= maxRedirects) {
      return { ok: false, reason: 'error' }
    }

    let next: URL

    try {
      next = new URL(location, current)
    } catch {
      return { ok: false, reason: 'error' }
    }

    const nextHop = await guardHop(next, io)

    if (nextHop.refusal) {
      return { ok: false, reason: nextHop.refusal }
    }

    current = next
    redirectsFollowed += 1
    // The vetted addresses ride along for THIS hop's request, replacing the
    // pre-guard flow where fetchOnce re-resolved the (already vetted) name.
    addresses = nextHop.addresses
  }
}

/** Past this an og:image is a download, not a thumbnail. */
export const PREVIEW_IMAGE_MAX_BYTES = 2 * 1024 * 1024

/**
 * Fetch an og:image thumbnail under the SAME admission rules as the page
 * fetch, returning a validated data URL — or '' when the image cannot be
 * proven safe and real.
 *
 * The renderer must never GET `meta.imageUrl` itself: an <img src> is a
 * renderer-side private-network fetch with cookie/referrer context and no
 * guard at all (the reviewer's blocking item 2). So the main process fetches
 * the bytes here — every hop admitted, every hop pinned to vetted addresses,
 * the same walk the HTML fetch uses, just binary — and hands the renderer a
 * data URL it can paint without touching the network. The bytes must sniff as
 * a real image; a body that fails validation drops the thumbnail rather than
 * poisoning the card.
 */
export async function resolveThumbnail(
  rawUrl: string,
  io: {
    fetchOnce: (url: string, addresses: string[]) => Promise<HttpHopResponse<Uint8Array>>
    resolveHost: (hostname: string) => Promise<string[]>
  }
): Promise<string> {
  let result: GuardedRedirectResult<Uint8Array>

  try {
    result = await fetchWithGuardedRedirects<Uint8Array>(rawUrl, io)
  } catch {
    return ''
  }

  if (!result.ok) {
    return ''
  }

  const bytes = result.body

  if (!bytes || bytes.length === 0 || bytes.length > PREVIEW_IMAGE_MAX_BYTES) {
    return ''
  }

  return imageMimeFromBytes(bytes) ? toDataUrl(imageMimeFromBytes(bytes), bytes) : ''
}

/** Minimal magic-byte sniff shared by the runtime; favicon.ts owns the full one. */
function imageMimeFromBytes(bytes: Uint8Array): string {
  const at = (offset: number, ...signature: number[]) =>
    signature.every((byte, index) => bytes[offset + index] === byte)

  if (at(0, 0x89, 0x50, 0x4e, 0x47)) {
    return 'image/png'
  }

  if (at(0, 0xff, 0xd8, 0xff)) {
    return 'image/jpeg'
  }

  if (at(0, 0x47, 0x49, 0x46, 0x38)) {
    return 'image/gif'
  }

  if (at(0, 0x00, 0x00, 0x01, 0x00)) {
    return 'image/x-icon'
  }

  if (at(0, 0x52, 0x49, 0x46, 0x46) && at(8, 0x57, 0x45, 0x42, 0x50)) {
    return 'image/webp'
  }

  return ''
}

/**
 * Per-host pacing plus a global concurrency cap.
 *
 * Same-host fetches start at least `spacingMs` apart and no more than
 * `maxPerHost` run concurrently, so a burst of clicks becomes a paced
 * trickle the host can't tell from one careful reader. `maxConcurrent`
 * bounds the whole process. `acquire` resolves to a release function;
 * callers must release in a `finally`.
 */
export class HostRateLimiter {
  private active = 0
  private readonly perHost = new Map<string, number>()
  private readonly lastGrant = new Map<string, number>()
  private readonly chains = new Map<string, Promise<unknown>>()

  constructor(
    private readonly spacingMs: number,
    private readonly maxConcurrent: number,
    private readonly maxPerHost = 3
  ) {}

  acquire(host: string): Promise<() => void> {
    const previous = this.chains.get(host) ?? Promise.resolve()

    const run = previous.then(
      async () => {
        // Room behind the global cap and this host's own in-flight cap.
        for (;;) {
          const hostActive = this.perHost.get(host) ?? 0

          if (this.active < this.maxConcurrent && hostActive < this.maxPerHost) {
            break
          }

          await new Promise<void>(resolve => setTimeout(resolve, 50))
        }

        // Politeness spacing since this host's previous fetch STARTED.
        const last = this.lastGrant.get(host)

        if (last !== undefined && this.spacingMs > 0) {
          const remaining = this.spacingMs - (Date.now() - last)

          if (remaining > 0) {
            await new Promise<void>(resolve => setTimeout(resolve, remaining))
          }
        }

        this.active += 1
        this.perHost.set(host, (this.perHost.get(host) ?? 0) + 1)
        this.lastGrant.set(host, Date.now())

        let released = false

        return () => {
          if (released) {
            return
          }

          released = true
          this.active -= 1
          this.perHost.set(host, Math.max(0, (this.perHost.get(host) ?? 1) - 1))
        }
      },
      // A rejected predecessor must not wedge the host's chain forever; the
      // waiter gets a no-op release so its fetch proceeds unpaced once.
      async () => () => undefined
    )

    this.chains.set(host, run.catch(() => undefined))

    return run
  }
}

interface StoredPreview extends Omit<LinkPreviewMeta, 'fetchedAt'> {
  at: number
}

export interface PreviewPersistence {
  read: () => Record<string, StoredPreview>
  write: (entries: Record<string, StoredPreview>) => void
}

/**
 * The durable preview cache. Entries expire after `ttlMs`; a full cache
 * evicts oldest-first. Failed fetches are never stored (a site that was
 * down a minute ago may be up now — a miss costs one click, a stale failure
 * costs the feature).
 */
export class LinkPreviewStore {
  private readonly entries = new Map<string, StoredPreview>()
  private writeTimer: ReturnType<typeof setTimeout> | null | undefined = undefined

  constructor(
    private readonly persistence: PreviewPersistence | null,
    private readonly options: { capacity?: number; ttlMs?: number; writeDebounceMs?: number; now?: () => number } = {}
  ) {
    for (const [key, entry] of Object.entries(persistence?.read() ?? {})) {
      if (this.isFresh(entry)) {
        this.entries.set(key, entry)
      }
    }
  }

  private isFresh(entry: StoredPreview): boolean {
    const now = (this.options.now ?? Date.now)()

    return Number.isFinite(entry?.at) && now - entry.at < (this.options.ttlMs ?? PREVIEW_TTL_MS)
  }

  get(rawUrl: string): LinkPreviewMeta | null {
    const key = canonicalPreviewKey(rawUrl)

    if (!key) {
      return null
    }

    const entry = this.entries.get(key)

    if (!entry) {
      return null
    }

    if (!this.isFresh(entry)) {
      this.entries.delete(key)

      return null
    }

    const { at, ...meta } = entry

    return { ...meta, fetchedAt: at }
  }

  set(meta: Omit<LinkPreviewMeta, 'fetchedAt'>): void {
    const key = canonicalPreviewKey(meta.url)

    if (!key) {
      return
    }

    const now = (this.options.now ?? Date.now)()

    if (this.entries.size >= (this.options.capacity ?? PREVIEW_CACHE_LIMIT)) {
      const oldest = this.entries.keys().next().value

      if (oldest !== undefined) {
        this.entries.delete(oldest)
      }
    }

    this.entries.set(key, { ...meta, at: now })
    this.flushSoon()
  }

  /** Debounced persistence; the cache is an optimization, writes are best-effort. */
  flushSoon(): void {
    if (!this.persistence || this.writeTimer) {
      return
    }

    this.writeTimer = setTimeout(() => {
      this.writeTimer = null

      try {
        this.persistence.write(Object.fromEntries(this.entries))
      } catch {
        // Losing a cache write costs one refetch.
      }
    }, this.options.writeDebounceMs ?? 3000)

    this.writeTimer.unref?.()
  }
}

export interface LinkPreviewIo {
  /** Raw tier-1 page HTML, or '' on any refusal. Budget-capped upstream. */
  fetchHtml: (url: string) => Promise<string>
  /** Tier-2 rendered <title>, or '' when the page will not render one. */
  fetchRenderedTitle: (url: string) => Promise<string>
  /** Resolved addresses for a hostname; [] when resolution fails. */
  resolveHost: (hostname: string) => Promise<string[]>
  /**
   * og:image bytes as a validated data URL, or '' on any refusal. MUST run the
   * same per-hop admission + pinning as fetchHtml — the whole point of this
   * leg is that the renderer never GETs the image URL itself.
   */
  fetchThumbnail: (url: string) => Promise<string>
}

/**
 * The whole policy in one function: guard, cache, pace, fetch, parse,
 * backfill, store. Never throws — every failure leg is an envelope.
 */
export async function resolveLinkPreview(
  rawUrl: string,
  io: LinkPreviewIo,
  deps: { store: LinkPreviewStore; limiter: HostRateLimiter; now?: () => number }
): Promise<LinkPreviewResult> {
  const url = String(rawUrl || '').trim()

  let parsed: URL

  try {
    parsed = new URL(url)
  } catch {
    return { ok: false, reason: 'error' }
  }

  if ((parsed.protocol !== 'http:' && parsed.protocol !== 'https:') || isPrivateHostname(parsed.hostname)) {
    return { ok: false, reason: 'private-url' }
  }

  const cached = deps.store.get(url)

  if (cached) {
    return { ok: true, meta: cached }
  }

  // Name looked public — now check where it actually points. A DNS answer
  // inside private space is the SSRF doorway; any private address refuses
  // the whole fetch.
  let addresses: string[] = []

  try {
    addresses = await io.resolveHost(parsed.hostname)
  } catch {
    addresses = []
  }

  if (!addresses.length || addresses.some(isPrivateAddress)) {
    return { ok: false, reason: 'private-url' }
  }

  // The DNS rebinding window the pre-pinning version documented here is
  // closed: every hop's request now carries the addresses the guard just
  // vetted (see fetchWithGuardedRedirects), so the transport never resolves
  // the name on its own.

  const release = await deps.limiter.acquire(parsed.hostname.toLowerCase())

  try {
    // Re-check the cache after queueing: a paced-out duplicate click should
    // reuse whatever the first one fetched instead of refetching.
    const cachedAfterWait = deps.store.get(url)

    if (cachedAfterWait) {
      return { ok: true, meta: cachedAfterWait }
    }

    const html = await io.fetchHtml(url).catch(() => '')
    const tier1 = parseLinkMeta(html, url)
    let meta = tier1

    if (!meta.title) {
      // Tier 2: JS-rendered pages. The hidden window blocks image loads and
      // cancels downloads already, so it contributes a title only — the card
      // renders honestly without description/thumbnail in this leg.
      const rendered = await io.fetchRenderedTitle(url).catch(() => '')

      meta = { ...tier1, title: usableTitle((rendered || '').slice(0, PREVIEW_TITLE_MAX)) }
    }

    // Thumbnail: fetched main-process-side under the same admission as the
    // page (see fetchThumbnail's contract). A private og:image is refused
    // HERE — the policy layer never even asks the I/O layer for it — and the
    // pinned hop walk inside fetchThumbnail re-admits every redirect hop. An
    // image that cannot be proven leaves `image` empty and the card renders
    // without it; the raw URL is still shown by the chip tooltip.
    let image = ''

    if (meta.imageUrl) {
      let imageHostPrivate = true

      try {
        const imageUrl = new URL(meta.imageUrl)

        imageHostPrivate =
          (imageUrl.protocol !== 'http:' && imageUrl.protocol !== 'https:') || isPrivateHostname(imageUrl.hostname)
      } catch {
        imageHostPrivate = true
      }

      if (!imageHostPrivate) {
        image = await io.fetchThumbnail(meta.imageUrl).catch(() => '')
      }
    }

    if (!meta.title && !meta.description && !meta.siteName && !image) {
      // A fetch that yields nothing readable is a miss: not cached.
      return { ok: false, reason: 'error' }
    }

    const stored: Omit<LinkPreviewMeta, 'fetchedAt'> = {
      url,
      title: meta.title,
      description: meta.description,
      siteName: meta.siteName,
      imageUrl: meta.imageUrl,
      // The durable store never holds image bytes (a few hundred thumbnails
      // would weight the cache file enormously); the data URL rides the
      // returned envelope and the runtime keeps a small memory LRU so
      // re-expands and cache hits still render the thumbnail.
      image: ''
    }

    deps.store.set(stored)

    return { ok: true, meta: { ...stored, image, fetchedAt: (deps.now ?? Date.now)() } }
  } catch {
    return { ok: false, reason: 'error' }
  } finally {
    release()
  }
}
