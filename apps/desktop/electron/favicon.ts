/**
 * Favicon resolution, the thorough way.
 *
 * `<origin>/favicon.ico` answers for maybe half the web. Everyone else
 * declares their marks in the page head (`<link rel="icon">`, apple-touch,
 * SVG mask icons) or in a web app manifest, at paths that are frequently
 * hashed build artifacts on a CDN — unguessable. So: read the page, collect
 * every declared icon, rank them, and only then fall back to the well-known
 * paths.
 *
 * Only ever the site's own marks. A public icon service would answer for the
 * hosts that hide behind a bot wall, but asking one means telling a third
 * party which connector a user is wiring up — so a site we can't read keeps
 * its monogram instead.
 *
 * This lives in the main process because none of it is possible from the
 * renderer: cross-origin HTML is unreadable under CORS, and the whole point
 * is reading someone else's markup.
 *
 * Everything here is pure — URL math, parsing, ranking. The I/O is injected
 * so the ladder can be tested without a network.
 */

export interface IconCandidate {
  url: string
  /** Rough pixel edge, or a synthetic rank for scalable/unsized marks. */
  score: number
}

export interface FaviconIo {
  /** Page/manifest text, or '' when it can't be read. */
  fetchText: (url: string) => Promise<string>
  /** Image bytes plus the server's content type, or null on any refusal. */
  fetchImage: (url: string) => Promise<null | { bytes: Uint8Array; mime: string }>
}

/** Scalable beats every raster size; below it, bigger wins up to a point. */
const SCORE_SVG = 1024
/** `sizes="any"` — usually an SVG or a multi-res ICO. */
const SCORE_ANY = 512
/** Apple's spec size, which is what unsized apple-touch links almost always are. */
const SCORE_APPLE_TOUCH = 180
/** A declared icon with no size at all still beats a guessed path. */
const SCORE_UNSIZED = 96
/** Guessed well-known paths, tried only after everything declared. */
const SCORE_GUESS = 48

/** Past this the file is a download, not an icon. */
const SCORE_CEILING = 512

/**
 * A host worth asking for an icon.
 *
 * Loopback and RFC1918 addresses serve MCP endpoints, not brands, and an
 * icon service can't see them anyway. Refusing them here is also what keeps
 * a private hostname from being handed to that service.
 */
export function isPublicHttpUrl(raw: string): boolean {
  let url: URL

  try {
    url = new URL(raw)
  } catch {
    return false
  }

  if (url.protocol !== 'http:' && url.protocol !== 'https:') {
    return false
  }

  const host = url.hostname.toLowerCase()

  return !(
    host === 'localhost' ||
    host === '::1' ||
    host.endsWith('.local') ||
    host.endsWith('.internal') ||
    !host.includes('.') ||
    /^127\./.test(host) ||
    /^10\./.test(host) ||
    /^192\.168\./.test(host) ||
    /^169\.254\./.test(host) ||
    /^172\.(1[6-9]|2\d|3[01])\./.test(host)
  )
}

const absolute = (href: string, base: string): string => {
  try {
    // HTML attributes commonly encode query separators as `&amp;`. Decode the
    // small set of entities that affect URL resolution before handing the
    // value to URL; the favicon parser intentionally remains dependency-free.
    const decoded = href
      .trim()
      .replace(/&amp;/gi, '&')
      .replace(/&#38;/gi, '&')
      .replace(/&#x26;/gi, '&')

    const url = new URL(decoded, base)

    return url.protocol === 'http:' || url.protocol === 'https:' ? url.toString() : ''
  } catch {
    return ''
  }
}

/** "180x180 32x32" → 180. Takes the largest declared square edge. */
export function largestDeclaredSize(sizes: string): number {
  let best = 0

  for (const token of sizes.toLowerCase().split(/\s+/)) {
    if (token === 'any') {
      best = Math.max(best, SCORE_ANY)

      continue
    }

    const edge = Number(token.split('x')[0])

    if (Number.isFinite(edge)) {
      best = Math.max(best, edge)
    }
  }

  return best
}

const attr = (tag: string, name: string): string =>
  tag
    .match(new RegExp(`\\b${name}\\s*=\\s*("([^"]*)"|'([^']*)'|([^\\s"'>]+))`, 'i'))
    ?.slice(2)
    .find(Boolean) ?? ''

const scoreFor = (rel: string, type: string, sizes: string): number => {
  if (type.includes('svg') || rel.includes('mask-icon')) {
    return SCORE_SVG
  }

  const declared = largestDeclaredSize(sizes)

  if (declared > 0) {
    return Math.min(declared, SCORE_CEILING)
  }

  return rel.includes('apple-touch-icon') ? SCORE_APPLE_TOUCH : SCORE_UNSIZED
}

/** Every icon the page declares, absolute and ranked. */
export function iconCandidatesFromHtml(html: string, pageUrl: string): IconCandidate[] {
  const base = absolute(attr(html.match(/<base\b[^>]*>/i)?.[0] ?? '', 'href'), pageUrl) || pageUrl
  const candidates: IconCandidate[] = []

  for (const tag of html.match(/<link\b[^>]*>/gi) ?? []) {
    const rel = attr(tag, 'rel').toLowerCase()

    if (!/\b(icon|shortcut icon|apple-touch-icon|apple-touch-icon-precomposed|fluid-icon|mask-icon)\b/.test(rel)) {
      continue
    }

    const url = absolute(attr(tag, 'href'), base)

    if (url) {
      candidates.push({ score: scoreFor(rel, attr(tag, 'type').toLowerCase(), attr(tag, 'sizes')), url })
    }
  }

  return candidates
}

/** The page's web app manifest, if it links one. */
export function manifestUrlFromHtml(html: string, pageUrl: string): string {
  for (const tag of html.match(/<link\b[^>]*>/gi) ?? []) {
    if (/\bmanifest\b/i.test(attr(tag, 'rel'))) {
      return absolute(attr(tag, 'href'), pageUrl)
    }
  }

  return ''
}

/** PWA manifests carry the best assets a site has — 192px and 512px marks
 *  designed to stand alone on a home screen, which is exactly our use. */
export function iconCandidatesFromManifest(raw: string, manifestUrl: string): IconCandidate[] {
  let parsed: unknown

  try {
    parsed = JSON.parse(raw)
  } catch {
    return []
  }

  const icons = (parsed as { icons?: unknown })?.icons

  if (!Array.isArray(icons)) {
    return []
  }

  const candidates: IconCandidate[] = []

  for (const icon of icons) {
    const src = typeof icon?.src === 'string' ? absolute(icon.src, manifestUrl) : ''

    if (!src) {
      continue
    }

    const type = typeof icon?.type === 'string' ? icon.type.toLowerCase() : ''
    const sizes = typeof icon?.sizes === 'string' ? icon.sizes : ''

    candidates.push({ score: scoreFor('', type, sizes), url: src })
  }

  return candidates
}

/** The well-known paths, on the origin and on its apex — vendors routinely
 *  serve icons from `example.com` and nothing from `api.example.com`. */
export function fallbackIconCandidates(pageUrl: string): IconCandidate[] {
  let url: URL

  try {
    url = new URL(pageUrl)
  } catch {
    return []
  }

  const labels = url.hostname.split('.')
  const apex = labels.length > 2 ? `${url.protocol}//${labels.slice(-2).join('.')}` : url.origin
  const origins = [...new Set([url.origin, apex])]

  return origins.flatMap(origin => [
    { score: SCORE_GUESS + 2, url: `${origin}/apple-touch-icon.png` },
    { score: SCORE_GUESS + 1, url: `${origin}/apple-touch-icon-precomposed.png` },
    { score: SCORE_GUESS, url: `${origin}/favicon.ico` },
    { score: SCORE_GUESS - 1, url: `${origin}/favicon.png` }
  ])
}

/** Highest-ranked first, one entry per URL, capped so a page declaring
 *  twenty icons can't turn one card into twenty requests. */
export function rankCandidates(candidates: IconCandidate[], limit = 6): IconCandidate[] {
  const best = new Map<string, number>()

  for (const candidate of candidates) {
    best.set(candidate.url, Math.max(best.get(candidate.url) ?? 0, candidate.score))
  }

  return [...best.entries()]
    .map(([url, score]) => ({ score, url }))
    .sort((a, b) => b.score - a.score)
    .slice(0, limit)
}

/** Magic bytes, because plenty of servers hand back an icon as
 *  `application/octet-stream` — and an HTML error page as `image/png`. */
export function sniffImageMime(bytes: Uint8Array): string {
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

  const head = new TextDecoder().decode(bytes.subarray(0, 1024)).toLowerCase()

  return head.includes('<svg') ? 'image/svg+xml' : ''
}

/**
 * The mime to trust for these bytes, or '' if they aren't an image.
 *
 * The bytes decide, never the header. A blocked request answers 200 with an
 * HTML challenge page under `content-type: image/png` often enough that
 * believing the server is how you end up rendering a broken-image box.
 */
export function imageMime(declared: null | string | undefined, bytes: Uint8Array): string {
  if (bytes.length < 48) {
    return ''
  }

  const sniffed = sniffImageMime(bytes)

  // An SVG that opens with a license comment long enough to push `<svg` past
  // the sniff window is still an SVG if the server said so.
  return sniffed || (declared?.toLowerCase().includes('svg') ? 'image/svg+xml' : '')
}

export const toDataUrl = (mime: string, bytes: Uint8Array): string =>
  `data:${mime};base64,${Buffer.from(bytes).toString('base64')}`

/**
 * Walk the ladder and return the first real image, as a data URL.
 *
 * Data URL rather than a link so the renderer paints without a second
 * network trip, the icon survives a site going down, and one cached string
 * covers every surface showing that connector.
 */
export async function resolveFavicon(pageUrl: string, io: FaviconIo): Promise<string> {
  if (!isPublicHttpUrl(pageUrl)) {
    return ''
  }

  const candidates: IconCandidate[] = []
  const html = await io.fetchText(pageUrl).catch(() => '')

  if (html) {
    candidates.push(...iconCandidatesFromHtml(html, pageUrl))

    const manifestUrl = manifestUrlFromHtml(html, pageUrl)

    if (manifestUrl) {
      const manifest = await io.fetchText(manifestUrl).catch(() => '')

      if (manifest) {
        candidates.push(...iconCandidatesFromManifest(manifest, manifestUrl))
      }
    }
  }

  candidates.push(...fallbackIconCandidates(pageUrl))

  // Only the site's own marks. A third-party icon service would answer for
  // the hosts that serve nothing readable, but asking it means naming a
  // connector's host to someone else — so a site that won't show us its icon
  // simply keeps its monogram.
  for (const candidate of rankCandidates(candidates)) {
    const image = await io.fetchImage(candidate.url).catch(() => null)

    if (!image) {
      continue
    }

    const mime = imageMime(image.mime, image.bytes)

    if (mime) {
      return toDataUrl(mime, image.bytes)
    }
  }

  return ''
}
