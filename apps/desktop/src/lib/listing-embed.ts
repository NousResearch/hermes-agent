/**
 * LISTING EMBEDS — property/rental listings as native transcript cards.
 *
 * The rental portals (Zillow, Redfin, Compass, HAR, homes.com) publish no
 * embeddable surface: the public APIs are deprecated or broker-gated, there is
 * no oEmbed endpoint, and every listing page ships `X-Frame-Options`. So a URL
 * cannot be framed and there is nothing to steal at the transport layer.
 *
 * What IS available is the data the agent already gathered to answer the
 * question, plus the portals' own image CDNs (which serve hotlinked `<img>`
 * fine). So the embed is authored, not fetched: the agent emits a ```listing
 * fence holding normalized JSON and it renders as a browsable card — gallery,
 * facts, and the catches that decide a tour.
 *
 * This module is the pure half: parse + validate + normalize. It runs during
 * render, so it stays synchronous and dependency-free, and it treats every
 * field as untrusted model output — bad input degrades to a plain code block
 * rather than a broken card.
 */

/** Cap the gallery so a pathological payload can't mount 400 `<img>` tags. */
const MAX_IMAGES = 40
/** Cap list-shaped fields (facts, catches) for the same reason. */
const MAX_LIST_ITEMS = 12
/** Cap a whole fence — several listings, not a database dump. */
const MAX_LISTINGS = 24
/** Longest single string we'll render; addresses and notes, not essays. */
const MAX_TEXT = 400

export interface ListingLink {
  label: string
  url: string
}

export interface Listing {
  /** Street address / headline. The card's primary identity. */
  address: string
  /** Formatted as given ("$12,500/mo") — the agent knows the market's units. */
  price?: string
  beds?: number
  baths?: number
  /** Interior size, already formatted ("2,136 sqft"). */
  size?: string
  /** Why this one is worth the tour — the agent's own pitch. */
  note?: string
  /** Short amenity/spec chips ("private dock", "12-mo lease"). */
  facts: string[]
  /** Risks worth verifying before touring — the honest half of the card. */
  catches: string[]
  /** Gallery image URLs (https only). */
  images: string[]
  /** Canonical detail-page links; multiple portals mirror one property. */
  links: ListingLink[]
  /** Stable key for React lists. */
  id: string
}

function text(value: unknown): string | undefined {
  if (typeof value !== 'string') {
    return undefined
  }

  const trimmed = value.trim().slice(0, MAX_TEXT)

  return trimmed || undefined
}

/** Beds/baths accept `3` and `"3.5"` — models emit both. Rejects absurd counts. */
function count(value: unknown): number | undefined {
  const parsed = typeof value === 'number' ? value : typeof value === 'string' ? Number(value.trim()) : NaN

  if (!Number.isFinite(parsed) || parsed <= 0 || parsed > 99) {
    return undefined
  }

  // One decimal place: half-baths are real, quarter-baths are not.
  return Math.round(parsed * 10) / 10
}

/** Only absolute http(s) — a card must never smuggle `javascript:` into an href. */
function webUrl(value: unknown): string | undefined {
  const raw = text(value)

  if (!raw) {
    return undefined
  }

  try {
    const url = new URL(raw)

    return url.protocol === 'https:' || url.protocol === 'http:' ? url.toString() : undefined
  } catch {
    return undefined
  }
}

function stringList(value: unknown, limit = MAX_LIST_ITEMS): string[] {
  if (!Array.isArray(value)) {
    // A single string is a one-item list — models drop the array constantly.
    const single = text(value)

    return single ? [single] : []
  }

  const items: string[] = []

  for (const entry of value) {
    const item = text(entry)

    if (item) {
      items.push(item)
    }

    if (items.length >= limit) {
      break
    }
  }

  return items
}

function urlList(value: unknown, limit = MAX_IMAGES): string[] {
  const urls: string[] = []

  for (const entry of Array.isArray(value) ? value : [value]) {
    const url = webUrl(entry)

    if (url && !urls.includes(url)) {
      urls.push(url)
    }

    if (urls.length >= limit) {
      break
    }
  }

  return urls
}

/** Links accept `["https://…"]` and `[{label, url}]`; the host is a fine
 *  default label when the model didn't name the portal. */
function linkList(value: unknown): ListingLink[] {
  const links: ListingLink[] = []

  for (const entry of Array.isArray(value) ? value : [value]) {
    const url = webUrl(typeof entry === 'object' && entry !== null ? (entry as { url?: unknown }).url : entry)

    if (!url || links.some(existing => existing.url === url)) {
      continue
    }

    const label =
      (typeof entry === 'object' && entry !== null ? text((entry as { label?: unknown }).label) : undefined) ??
      hostLabel(url)

    links.push({ label, url })

    if (links.length >= MAX_LIST_ITEMS) {
      break
    }
  }

  return links
}

/** `www.homes.com` → `homes.com`. Bare host reads as the portal's name. */
export function hostLabel(url: string): string {
  try {
    return new URL(url).hostname.replace(/^www\./i, '')
  } catch {
    return 'Listing'
  }
}

function normalizeListing(raw: unknown, index: number): Listing | null {
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) {
    return null
  }

  const source = raw as Record<string, unknown>
  // `title` is the common synonym when the property isn't street-addressed.
  const address = text(source.address) ?? text(source.title)

  // A card with no identity is not a card. Everything else is optional, so a
  // sparse listing still renders (and shows what it does have).
  if (!address) {
    return null
  }

  return {
    address,
    baths: count(source.baths),
    beds: count(source.beds),
    catches: stringList(source.catches),
    facts: stringList(source.facts),
    id: text(source.id) ?? `${address}:${index}`,
    images: urlList(source.images ?? source.image),
    links: linkList(source.links ?? source.url),
    note: text(source.note) ?? text(source.why),
    price: text(source.price),
    size: text(source.size) ?? text(source.sqft)
  }
}

/**
 * Parse a ```listing fence. Accepts a single object, a bare array, or
 * `{ listings: [...] }` — all three shapes show up in practice. Returns an
 * empty array on anything unparseable so the caller can fall back to the
 * plain code block.
 */
export function parseListings(code: string): Listing[] {
  let data: unknown

  try {
    data = JSON.parse(code)
  } catch {
    return []
  }

  const entries = Array.isArray(data)
    ? data
    : typeof data === 'object' && data !== null && Array.isArray((data as { listings?: unknown }).listings)
      ? (data as { listings: unknown[] }).listings
      : [data]

  const listings: Listing[] = []

  for (const [index, entry] of entries.slice(0, MAX_LISTINGS).entries()) {
    const listing = normalizeListing(entry, index)

    if (listing) {
      listings.push(listing)
    }
  }

  return listings
}

/** The metadata line under the address: "3 bd · 2.5 ba · 2,136 sqft". */
export function specLine(listing: Listing): string {
  const parts: string[] = []

  if (listing.beds !== undefined) {
    parts.push(`${listing.beds} bd`)
  }

  if (listing.baths !== undefined) {
    parts.push(`${listing.baths} ba`)
  }

  if (listing.size) {
    parts.push(listing.size)
  }

  return parts.join(' · ')
}
