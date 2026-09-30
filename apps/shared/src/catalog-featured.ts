/** The catalog hero pick, shared by the website and the desktop app so both
 *  feature the same entry at the same time.
 *
 *  Curated picks come from `catalog-curation.json` → `featured`, stamped into
 *  the published catalog as a 1-based `featured` rank by the extractors. The
 *  hero rotates through them weekly. With no curated pick in the feed it falls
 *  back to the newest pictured official entry of the last 30 days, then the
 *  official entry with art and the most stars. Nothing here reads a download
 *  or install count; the feed has none. */

export interface FeaturedCandidate {
  /** 1-based curated rank from the feed; absent for everything uncurated. */
  featured?: number | null
  official: boolean
  pictured: boolean
  addedAt?: string | null
  stars?: number | null
}

const DAY_MS = 24 * 60 * 60 * 1000
const WEEK_MS = 7 * DAY_MS
const FRESH_MS = 30 * DAY_MS

const rank = (value: FeaturedCandidate) => (typeof value.featured === 'number' && value.featured > 0 ? value.featured : 0)
const added = (value: FeaturedCandidate) => Date.parse(value.addedAt ?? '') || 0

export function pickFeatured<T>(entries: readonly T[], read: (entry: T) => FeaturedCandidate, now = Date.now()): T | undefined {
  const rows = entries.map(entry => ({ entry, value: read(entry) }))
  const curated = rows.filter(row => rank(row.value)).sort((a, b) => rank(a.value) - rank(b.value))

  if (curated.length) {
    return curated[Math.floor(now / WEEK_MS) % curated.length].entry
  }

  const official = rows.filter(row => row.value.official)
  const fresh = official.filter(row => row.value.pictured && added(row.value) > now - FRESH_MS).sort((a, b) => added(b.value) - added(a.value))

  return (fresh[0] ?? official.sort((a, b) => Number(b.value.pictured) - Number(a.value.pictured) || (b.value.stars ?? 0) - (a.value.stars ?? 0))[0])?.entry
}
