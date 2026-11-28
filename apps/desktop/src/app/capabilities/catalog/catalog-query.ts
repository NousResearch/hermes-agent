import { sortCatalogPlugins } from '@hermes/shared'

import { PLUGIN_TIER_ORDER, pluginTier } from './catalog-curation'
import type { CatalogEntry, CatalogKind } from './catalog-data'
import type { FacetRow } from './catalog-filters'

export type CatalogSort = 'discover' | 'usage' | 'stars' | 'newest' | 'updated' | 'name'

export interface CatalogFacets {
  sources: string[]
  categories: string[]
  tags: string[]
  installedOnly: boolean
  updatesOnly: boolean
}

export const EMPTY_FACETS: CatalogFacets = { sources: [], categories: [], tags: [], installedOnly: false, updatesOnly: false }

/** Facet value an entry files under: the normalized tier for plugins, the registry for skills. */
export const facetSource = (entry: CatalogEntry, kind: CatalogKind) => (kind === 'plugins' ? pluginTier(entry) : entry.source)

/** Source facet rows; for plugins, the tiers present, in trust order. */
export function catalogSources(entries: CatalogEntry[], kind: CatalogKind) {
  const present = new Set(entries.map(entry => facetSource(entry, kind)))

  return kind === 'plugins' ? PLUGIN_TIER_ORDER.filter(tier => present.has(tier)) : [...present]
}

/** Category facet rows with counts, A–Z with "Other" last. */
export function catalogCategories(entries: CatalogEntry[]) {
  const values = new Map<string, { label: string; count: number }>()

  for (const entry of entries) {
    values.set(entry.category, { label: entry.categoryLabel, count: (values.get(entry.category)?.count ?? 0) + 1 })
  }

  return [...values].sort(byCategoryOrder)
}

const categoryRank = (row: [string, { label: string }]) => (row[0] === 'other' || row[1].label.toLowerCase() === 'other' ? 1 : 0)

const byCategoryOrder = (a: [string, { label: string }], b: [string, { label: string }]) =>
  categoryRank(a) - categoryRank(b) || a[1].label.localeCompare(b[1].label)

/** One `[key, label, entries]` section per category, in `catalogCategories` order, entries in feed order.
 *  Grouped in a single pass, because a filter per category rereads the whole catalog once per category. */
export function categorySections(entries: CatalogEntry[]): [string, string, CatalogEntry[]][] {
  const sections = new Map<string, { label: string; items: CatalogEntry[] }>()

  for (const entry of entries) {
    const section = sections.get(entry.category)

    if (section) {
      // The last label seen wins, as in `catalogCategories`.
      section.label = entry.categoryLabel
      section.items.push(entry)
    } else {
      sections.set(entry.category, { label: entry.categoryLabel, items: [entry] })
    }
  }

  return [...sections].sort(byCategoryOrder).map(([key, { label, items }]) => [key, label, items])
}

/** The most common tags plus whatever is selected; search covers the long tail. */
export function catalogTags(entries: CatalogEntry[], selected: string[], limit: number): FacetRow[] {
  const counts = new Map<string, number>()

  for (const entry of entries) {
    for (const value of new Set(entry.tags)) {
      counts.set(value, (counts.get(value) ?? 0) + 1)
    }
  }

  const ranked = [...counts].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
  const shown = new Set([...ranked.slice(0, limit).map(([value]) => value), ...selected])

  return ranked.filter(([value]) => shown.has(value)).map(([value, count]) => ({ value, label: value, count }))
}

/** Facets combine as AND across groups and OR within one. */
export function filterCatalog(
  entries: CatalogEntry[],
  facets: CatalogFacets,
  query: string,
  isInstalled: (entry: CatalogEntry) => boolean,
  kind: CatalogKind
) {
  const { sources, categories, tags, installedOnly, updatesOnly } = facets

  return entries.filter(
    entry =>
      (!sources.length || sources.includes(facetSource(entry, kind))) &&
      (!categories.length || categories.includes(entry.category)) &&
      (!tags.length || tags.some(tag => entry.tags.includes(tag))) &&
      (!installedOnly || isInstalled(entry)) &&
      (!updatesOnly || Boolean(entry.updateAvailable)) &&
      (!query || entry.search.includes(query))
  )
}

/** Returns a new array; `discover` keeps the feed's curated order. */
export function sortCatalog(entries: CatalogEntry[], sort: CatalogSort, kind: CatalogKind) {
  if (sort === 'discover') {
    return entries
  }

  if (kind === 'plugins' && sort !== 'name' && sort !== 'usage') {
    return sortCatalogPlugins(entries, sort)
  }

  if (sort === 'usage') {
    return [...entries].sort((a, b) => (b.usage ?? -1) - (a.usage ?? -1) || a.name.localeCompare(b.name))
  }

  if (sort === 'name') {
    return [...entries].sort((a, b) => a.name.localeCompare(b.name))
  }

  if (sort === 'stars') {
    return [...entries].sort((a, b) => (b.stars ?? -1) - (a.stars ?? -1) || a.name.localeCompare(b.name))
  }

  const field = sort === 'newest' ? 'addedAt' : 'updatedAt'
  const time = new Map(entries.map(entry => [entry.id, Date.parse(entry[field] ?? '') || 0]))

  return [...entries].sort((a, b) => time.get(b.id)! - time.get(a.id)! || a.name.localeCompare(b.name))
}

/** First `limit` entries sharing a category or tag with `entry`; stops early on large feeds. */
export function relatedEntries(entries: CatalogEntry[], entry: CatalogEntry, limit: number) {
  const matches: CatalogEntry[] = []

  for (const candidate of entries) {
    if (matches.length === limit) {
      break
    }

    if (
      candidate.id !== entry.id &&
      (candidate.category === entry.category || candidate.tags.some(tag => entry.tags.includes(tag)))
    ) {
      matches.push(candidate)
    }
  }

  return matches
}

export function catalogSortOptions(entries: CatalogEntry[], kind: CatalogKind, labels: Record<CatalogSort, string>) {
  const available: Record<CatalogSort, boolean> = {
    discover: kind === 'skills',
    usage: entries.some(entry => (entry.usage ?? 0) > 0),
    stars: entries.some(entry => entry.stars !== null),
    newest: entries.some(entry => entry.addedAt),
    updated: entries.some(entry => entry.updatedAt),
    name: kind === 'skills'
  }

  return (Object.keys(available) as CatalogSort[])
    .filter(value => available[value])
    .map(value => ({ value, label: labels[value] }))
}
