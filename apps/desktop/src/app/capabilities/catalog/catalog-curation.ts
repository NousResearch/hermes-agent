import { isOfficialSkillSource } from '@hermes/shared'

import curation from '../../../../../shared/src/catalog-curation.json'

const { artwork, featured } = curation

const images = import.meta.glob<string>('../../../../../../website/static/img/catalog/*.jpg', {
  eager: true,
  import: 'default',
  query: '?url'
})

const artworkUrl = (filename: string | undefined) =>
  filename ? images[`../../../../../../website/static/img/catalog/${filename}`] ?? null : null

interface CurationRow {
  name?: string
  category?: string
  identifier?: string
  source?: string
  tier?: string
}

/** Official identities whose art the curation file owns: a reassignment there
 *  must win over a feed published with the old mapping. */
export function curationOwnsArtwork(kind: 'skills' | 'plugins', row: CurationRow): boolean {
  return kind === 'skills' ? isOfficialSkillSource(row.source) : row.tier === 'bundled'
}

/** Only first-party catalog identities receive editorial artwork. */
export function officialCatalogArtwork(kind: 'skills' | 'plugins', row: CurationRow): string | null {
  if (!curationOwnsArtwork(kind, row)) {return null}

  return kind === 'skills'
    ? artworkUrl((artwork.skills as Record<string, string>)[`${row.category}/${row.name}`])
    : artworkUrl((artwork.plugins as Record<string, string>)[row.identifier ?? ''])
}

/** Curated hero rank from the bundled curation file, for feeds published before
 *  the extractors stamped `featured`. Same keys the extractors use, so the app
 *  and the site agree once the feed catches up. */
export function curatedFeaturedRank(kind: 'skills' | 'plugins', row: CurationRow): number | undefined {
  const key = kind === 'skills'
    ? (isOfficialSkillSource(row.source) ? `${row.category}/${row.name}` : '')
    : (row.tier === 'official' ? row.name ?? '' : '')

  const rank = (featured[kind] as string[]).indexOf(key) + 1

  return key && rank ? rank : undefined
}

export type PluginTier = 'official' | 'verified' | 'community'

/** Trust order: the tier filter, chips and shelves all read it. */
export const PLUGIN_TIER_ORDER: readonly PluginTier[] = ['official', 'verified', 'community']

/** Catalog tier straight from the feed (`tier` in plugin-catalog/*.yaml):
 *  official and bundled are Nous, verified is granted by Nous to plugins from
 *  the company behind the integration, everything else is community —
 *  including installed-only values (`user`, `git`, desktop kinds). */
export function pluginTier(entry: { source: string }): PluginTier {
  if (entry.source === 'official' || entry.source === 'bundled') {
    return 'official'
  }

  return entry.source === 'verified' ? 'verified' : 'community'
}
