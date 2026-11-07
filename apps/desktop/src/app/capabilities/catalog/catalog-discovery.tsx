import './catalog-discovery.css'

import { featuredPicks, groupCatalogPlugins, isOfficialSkillSource, PLUGIN_CATEGORIES } from '@hermes/shared'
import { type ReactNode, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { useI18n } from '@/i18n'

import { BENTO_COLUMNS, packBento } from './catalog-bento'
import type { CatalogCardVariant } from './catalog-card'
import { pluginTier, type PluginTier } from './catalog-curation'
import { type CatalogEntry, type CatalogKind, catalogLabel } from './catalog-data'
import { CatalogDeveloper } from './catalog-developer'
import { CatalogFeaturedSlider } from './catalog-featured-slider'
import { catalogCategories } from './catalog-query'

/** Fewest cards a category needs for its own shelf; smaller ones share "More to explore". */
const MIN_SHELF = 3


/** A shelf ranks by its most-starred entry, so the strongest categories lead. */
const topStars = (items: CatalogEntry[]) => Math.max(-1, ...items.map(entry => entry.stars ?? -1))

/** Plugin tiers in page order; community keeps its category shelves. */
const PLUGIN_TIERS: Exclude<PluginTier, 'community'>[] = ['official', 'verified']

/** Category shelves shown at first, and how many more each "Show more" adds. */
const CATEGORY_PAGE = 8

/** Rows of the mixed "More to explore" shelf shown at first, and per "Show more". */
const LOOSE_ROWS_PAGE = 3

export function CatalogDiscovery({ entries, kind, card, onCategory, actions, narrowed, byCategory }: {
  entries: CatalogEntry[]
  kind: CatalogKind
  card: (entry: CatalogEntry, index: number, variant: CatalogCardVariant, placement?: React.CSSProperties) => ReactNode
  onCategory: (category: string) => void
  actions?: ReactNode
  /** A tier filter is on: the shelves show only those tiers, so no hero pick. */
  narrowed?: boolean
  /** Several categories picked: one shelf per category, tiers and small-shelf folding off. */
  byCategory?: boolean
}) {
  const { t } = useI18n()
  const [categoryLimit, setCategoryLimit] = useState(CATEGORY_PAGE)
  const [looseRows, setLooseRows] = useState(LOOSE_ROWS_PAGE)
  const title = kind === 'plugins' ? t.skills.tabPlugins : t.skills.tabSkills

  // Same pick as the website: curated ranks from the feed, rotated weekly.
  // Every curated pick, this week's first; more than one becomes a slider.
  const featured = narrowed ? [] : featuredPicks(entries, entry => ({
    featured: entry.featured,
    official: kind === 'plugins' ? pluginTier(entry) === 'official' : isOfficialSkillSource(entry.source),
    pictured: Boolean(entry.imageUrl),
    addedAt: entry.addedAt,
    stars: entry.stars
  }))

  // A slide leaves its shelf, as the single hero always did.
  const others = entries.filter(entry => !featured.includes(entry))
  const tiered = kind === 'plugins' && !byCategory ? PLUGIN_TIERS.map(tier => ({ tier, ...t.catalog.tiers[tier], items: others.filter(entry => pluginTier(entry) === tier) })).filter(({ items }) => items.length) : []
  const remaining = kind === 'plugins' && !byCategory ? others.filter(entry => pluginTier(entry) === 'community') : others

  // Plugins with an unlisted category land in `general`, as in every other plugin grouping.
  const sections: [string, string, CatalogEntry[]][] = (
    kind === 'plugins'
      ? groupCatalogPlugins(remaining).map(([key, items]) => [key, PLUGIN_CATEGORIES[key].label, items] as [string, string, CatalogEntry[]])
      : catalogCategories(remaining).map(([key, meta]) => [key, meta.label, remaining.filter(entry => entry.category === key)] as [string, string, CatalogEntry[]])
  ).sort((a, b) => topStars(b[2]) - topStars(a[2]))

  // A category too small to fill a row reads as a stray full-width strip; those
  // fold into one mixed shelf at the end instead.
  // (A category the user picked always gets its own shelf, however small.)
  const minShelf = byCategory ? 1 : MIN_SHELF
  const shelves = sections.filter(([, , items]) => items.length >= minShelf)
  const loose = sections.filter(([, , items]) => items.length < minShelf).flatMap(([, , items]) => items)

  const bento = (items: CatalogEntry[], rows?: number, flip = false) => (
    <div className="catalog-bento" data-catalog-hover-group>
      {packBento(items, BENTO_COLUMNS, rows, flip).map(tile => card(tile.entry, tile.row, tile.variant, {
        '--bento-x': tile.column + 1,
        '--bento-y': tile.row + 1,
        '--bento-w': tile.w,
        '--bento-h': tile.h
      } as React.CSSProperties))}
    </div>
  )

  return (
    <div className="catalog-discovery">
      {featured.length > 0 && <section data-catalog-section="featured">
        <header className="catalog-section-heading"><h2>{t.catalog.discoverKind(title)}</h2></header>
        <CatalogFeaturedSlider slides={featured.map((entry, index) => card(entry, index, 'hero'))} />
      </section>}
      {tiered.map(({ tier, title: tierTitle, blurb, items }, index) => (
        <section data-catalog-section={tier} data-catalog-tier={tier} key={tier}>
          <header className="catalog-section-heading">
            <div>
              <h2>{tierTitle}</h2>
              <p>{blurb}</p>
            </div>
          </header>
          {bento(items, Math.ceil(items.length / BENTO_COLUMNS) + 1, index % 2 === 1)}
        </section>
      ))}
      {shelves.slice(0, categoryLimit).map(([category, label, items], index) => {
        return <section data-catalog-section={category} key={category}>
          <header className="catalog-section-heading">
            <div>
              <h2><Button className="text-[length:inherit] font-[inherit] text-(--ui-text-primary) hover:text-(--ui-text-primary)" onClick={() => onCategory(category)} size="inline" variant="text">{catalogLabel(label)}</Button></h2>
              <p>{kind === 'plugins' ? PLUGIN_CATEGORIES[category]?.blurb : t.catalog.skillsFor(catalogLabel(label))}</p>
            </div>
            {items.length > packBento(items).length && <Button onClick={() => onCategory(category)} size="inline" variant="text">{t.catalog.seeAll}<Codicon name="arrow-right" /></Button>}
          </header>
          {bento(items, undefined, index % 2 === 1)}
        </section>
      })}
      {shelves.length > categoryLimit && <Button onClick={() => setCategoryLimit(value => value + CATEGORY_PAGE)} size="sm" variant="text">{t.catalog.more}</Button>}
      {loose.length > 0 && shelves.length <= categoryLimit && <section data-catalog-section="more">
        <header className="catalog-section-heading">
          <div>
            <h2>{t.catalog.moreToExplore}</h2>
            <p>{t.catalog.moreToExploreBlurb}</p>
          </div>
        </header>
        {bento(loose, looseRows)}
        {loose.length > packBento(loose, BENTO_COLUMNS, looseRows).length && <Button className="mt-4" onClick={() => setLooseRows(value => value + LOOSE_ROWS_PAGE)} size="sm" variant="text">{t.catalog.more}</Button>}
      </section>}
      <CatalogDeveloper actions={actions} kind={kind} />
    </div>
  )
}
