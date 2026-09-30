import './catalog-discovery.css'

import { groupCatalogPlugins, pickFeatured, PLUGIN_CATEGORIES } from '@hermes/shared'
import { type ReactNode, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { useI18n } from '@/i18n'

import { BENTO_COLUMNS, packBento } from './catalog-bento'
import type { CatalogCardVariant } from './catalog-card'
import { type CatalogEntry, type CatalogKind, catalogLabel } from './catalog-data'
import { CatalogDeveloper } from './catalog-developer'
import { catalogCategories } from './catalog-query'

/** Fewest cards a category needs for its own shelf; smaller ones share "More to explore". */
const MIN_SHELF = 3

/** First-party sources, per kind, for the hero's fallback when the feed curates nothing. */
const OFFICIAL = { plugins: ['official'], skills: ['built-in', 'bundled', 'optional', 'official'] } as const

export function CatalogDiscovery({ entries, kind, card, onCategory, actions }: {
  entries: CatalogEntry[]
  kind: CatalogKind
  card: (entry: CatalogEntry, index: number, variant: CatalogCardVariant, placement?: React.CSSProperties) => ReactNode
  onCategory: (category: string) => void
  actions?: ReactNode
}) {
  const { t } = useI18n()
  const [categoryLimit, setCategoryLimit] = useState(8)
  const [looseRows, setLooseRows] = useState(3)
  const title = kind === 'plugins' ? t.skills.tabPlugins : t.skills.tabSkills

  // Same pick as the website: curated ranks from the feed, rotated weekly.
  const featured = pickFeatured(entries, entry => ({
    featured: entry.featured,
    official: (OFFICIAL[kind] as readonly string[]).includes(entry.source),
    pictured: Boolean(entry.imageUrl),
    addedAt: entry.addedAt,
    stars: entry.stars
  }))

  const remaining = entries.filter(entry => entry !== featured)

  // Plugins with an unlisted category land in `general`, as in every other plugin grouping.
  const sections: [string, string, CatalogEntry[]][] = (
    kind === 'plugins'
      ? groupCatalogPlugins(remaining).map(([key, items]) => [key, PLUGIN_CATEGORIES[key].label, items] as [string, string, CatalogEntry[]])
      : catalogCategories(remaining).map(([key, meta]) => [key, meta.label, remaining.filter(entry => entry.category === key)] as [string, string, CatalogEntry[]])
  ).sort((a, b) => Number(b[2].some(entry => entry.imageUrl)) - Number(a[2].some(entry => entry.imageUrl)))

  // A category too small to fill a row reads as a stray full-width strip; those
  // fold into one mixed shelf at the end instead.
  const shelves = sections.filter(([, , items]) => items.length >= MIN_SHELF)
  const loose = sections.filter(([, , items]) => items.length < MIN_SHELF).flatMap(([, , items]) => items)

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
      {featured && <section data-catalog-section="featured">
        <header className="catalog-section-heading"><h2>{t.catalog.discover} {title}</h2></header>
        {card(featured, 0, 'hero')}
      </section>}
      {shelves.slice(0, categoryLimit).map(([category, label, items], index) => {
        return <section data-catalog-section={category} key={category}>
          <header className="catalog-section-heading">
            <div>
              <h2><Button className="text-[length:inherit] font-[inherit] text-(--ui-text-primary) hover:text-(--ui-text-primary)" onClick={() => onCategory(category)} size="inline" variant="text">{catalogLabel(label)}</Button></h2>
              <p>{kind === 'plugins' ? PLUGIN_CATEGORIES[category]?.blurb : `Skills for ${catalogLabel(label).toLowerCase()}.`}</p>
            </div>
            {items.length > packBento(items).length && <Button onClick={() => onCategory(category)} size="inline" variant="text">{t.catalog.seeAll}<Codicon name="arrow-right" /></Button>}
          </header>
          {bento(items, undefined, index % 2 === 1)}
        </section>
      })}
      {shelves.length > categoryLimit && <Button onClick={() => setCategoryLimit(value => value + 8)} size="sm" variant="text">{t.catalog.more}</Button>}
      {loose.length > 0 && shelves.length <= categoryLimit && <section data-catalog-section="more">
        <header className="catalog-section-heading">
          <div>
            <h2>More to explore</h2>
            <p>Smaller corners of the catalog, all in one place.</p>
          </div>
        </header>
        {bento(loose, looseRows)}
        {loose.length > packBento(loose, BENTO_COLUMNS, looseRows).length && <Button className="mt-4" onClick={() => setLooseRows(value => value + 3)} size="sm" variant="text">{t.catalog.more}</Button>}
      </section>}
      <CatalogDeveloper actions={actions} kind={kind} />
    </div>
  )
}
