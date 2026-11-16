import '../catalog/catalog-discovery.css'

import type { ReactNode } from 'react'

import { CatalogCard, type CatalogCardVariant } from '../catalog/catalog-card'
import type { CatalogEntry, CatalogKind } from '../catalog/catalog-data'
import { CatalogDeveloper } from '../catalog/catalog-developer'

export interface CatalogSurfaceItem {
  id: string
  name: string
  description: string
  /** Topline label: who or what the item belongs to. */
  eyebrow?: string
  /** Footer badge: the one piece of meta the item carries. */
  badge?: string
  action?: ReactNode
  onOpen: () => void
}

export interface CatalogSurfaceSection {
  id: string
  label: string
  blurb?: string
  /** `feature` is the tall-card row; `list` is the two-up small-banner list. */
  layout: 'feature' | 'list'
  items: CatalogSurfaceItem[]
}

const VARIANT: Record<CatalogSurfaceSection['layout'], CatalogCardVariant> = { feature: 'compact', list: 'horizontal' }

const entryFor = (item: CatalogSurfaceItem, category: string): CatalogEntry => ({
  id: item.id,
  name: item.name,
  description: item.description,
  overview: item.description,
  category,
  categoryLabel: category,
  source: item.badge ?? '',
  author: item.eyebrow ?? '',
  identifier: item.id,
  repo: '',
  sha: '',
  subdir: '',
  version: '',
  requiresHermes: '',
  tags: [],
  platforms: [],
  requirements: [],
  tools: [],
  hooks: [],
  sourceUrl: null,
  docsUrl: null,
  imageUrl: null,
  stars: null,
  search: ''
})

const noop = () => {}

/** The catalog page for a capability that isn't a registry feed: the same
 *  headings, cards and developer banner, laid out as the board's feature row
 *  and small-banner lists. State and actions stay with the caller. */
export function CatalogSurface({ sections, kind, actions }: { sections: CatalogSurfaceSection[]; kind: CatalogKind; actions?: ReactNode }) {
  return (
    <div className="capabilities-scroll">
      <div className="catalog-discovery">
        {sections.filter(section => section.items.length).map(section => (
          <section data-catalog-section={section.id} key={section.id}>
            <header className="catalog-section-heading">
              <div>
                <h2>{section.label}</h2>
                {section.blurb && <p>{section.blurb}</p>}
              </div>
            </header>
            <div className={section.layout === 'feature' ? 'catalog-feature-row' : 'catalog-banner-list'}>
              {section.items.map((item, index) => (
                <CatalogCard
                  accentIndex={index}
                  action={item.action}
                  entry={entryFor(item, section.id)}
                  key={item.id}
                  onCategory={noop}
                  onOpen={item.onOpen}
                  onSearch={noop}
                  onTag={noop}
                  variant={VARIANT[section.layout]}
                />
              ))}
            </div>
          </section>
        ))}
        <CatalogDeveloper actions={actions} kind={kind} />
      </div>
    </div>
  )
}
