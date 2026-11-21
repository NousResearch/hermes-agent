import { Skeleton } from '@/components/ui/skeleton'
import { useI18n } from '@/i18n'

import type { CatalogKind } from './catalog-data'

/** Cards in the first shelf's row: the bento's four columns. */
const SHELF_CARDS = 4

function ShelfSkeleton() {
  return (
    <section>
      <header className="catalog-section-heading">
        <div>
          <Skeleton className="catalog-skeleton-title" />
          <Skeleton className="catalog-skeleton-blurb" />
        </div>
      </header>
      <div className="catalog-bento">
        {Array.from({ length: SHELF_CARDS }, (_, index) => (
          <Skeleton className="catalog-skeleton-card" key={index} />
        ))}
      </div>
    </section>
  )
}

/** A library's first paint while its list loads (redesign only): the page's own frame, so it holds
 *  its final layout and the shelves fill in place instead of replacing a spinner. A catalog feed
 *  (`kind`) opens on its real "Discover …" heading, the hero's slot and the slider row; the Tools
 *  library opens on its shelves. */
export function CatalogSkeleton({ kind }: { kind?: CatalogKind }) {
  const { t } = useI18n()
  const title = kind === 'plugins' ? t.skills.tabPlugins : t.skills.tabSkills

  return (
    <div aria-busy="true" aria-label={t.skills.loading} className="capabilities-scroll" role="status">
      <div className="catalog-discovery" data-catalog-skeleton>
        {kind && (
          <section>
            <header className="catalog-section-heading">
              <h2>{t.catalog.discoverKind(title)}</h2>
            </header>
            <Skeleton className="catalog-skeleton-hero" />
            <div aria-hidden className="catalog-featured-controls">
              <Skeleton className="catalog-skeleton-dots" />
            </div>
          </section>
        )}
        <ShelfSkeleton />
        {!kind && <ShelfSkeleton />}
      </div>
    </div>
  )
}
