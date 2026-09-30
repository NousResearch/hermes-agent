import { useStore } from '@nanostores/react'
import { memo, type ReactNode, useDeferredValue, useEffect, useRef, useState } from 'react'

import { PageLoader } from '@/components/page-loader'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { ErrorState } from '@/components/ui/error-state'
import { Masonry } from '@/components/ui/masonry'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'

import { DetailColumn, ListColumn, MasterDetail } from '../../master-detail'
import { PanelEmpty } from '../../overlays/panel'
import { CapabilitySearch } from '../ui/capability-search'

import { CatalogAlert } from './catalog-alert'
import { CatalogCard, type CatalogCardVariant } from './catalog-card'
import { type CatalogEntry, type CatalogKind, useCatalog } from './catalog-data'
import { CatalogDetail } from './catalog-detail'
import { CatalogDetailDialog } from './catalog-detail-dialog'
import { CatalogDiscovery } from './catalog-discovery'
import { CatalogFilterBar } from './catalog-filter-bar'
import { CatalogInstallSwitch } from './catalog-install-switch'
import { CatalogListRow } from './catalog-list-row'
import { CATALOG_POINTER_ENABLED, trackCatalogPointer } from './catalog-pointer'
import {
  catalogCategories,
  type CatalogSort,
  catalogSortOptions,
  catalogSources,
  catalogTags,
  filterCatalog,
  relatedEntries,
  sortCatalog
} from './catalog-query'
import { $catalogCardView } from './store'
import { useCatalogFilters } from './use-catalog-filters'

interface CatalogBrowserProps {
  kind: CatalogKind
  query?: string
  onQueryChange?: (value: string) => void
  /** Data-derived "Try …" placeholder nudges for the search field. */
  searchHints?: string[]
  isInstalled: (entry: CatalogEntry) => boolean
  onInstall: (entry: CatalogEntry) => void
  isInstalling?: (entry: CatalogEntry) => boolean
  /** Feed rows the caller drops from the list entirely (e.g. a community
   *  lookalike whose name is taken by an installed skill). */
  isSuperseded?: (entry: CatalogEntry) => boolean
  installedEntries?: CatalogEntry[]
  matchInstalled?: (entry: CatalogEntry) => CatalogEntry | undefined
  actions?: ReactNode
  headerActions?: ReactNode
  installedPending?: boolean
  notice?: ReactNode
  renderInstalledDetail?: (entry: CatalogEntry) => ReactNode
  renderInstalledAction?: (entry: CatalogEntry) => ReactNode
  selectedEntryId?: string | null
}

const PAGE_SIZE = 60
// The skills hub carries ~10k distinct tags; the rail is for browsing, search covers the tail.
const TAG_LIMIT = 16
/** On: masonry lanes. Off: fallback grid whose rows share one height (`.catalog-grid`). */
const CATALOG_MASONRY = true

/** Merge the public feed with installed rows: a catalog entry that matches an
 *  installed one takes the installed id (so selection and install state line
 *  up); installed rows with no catalog match are appended. */
function mergeInstalled(
  data: CatalogEntry[],
  installedEntries: CatalogEntry[],
  matchInstalled?: (entry: CatalogEntry) => CatalogEntry | undefined
) {
  const remaining = new Map(installedEntries.map(entry => [entry.id, entry]))
  const seen = new Set<string>()

  const catalog = data.flatMap(entry => {
    const installed = matchInstalled?.(entry)

    if (!installed) {
      return [entry]
    }

    remaining.delete(installed.id)

    if (seen.has(installed.id)) {
      return []
    }

    seen.add(installed.id)

    return [
      {
        ...entry,
        id: installed.id,
        version: installed.version || entry.version,
        search: `${entry.search} ${installed.search}`
      }
    ]
  })

  return [...catalog, ...remaining.values()]
}

// Props take no destructuring defaults: React Compiler 1.0 skips components that have them.
export const CatalogBrowser = memo(function CatalogBrowser({
  kind,
  isInstalled,
  onInstall,
  isInstalling,
  isSuperseded,
  installedEntries,
  matchInstalled,
  actions,
  headerActions,
  installedPending,
  notice,
  renderInstalledDetail,
  renderInstalledAction,
  selectedEntryId,
  query,
  onQueryChange,
  searchHints
}: CatalogBrowserProps) {
  const { t } = useI18n()
  const c = t.catalog
  const cardView = useStore($catalogCardView)
  const { data, isPending, error, refetch } = useCatalog(kind)
  const deferredQuery = useDeferredValue((query ?? '').trim().toLowerCase())
  const [selectedId, setSelectedId] = useState<string | null>(selectedEntryId ?? null)
  const [detailOpen, setDetailOpen] = useState(false)
  const [limit, setLimit] = useState(PAGE_SIZE)

  const resetSelection = () => {
    setLimit(PAGE_SIZE)
    setSelectedId(null)
    setDetailOpen(false)
  }

  const filters = useCatalogFilters(kind, resetSelection)
  const { facets, sort } = filters

  // Clearing the query only widens results, and a deep link clears it while selecting its target.
  useEffect(() => {
    setLimit(PAGE_SIZE)

    if (deferredQuery) {
      setSelectedId(null)
    }
  }, [deferredQuery])

  // A deep link asks to open one entry; selection stays local afterwards.
  useEffect(() => {
    if (!selectedEntryId) {
      return
    }

    setSelectedId(selectedEntryId)
    setDetailOpen(true)
  }, [selectedEntryId])

  const root = useRef<HTMLDivElement>(null)
  useEffect(() => (CATALOG_POINTER_ENABLED && root.current ? trackCatalogPointer(root.current) : undefined), [])

  const entries = mergeInstalled(data ?? [], installedEntries ?? [], matchInstalled)
  const visible = isSuperseded ? entries.filter(entry => !isSuperseded(entry)) : entries
  const filtered = sortCatalog(filterCatalog(visible, facets, deferredQuery, isInstalled), sort, kind)

  const discover = !facets.sources.length && !facets.categories.length && !facets.tags.length && !deferredQuery && !facets.installedOnly
  const pageOrder = filtered

  const selected = visible.find(entry => entry.id === selectedId) ?? filtered[0]
  const related = selected && (!cardView || detailOpen) ? relatedEntries(filtered, selected, 3) : []

  const searchFor = (value: string) => {
    onQueryChange?.(value)
    resetSelection()
  }

  const clearFilters = () => {
    onQueryChange?.('')
    filters.clear()
  }

  const openEntry = (entry: CatalogEntry) => {
    setSelectedId(entry.id)
    setDetailOpen(true)
  }

  // Installed entries get the owner's on/off switch; the rest install one-way.
  const entryAction = (entry: CatalogEntry) => {
    const custom = renderInstalledAction?.(entry)

    if (custom) {
      return custom
    }

    const installed = isInstalled(entry)

    return (
      <CatalogInstallSwitch
        disabled={Boolean(installedPending) || installed || (kind === 'skills' && !entry.installIdentifier)}
        installed={installed}
        installing={isInstalling?.(entry) ?? false}
        name={entry.name}
        onInstall={() => {
          if (kind === 'plugins') {
            setDetailOpen(false)
          }

          onInstall(entry)
        }}
      />
    )
  }

  const card = (entry: CatalogEntry, accentIndex: number, variant: CatalogCardVariant = 'default', placement?: React.CSSProperties) => (
    <CatalogCard
      accentIndex={accentIndex}
      action={entryAction(entry)}
      entry={entry}
      key={entry.id}
      onCategory={filters.chooseCategory}
      onOpen={openEntry}
      onSearch={searchFor}
      onTag={filters.toggleTag}
      style={placement}
      variant={variant}
    />
  )

  const details = selected ? (
    <CatalogDetail
      action={entryAction(selected)}
      dialog={cardView}
      entry={selected}
      kind={kind}
      management={renderInstalledDetail?.(selected)}
      onCategory={filters.chooseCategory}
      onRelated={entry => setSelectedId(entry.id)}
      onSearch={searchFor}
      onTag={filters.toggleTag}
      related={related}
    />
  ) : null

  const sortLabels: Record<CatalogSort, string> = {
    discover: c.featured,
    stars: c.mostStarred,
    newest: c.newest,
    updated: c.recentlyUpdated,
    name: c.alphabetical
  }

  const sortControl = (
    <Select onValueChange={value => filters.setSort(value as CatalogSort)} value={sort}>
      <SelectTrigger aria-label={c.sortBy} className="w-full" size="sm">
        <SelectValue placeholder={c.sortBy} />
      </SelectTrigger>
      <SelectContent>
        {catalogSortOptions(visible, kind, sortLabels).map(({ value, label }) => (
          <SelectItem key={value} value={value}>
            {label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )

  const viewToggleLabel = cardView ? c.listView : c.cardView
  const filterKey = [facets.sources, facets.categories, facets.tags].map(list => list.join(',')).join(':')

  return (
    <div className="@container/catalog flex h-full min-h-0 min-w-0" data-catalog={kind} ref={root}>
      <div className="flex min-h-0 min-w-0 flex-1 flex-col">
        <div className="flex min-h-0 min-w-0 flex-1 flex-col" data-catalog-results>
          {/* Search owns the header; kind-specific setup actions and the view toggle share the end slot. */}
          <header
            className="catalog-search-header shrink-0"
            data-catalog-header
          >
            <CapabilitySearch
              actions={<>
                {headerActions}
                <Tip label={viewToggleLabel}>
                  <Button
                    aria-label={viewToggleLabel}
                    onClick={() => {
                      setDetailOpen(false)
                      $catalogCardView.set(!cardView)
                    }}
                    size="icon-xs"
                    variant="ghost"
                  >
                    <Codicon name={cardView ? 'list-unordered' : 'extensions'} />
                  </Button>
                </Tip>
              </>}
              hints={searchHints}
              onChange={value => onQueryChange?.(value)}
              placeholder={kind === 'plugins' ? c.searchPlugins : c.searchSkills}
              scope={kind === 'plugins' ? t.skills.tabPlugins : t.skills.tabSkills}
              value={query ?? ''}
            />
          </header>
          <CatalogFilterBar
            actions={actions}
            categories={catalogCategories(visible)}
            facets={facets}
            onCategory={filters.toggleCategory}
            onClear={clearFilters}
            onInstalled={filters.toggleInstalled}
            onSource={filters.toggleSource}
            onTag={filters.toggleTag}
            resultCount={filtered.length}
            sortControl={sortControl}
            sources={catalogSources(visible)}
            tags={catalogTags(visible, facets.tags, TAG_LIMIT)}
          />
          {notice}
          {error && entries.length > 0 && (
            <CatalogAlert onRetry={() => void refetch()} retryLabel={c.retry} title={c.loadFailed}>
              {error.message}
            </CatalogAlert>
          )}
          <div className="min-h-0 flex-1">
            {isPending && !entries.length ? (
              <PageLoader label={t.skills.loading} />
            ) : error && !entries.length ? (
              <div className="grid h-full place-items-center p-5">
                <ErrorState description={error.message} title={c.loadFailed}>
                  <Button onClick={() => void refetch()} size="sm" variant="secondary">
                    {c.retry}
                  </Button>
                </ErrorState>
              </div>
            ) : !selected ? (
              <PanelEmpty
                action={
                  <Button onClick={clearFilters} size="sm" variant="secondary">
                    {c.clearFilters}
                  </Button>
                }
                description={c.tryAnother}
                icon="search"
                title={c.noResults}
              />
            ) : cardView ? (
              <>
                <div className="flex h-full min-h-0 flex-col" data-catalog-cards={kind}>
                  <div
                    className="capabilities-scroll"
                    data-catalog-scroll
                    key={`${filterKey}:${sort}:${deferredQuery}:${facets.installedOnly}`}
                  >
                    {discover ? (
                      <CatalogDiscovery actions={actions} card={card} entries={filtered} kind={kind} onCategory={filters.chooseCategory} />
                    ) : (
                      <div className="catalog-filtered-results pt-(--capabilities-gap-lead) pb-2">
                        {CATALOG_MASONRY ? (
                          <Masonry data-catalog-hover-group>{filtered.slice(0, limit).map((entry, index) => card(entry, index))}</Masonry>
                        ) : (
                          <div className="catalog-grid" data-catalog-hover-group>
                            {filtered.slice(0, limit).map((entry, index) => card(entry, index))}
                          </div>
                        )}
                        {filtered.length > limit && (
                          <Button onClick={() => setLimit(value => value + PAGE_SIZE)} size="sm" variant="text">
                            {c.more}
                          </Button>
                        )}
                      </div>
                    )}
                  </div>
                </div>
                <CatalogDetailDialog
                  onOpenChange={setDetailOpen}
                  onSelect={setSelectedId}
                  open={detailOpen}
                  order={pageOrder}
                  selectedId={selected.id}
                >
                  {details}
                </CatalogDetailDialog>
              </>
            ) : (
              <div
                className={cn('h-full min-h-0', detailOpen ? '[&_aside]:max-sm:hidden' : '[&_main]:max-sm:hidden')}
                data-catalog-list={kind}
              >
                <MasterDetail resizeId="capabilities-split" split="wide">
                  <ListColumn key={`${filterKey}:${deferredQuery}`}>
                    {filtered.slice(0, limit).map(entry => (
                      <CatalogListRow
                        entry={entry}
                        installed={isInstalled(entry)}
                        key={entry.id}
                        kind={kind}
                        onCategory={filters.chooseCategory}
                        onOpen={openEntry}
                        onSearch={searchFor}
                        onTag={filters.toggleTag}
                        selected={entry.id === selected.id}
                      />
                    ))}
                    {filtered.length > limit && (
                      <Button onClick={() => setLimit(value => value + PAGE_SIZE)} size="sm" variant="text">
                        {c.more}
                      </Button>
                    )}
                  </ListColumn>
                  <DetailColumn footer={c.snapshotHint}>
                    <div className="sm:hidden">
                      <Button onClick={() => setDetailOpen(false)} size="sm" variant="text">
                        <Codicon name="arrow-left" />
                        {c.back}
                      </Button>
                    </div>
                    {details}
                  </DetailColumn>
                </MasterDetail>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
})
