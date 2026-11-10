import { compactNumber, groupCatalogPlugins, PLUGIN_CATEGORIES } from '@hermes/shared'
import { useStore } from '@nanostores/react'
import { memo, type ReactNode, useDeferredValue, useEffect, useRef, useState } from 'react'

import { PageLoader } from '@/components/page-loader'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { ErrorState } from '@/components/ui/error-state'
import { Masonry } from '@/components/ui/masonry'
import { Reel } from '@/components/ui/reel'
import { SearchField } from '@/components/ui/search-field'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'

import { DetailColumn, ListColumn, MasterDetail } from '../../master-detail'
import { PanelEmpty } from '../../overlays/panel'
import { brandingLockedSlice } from '../free-tier'
import { LockFog, useLockFogTop } from '../lock-fog'

import { CatalogAlert } from './catalog-alert'
import { CatalogCard } from './catalog-card'
import { type CatalogEntry, type CatalogKind, useCatalog } from './catalog-data'
import { CatalogDetail } from './catalog-detail'
import { CatalogDetailDialog } from './catalog-detail-dialog'
import { CatalogFilters } from './catalog-filters'
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
  /** Free-plan gate: unlocked entries stay interactive; the rest sit under one page blur + lock. */
  isLocked?: (entry: CatalogEntry) => boolean
  lockedLabel?: string
  selectedEntryId?: string | null
}

const PAGE_SIZE = 60
// The skills hub carries ~10k distinct tags; the rail is for browsing, search covers the tail.
const TAG_LIMIT = 16
// Discovery shelves preview a category; "See all" opens the full list. Every
// rendered card is restyled whenever a modal locks the page.
const SHELF_SIZE = 6
/** On: masonry lanes. Off: fallback grid whose rows share one height (`.catalog-grid`).
 *  Marketplace mock is a uniform card grid, so keep masonry off. */
const CATALOG_MASONRY = false

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
  isLocked,
  lockedLabel,
  selectedEntryId,
  query,
  onQueryChange
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

  const discover =
    kind === 'plugins' && !facets.categories.length && !facets.tags.length && !deferredQuery && !facets.installedOnly

  const sections =
    discover && cardView
      ? groupCatalogPlugins(filtered).map(([key, items]) => ({ key, ...PLUGIN_CATEGORIES[key], entries: items }))
      : []

  const freeEntries = isLocked ? filtered.filter(entry => !isLocked(entry)) : filtered
  const lockedEntries = isLocked ? filtered.filter(entry => isLocked(entry)) : []
  const lockedBehind = Boolean(isLocked) && lockedEntries.length > 0
  const { freeAnchor, fogTop } = useLockFogTop(
    lockedBehind,
    `${freeEntries.length}:${cardView ? 'card' : 'list'}:${deferredQuery}`
  )

  const pageOrder = lockedBehind
    ? freeEntries
    : sections.length
      ? sections.flatMap(section => section.entries)
      : filtered

  const selected = pageOrder.find(entry => entry.id === selectedId) ?? pageOrder[0] ?? filtered[0]
  const related = selected && (!cardView || detailOpen) ? relatedEntries(pageOrder, selected, 3) : []
  const lockTip = lockedLabel || 'Upgrade to unlock more'

  const searchFor = (value: string) => {
    onQueryChange?.(value)
    resetSelection()
  }

  const clearFilters = () => {
    onQueryChange?.('')
    filters.clear()
  }

  const openEntry = (entry: CatalogEntry) => {
    if (isLocked?.(entry)) {
      return
    }

    setSelectedId(entry.id)
    setDetailOpen(true)
  }

  // Installed entries get the owner's on/off switch; the rest install one-way.
  const entryAction = (entry: CatalogEntry) => {
    if (isLocked?.(entry)) {
      return null
    }

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

  const card = (entry: CatalogEntry, accentIndex: number) => (
    <CatalogCard
      accentIndex={accentIndex}
      action={entryAction(entry)}
      entry={entry}
      key={entry.id}
      onCategory={filters.chooseCategory}
      onOpen={openEntry}
      onSearch={searchFor}
      onTag={filters.toggleTag}
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
      <SelectTrigger
        aria-label={c.sortBy}
        className="h-[38px] w-full rounded-xl border-[#e2e8f0] bg-[#f7f9fc] px-3 text-[13px] font-semibold text-[#0f172a] shadow-none hover:bg-[#f7f9fc]"
        size="sm"
      >
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
    <div
      className="@container/catalog flex h-full min-h-0 min-w-0 flex-col bg-gradient-to-b from-[#e6effb] to-[#f7f9fc] to-45% dark:bg-(--ui-chat-surface-background) dark:bg-none"
      data-catalog={kind}
      ref={root}
    >
      {/* Figma: page header is full-width above Discover + grid */}
      <header className="shrink-0 space-y-4 px-7 pb-4 pt-5" data-catalog-header>
        <div className="flex min-w-0 items-start justify-between gap-3">
          <div className="min-w-0 space-y-1.5">
            <h1 className="text-[40px] font-extrabold leading-none tracking-tight text-[#0f172a] dark:text-foreground">
              {kind === 'plugins' ? c.pageTitlePlugins : c.pageTitleSkills}
            </h1>
            <p className="max-w-2xl text-[15px] text-[#475569] dark:text-(--ui-text-tertiary)">
              {kind === 'plugins' ? c.pageSubtitlePlugins : c.pageSubtitleSkills}
            </p>
          </div>
          <div className="flex shrink-0 flex-wrap items-center justify-end gap-2.5" data-catalog-actions>
            {headerActions}
            {kind === 'skills' ? (
              <Tip label={viewToggleLabel}>
                <Button
                  aria-label={viewToggleLabel}
                  className="size-auto rounded-full border border-[#e2e8f0] bg-white px-3 py-[11px] text-[#0f172a] shadow-none hover:bg-[#f8fafc]"
                  onClick={() => {
                    setDetailOpen(false)
                    $catalogCardView.set(!cardView)
                  }}
                  size="icon-xs"
                  type="button"
                  variant="outline"
                >
                  <Codicon name={cardView ? 'list-unordered' : 'extensions'} />
                </Button>
              </Tip>
            ) : null}
          </div>
        </div>
        <SearchField
          containerClassName="w-full min-w-0"
          onChange={value => onQueryChange?.(value)}
          placeholder={kind === 'plugins' ? c.searchPlugins : c.searchSkills}
          shortcutHint={c.searchShortcut}
          value={query ?? ''}
          variant="box"
        />
      </header>

      <div className="flex min-h-0 min-w-0 flex-1 flex-col gap-5 px-7 pb-5 @[48rem]/catalog:flex-row">
        <CatalogFilters
          categories={catalogCategories(visible, kind)}
          className="max-h-64 shrink-0 overflow-y-auto @[48rem]/catalog:max-h-none"
          facets={facets}
          onCategory={filters.toggleCategory}
          onClear={clearFilters}
          onInstalled={filters.toggleInstalled}
          onSource={filters.toggleSource}
          onTag={filters.toggleTag}
          resultCount={visible.length}
          sidebarActions={actions && <div className="flex flex-wrap items-center gap-1.5 pt-2">{actions}</div>}
          sortControl={sortControl}
          sources={catalogSources(visible)}
          tags={[]}
        />
        <div className="flex min-h-0 min-w-0 flex-1 flex-col" data-catalog-results>
          <div className="mb-3 flex min-w-0 items-center justify-between gap-3 text-[13px] text-[#64748b]">
            <span>
              {c.showingCount(Math.min(filtered.length, cardView ? limit : filtered.length), filtered.length)}
            </span>
            <span className="hidden sm:inline">
              {c.sortBy}: <span className="font-semibold text-[#0f172a] dark:text-foreground">{sortLabels[sort]}</span>
            </span>
          </div>
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
                <div className="flex h-full min-h-0 flex-col px-3 pb-3" data-catalog-cards={kind}>
                  {lockedBehind ? (
                    <div
                      className="relative min-h-0 flex-1 overflow-hidden"
                      data-catalog-scroll
                      key={`${filterKey}:${sort}:${deferredQuery}:gated`}
                    >
                      <div className="h-full overflow-hidden py-2">
                        <div ref={freeAnchor}>
                          <div className="catalog-grid" data-catalog-hover-group>
                            {freeEntries.map(card)}
                          </div>
                        </div>
                        <div className="catalog-grid mt-3" data-catalog-hover-group>
                          {brandingLockedSlice(lockedEntries).map((entry, index) => card(entry, freeEntries.length + index))}
                        </div>
                      </div>
                      {fogTop > 0 ? (
                        <div className="absolute inset-x-0 bottom-0" style={{ top: fogTop }}>
                          <LockFog label={lockTip} />
                        </div>
                      ) : null}
                    </div>
                  ) : (
                    <div
                      className="min-h-0 flex-1 overflow-y-auto overscroll-contain [scrollbar-gutter:stable]"
                      data-catalog-scroll
                      key={`${filterKey}:${sort}:${deferredQuery}:${facets.installedOnly}`}
                    >
                      {discover ? (
                        <div className="space-y-7 py-3">
                          {sections.map((section, sectionIndex) => (
                            <section className="space-y-3" data-catalog-section={section.key} key={section.key}>
                              <header className="flex items-center justify-between gap-3">
                                <h3 className="flex items-baseline gap-2 text-sm font-semibold">
                                  <Button
                                    className="text-sm font-semibold text-(--ui-text-primary)"
                                    onClick={() => filters.chooseCategory(section.key)}
                                    size="inline"
                                    variant="text"
                                  >
                                    {section.label}
                                  </Button>
                                  <span className="text-xs font-normal text-(--ui-text-tertiary)">
                                    {compactNumber(section.entries.length)}
                                  </span>
                                </h3>
                                <Button onClick={() => filters.chooseCategory(section.key)} size="inline" variant="text">
                                  {c.seeAll}
                                  <Codicon name="arrow-right" />
                                </Button>
                              </header>
                              <p className="text-xs text-(--ui-text-tertiary)">{section.blurb}</p>
                              <Reel className="*:w-68" data-catalog-hover-group>
                                {section.entries
                                  .slice(0, SHELF_SIZE)
                                  .map((entry, index) => card(entry, sectionIndex * SHELF_SIZE + index))}
                              </Reel>
                            </section>
                          ))}
                        </div>
                      ) : (
                        <div className="py-2">
                          {CATALOG_MASONRY ? (
                            <Masonry data-catalog-hover-group>{filtered.slice(0, limit).map(card)}</Masonry>
                          ) : (
                            <div className="catalog-grid" data-catalog-hover-group>
                              {filtered.slice(0, limit).map(card)}
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
                  )}
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
                    {lockedBehind ? (
                      <div className="relative min-h-full">
                        <div ref={freeAnchor}>
                          {freeEntries.map(entry => (
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
                        </div>
                        {brandingLockedSlice(lockedEntries).map(entry => (
                          <CatalogListRow
                            entry={entry}
                            installed={isInstalled(entry)}
                            key={entry.id}
                            kind={kind}
                            onCategory={filters.chooseCategory}
                            onOpen={openEntry}
                            onSearch={searchFor}
                            onTag={filters.toggleTag}
                            selected={false}
                          />
                        ))}
                        {fogTop > 0 ? (
                          <div className="absolute inset-x-0 bottom-0" style={{ top: fogTop }}>
                            <LockFog label={lockTip} />
                          </div>
                        ) : null}
                      </div>
                    ) : (
                      <>
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
                      </>
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
