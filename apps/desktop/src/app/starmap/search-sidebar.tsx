import { useStore } from '@nanostores/react'
import { useEffect, useMemo, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { Bookmark, BookmarkFilled, Trash2, X } from '@/lib/icons'
import { fmtDate } from '@/lib/time'
import type { StarmapGraph, StarmapNode } from '@/types/hermes'

import {
  $savedSearches,
  $searchHistory,
  commitSearchHistory,
  type DateMode,
  distinctOrigins,
  EMPTY_FILTERS,
  filterNodes,
  hasActiveNarrowing,
  isConclusion,
  nodeOrigin,
  removeSavedSearch,
  type SavedSearch,
  saveSearch,
  type SearchFilters
} from './search'

interface SearchSidebarProps {
  /** Legend/row color for conclusion nodes (matches the canvas hexagon ink). */
  conclusionColor: string
  /** Which section to open focused on: 'search' focuses the query input,
   *  'filter' scrolls to + highlights the filters. Both share this one panel. */
  focusSection: 'filter' | 'search'
  graph: StarmapGraph
  memoryColor: string
  onClose: () => void
  /** Row click: focus the node on the canvas. */
  onFocusNode: (id: string) => void
  /** Push the current match set up to the canvas (pulse + filtered rendering).
   *  Called with null when the sidebar closes or nothing narrows the graph. */
  onMatchesChange: (ids: null | Set<string>) => void
  /** Row context/⋯ action: open the same right-click menu nodes get. */
  onNodeMenu: (id: string, x: number, y: number) => void
  /** Hand the host a stable "reset my query + filters" callback (for the
   *  canvas "showing N of M · Clear" chip). */
  onRegisterClear: (clear: () => void) => void
  /** Whether the Conclusions category is offered (Honcho active). */
  showConclusions: boolean
}

function fmtTs(ts: null | number | undefined): string {
  if (!ts) {
    return ''
  }

  try {
    return fmtDate.format(new Date(ts * 1000))
  } catch {
    return ''
  }
}

const selectCls =
  'h-6 rounded-md border border-(--ui-stroke-secondary) bg-transparent px-1 text-[0.65rem] text-muted-foreground outline-none focus-visible:ring-1 focus-visible:ring-ring/40'

// Short month labels for the year-month picker. Index 0 = January; the control
// stores a 1-based '01'–'12' string.
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

// The set of calendar years that actually have nodes, newest first — so the
// year picker only offers years the map contains.
function yearsWithNodes(nodes: StarmapNode[]): number[] {
  const set = new Set<number>()

  for (const n of nodes) {
    if (n.timestamp) {
      set.add(new Date(n.timestamp * 1000).getFullYear())
    }
  }

  return [...set].sort((a, b) => b - a)
}

// Search / filter sidebar for the star map. Opens from the search icon;
// filters narrow live, matches pulse on the canvas, and the result list is
// chronological (newest first). Committed queries build a Google-style
// recents dropdown; the bookmark saves query+filters.
export function SearchSidebar({
  conclusionColor,
  focusSection,
  graph,
  memoryColor,
  onClose,
  onFocusNode,
  onMatchesChange,
  onNodeMenu,
  onRegisterClear,
  showConclusions
}: SearchSidebarProps) {
  const { t } = useI18n()
  const history = useStore($searchHistory)
  const saved = useStore($savedSearches)

  const [query, setQuery] = useState('')
  const [filters, setFilters] = useState<SearchFilters>(EMPTY_FILTERS)
  const [dropdownOpen, setDropdownOpen] = useState(false)
  const inputRef = useRef<HTMLInputElement | null>(null)
  const filtersRef = useRef<HTMLDivElement | null>(null)

  const origins = useMemo(() => distinctOrigins(graph.nodes), [graph.nodes])
  const years = useMemo(() => yearsWithNodes(graph.nodes), [graph.nodes])
  const narrowed = hasActiveNarrowing(query, filters)
  const results = useMemo(() => filterNodes(graph, query, filters).reverse(), [filters, graph, query])

  const conclusionSet = useMemo(() => {
    const s = new Set<string>()

    if (showConclusions) {
      for (const n of graph.nodes) {
        if (isConclusion(n, graph.memoryProvider)) {
          s.add(n.id)
        }
      }
    }

    return s
  }, [graph.memoryProvider, graph.nodes, showConclusions])

  // Publish matches to the canvas only when something narrows — an idle open
  // sidebar must not pulse the whole map.
  useEffect(() => {
    onMatchesChange(narrowed ? new Set(results.map(n => n.id)) : null)
  }, [narrowed, onMatchesChange, results])

  // Clear the canvas highlight when the sidebar unmounts.
  useEffect(() => () => onMatchesChange(null), [onMatchesChange])

  // Hand the host a stable "clear my query + filters" callback for the canvas
  // "showing N of M · Clear" chip.
  useEffect(() => {
    onRegisterClear(() => {
      setQuery('')
      setFilters(EMPTY_FILTERS)
    })
  }, [onRegisterClear])

  // Open focused on the requested section: the search icon lands on the query
  // box; the filter icon scrolls the filters into view (nothing to focus, since
  // native selects don't take text focus meaningfully).
  useEffect(() => {
    if (focusSection === 'search') {
      inputRef.current?.focus()
    } else {
      filtersRef.current?.scrollIntoView({ block: 'nearest' })
    }
  }, [focusSection])

  const commit = () => {
    commitSearchHistory(query)
    setDropdownOpen(false)
  }

  const applySaved = (s: SavedSearch) => {
    setQuery(s.query)
    setFilters({
      dateMode: s.dateMode,
      from: s.from,
      kind: s.kind,
      month: s.month,
      source: s.source,
      to: s.to,
      year: s.year
    })
    setDropdownOpen(false)
  }

  const currentSaved: SavedSearch = { ...filters, query }
  const recents = history.filter(h => h.toLowerCase().includes(query.trim().toLowerCase()) && h !== query.trim())

  return (
    <div className="pointer-events-auto flex h-full w-72 min-w-0 flex-col gap-2 border-l border-(--ui-stroke-secondary) bg-[color-mix(in_srgb,var(--ui-bg-elevated)_94%,transparent)] p-3 backdrop-blur-md [-webkit-app-region:no-drag]">
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs font-medium">{t.starmap.searchTitle}</span>
        <Button aria-label={t.starmap.close} onClick={onClose} size="icon-xs" variant="ghost">
          <X className="size-3.5" />
        </Button>
      </div>

      {/* Query input + recents dropdown (committed searches, newest first). */}
      <div className="relative">
        <input
          aria-label={t.starmap.searchTitle}
          className="h-7 w-full rounded-md border border-(--ui-stroke-secondary) bg-foreground/5 px-2 text-xs outline-none placeholder:text-muted-foreground/60 focus-visible:ring-1 focus-visible:ring-ring/40"
          onBlur={() => setTimeout(() => setDropdownOpen(false), 120)}
          onChange={e => {
            setQuery(e.target.value)
            setDropdownOpen(true)
          }}
          onFocus={() => setDropdownOpen(true)}
          onKeyDown={e => {
            if (e.key === 'Enter') {
              commit()
            }

            if (e.key === 'Escape') {
              setDropdownOpen(false)
            }
          }}
          placeholder={t.starmap.searchPlaceholder}
          ref={inputRef}
          spellCheck={false}
          value={query}
        />

        {dropdownOpen && (recents.length > 0 || saved.length > 0) ? (
          <div className="absolute inset-x-0 top-8 z-30 max-h-56 overflow-y-auto rounded-md border border-(--ui-stroke-secondary) bg-(--ui-bg-elevated) py-1 shadow-md">
            {saved.length > 0 ? (
              <>
                <div className="px-2 py-0.5 text-[0.6rem] uppercase tracking-wide text-muted-foreground/70">
                  {t.starmap.searchSaved}
                </div>
                {saved.map(s => (
                  <div
                    className="group flex items-center"
                    key={`saved:${s.query}:${s.kind}:${s.source}:${s.dateMode}:${s.from}:${s.to}:${s.year}:${s.month}`}
                  >
                    <button
                      className="flex min-w-0 flex-1 cursor-pointer items-center gap-1.5 px-2 py-1 text-left text-xs hover:bg-(--ui-control-active-background)"
                      onMouseDown={e => {
                        e.preventDefault()
                        applySaved(s)
                      }}
                      type="button"
                    >
                      <BookmarkFilled className="size-3 shrink-0 text-muted-foreground" />
                      <span className="truncate">{s.query || t.starmap.searchFiltersOnly}</span>
                      <span className="ml-auto shrink-0 text-[0.6rem] text-muted-foreground/70">
                        {[
                          s.kind !== 'all' ? s.kind : null,
                          s.source !== 'all' ? s.source : null,
                          hasActiveNarrowing('', s) && (s.from || s.to || s.year) ? '📅' : null
                        ]
                          .filter(Boolean)
                          .join(' · ')}
                      </span>
                    </button>
                    <button
                      aria-label={t.starmap.searchDeleteSaved}
                      className="cursor-pointer px-1.5 py-1 text-muted-foreground/50 opacity-0 hover:text-destructive group-hover:opacity-100"
                      onMouseDown={e => {
                        e.preventDefault()
                        removeSavedSearch(s)
                      }}
                      type="button"
                    >
                      <Trash2 className="size-3" />
                    </button>
                  </div>
                ))}
              </>
            ) : null}
            {recents.length > 0 ? (
              <>
                <div className="px-2 py-0.5 text-[0.6rem] uppercase tracking-wide text-muted-foreground/70">
                  {t.starmap.searchRecent}
                </div>
                {recents.map(h => (
                  <button
                    className="block w-full cursor-pointer truncate px-2 py-1 text-left text-xs hover:bg-(--ui-control-active-background)"
                    key={`recent:${h}`}
                    onMouseDown={e => {
                      e.preventDefault()
                      setQuery(h)
                      commitSearchHistory(h)
                      setDropdownOpen(false)
                    }}
                    type="button"
                  >
                    {h}
                  </button>
                ))}
              </>
            ) : null}
          </div>
        ) : null}
      </div>

      {/* Filters: kind, source (open-ended origins), date (range/year/month). */}
      <div className="flex flex-wrap items-center gap-1.5" ref={filtersRef}>
        <select
          aria-label={t.starmap.searchKind}
          className={selectCls}
          onChange={e => setFilters(f => ({ ...f, kind: e.target.value as SearchFilters['kind'] }))}
          value={filters.kind}
        >
          <option value="all">{t.starmap.searchKindAll}</option>
          <option value="memory">{t.starmap.memory}</option>
          <option value="skill">{t.starmap.searchKindSkills}</option>
          {showConclusions ? <option value="conclusion">{t.starmap.conclusions}</option> : null}
        </select>

        <select
          aria-label={t.starmap.searchSource}
          className={selectCls}
          onChange={e => setFilters(f => ({ ...f, source: e.target.value }))}
          value={filters.source}
        >
          <option value="all">{t.starmap.searchSourceAll}</option>
          {origins.map(o => (
            <option key={o} value={o}>
              {o === 'hermes' ? t.starmap.searchSourceHermes : t.starmap.searchSourceImported(o)}
            </option>
          ))}
        </select>

        <Tip label={t.starmap.searchSave}>
          <Button
            aria-label={t.starmap.searchSave}
            disabled={!narrowed}
            onClick={() => saveSearch(currentSaved)}
            size="icon-xs"
            variant="ghost"
          >
            <Bookmark className="size-3.5" />
          </Button>
        </Tip>
      </div>

      {/* Date filter: a mode dropdown swaps the controls between an explicit
          custom range, a whole year, or a single year+month. */}
      <div className="flex items-center gap-1.5">
        <select
          aria-label={t.starmap.dateModeLabel}
          className={selectCls}
          onChange={e => setFilters(f => ({ ...f, dateMode: e.target.value as DateMode }))}
          value={filters.dateMode}
        >
          <option value="range">{t.starmap.dateModeRange}</option>
          <option value="year">{t.starmap.dateModeYear}</option>
          <option value="yearMonth">{t.starmap.dateModeYearMonth}</option>
        </select>

        {filters.dateMode === 'range' ? (
          <>
            <input
              aria-label={t.starmap.searchFrom}
              className={`${selectCls} min-w-0 flex-1`}
              onChange={e => setFilters(f => ({ ...f, from: e.target.value }))}
              type="date"
              value={filters.from}
            />
            <span className="text-[0.6rem] text-muted-foreground/60">→</span>
            <input
              aria-label={t.starmap.searchTo}
              className={`${selectCls} min-w-0 flex-1`}
              onChange={e => setFilters(f => ({ ...f, to: e.target.value }))}
              type="date"
              value={filters.to}
            />
          </>
        ) : (
          <>
            <select
              aria-label={t.starmap.dateModeYear}
              className={`${selectCls} min-w-0 flex-1`}
              onChange={e => setFilters(f => ({ ...f, year: e.target.value }))}
              value={filters.year}
            >
              <option value="">{t.starmap.dateYearAny}</option>
              {years.map(y => (
                <option key={y} value={String(y)}>
                  {y}
                </option>
              ))}
            </select>
            {filters.dateMode === 'yearMonth' ? (
              <select
                aria-label={t.starmap.dateMonthLabel}
                className={`${selectCls} min-w-0 flex-1`}
                onChange={e => setFilters(f => ({ ...f, month: e.target.value }))}
                value={filters.month}
              >
                <option value="">{t.starmap.dateMonthAny}</option>
                {MONTHS.map((label, i) => (
                  <option key={label} value={String(i + 1).padStart(2, '0')}>
                    {label}
                  </option>
                ))}
              </select>
            ) : null}
          </>
        )}
      </div>

      {/* Chronological results (newest first). Click focuses the node; the ⋯
          opens the same context menu the canvas node gets. */}
      <div className="text-[0.62rem] text-muted-foreground/70">{t.starmap.searchCount(results.length)}</div>
      <div className="min-h-0 flex-1 space-y-0.5 overflow-y-auto pr-0.5">
        {results.length === 0 ? (
          <p className="pt-2 text-xs text-muted-foreground">{t.starmap.searchEmpty}</p>
        ) : (
          results.map(n => (
            <SearchRow
              conclusionColor={conclusionColor}
              isConclusion={conclusionSet.has(n.id)}
              key={n.id}
              memoryColor={memoryColor}
              node={n}
              onFocusNode={onFocusNode}
              onNodeMenu={onNodeMenu}
            />
          ))
        )}
      </div>
    </div>
  )
}

function SearchRow({
  conclusionColor,
  isConclusion,
  memoryColor,
  node,
  onFocusNode,
  onNodeMenu
}: {
  conclusionColor: string
  isConclusion: boolean
  memoryColor: string
  node: StarmapNode
  onFocusNode: (id: string) => void
  onNodeMenu: (id: string, x: number, y: number) => void
}) {
  const origin = nodeOrigin(node)

  return (
    <div
      className="group flex w-full cursor-pointer items-start gap-1.5 rounded-md border border-transparent px-1.5 py-1 text-left hover:border-(--ui-stroke-secondary) hover:bg-(--ui-control-active-background)"
      onClick={() => onFocusNode(node.id)}
      onContextMenu={e => {
        e.preventDefault()
        onNodeMenu(node.id, e.clientX, e.clientY)
      }}
      role="button"
      tabIndex={0}
    >
      {/* Kind glyph mirrors the canvas: hexagon = conclusion, diamond = memory,
          circle = skill. */}
      {isConclusion ? (
        <span
          className="mt-1 inline-block size-2 shrink-0"
          style={{ backgroundColor: conclusionColor, clipPath: 'polygon(25% 0,75% 0,100% 50%,75% 100%,25% 100%,0 50%)' }}
        />
      ) : node.kind === 'memory' ? (
        <span className="mt-1 inline-block size-2 shrink-0 rotate-45" style={{ backgroundColor: memoryColor }} />
      ) : (
        <span className="mt-1 inline-block size-2 shrink-0 rounded-full bg-[var(--theme-primary)]/80" />
      )}
      <span className="min-w-0 flex-1">
        <span className="block truncate text-xs leading-tight">{node.label}</span>
        <span className="block text-[0.6rem] text-muted-foreground/70">
          {[fmtTs(node.timestamp), origin !== 'hermes' ? origin : null].filter(Boolean).join(' · ')}
        </span>
      </span>
      <button
        aria-label="⋯"
        className="mt-0.5 shrink-0 cursor-pointer rounded px-1 text-xs text-muted-foreground/60 opacity-0 hover:text-foreground group-hover:opacity-100"
        onClick={e => {
          e.stopPropagation()
          onNodeMenu(node.id, e.clientX, e.clientY)
        }}
        type="button"
      >
        ⋯
      </button>
    </div>
  )
}
