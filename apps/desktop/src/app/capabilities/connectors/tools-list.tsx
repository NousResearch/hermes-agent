// The right column of the connector dialog: the filter bar, the tool list, and
// the one-line dirty footer.
//
// The list is windowed by hand rather than by a virtualiser, because the shape is
// simple enough to be exact: every closed row is `TOOL_ROW_HEIGHT`, and at most
// one disclosure is open, so the offset of any row is its index times the row
// height plus the single measured detail height when the row is below it.
// github ships 896 rows of live switches; painting them all costs a frame the
// person can feel on every keystroke in the search field.

import { useCallback, useLayoutEffect, useMemo, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'

import {
  availableQuickActions,
  categoryCounts,
  deprecatedCount,
  EMPTY_TOOLS_FILTER,
  facetChips,
  filterTools,
  hintChips,
  isTinyConnector
} from './derive-tools'
import { TOOL_ROW_HEIGHT, ToolRow } from './tool-row'
import { ToolsFilterBar } from './tools-filter-bar'
import { ToolsConflict, ToolsGone, ToolsSignedOut, ToolsUnavailable, ToolsWash } from './tools-status'
import type {
  ConflictDifference,
  QuickAction,
  QuickActionId,
  ToolRowModel,
  ToolsEditorCounts,
  ToolsEditorPhase,
  ToolsFilter,
  ToolsFreshness
} from './types'

const OVERSCAN = 4
const MIN_VIEWPORT = 200

export interface ToolsListProps {
  /** Only read in the `conflict` phase. */
  conflict?: ConflictDifference
  connectorName: string
  counts: ToolsEditorCounts
  currentAction: null | QuickAction
  dirty: boolean
  freshness?: ToolsFreshness
  isOn: (slug: string) => boolean
  onApplyQuickAction: (id: QuickActionId) => void
  onDiscard: () => void
  onKeepMine: () => void
  onRefresh: () => void
  onReload: () => void
  onRemove: () => void
  onRetry: () => void
  onSave: () => void
  onSignIn: () => void
  onToggle: (slug: string) => void
  phase: ToolsEditorPhase
  tools: ToolRowModel[]
}

export function ToolsList(props: ToolsListProps) {
  const [filter, setFilter] = useState<ToolsFilter>(EMPTY_TOOLS_FILTER)

  const { conflict, connectorName, onKeepMine, onReload, onRemove, onRetry, onSignIn, phase, tools } = props

  // Derived from the whole list, not the filtered one: a chip must not vanish
  // because the chip beside it is already narrowing the rows.
  //
  // "The whole list" is what the reader can see, though. Deprecated rows are
  // hidden until asked for, so counting them here would print a count no row
  // adds up to — and could raise a chip to two on the strength of a row that is
  // not on screen. `deprecatedCount` keeps the full list: it is the toggle's own
  // number, and the toggle is how the hidden rows come back.
  const chrome = useMemo(() => {
    const counted = filter.showDeprecated ? tools : tools.filter(tool => !tool.deprecated)

    return {
      categories: categoryCounts(counted),
      deprecated: deprecatedCount(tools),
      facets: facetChips(counted),
      hints: hintChips(counted),
      quickActions: availableQuickActions(tools),
      tiny: isTinyConnector(tools)
    }
  }, [filter.showDeprecated, tools])

  const visible = useMemo(() => filterTools(tools, filter), [tools, filter])

  if (phase === 'loading') {
    return <ToolsWash />
  }

  if (phase === 'unavailable') {
    return <ToolsUnavailable onRetry={onRetry} />
  }

  if (phase === 'gone') {
    return <ToolsGone connectorName={connectorName} onRemove={onRemove} />
  }

  if (phase === 'signedOut') {
    return <ToolsSignedOut onSignIn={onSignIn} />
  }

  if (phase === 'conflict') {
    return (
      <ToolsConflict difference={conflict ?? { theyOff: 0, theyOn: 0 }} onKeepMine={onKeepMine} onReload={onReload} />
    )
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col" data-slot="tools-list">
      <ToolsFilterBar
        categories={chrome.categories}
        currentAction={props.currentAction}
        deprecated={chrome.deprecated}
        facets={chrome.facets}
        filter={filter}
        freshness={props.freshness}
        hints={chrome.hints}
        onApplyQuickAction={props.onApplyQuickAction}
        onFilterChange={setFilter}
        onRefresh={props.onRefresh}
        quickActions={chrome.quickActions}
        tiny={chrome.tiny}
        total={tools.length}
      />

      {/* The viewport stays mounted through a query that matches nothing: it owns
          the scroll offset and the open row, and losing them on one keystroke
          and keeping them on the next is a list that behaves differently for no
          reason the reader can see. */}
      <ToolViewport isOn={props.isOn} onToggle={props.onToggle} tools={visible} />

      {props.dirty ? (
        <DirtyFooter
          counts={props.counts}
          onDiscard={props.onDiscard}
          onSave={props.onSave}
          saving={phase === 'saving'}
        />
      ) : null}
    </div>
  )
}

/** The windowed viewport. Scroll anchoring is off on purpose: the browser reads
 *  the recycled slice moving under the viewport as content shifting and
 *  "corrects" the scroll position by a row. */
function ToolViewport({
  isOn,
  onToggle,
  tools
}: {
  isOn: (slug: string) => boolean
  onToggle: (slug: string) => void
  tools: ToolRowModel[]
}) {
  const { t } = useI18n()
  const [scrollTop, setScrollTop] = useState(0)
  const [expanded, setExpanded] = useState<null | string>(null)
  /** The open detail's real height. A description runs from one line to a
   *  paragraph and is shown whole, so it cannot be a constant. */
  const [detailHeight, setDetailHeight] = useState(0)
  const viewport = useViewportHeight()

  /** Measures whatever the open row added beyond one closed row. A ref callback
   *  rather than an effect: the detail unmounts whenever its row scrolls out of
   *  the window and mounts again when it returns, and the height must be re-read
   *  each time without the expanded row having changed. Stable, so React calls it
   *  only when the node actually moves. */
  const measureDetail = useCallback((node: HTMLDivElement | null) => {
    if (node) {
      setDetailHeight(Math.max(0, node.getBoundingClientRect().height - TOOL_ROW_HEIGHT))
    }
  }, [])

  const expandedIndex = expanded === null ? -1 : tools.findIndex(tool => tool.slug === expanded)
  const extra = expandedIndex >= 0 ? detailHeight : 0
  const total = tools.length * TOOL_ROW_HEIGHT + extra
  // Everything below the open row sits `extra` lower; everything above is where
  // it always was. One cut point is all the arithmetic this needs.
  const cut = expandedIndex >= 0 ? (expandedIndex + 1) * TOOL_ROW_HEIGHT : Number.POSITIVE_INFINITY

  const indexAt = (y: number) => Math.max(0, Math.floor((y < cut ? y : Math.max(cut, y - extra)) / TOOL_ROW_HEIGHT))

  const offsetAt = (index: number) =>
    index * TOOL_ROW_HEIGHT + (expandedIndex >= 0 && index > expandedIndex ? extra : 0)

  // The offset the browser is about to settle on, not the one it last reported:
  // a filter that shortens the list leaves `scrollTop` past the new bottom, and
  // a window taken from it starts after the last row and paints nothing.
  const top = Math.min(scrollTop, Math.max(0, total - viewport.height))

  const start = Math.max(0, indexAt(top) - OVERSCAN)
  const end = Math.min(tools.length, indexAt(top + viewport.height) + OVERSCAN + 1)

  const toggleExpanded = (slug: string) =>
    setExpanded(previous => {
      if (previous === slug) {
        setDetailHeight(0)

        return null
      }

      return slug
    })

  return (
    <div
      className="min-h-0 flex-1 overflow-y-auto overscroll-contain [overflow-anchor:none]"
      data-slot="tools-viewport"
      onScroll={event => setScrollTop(event.currentTarget.scrollTop)}
      ref={viewport.ref}
    >
      {tools.length === 0 ? (
        <p className="px-3.5 py-8 text-center text-xs text-(--ui-text-tertiary)">{t.connectorsPage.tools.noMatch}</p>
      ) : null}

      <div className="relative" style={{ height: total }}>
        <div className="absolute inset-x-0" style={{ top: offsetAt(start) }}>
          {tools.slice(start, end).map(tool => (
            <div key={tool.slug} ref={expanded === tool.slug ? measureDetail : undefined}>
              <ToolRow
                expanded={expanded === tool.slug}
                on={isOn(tool.slug)}
                onExpand={() => toggleExpanded(tool.slug)}
                onToggle={() => onToggle(tool.slug)}
                tool={tool}
              />
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

/** How tall the window is. Measured, so the list ends where the dialog ends and
 *  is never a small scroller nested inside a scrolling column. */
function useViewportHeight() {
  const ref = useRef<HTMLDivElement | null>(null)
  const [height, setHeight] = useState(MIN_VIEWPORT)

  const read = useCallback(() => {
    const node = ref.current

    if (node) {
      setHeight(previous => {
        const next = Math.max(MIN_VIEWPORT, Math.round(node.clientHeight))

        return previous === next ? previous : next
      })
    }
  }, [])

  useLayoutEffect(() => {
    read()

    const node = ref.current

    if (!node || typeof ResizeObserver === 'undefined') {
      return
    }

    const observer = new ResizeObserver(read)

    observer.observe(node)

    return () => observer.disconnect()
  }, [read])

  return { height, ref }
}

/** One line, always the same line: what changes, then the two ways out. */
function DirtyFooter({
  counts,
  onDiscard,
  onSave,
  saving
}: {
  counts: ToolsEditorCounts
  onDiscard: () => void
  onSave: () => void
  saving: boolean
}) {
  const { t } = useI18n()
  const copy = t.connectorsPage.tools

  return (
    <div
      className="flex items-center gap-2.5 border-t border-(--ui-stroke-tertiary) bg-(--ui-bg-chrome) px-3.5 py-2"
      data-slot="tools-dirty-footer"
    >
      <span aria-hidden className="size-1.5 shrink-0 rounded-full bg-(--theme-primary)" />
      <span className="min-w-0 flex-1 truncate text-xs text-(--ui-text-primary)">
        {copy.footerDirty(counts.off, counts.backOn)}
      </span>
      <Button disabled={saving} onClick={onDiscard} size="xs" variant="text">
        {copy.discard}
      </Button>
      <Button disabled={saving} loading={saving} onClick={onSave} size="xs">
        {saving ? copy.saving : copy.save}
      </Button>
    </div>
  )
}
