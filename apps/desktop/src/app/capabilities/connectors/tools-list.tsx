import type * as React from 'react'
import { type ReactNode, useCallback, useLayoutEffect, useMemo, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'

import {
  availableQuickActions,
  categoryCounts,
  deprecatedCount,
  EMPTY_TOOLS_FILTER,
  facetChips,
  facetSummary,
  filterTools,
  hintChips,
  isTinyConnector
} from './derive-tools'
import { TOOL_ROW_HEIGHT, ToolRow } from './tool-row'
import { ToolsFilterBar } from './tools-filter-bar'
import {
  isToolsStatusPhase,
  ToolsStatus,
  type ToolsStatusAction,
  type ToolsStatusPhase,
  ToolsWash
} from './tools-status'
import { ToolsSummary, useShowAllTools } from './tools-summary'
import type { ConflictDifference, QuickAction, ToolRowModel, ToolsEditorCounts, ToolsFilter } from './types'
import type { ToolsEditor } from './use-tools-editor'

const OVERSCAN = 4
const MIN_VIEWPORT = 200

const EMPTY_QUICK_ACTIONS: QuickAction[] = []

export interface ToolsListProps {
  appOff?: boolean
  conflict?: ConflictDifference
  connectorName: string
  editor: ToolsEditor
  listKey: string
  onReload: () => void
  onRemove?: () => void
  onRetry: () => void
  onRetryRules?: () => void
  onSignIn?: () => void
  preview?: boolean
  readOnly?: boolean
  rulesSignedOut?: boolean
  signedOut?: boolean
  summaryTitle?: string
  tools: ToolRowModel[]
}

export function ToolsList(props: ToolsListProps) {
  const phase = props.editor.phase

  if (phase === 'loading') {
    return <ToolsWash />
  }

  if (phase === 'unavailable') {
    return <UnavailableLine onRetry={props.onRetry} />
  }

  if (isToolsStatusPhase(phase)) {
    return (
      <StatusColumn
        connectorName={props.connectorName}
        difference={props.conflict ?? { theyOff: 0, theyOn: 0 }}
        editor={props.editor}
        onReload={props.onReload}
        onRemove={props.onRemove}
        onSignIn={props.onSignIn}
        phase={phase}
      />
    )
  }

  return <ToolsListBody {...props} />
}

function ToolsListBody(props: ToolsListProps) {
  const { t } = useI18n()
  const [filter, setFilter] = useState<ToolsFilter>(EMPTY_TOOLS_FILTER)

  const {
    appOff = false,
    connectorName,
    editor,
    listKey,
    onRetryRules,
    onSignIn,
    preview = false,
    readOnly = false,
    rulesSignedOut = false,
    signedOut = false,
    summaryTitle,
    tools
  } = props

  const frozen = readOnly || preview || appOff
  const all = useShowAllTools(listKey, editor.dirty)

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
  const summary = useMemo(() => facetSummary(tools, editor.isOn), [tools, editor.isOn])

  return (
    <div className="flex min-h-0 flex-1 flex-col" data-slot="tools-list">
      {all.open ? (
        <ToolsFilterBar
          categories={chrome.categories}
          currentAction={editor.currentAction}
          facets={chrome.facets}
          filter={filter}
          hints={chrome.hints}
          onApplyQuickAction={editor.applyQuickAction}
          onFilterChange={setFilter}
          onShowSummary={all.hide}
          quickActions={frozen ? EMPTY_QUICK_ACTIONS : chrome.quickActions}
          tiny={chrome.tiny}
          total={tools.length}
        />
      ) : (
        <ToolsSummary
          connectorName={connectorName}
          onShowAll={all.show}
          onToggleFacet={editor.toggleFacet}
          preview={preview}
          readOnly={readOnly || appOff}
          rows={summary}
          title={summaryTitle}
          total={tools.length}
        />
      )}

      <FrozenLines
        appOffName={appOff ? connectorName : undefined}
        frozen={readOnly || appOff}
        onRetryRules={onRetryRules}
        onSignIn={onSignIn}
        preview={preview}
        rulesSignedOut={rulesSignedOut}
        signedOut={signedOut}
      />

      {all.open ? (
        <ToolViewport
          isOn={editor.isOn}
          label={t.connectorsPage.tools.toolList(connectorName)}
          onToggle={editor.toggle}
          preview={preview}
          readOnly={frozen}
          tools={visible}
        />
      ) : null}

      {all.open && chrome.deprecated > 0 ? (
        <DeprecatedToggle
          count={chrome.deprecated}
          onToggle={() => setFilter({ ...filter, showDeprecated: !filter.showDeprecated })}
          shown={filter.showDeprecated}
        />
      ) : null}

      {editor.dirty && !frozen ? (
        <DirtyFooter
          counts={editor.counts}
          onDiscard={editor.discard}
          onSave={() => void editor.save()}
          saving={editor.phase === 'saving'}
        />
      ) : null}
    </div>
  )
}

function StatusColumn({
  connectorName,
  difference,
  editor,
  onReload,
  onRemove,
  onSignIn,
  phase
}: {
  connectorName: string
  difference: ConflictDifference
  editor: ToolsEditor
  onReload: () => void
  onRemove?: () => void
  onSignIn?: () => void
  phase: ToolsStatusPhase
}) {
  const act = {
    keepMine: () => void editor.keepMine(),
    reload: () => {
      editor.discard()
      onReload()
    },
    remove: onRemove,
    signIn: onSignIn
  } satisfies Record<ToolsStatusAction, (() => void) | undefined>

  return (
    <ToolsStatus
      connectorName={connectorName}
      difference={difference}
      onAction={id => act[id]?.()}
      phase={phase}
    />
  )
}

function Line({ action, label }: { action?: ReactNode; label: string }) {
  return (
    <div className="flex shrink-0 items-center gap-2 border-b border-(--ui-stroke-tertiary) px-3.5 py-1.5">
      <span className="min-w-0 flex-1 text-[0.7rem] text-(--ui-text-tertiary)">{label}</span>
      {action}
    </div>
  )
}

function UnavailableLine({ onRetry }: { onRetry: () => void }) {
  const { t } = useI18n()

  return (
    <Line
      action={
        <Button onClick={onRetry} size="inline" variant="textStrong">
          {t.connectorsPage.tools.retry}
        </Button>
      }
      label={t.connectorsPage.tools.unavailableLine}
    />
  )
}

function SignInLine({ onSignIn }: { onSignIn: () => void }) {
  const { t } = useI18n()

  return (
    <Line
      action={
        <Button onClick={onSignIn} size="inline" variant="textStrong">
          {t.connectorsPage.page.signIn}
        </Button>
      }
      label={t.connectorsPage.tools.staleSignIn}
    />
  )
}

function FrozenLines({
  appOffName,
  frozen,
  onRetryRules,
  onSignIn,
  preview,
  rulesSignedOut,
  signedOut
}: {
  appOffName?: string
  frozen: boolean
  onRetryRules?: () => void
  onSignIn?: () => void
  preview: boolean
  rulesSignedOut: boolean
  signedOut: boolean
}) {
  return (
    <>
      {frozen && !preview ? (
        <RulesLine
          appOffName={appOffName}
          onRetryRules={onRetryRules}
          onSignIn={onSignIn}
          signedOut={rulesSignedOut}
        />
      ) : null}

      {signedOut && onSignIn ? <SignInLine onSignIn={onSignIn} /> : null}
    </>
  )
}

function RulesLine({
  appOffName,
  onRetryRules,
  onSignIn,
  signedOut
}: {
  appOffName?: string
  onRetryRules?: () => void
  onSignIn?: () => void
  signedOut: boolean
}) {
  const { t } = useI18n()
  const copy = t.connectorsPage

  if (appOffName) {
    return <Line label={copy.dialog.rulesAppOff(appOffName)} />
  }

  return (
    <Line
      action={
        signedOut && onSignIn ? (
          <Button onClick={onSignIn} size="inline" variant="textStrong">
            {copy.page.signIn}
          </Button>
        ) : onRetryRules ? (
          <Button onClick={onRetryRules} size="inline" variant="textStrong">
            {copy.page.retry}
          </Button>
        ) : null
      }
      label={signedOut ? copy.dialog.rulesSignIn : copy.dialog.rulesReadOnly}
    />
  )
}

function DeprecatedToggle({ count, onToggle, shown }: { count: number; onToggle: () => void; shown: boolean }) {
  const { t } = useI18n()
  const copy = t.connectorsPage.tools

  return (
    <div className="flex shrink-0 border-t border-(--ui-stroke-tertiary) px-3.5 py-1">
      <Button aria-pressed={shown} onClick={onToggle} size="xs" variant="text">
        {shown ? copy.hideDeprecated(count) : copy.showDeprecated(count)}
      </Button>
    </div>
  )
}

function ToolViewport({
  isOn,
  label,
  onToggle,
  preview,
  readOnly,
  tools
}: {
  isOn: (slug: string) => boolean
  label: string
  onToggle: (slug: string) => void
  preview: boolean
  readOnly: boolean
  tools: ToolRowModel[]
}) {
  const { t } = useI18n()
  const [scrollTop, setScrollTop] = useState(0)
  const [expanded, setExpanded] = useState<null | string>(null)
  const [detailHeight, setDetailHeight] = useState(0)
  const viewport = useViewportHeight()

  const measureDetail = useCallback((node: HTMLDivElement | null) => {
    if (node) {
      setDetailHeight(Math.max(0, node.getBoundingClientRect().height - TOOL_ROW_HEIGHT))
    }
  }, [])

  const expandedIndex = expanded === null ? -1 : tools.findIndex(tool => tool.slug === expanded)
  const extra = expandedIndex >= 0 ? detailHeight : 0
  const total = tools.length * TOOL_ROW_HEIGHT + extra
  const cut = expandedIndex >= 0 ? (expandedIndex + 1) * TOOL_ROW_HEIGHT : Number.POSITIVE_INFINITY

  const indexAt = (y: number) => Math.max(0, Math.floor((y < cut ? y : Math.max(cut, y - extra)) / TOOL_ROW_HEIGHT))

  const offsetAt = (index: number) =>
    index * TOOL_ROW_HEIGHT + (expandedIndex >= 0 && index > expandedIndex ? extra : 0)

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

  const onKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    const node = event.currentTarget
    const step = scrollStepFor(event.key, viewport.height)

    if (step === undefined || event.target !== node) {
      return
    }

    event.preventDefault()
    node.scrollTop = step === 'top' ? 0 : step === 'bottom' ? total : node.scrollTop + step
  }

  return (
    <div
      aria-label={label}
      className="min-h-0 flex-1 overflow-y-auto overscroll-contain [overflow-anchor:none] outline-none focus-visible:ring-[0.1875rem] focus-visible:ring-ring/50"
      data-slot="tools-viewport"
      onKeyDown={onKeyDown}
      onScroll={event => setScrollTop(event.currentTarget.scrollTop)}
      ref={viewport.ref}
      role="list"
      tabIndex={0}
    >
      {tools.length === 0 ? (
        <p className="px-3.5 py-8 text-center text-xs text-(--ui-text-tertiary)">{t.connectorsPage.tools.noMatch}</p>
      ) : null}

      <div className="relative" style={{ height: total }}>
        <div className="absolute inset-x-0" style={{ top: offsetAt(start) }}>
          {tools.slice(start, end).map((tool, offset) => (
            <div
              aria-posinset={start + offset + 1}
              aria-setsize={tools.length}
              key={tool.slug}
              ref={expanded === tool.slug ? measureDetail : undefined}
              role="listitem"
            >
              <ToolRow
                expanded={expanded === tool.slug}
                on={isOn(tool.slug)}
                onExpand={() => toggleExpanded(tool.slug)}
                onToggle={() => onToggle(tool.slug)}
                preview={preview}
                readOnly={readOnly}
                tool={tool}
              />
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

type ScrollStep = 'bottom' | 'top' | number

const SCROLL_KEYS = {
  ArrowDown: () => TOOL_ROW_HEIGHT,
  ArrowUp: () => -TOOL_ROW_HEIGHT,
  End: () => 'bottom',
  Home: () => 'top',
  PageDown: (viewport: number) => viewport,
  PageUp: (viewport: number) => -viewport
} satisfies Record<string, (viewport: number) => ScrollStep>

function scrollStepFor(key: string, viewport: number): ScrollStep | undefined {
  if (!Object.hasOwn(SCROLL_KEYS, key)) {
    return undefined
  }

  // SAFETY: guarded by `Object.hasOwn` on the line above.
  return SCROLL_KEYS[key as keyof typeof SCROLL_KEYS](viewport)
}

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

    // oxlint-disable-next-line anti-slop/no-runtime-typeof -- SAFETY: an environment guard, not a value parse: `ResizeObserver` is absent in jsdom and in an Electron renderer before layout, and `typeof` is the only read that does not throw.
    if (!node || typeof ResizeObserver === 'undefined') {
      return
    }

    const observer = new ResizeObserver(read)

    observer.observe(node)

    return () => observer.disconnect()
  }, [read])

  return { height, ref }
}

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
      className="flex shrink-0 items-center gap-2.5 border-t border-(--ui-stroke-tertiary) bg-(--ui-bg-chrome) px-3.5 py-2"
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
