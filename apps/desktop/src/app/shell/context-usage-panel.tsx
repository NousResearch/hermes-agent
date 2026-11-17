import { compactNumber } from '@hermes/shared'
import { useContext, useMemo, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { useI18n } from '@/i18n'
import { Loader2 } from '@/lib/icons'
import { cn } from '@/lib/utils'
import type { ContextBreakdown, ContextUsageCategory, UsageStats } from '@/types/hermes'

import type { CompactConversationAction } from './hooks/use-compact-conversation'
import { StatusbarMenuCloseContext } from './statusbar-controls'
import { type RequestGateway, TurnInspector } from './turn-inspector'

interface ContextUsagePanelProps {
  breakdown: ContextBreakdown | null
  /** Omitted by surfaces that cannot act on the session: numbers only. */
  compact?: CompactConversationAction
  /** The focused session the turn inspector reads; omitted by surfaces with no session to inspect. */
  inspector?: { requestGateway: RequestGateway; sessionId: null | string }
  loading: boolean
  usage: UsageStats
}

type PanelTab = 'inspector' | 'usage'

/** Presentational: the breakdown is fetched by the statusbar (see
 *  `useContextBreakdown`) because the gauge's own label needs it, so the
 *  popover opens with its numbers already in hand. `usage` is the gauge's
 *  merged figure — measured occupancy when the backend has it, the estimate
 *  otherwise — so the header and the bar can never disagree. */
export function ContextUsagePanel({ breakdown, compact, inspector, loading, usage }: ContextUsagePanelProps) {
  const { t } = useI18n()
  const copy = t.shell.statusbar.contextUsagePanel
  const [tab, setTab] = useState<PanelTab>('usage')
  const contextMax = usage.context_max ?? 0
  const contextUsed = usage.context_used ?? 0
  const contextPercent = Math.max(0, Math.min(100, Math.round(usage.context_percent ?? 0)))

  const categories = useMemo(
    () =>
      (breakdown?.categories ?? []).map(category => ({
        ...category,
        label: copy.categories[category.id as keyof typeof copy.categories] ?? category.label
      })),
    [breakdown?.categories, copy]
  )

  const segmentTotal = categories.reduce((sum, category) => sum + category.tokens, 0) || contextUsed || 1

  const tabs = inspector && <PanelTabs onSelect={setTab} tab={tab} />

  if (inspector && tab === 'inspector') {
    return (
      <div className="flex w-96 flex-col gap-3 p-3 text-[0.75rem]" data-slot="context-usage-panel">
        {tabs}
        <TurnInspector requestGateway={inspector.requestGateway} sessionId={inspector.sessionId} />
      </div>
    )
  }

  return (
    <div className="flex w-72 flex-col gap-3 p-3 text-[0.75rem]" data-slot="context-usage-panel">
      {tabs}
      <div className="flex items-baseline justify-between gap-2">
        <p className="font-medium text-foreground">{copy.title}</p>

        <span className="text-[0.6875rem] text-muted-foreground">
          {copy.tokenSummary(
            `${usage.context_estimated ? '~' : ''}${compactNumber(contextUsed)}`,
            compactNumber(contextMax)
          )}
        </span>
      </div>

      <p className="flex items-center justify-between gap-2 text-[0.6875rem] text-foreground">
        <span>
          {usage.context_estimated ? '~' : ''}
          {copy.percentFull(contextPercent)}
        </span>

        {usage.compressions !== undefined && (
          <span
            className="inline-flex items-center gap-1 text-muted-foreground"
            data-testid="context-panel-compressions"
          >
            <Codicon aria-hidden="true" name="layers" size="0.6875rem" />
            {t.shell.statusbar.compressions(usage.compressions)}
          </span>
        )}
      </p>

      <ContextUsageBar categories={categories} segmentTotal={segmentTotal} />

      <ul className="flex flex-col gap-1.5">
        {categories.map(category => (
          <li className="flex items-center justify-between gap-2" key={category.id}>
            <span className="flex min-w-0 items-center gap-2">
              <span className="size-2 shrink-0 rounded-[2px]" style={{ background: category.color }} />

              <span className="truncate text-muted-foreground">{category.label}</span>
            </span>

            <span className="shrink-0 tabular-nums text-foreground">~{compactNumber(category.tokens)}</span>
          </li>
        ))}
      </ul>

      {loading && !categories.length && <p className="text-[0.6875rem] text-muted-foreground">{copy.loading}</p>}

      {!loading && !categories.length && <p className="text-[0.6875rem] text-muted-foreground">{copy.empty}</p>}

      {compact && <CompactConversationRow compact={compact} />}
    </div>
  )
}

function PanelTabs({ onSelect, tab }: { onSelect: (tab: PanelTab) => void; tab: PanelTab }) {
  const { t } = useI18n()
  const copy = t.shell.statusbar.contextUsagePanel.tabs

  return (
    <div className="flex gap-1 rounded-md bg-muted p-0.5" role="tablist">
      {(['usage', 'inspector'] as const).map(id => (
        <button
          aria-selected={tab === id}
          className={cn(
            'flex-1 rounded-[5px] px-2 py-0.5 text-[0.6875rem] transition-colors',
            tab === id ? 'bg-background text-foreground shadow-xs' : 'text-muted-foreground hover:text-foreground'
          )}
          key={id}
          // Radix's menu treats a click as a selection and closes; the tab switch must keep it open.
          onClick={event => {
            event.preventDefault()
            event.stopPropagation()
            onSelect(id)
          }}
          role="tab"
          type="button"
        >
          {copy[id]}
        </button>
      ))}
    </div>
  )
}

function CompactConversationRow({ compact }: { compact: CompactConversationAction }) {
  const { t } = useI18n()
  const closeMenu = useContext(StatusbarMenuCloseContext)
  const copy = t.shell.statusbar.contextUsagePanel.compact

  return (
    <div className="flex flex-col gap-1.5 border-t border-(--ui-stroke-tertiary) pt-2" data-slot="context-compact">
      <p className="text-[0.6875rem] text-muted-foreground">{compact.busy ? copy.busyHint : copy.hint}</p>

      {compact.pending ? (
        <p
          className="inline-flex items-center gap-1.5 text-[0.6875rem] text-muted-foreground"
          data-testid="context-compact-pending"
        >
          <Loader2 className="size-3 animate-spin" />
          {copy.working}
        </p>
      ) : (
        // Close the dropdown first: left open under the shared confirmation, its
        // modal layer keeps trapping pointer events and the meter cannot reopen.
        <Button
          className="w-full justify-start"
          disabled={compact.busy || compact.unavailable}
          onClick={() => {
            closeMenu()
            compact.onRequest()
          }}
          size="xs"
          type="button"
          variant="ghost"
        >
          <Codicon name="archive" size="0.75rem" />
          {copy.action}
        </Button>
      )}
    </div>
  )
}

/** The statusbar meter's trailing `[bar] N%`, plus a quiet `layers N` once the
 *  live session has compacted — the same glyph the sidebar puts on rows that
 *  came from a compression, so the two read as one idea. Zero stays out of the
 *  meter (every fresh session would carry noise); the panel shows it. */
export function ContextMeterDetail({ bar, compressions }: { bar: string; compressions?: number }) {
  if (!compressions) {
    return bar
  }

  return (
    <span className="inline-flex items-center gap-1.5">
      {bar}
      <span className="inline-flex items-center gap-0.5 tabular-nums" data-testid="context-meter-compressions">
        <Codicon aria-hidden="true" name="layers" size="0.6875rem" />
        {compressions}
      </span>
    </span>
  )
}

function ContextUsageBar({
  categories,
  segmentTotal
}: {
  categories: readonly ContextUsageCategory[]
  segmentTotal: number
}) {
  return (
    <div
      className={cn(
        'flex h-1.5 overflow-hidden rounded-full',
        categories.length ? 'bg-(--ui-stroke-tertiary)' : 'dither bg-(--ui-bg-elevated)'
      )}
      data-slot="context-usage-bar"
    >
      {categories.map(category => (
        <span
          className="h-full min-w-px"
          key={category.id}
          style={{
            background: category.color,
            width: `${(category.tokens / segmentTotal) * 100}%`
          }}
        />
      ))}
    </div>
  )
}
