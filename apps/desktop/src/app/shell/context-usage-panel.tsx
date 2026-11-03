import { compactNumber } from '@hermes/shared'
import { useMemo } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { ConfirmDialog } from '@/components/ui/confirm-dialog'
import { useI18n } from '@/i18n'
import { Loader2 } from '@/lib/icons'
import { cn } from '@/lib/utils'
import type { ContextBreakdown, ContextUsageCategory, UsageStats } from '@/types/hermes'

import type { CompactConversationAction } from './hooks/use-compact-conversation'

interface ContextUsagePanelProps {
  breakdown: ContextBreakdown | null
  /** The popover's compact action. Omitted by surfaces that cannot act on the
   *  session (no controller, no runtime id) — the panel then shows numbers only. */
  compact?: CompactConversationAction
  loading: boolean
  usage: UsageStats
}

/** Presentational: the breakdown is fetched by the statusbar (see
 *  `useContextBreakdown`) because the gauge's own label needs it, so the
 *  popover opens with its numbers already in hand. `usage` is the gauge's
 *  merged figure — measured occupancy when the backend has it, the estimate
 *  otherwise — so the header and the bar can never disagree. */
export function ContextUsagePanel({ breakdown, compact, loading, usage }: ContextUsagePanelProps) {
  const { t } = useI18n()
  const copy = t.shell.statusbar.contextUsagePanel
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

  return (
    <div className="flex w-72 flex-col gap-3 p-3 text-[0.75rem]" data-slot="context-usage-panel">
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

      {compact && <CompactConversationRow compact={compact} />}

      {loading && !categories.length && <p className="text-[0.6875rem] text-muted-foreground">{copy.loading}</p>}

      {!loading && !categories.length && <p className="text-[0.6875rem] text-muted-foreground">{copy.empty}</p>}
    </div>
  )
}

/** The popover's one verb: the numbers are read-only, compaction changes them.
 *  Two steps (ask, then run) because the RPC rewrites the transcript and can
 *  take minutes, and the running state outlives this popover — see
 *  `useCompactConversation`. Every branch is a real state, never a guess: the
 *  button says why it cannot run, the pending label outlives the dropdown, and
 *  a failure is printed where the user opened the action. */
function CompactConversationRow({ compact }: { compact: CompactConversationAction }) {
  const { t } = useI18n()
  const copy = t.shell.statusbar.contextUsagePanel.compact

  return (
    <div className="flex flex-col gap-1.5 border-t border-(--ui-stroke-tertiary) pt-2" data-slot="context-compact">
      <p className="text-[0.6875rem] text-muted-foreground">{copy.hint}</p>

      {compact.pending ? (
        <p
          className="inline-flex items-center gap-1.5 text-[0.6875rem] text-muted-foreground"
          data-testid="context-compact-pending"
        >
          <Loader2 className="size-3 animate-spin" />
          {copy.working}
        </p>
      ) : (
        <>
          <Button
            className="w-full justify-start"
            disabled={compact.busy || compact.unavailable}
            onClick={compact.onRequest}
            size="xs"
            type="button"
            variant="ghost"
          >
            <Codicon name="archive" size="0.75rem" />
            {copy.action}
          </Button>
        </>
      )}

      <ConfirmDialog
        confirmLabel={copy.confirm}
        dismissOnConfirm
        onClose={compact.onCancel}
        onConfirm={compact.onConfirm}
        open={compact.confirming && !compact.busy && !compact.unavailable}
        title={copy.question}
      />

      {compact.busy && <p className="text-[0.6875rem] text-muted-foreground">{copy.busyHint}</p>}

      {compact.error && (
        <p className="whitespace-pre-line text-[0.6875rem] text-destructive" data-testid="context-compact-error">
          {compact.error}
        </p>
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
