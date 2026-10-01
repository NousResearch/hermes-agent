import { compactNumber } from '@hermes/shared'
import { useId, useMemo, useState } from 'react'

import { Codicon } from '@/components/ui/codicon'
import { DisclosureCaret } from '@/components/ui/disclosure-caret'
import { DropdownMenuItem } from '@/components/ui/dropdown-menu'
import { OverflowTip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import type { ContextBreakdown, ContextUsageCategory, UsageStats } from '@/types/hermes'

interface ContextUsagePanelProps {
  breakdown: ContextBreakdown | null
  loading: boolean
  usage: UsageStats
}

/** Presentational: the breakdown is fetched by the statusbar (see
 *  `useContextBreakdown`) because the gauge's own label needs it, so the
 *  popover opens with its numbers already in hand. `usage` is the gauge's
 *  merged figure — measured occupancy when the backend has it, the estimate
 *  otherwise — so the header and the bar can never disagree. */
export function ContextUsagePanel({ breakdown, loading, usage }: ContextUsagePanelProps) {
  const { t } = useI18n()
  const copy = t.shell.statusbar.contextUsagePanel
  const contextFilesId = useId()
  const [contextFilesOpen, setContextFilesOpen] = useState(false)
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

  const contextFiles = (breakdown?.context_files ?? []).map(source => {
    const status = Object.prototype.hasOwnProperty.call(copy.contextFileStatuses, source.status)
      ? (source.status as keyof typeof copy.contextFileStatuses)
      : 'unknown'

    return { ...source, statusLabel: copy.contextFileStatuses[status] }
  })

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

      {loading && !categories.length && <p className="text-[0.6875rem] text-muted-foreground">{copy.loading}</p>}

      {!loading && !categories.length && <p className="text-[0.6875rem] text-muted-foreground">{copy.empty}</p>}

      {contextFiles.length > 0 && (
        <section className="border-t border-(--ui-stroke-tertiary) pt-2" data-slot="context-files">
          <DropdownMenuItem
            aria-controls={contextFilesId}
            aria-expanded={contextFilesOpen}
            onSelect={event => {
              // This is a disclosure inside the statusbar menu, not a terminal
              // command. Keep the menu open while its inline details expand.
              event.preventDefault()
              setContextFilesOpen(open => !open)
            }}
          >
            <DisclosureCaret open={contextFilesOpen} />
            {copy.contextFiles(contextFiles.length)}
          </DropdownMenuItem>

          {contextFilesOpen && (
            <div id={contextFilesId}>
              <p className="mt-2 text-[0.6875rem] leading-snug text-muted-foreground">{copy.fullFileEstimateNote}</p>

              <ul className="mt-2 flex flex-col gap-2">
                {contextFiles.map(source => (
                  <li className="min-w-0" data-status={source.status} key={`${source.path}:${source.label}`}>
                    {/* Beside the name, not the category total column: raw file size, not that file's share. */}
                    <div className="flex min-w-0 flex-wrap items-baseline gap-x-2">
                      <span className="min-w-0 max-w-full truncate font-medium text-foreground">{source.label}</span>
                      <span className="text-[0.6875rem] tabular-nums text-muted-foreground">
                        {copy.fullFileEstimate(compactNumber(source.est_tokens))}
                      </span>
                    </div>

                    <OverflowTip boundary="viewport" label={source.path} side="left">
                      <span className="block truncate text-[0.6875rem] text-muted-foreground">{source.path}</span>
                    </OverflowTip>

                    <p className="text-[0.6875rem] text-muted-foreground">{source.statusLabel}</p>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </section>
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
