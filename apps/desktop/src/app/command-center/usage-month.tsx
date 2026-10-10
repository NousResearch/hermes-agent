import { compactNumber } from '@hermes/shared'
import { useStore } from '@nanostores/react'
import { type ReactElement, useEffect, useState } from 'react'

import { useElapsedSeconds } from '@/components/chat/activity-timer'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Progress } from '@/components/ui/progress'
import { SegmentedControl } from '@/components/ui/segmented-control'
import { setUsageBudget } from '@/hermes'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import { $usageMonth, $usageMonthState, refreshUsageMonth } from '@/store/usage-month'
import type { UsageBudget, UsageBudgetUpdate, UsageMonthProvider } from '@/types/hermes'

const usd = new Intl.NumberFormat(undefined, { currency: 'USD', style: 'currency' })

const headingClass = 'text-[0.625rem] font-medium uppercase tracking-[0.08em] text-(--ui-text-tertiary)'
const captionClass = 'text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)'
const errorClass = 'text-[length:var(--conversation-caption-font-size)] text-destructive'

/** A budget figure in its own unit: compact tokens, or dollars. */
function amount(kind: UsageBudget['kind'], value: number): string {
  return kind === 'usd' ? usd.format(value) : compactNumber(value)
}

/** `2026-10-26` as the viewer's short date, read as a local calendar day (not UTC midnight). */
export function formatRunsOut(isoDate: string): string {
  const [year, month, day] = isoDate.split('-').map(Number)

  return new Date(year, month - 1, day).toLocaleDateString(undefined, { day: 'numeric', month: 'short' })
}

/** "This month" in Command Center → Usage. Per billing provider: tokens used (always known), money
 *  where the price is known, and the count of sessions with no known price, which is never shown as
 *  $0. A budgeted provider adds its bar, the month-end pace and the day the pace crosses the limit. */
export function UsageMonthSection(): ReactElement {
  const { t } = useI18n()
  const cc = t.commandCenter
  const month = useStore($usageMonth)
  const state = useStore($usageMonthState)
  const loadingFor = useElapsedSeconds(state.loading, undefined, state.startedAt || undefined)

  useEffect(() => {
    void refreshUsageMonth()
  }, [])

  return (
    <section className="flex flex-col gap-3">
      <div className="flex items-baseline justify-between">
        <span className={headingClass}>{cc.thisMonth}</span>
        {month ? (
          <span className={captionClass}>{cc.monthProgress(Math.ceil(month.days_elapsed), month.days_in_month)}</span>
        ) : null}
      </div>
      {state.loading && !month ? <span className={captionClass}>{cc.loadingMonth(loadingFor)}</span> : null}
      {state.error ? <span className={errorClass}>{state.error}</span> : null}
      {month?.providers.length === 0 ? <span className={captionClass}>{cc.noUsageThisMonth}</span> : null}
      {month?.providers.map(row => (
        <ProviderUsageRow key={row.provider} row={row} />
      ))}
    </section>
  )
}

function ProviderUsageRow({ row }: { row: UsageMonthProvider }): ReactElement {
  const { t } = useI18n()
  const cc = t.commandCenter
  const [editing, setEditing] = useState(false)

  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-baseline justify-between gap-3">
        <span className="min-w-0 truncate text-sm font-medium">{row.provider}</span>
        <span className={cn(captionClass, 'flex shrink-0 gap-2 tabular-nums')}>
          <span>{cc.tokensUsed(compactNumber(row.tokens))}</span>
          {row.estimated_cost > 0 ? <span>{cc.spentKnown(usd.format(row.estimated_cost))}</span> : null}
        </span>
      </div>
      {row.budget ? <BudgetMeter budget={row.budget} /> : null}
      <div className="flex items-center justify-between gap-3">
        <span className={captionClass}>
          {row.unpriced_sessions > 0 ? cc.unpricedSessions(row.unpriced_sessions) : null}
        </span>
        {editing ? null : (
          <Button onClick={() => setEditing(true)} size="xs" variant="text">
            {cc.setBudget}
          </Button>
        )}
      </div>
      {editing ? <BudgetEditor budget={row.budget} onDone={() => setEditing(false)} provider={row.provider} /> : null}
    </div>
  )
}

function BudgetMeter({ budget }: { budget: UsageBudget }): ReactElement {
  const { t } = useI18n()
  const cc = t.commandCenter
  const over = budget.used_ratio >= 1

  return (
    <div className="flex flex-col gap-1">
      <Progress
        destructive={over}
        fillClassName={!over && budget.projected_ratio >= 1 ? 'bg-amber-500' : undefined}
        size="sm"
        value={Math.min(budget.used_ratio, 1)}
      />
      <span className={cn(captionClass, 'flex flex-wrap gap-x-2 tabular-nums')}>
        <span>{cc.budgetOf(amount(budget.kind, budget.used), amount(budget.kind, budget.limit))}</span>
        <span>{cc.projected(Math.round(budget.projected_ratio * 100))}</span>
        {budget.runs_out_on ? <span>{cc.runsOutOn(formatRunsOut(budget.runs_out_on))}</span> : null}
      </span>
    </div>
  )
}

interface BudgetEditorProps {
  budget: null | UsageBudget
  onDone: () => void
  provider: string
}

function BudgetEditor({ budget, onDone, provider }: BudgetEditorProps): ReactElement {
  const { t } = useI18n()
  const cc = t.commandCenter
  const [kind, setKind] = useState<UsageBudget['kind']>(budget?.kind ?? 'tokens')
  const [value, setValue] = useState(budget ? String(budget.limit) : '')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const savingFor = useElapsedSeconds(saving)
  const limit = Number(value)

  const save = async (update: UsageBudgetUpdate) => {
    setSaving(true)
    setError('')

    try {
      await setUsageBudget(update)
      await refreshUsageMonth()
      onDone()
    } catch (failure) {
      setError(cc.budgetSaveFailed(failure instanceof Error ? failure.message : String(failure)))
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="flex flex-wrap items-center gap-2">
      <SegmentedControl
        onChange={setKind}
        options={[
          { id: 'tokens', label: cc.budgetTokens },
          { id: 'usd', label: cc.budgetUsd }
        ]}
        value={kind}
      />
      <Input
        aria-label={cc.setBudget}
        className="w-36"
        min={0}
        onChange={event => setValue(event.target.value)}
        type="number"
        value={value}
      />
      <Button
        disabled={saving || !(limit > 0)}
        onClick={() =>
          void save(kind === 'usd' ? { monthly_usd: limit, provider } : { monthly_tokens: limit, provider })
        }
        size="xs"
      >
        {saving ? cc.savingBudget(savingFor) : cc.saveBudget}
      </Button>
      {budget ? (
        <Button disabled={saving} onClick={() => void save({ provider })} size="xs" variant="text">
          {cc.clearBudget}
        </Button>
      ) : null}
      {error ? <span className={errorClass}>{error}</span> : null}
    </div>
  )
}
