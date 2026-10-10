import { useStore } from '@nanostores/react'
import { useEffect } from 'react'

import { COMMAND_CENTER_ROUTE } from '@/app/routes'
import type { StatusbarItem } from '@/app/shell/statusbar-controls'
import { useI18n } from '@/i18n'
import type { Translations } from '@/i18n/types'
import { Gauge, Layers3, Zap } from '@/lib/icons'
import { cacheHitLabel, tokensPerSecondLabel } from '@/lib/statusbar'
import { $usageMonth, refreshUsageMonth } from '@/store/usage-month'
import type { UsageMonthResponse, UsageStats } from '@/types/hermes'

/** Output speed of the active session's last turn. */
export function tokensPerSecondItem(copy: Translations['shell']['statusbar'], usage: UsageStats): StatusbarItem {
  return {
    icon: <Zap className="size-3" />,
    id: 'tokens-per-second',
    label: tokensPerSecondLabel(usage) || '—',
    title: copy.tokensPerSecondTitle,
    toggleLabel: copy.toggleTokensPerSecond,
    variant: 'text'
  }
}

/** Prompt-cache hit rate for the active session's last turns. */
export function cacheHitRateItem(copy: Translations['shell']['statusbar'], usage: UsageStats): StatusbarItem {
  return {
    icon: <Layers3 className="size-3" />,
    id: 'cache-hit-rate',
    // Same never-self-hide rule as the context meter: opted in means a
    // placeholder until the first cached turn reports, not a vanished item.
    label: cacheHitLabel(usage) || '—',
    title: copy.cacheHitRateTitle,
    toggleLabel: copy.toggleCacheHitRate,
    variant: 'text'
  }
}

// Month-to-date usage moves slowly; a periodic refresh keeps the chip honest without polling per turn.
const REFRESH_MS = 5 * 60_000

/** The chip's state: the budget nearest its limit ("xiaomi 62%"; the backend sorts that one first).
 *  It stays out of the bar until a budget exists, so a user who never set one sees no new noise. */
export function usageBudgetChip(month: null | UsageMonthResponse): Pick<StatusbarItem, 'hidden' | 'label' | 'to'> {
  const nearest = month?.providers.find(row => row.budget)

  return {
    hidden: !nearest?.budget,
    label: nearest?.budget ? `${nearest.provider} ${Math.round(nearest.budget.used_ratio * 100)}%` : undefined,
    to: `${COMMAND_CENTER_ROUTE}?section=usage`
  }
}

/** Status-bar item for this month's usage against the user's budgets; opens Command Center → Usage. */
export function useUsageBudgetItem(): StatusbarItem {
  const { t } = useI18n()
  const copy = t.shell.statusbar
  const month = useStore($usageMonth)

  useEffect(() => {
    void refreshUsageMonth()
    const timer = window.setInterval(() => void refreshUsageMonth(), REFRESH_MS)

    return () => window.clearInterval(timer)
  }, [])

  return {
    ...usageBudgetChip(month),
    icon: <Gauge className="size-3" />,
    id: 'usage-budget',
    title: copy.usageBudgetTitle,
    toggleLabel: copy.toggleUsageBudget,
    variant: 'link'
  }
}
