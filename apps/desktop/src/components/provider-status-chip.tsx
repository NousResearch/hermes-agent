import type { ModelOptionProvider } from '@hermes/shared'
import type { ReactElement, ReactNode } from 'react'

import { useI18n } from '@/i18n'
import type { ModelMenuTranslations } from '@/i18n/types_model_menu'
import {
  accountResetMs,
  formatReset,
  type PoolAccountView,
  poolUsage,
  USAGE_NOTICE_PERCENT,
  USAGE_WARN_PERCENT,
  usageWindows
} from '@/lib/provider-limit'
import { cn } from '@/lib/utils'

import { Badge } from './ui/badge'
import { Tip } from './ui/tooltip'

interface ChipState {
  label: string
  remaining: number
  tip: ReactNode
  tipPlacement?: 'row'
  warn: boolean
}

function poolChipState(provider: ModelOptionProvider, copy: ModelMenuTranslations): ChipState | null {
  const pool = poolUsage(provider)

  if (!pool) {
    return null
  }

  const resetMs = accountResetMs(provider) ?? pool.resetMs
  const time = resetMs === null ? null : formatReset(resetMs)

  const count =
    pool.limited > 0 ? copy.poolLimited(pool.limited, pool.accounts.length) : copy.poolAccounts(pool.accounts.length)

  const label = resetMs === null ? count : time ? copy.limitedUntil(time) : copy.limited

  return {
    label,
    remaining: 0,
    tip: <PoolUsageTip accounts={pool.accounts} copy={copy} />,
    tipPlacement: 'row',
    warn: resetMs !== null
  }
}

/** A scan-first account list: each row shows its tightest live window as a bar, with exact
 *  remaining/reset beside it. Unknown and signed-out rows stay empty rather than looking full. */
function PoolUsageTip({ accounts, copy }: { accounts: PoolAccountView[]; copy: ModelMenuTranslations }) {
  return (
    <span className="grid w-60 gap-2 py-0.5">
      {accounts.map((entry, index) => {
        const tightest = entry.windows[0]
        const reset = (ms: null | number) => (ms === null ? null : formatReset(ms))
        const limited = entry.state === 'limited'
        const unavailable = entry.state === 'unknown' || entry.state === 'unavailable'
        const width = limited || unavailable ? 0 : (tightest?.remaining ?? 0)
        const low = !limited && !unavailable && width <= USAGE_NOTICE_PERCENT
        const accountReset = reset(entry.resetMs)

        const status = unavailable
          ? entry.state === 'unknown'
            ? copy.poolUnknown
            : copy.poolUnavailable
          : limited
            ? accountReset
              ? copy.limitedUntil(accountReset)
              : copy.limited
            : tightest
              ? copy.usageWindow(tightest.label, tightest.remaining, null)
              : copy.poolUnknown

        return (
          <span className="grid gap-1" key={entry.id}>
            <span className="flex min-w-0 items-baseline justify-between gap-3 leading-tight">
              <span className="truncate">{entry.label || copy.poolAccount(index + 1)}</span>
              <span
                className={cn(
                  'shrink-0 tabular-nums opacity-65',
                  // The bubble inverts the theme, so its warning tone does too.
                  limited && 'text-amber-300 opacity-100 dark:text-amber-700'
                )}
              >
                {status}
              </span>
            </span>
            <span
              aria-hidden
              className={cn(
                'h-1 overflow-hidden rounded-full bg-current/15',
                unavailable && 'bg-transparent outline-1 outline-dashed outline-current/25'
              )}
            >
              <span
                className={cn('block h-full rounded-full bg-current/55', low && 'bg-amber-300 dark:bg-amber-700')}
                style={{ width: `${width}%` }}
              />
            </span>
          </span>
        )
      })}
    </span>
  )
}

function useChipState(provider: ModelOptionProvider): ChipState | null {
  const { t } = useI18n()
  const copy = t.shell.modelMenu
  const pool = poolChipState(provider, copy)

  if (pool) {
    return pool
  }

  const resetMs = accountResetMs(provider)

  if (resetMs !== null) {
    const time = formatReset(resetMs)

    return {
      label: time ? copy.limitedUntil(time) : copy.limited,
      remaining: 0,
      tip: copy.limitedTip(provider.name, time),
      warn: true
    }
  }

  const windows = usageWindows(provider)
  const tightest = windows?.[0]

  if (!windows || !tightest || tightest.remaining > USAGE_NOTICE_PERCENT) {
    return null
  }

  const reset = (ms: null | number) => (ms === null ? null : formatReset(ms))

  const lines = [
    copy.usageTip(provider.name),
    ...windows.map(w => copy.usageWindow(w.label, w.remaining, reset(w.resetMs)))
  ]

  return {
    label: copy.usageLeft(tightest.remaining, reset(tightest.resetMs)),
    remaining: tightest.remaining,
    tip: <span className="whitespace-pre-line">{lines.join('\n')}</span>,
    warn: tightest.remaining <= USAGE_WARN_PERCENT
  }
}

/** One status slot in both pickers: a single account's gauge, or a pool's limited-account count.
 *  A pooled percentage would mistake one exhausted subscription for an exhausted provider. */
export function ProviderStatusChip({
  className,
  provider
}: {
  className?: string
  provider: ModelOptionProvider
}): null | ReactElement {
  const state = useChipState(provider)

  if (!state) {
    return null
  }

  // The bar along the bottom edge is what's LEFT, so it matches the label and empties toward the
  // wall; a login that's already out says so in words and drops the bar.
  return (
    <Tip label={state.tip} placement={state.tipPlacement}>
      <Badge
        className={cn('relative shrink-0 overflow-hidden normal-case tracking-normal tabular-nums', className)}
        size="xs"
        variant={state.warn ? 'warn' : 'muted'}
      >
        {state.remaining > 0 && (
          <span aria-hidden className="absolute inset-x-0 bottom-0 h-0.5 bg-current/20">
            <span className="absolute inset-y-0 left-0 bg-current" style={{ width: `${state.remaining}%` }} />
          </span>
        )}
        <span className="relative">{state.label}</span>
      </Badge>
    </Tip>
  )
}
