import type { ModelOptionProvider } from '@hermes/shared'
import type { ReactElement, ReactNode } from 'react'

import { useI18n } from '@/i18n'
import {
  accountResetMs,
  formatReset,
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
  warn: boolean
}

function useChipState(provider: ModelOptionProvider): ChipState | null {
  const { t } = useI18n()
  const copy = t.shell.modelMenu
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

/** The one status slot on a provider heading, a fuel gauge: hidden while there's room, a progress
 *  chip once the tightest usage window is nearly spent (`8% left · resets 4:30 PM`, amber from
 *  10%), and `Limited until 4:30 PM` once the login is out, so you see the wall coming. */
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
    <Tip label={state.tip}>
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
