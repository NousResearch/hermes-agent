import { useMemo } from 'react'

import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { type CodexAccountUsageOptions, useCodexAccountUsage } from '@/hooks/use-codex-account-usage'
import { ExternalLink } from '@/lib/external-link'
import { AlertCircle, BarChart3, Loader2, RefreshCw } from '@/lib/icons'
import { fmtDateTime, relativeTime } from '@/lib/time'
import { cn } from '@/lib/utils'
import type { AccountUsageSnapshot, AccountUsageWindow } from '@/types/hermes'

import type { StatusbarItem } from './statusbar-controls'

const CODEX_USAGE_URL = 'https://chatgpt.com/codex/settings/usage'

function finitePercent(value: null | number | undefined): number | null {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return null
  }

  return Math.max(0, Math.min(100, value))
}

export function accountUsageRemaining(window: AccountUsageWindow): number | null {
  const used = finitePercent(window.used_percent)

  return used === null ? null : Math.round(100 - used)
}

export function primaryAccountUsageWindow(snapshot: AccountUsageSnapshot): AccountUsageWindow | null {
  return snapshot.windows.find(window => accountUsageRemaining(window) !== null) ?? null
}

export function useCodexUsageStatusbarItem(options: CodexAccountUsageOptions): StatusbarItem {
  const { t } = useI18n()
  const copy = t.shell.statusbar
  const { error, loading, refresh, snapshot } = useCodexAccountUsage(options)
  const primaryWindow = snapshot ? primaryAccountUsageWindow(snapshot) : null
  const remaining = primaryWindow ? accountUsageRemaining(primaryWindow) : null
  const codexActive = options.provider.trim().toLowerCase() === 'openai-codex'

  return useMemo(
    () => ({
      className: cn(
        error && 'text-amber-600 hover:text-amber-600',
        remaining !== null && remaining <= 10 && 'text-destructive hover:text-destructive'
      ),
      hidden: !codexActive || !snapshot || remaining === null,
      icon: loading ? <Loader2 className="size-3 animate-spin" /> : <BarChart3 className="size-3" />,
      id: 'codex-account-usage',
      label: remaining === null ? copy.codexUsage : copy.codexUsageLabel(remaining),
      menuAlign: 'end',
      menuClassName: 'w-auto border-(--ui-stroke-secondary) p-0',
      menuContent: snapshot ? (
        <CodexUsagePanel error={error} loading={loading} onRefresh={() => void refresh()} snapshot={snapshot} />
      ) : undefined,
      title: copy.openCodexUsage,
      variant: 'menu'
    }),
    [codexActive, copy, error, loading, refresh, remaining, snapshot]
  )
}

export function CodexUsagePanel({
  error,
  loading,
  onRefresh,
  snapshot
}: {
  error: boolean
  loading: boolean
  onRefresh: () => void
  snapshot: AccountUsageSnapshot
}) {
  const { t } = useI18n()
  const copy = t.shell.statusbar.codexUsagePanel
  const fetchedAt = Date.parse(snapshot.fetched_at)

  return (
    <div className="flex w-80 flex-col text-[0.75rem]" data-slot="codex-usage-panel">
      <div className="flex items-center justify-between gap-3 border-b border-(--ui-stroke-tertiary) px-3 py-2.5">
        <div className="min-w-0">
          <p className="font-medium text-foreground">{copy.title}</p>
          <p className="truncate text-[0.6875rem] text-muted-foreground">
            {snapshot.plan ? copy.plan(snapshot.plan) : copy.subscription}
          </p>
        </div>

        <Button
          aria-label={copy.refresh}
          className="text-muted-foreground hover:text-foreground"
          disabled={loading}
          onClick={onRefresh}
          size="icon-xs"
          variant="ghost"
        >
          <RefreshCw className={cn(loading && 'animate-spin')} />
        </Button>
      </div>

      <ul className="flex flex-col gap-3 px-3 py-3">
        {snapshot.windows.map(window => {
          const used = finitePercent(window.used_percent)
          const remaining = accountUsageRemaining(window)
          const resetAt = window.reset_at ? Date.parse(window.reset_at) : Number.NaN

          return (
            <li className="flex flex-col gap-1.5" key={window.label}>
              <div className="flex items-baseline justify-between gap-3">
                <span className="font-medium text-foreground">{window.label}</span>
                <span className="tabular-nums text-foreground">
                  {remaining === null ? copy.unavailable : copy.remaining(remaining)}
                </span>
              </div>

              <div className="h-1.5 overflow-hidden rounded-full bg-(--ui-stroke-tertiary)">
                <span
                  className="block h-full rounded-full bg-primary transition-[width]"
                  style={{ width: `${used ?? 0}%` }}
                />
              </div>

              <div className="flex items-center justify-between gap-3 text-[0.6875rem] text-muted-foreground">
                <span>{used === null ? window.detail : copy.used(Math.round(used))}</span>
                {Number.isFinite(resetAt) && <span>{copy.resets(relativeTime(resetAt))}</span>}
              </div>
            </li>
          )
        })}
      </ul>

      {snapshot.details.length > 0 && (
        <ul className="border-t border-(--ui-stroke-tertiary) px-3 py-2 text-[0.6875rem] text-muted-foreground">
          {snapshot.details.map(detail => (
            <li key={detail}>{detail}</li>
          ))}
        </ul>
      )}

      {error && (
        <div className="flex items-start gap-2 border-t border-(--ui-stroke-tertiary) px-3 py-2 text-amber-700">
          <AlertCircle className="mt-0.5 size-3 shrink-0" />
          <span>{copy.stale}</span>
        </div>
      )}

      <div className="flex items-center justify-between gap-3 border-t border-(--ui-stroke-tertiary) px-3 py-2 text-[0.6875rem] text-muted-foreground">
        <span>{Number.isFinite(fetchedAt) ? copy.updated(fmtDateTime.format(fetchedAt)) : copy.updatedUnknown}</span>
        <ExternalLink href={CODEX_USAGE_URL} showExternalIcon={false}>
          {copy.openUsage}
        </ExternalLink>
      </div>
    </div>
  )
}
