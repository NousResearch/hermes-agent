import { compactNumber } from '@hermes/shared'
import { useStore } from '@nanostores/react'
import { useMemo } from 'react'

import { Button } from '@/components/ui/button'
import { ErrorState } from '@/components/ui/error-state'
import { GlyphSpinner } from '@/components/ui/glyph-spinner'
import { Tip } from '@/components/ui/tooltip'
import { type AccountUsageOptions, useAccountUsage } from '@/hooks/use-account-usage'
import { useI18n } from '@/i18n'
import { BarChart3, RefreshCw } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { $statusbarHiddenIds, $statusbarVisible } from '@/store/statusbar-prefs'
import type { AccountUsageSnapshot, AccountUsageWindow } from '@/types/account-usage'
import type { UsageStats } from '@/types/hermes'

import {
  accountUsageDetails,
  accountUsageMinRemaining,
  accountUsagePercent,
  accountUsageProviderLabel,
  accountUsageRemaining,
  accountUsageTimestamp,
  accountUsageWindowFooter,
  formatCreditsBalance
} from './account-usage-presentation'
import type { StatusbarItem } from './statusbar-controls'

interface AccountUsageStatusbarOptions extends Omit<AccountUsageOptions, 'enabled'> {
  usage: UsageStats
}

export function useAccountUsageStatusbarItem(options: AccountUsageStatusbarOptions): StatusbarItem {
  const { locale, t } = useI18n()
  const copy = t.shell.statusbar
  const hiddenIds = useStore($statusbarHiddenIds)
  const visible = useStore($statusbarVisible)

  const { available, error, loading, refresh, snapshot } = useAccountUsage({
    ...options,
    enabled: visible && !hiddenIds.includes('account-usage')
  })

  const remaining = snapshot ? accountUsageMinRemaining(snapshot) : null
  const credits = snapshot?.credits_balance
  const providerName = accountUsageProviderLabel(options.provider)
  const parts: string[] = []

  if (typeof credits === 'number' && Number.isFinite(credits)) {
    parts.push(formatCreditsBalance(credits, locale))
  }

  if (remaining !== null) {
    parts.push(copy.accountUsageLeft(accountUsagePercent(remaining, locale)))
  }

  if (error) {
    parts.push(copy.accountUsagePanel.unavailable)
  }

  const label = parts.length ? `${providerName}: ${parts.join(' · ')}` : providerName

  return useMemo(
    () => ({
      className:
        remaining !== null && remaining <= 5
          ? 'text-destructive hover:text-destructive'
          : remaining !== null && remaining <= 20
            ? 'text-amber-600 hover:text-amber-600 dark:text-amber-400'
            : undefined,
      hidden: !available,
      icon: loading ? (
        <GlyphSpinner ariaLabel={copy.accountUsagePanel.refresh} className="size-3" />
      ) : (
        <BarChart3 className="size-3" />
      ),
      id: 'account-usage',
      label,
      menuAlign: 'end',
      menuClassName: 'w-auto p-0',
      menuContent: (
        <AccountUsagePanel
          error={error}
          loading={loading}
          onRefresh={() => void refresh()}
          provider={options.provider}
          snapshot={snapshot}
          usage={options.usage}
        />
      ),
      title: copy.accountUsage,
      toggleLabel: copy.toggleAccountUsage,
      variant: 'menu'
    }),
    [available, copy, error, label, loading, options.provider, options.usage, refresh, remaining, snapshot]
  )
}

interface AccountUsageWindowRowProps {
  window: AccountUsageWindow
}

function AccountUsageWindowRow({ window }: AccountUsageWindowRowProps) {
  const { locale, t } = useI18n()
  const copy = t.shell.statusbar.accountUsagePanel
  const remaining = accountUsageRemaining(window)

  const used = remaining === null ? null : 100 - remaining

  return (
    <li className="flex min-w-0 flex-col gap-1">
      <div className="flex min-w-0 flex-wrap items-baseline justify-between gap-x-2 gap-y-1">
        <span className="min-w-0 break-words font-medium text-foreground">
          {copy.windowLabels[window.label_key]}
          {window.scope === 'model' ? ` · ${copy.modelQuota}` : ''}
        </span>
        <span className="min-w-0 break-words tabular-nums text-foreground">
          {remaining === null ? copy.unavailable : copy.remaining(accountUsagePercent(remaining, locale))}
        </span>
      </div>
      {used !== null && (
        <div className="h-1.5 overflow-hidden rounded-full bg-(--ui-stroke-tertiary)">
          <span className="block h-full rounded-full bg-primary" style={{ width: `${used}%` }} />
        </div>
      )}
      <div className="flex min-w-0 flex-wrap items-start justify-between gap-x-2 gap-y-1 text-[0.6875rem] text-muted-foreground">
        {used !== null && <span>{copy.used(accountUsagePercent(used, locale))}</span>}
        <span className="min-w-0 break-words">{accountUsageWindowFooter(window, copy, locale)}</span>
      </div>
    </li>
  )
}

interface AccountUsagePanelProps {
  error: boolean
  loading: boolean
  onRefresh: () => void
  provider: string
  snapshot: AccountUsageSnapshot | null
  usage: UsageStats
}

export function AccountUsagePanel({ error, loading, onRefresh, provider, snapshot, usage }: AccountUsagePanelProps) {
  const { locale, t } = useI18n()
  const copy = t.shell.statusbar.accountUsagePanel
  const updated = accountUsageTimestamp(snapshot?.fetched_at, locale)
  const details = snapshot ? accountUsageDetails(snapshot.rows, copy, locale) : []

  return (
    <div
      className="flex max-h-[70vh] w-80 max-w-[calc(100vw-2rem)] min-w-0 flex-col gap-3 overflow-y-auto whitespace-normal p-3 text-[0.75rem]"
      data-slot="account-usage-panel"
    >
      <div className="min-w-0">
        <p className="break-words font-medium text-foreground">{copy.title}</p>
        <p className="break-words text-[0.6875rem] text-muted-foreground">
          {accountUsageProviderLabel(provider)}
          {snapshot?.plan ? ` · ${copy.plan(snapshot.plan)}` : ''}
        </p>
      </div>
      {error ? (
        <ErrorState
          description={<p className="break-words text-muted-foreground">{copy.refreshFailed}</p>}
          title={<p>{copy.unavailable}</p>}
        />
      ) : snapshot ? (
        <>
          <ul className="flex min-w-0 flex-col gap-3">
            {snapshot.windows.map((window, index) => (
              <AccountUsageWindowRow key={`${window.label_key}:${index}`} window={window} />
            ))}
          </ul>
          {details.length > 0 && (
            <ul className="flex min-w-0 flex-col gap-2">
              {details.map((detail, index) => (
                <li className="flex min-w-0 flex-col gap-1" key={index}>
                  {detail.label && <p className="break-words font-medium text-foreground">{detail.label}</p>}
                  <p className="break-words text-[0.6875rem] text-muted-foreground">{detail.value}</p>
                </li>
              ))}
            </ul>
          )}
        </>
      ) : (
        <GlyphSpinner ariaLabel={copy.refresh} />
      )}
      {usage.total > 0 && (
        <div className="flex min-w-0 flex-col gap-1">
          <p className="break-words font-medium text-foreground">{copy.thisSession}</p>
          <p className="break-words tabular-nums text-[0.6875rem] text-muted-foreground">
            {copy.sessionLine(compactNumber(usage.input), compactNumber(usage.output), compactNumber(usage.total))}
          </p>
        </div>
      )}
      <div className="flex min-w-0 items-center justify-between gap-2 text-[0.6875rem] text-muted-foreground">
        <span className="min-w-0 break-words">{updated ? copy.updated(updated) : copy.updatedUnknown}</span>
        <Tip label={copy.refresh}>
          <Button aria-label={copy.refresh} disabled={loading} onClick={onRefresh} size="icon-xs" variant="ghost">
            <RefreshCw className={cn(loading && 'animate-spin motion-reduce:animate-none')} />
          </Button>
        </Tip>
      </div>
    </div>
  )
}
