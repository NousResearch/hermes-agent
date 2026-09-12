import { useMemo } from 'react'

import { Button } from '@/components/ui/button'
import { type AccountUsageOptions, useAccountUsage } from '@/hooks/use-account-usage'
import { useI18n } from '@/i18n'
import type { Translations } from '@/i18n/types'
import { ExternalLink } from '@/lib/external-link'
import { compactNumber } from '@/lib/format'
import { AlertCircle, BarChart3, Loader2, RefreshCw } from '@/lib/icons'
import { relativeTime } from '@/lib/time'
import { cn } from '@/lib/utils'
import type { AccountUsageRow, AccountUsageSnapshot, AccountUsageWindow, UsageStats } from '@/types/hermes'

import type { StatusbarItem } from './statusbar-controls'

// * Display names and billing URLs are presentation only — capability lives on
// * the backend `unsupported` status, not this map.
const PROVIDER_DISPLAY_NAMES: Record<string, string> = {
  anthropic: 'Anthropic',
  'openai-codex': 'Codex',
  openrouter: 'OpenRouter'
}

const ACCOUNT_USAGE_SETTINGS_URLS: Record<string, string> = {
  anthropic: 'https://console.anthropic.com/settings/billing',
  'openai-codex': 'https://chatgpt.com/codex/settings/usage',
  openrouter: 'https://openrouter.ai/settings/credits'
}

function finitePercent(value: null | number | undefined): number | null {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return null
  }

  return Math.max(0, Math.min(100, value))
}

function finiteCredits(value: null | number | undefined): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function creditsLocale(locale: string): string {
  if (locale === 'zh-hant') {
    return 'zh-Hant'
  }

  if (locale === 'zh') {
    return 'zh-CN'
  }

  if (locale === 'en') {
    return 'en-US'
  }

  return locale
}

type AccountUsagePanelCopy = Translations['shell']['statusbar']['accountUsagePanel']

type StructuredAccountUsageDetail =
  | { kind: 'metric'; label: string; value: string }
  | { kind: 'note'; text: string }

const ACCOUNT_USAGE_RESET_INTERVALS = ['daily', 'monthly', 'weekly'] as const

type AccountUsageResetInterval = (typeof ACCOUNT_USAGE_RESET_INTERVALS)[number]

export function formatCreditsBalance(value: number, locale: string, currency = 'USD'): string {
  const code = currency.trim() || 'USD'

  // * Unknown ISO codes throw RangeError; fall back so a bad wire value
  // * cannot crash the panel.
  try {
    return new Intl.NumberFormat(creditsLocale(locale), {
      currency: code,
      maximumFractionDigits: 2,
      minimumFractionDigits: 2,
      style: 'currency'
    }).format(value)
  } catch {
    return `${value.toFixed(2)} ${code}`
  }
}

export function accountUsageProviderLabel(provider: string): string {
  const slug = provider.trim().toLowerCase()
  const mapped = PROVIDER_DISPLAY_NAMES[slug]

  if (mapped) {
    return mapped
  }

  return slug
    .split(/[-_]+/)
    .filter(Boolean)
    .map(part => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ')
}

export function accountUsageRemaining(window: AccountUsageWindow): number | null {
  const used = finitePercent(window.used_percent)

  return used === null ? null : Math.round(100 - used)
}

export function accountUsageMinRemaining(snapshot: AccountUsageSnapshot): number | null {
  let min: number | null = null

  for (const window of snapshot.windows) {
    const remaining = accountUsageRemaining(window)

    if (remaining === null) {
      continue
    }

    min = min === null ? remaining : Math.min(min, remaining)
  }

  return min
}

export function tightestAccountUsageWindow(snapshot: AccountUsageSnapshot): AccountUsageWindow | null {
  let tightest: AccountUsageWindow | null = null
  let min: number | null = null

  for (const window of snapshot.windows) {
    const remaining = accountUsageRemaining(window)

    if (remaining === null) {
      continue
    }

    if (min === null || remaining < min) {
      min = remaining
      tightest = window
    }
  }

  return tightest
}

// * Compact chip ratio uses whole dollars so a $20 key reads as $19/20, not $19.29/20.00.
export function formatCompactQuotaRatio(
  remaining: null | number | undefined,
  limit: null | number | undefined
): string | null {
  const rem = finiteCredits(remaining)
  const lim = finiteCredits(limit)
  if (rem === null || lim === null) {
    return null
  }

  const roundedLimit = Math.round(lim)
  if (roundedLimit <= 0) {
    return null
  }

  const roundedRemaining = Math.min(roundedLimit, Math.max(0, Math.round(rem)))
  return `$${roundedRemaining}/${roundedLimit}`
}

function accountUsageSettingsUrl(provider: string): string | undefined {
  return ACCOUNT_USAGE_SETTINGS_URLS[provider.trim().toLowerCase()]
}

function isAccountUsageResetInterval(value: string): value is AccountUsageResetInterval {
  return (ACCOUNT_USAGE_RESET_INTERVALS as readonly string[]).includes(value)
}

function rowNumber(args: AccountUsageRow['args'], key: string): number | null {
  const raw = args?.[key]

  if (typeof raw === 'number') {
    return Number.isFinite(raw) ? raw : null
  }

  if (typeof raw === 'string' && raw.trim()) {
    const parsed = Number(raw)
    return Number.isFinite(parsed) ? parsed : null
  }

  return null
}

function rowString(args: AccountUsageRow['args'], key: string): string | undefined {
  const raw = args?.[key]
  return typeof raw === 'string' && raw.trim() ? raw.trim() : undefined
}

function structuredAccountUsageRows(snapshot: AccountUsageSnapshot): AccountUsageRow[] | null {
  if (snapshot.details_structured !== true || !snapshot.rows?.length) {
    return null
  }

  return snapshot.rows
}

function accountUsageWindowTitle(window: AccountUsageWindow, copy: AccountUsagePanelCopy): string {
  const key = window.label_key?.trim()
  // * Own-property check so unknown keys (including prototype names) stay on
  // * the raw English fallback instead of resolving Object.prototype.
  if (key && Object.hasOwn(copy.windowLabels, key)) {
    return copy.windowLabels[key as keyof typeof copy.windowLabels]
  }

  return window.label
}

function accountUsageWindowFooter(
  window: AccountUsageWindow,
  copy: AccountUsagePanelCopy,
  locale: string
): string {
  const resetAt = window.reset_at ? Date.parse(window.reset_at) : Number.NaN
  if (Number.isFinite(resetAt)) {
    return copy.resets(relativeTime(resetAt))
  }

  const remaining = finiteCredits(window.limit_remaining)
  const limit = finiteCredits(window.limit)
  if (remaining !== null && limit !== null) {
    const composed = copy.ofLimitRemaining(
      formatCreditsBalance(remaining, locale),
      formatCreditsBalance(limit, locale)
    )
    const interval = window.reset_interval?.trim()
    if (interval && isAccountUsageResetInterval(interval)) {
      return `${composed} · ${copy.resetIntervals[interval]}`
    }

    return composed
  }

  return window.detail ?? ''
}

function appendUsagePart(
  parts: string[],
  args: AccountUsageRow['args'],
  key: string,
  locale: string,
  format: (value: string) => string
): void {
  const amount = rowNumber(args, key)
  if (amount === null) {
    return
  }

  parts.push(format(formatCreditsBalance(amount, locale)))
}

function structuredDetailFromRow(
  row: AccountUsageRow,
  copy: AccountUsagePanelCopy,
  locale: string
): StructuredAccountUsageDetail | null {
  switch (row.key) {
    case 'credits_balance': {
      const value = rowNumber(row.args, 'value')
      if (value === null) {
        return null
      }

      return {
        kind: 'metric',
        label: copy.creditsBalance,
        value: formatCreditsBalance(value, locale, rowString(row.args, 'currency') ?? 'USD')
      }
    }
    case 'credits_unlimited':
      return { kind: 'metric', label: copy.creditsBalance, value: copy.creditsUnlimited }
    case 'api_key_usage': {
      const total = rowNumber(row.args, 'total')
      if (total === null) {
        return null
      }

      const parts = [copy.usageTotal(formatCreditsBalance(total, locale))]
      appendUsagePart(parts, row.args, 'daily', locale, copy.usageToday)
      appendUsagePart(parts, row.args, 'weekly', locale, copy.usageThisWeek)
      appendUsagePart(parts, row.args, 'monthly', locale, copy.usageThisMonth)
      return { kind: 'metric', label: copy.apiKeyUsage, value: parts.join(' · ') }
    }
    case 'extra_usage': {
      const used = rowNumber(row.args, 'used')
      const limit = rowNumber(row.args, 'limit')
      if (used === null || limit === null) {
        return null
      }

      const currency = rowString(row.args, 'currency') ?? 'USD'
      return {
        kind: 'metric',
        label: copy.extraUsage,
        value: copy.extraUsageValue(
          formatCreditsBalance(used, locale, currency),
          formatCreditsBalance(limit, locale, currency)
        )
      }
    }
    case 'banked_resets': {
      const count = rowNumber(row.args, 'count')
      if (count === null) {
        return null
      }

      return { kind: 'note', text: copy.bankedResets(count) }
    }
    default:
      return null
  }
}

function AccountUsageStructuredDetails({
  copy,
  locale,
  rows
}: {
  copy: AccountUsagePanelCopy
  locale: string
  rows: AccountUsageRow[]
}) {
  const items = rows.flatMap((row, index) => {
    const detail = structuredDetailFromRow(row, copy, locale)
    return detail ? [{ detail, index, row }] : []
  })

  if (items.length === 0) {
    return null
  }

  return (
    <ul className="flex min-w-0 flex-col gap-2">
      {items.map(({ detail, index, row }) => (
        <li className="flex min-w-0 flex-col gap-1" key={`${row.key}:${index}`}>
          {detail.kind === 'metric' ? (
            <>
              <p className="font-medium text-foreground">{detail.label}</p>
              <p className="break-words text-[0.6875rem] text-muted-foreground">{detail.value}</p>
            </>
          ) : (
            <p className="break-words text-[0.6875rem] text-muted-foreground">{detail.text}</p>
          )}
        </li>
      ))}
    </ul>
  )
}

export function accountUsageSnapshotMatchesProvider(
  snapshot: AccountUsageSnapshot | null,
  provider: string
): snapshot is AccountUsageSnapshot {
  if (!snapshot) {
    return false
  }

  return snapshot.provider.trim().toLowerCase() === provider.trim().toLowerCase()
}

export function accountUsageChipLabel({
  copy,
  credits,
  providerName,
  quotaRatio,
  remaining
}: {
  copy: { accountUsage: string; accountUsageLeft: (remaining: number) => string }
  credits: string | null
  providerName: string
  quotaRatio: string | null
  remaining: number | null
}): string {
  const parenParts: string[] = []
  if (quotaRatio) {
    parenParts.push(quotaRatio)
  }
  if (remaining !== null) {
    parenParts.push(`${remaining}%`)
  }
  const parens = parenParts.length > 0 ? ` (${parenParts.join(', ')})` : ''

  if (credits !== null) {
    return `${providerName}: ${credits}${parens}`
  }

  if (quotaRatio !== null && remaining !== null) {
    return `${providerName}: ${quotaRatio} (${remaining}%)`
  }

  if (remaining !== null) {
    return `${providerName}: ${copy.accountUsageLeft(remaining)}`
  }

  if (quotaRatio !== null) {
    return `${providerName}: ${quotaRatio}`
  }

  return copy.accountUsage
}

export function useAccountUsageStatusbarItem(options: AccountUsageOptions & { usage: UsageStats }): StatusbarItem {
  const { locale, t } = useI18n()
  const copy = t.shell.statusbar
  const { error, loading, methodUnavailable, refresh, snapshot, unsupported } = useAccountUsage(options)
  // * No placeholder: a never-fetched key hides the chip. A previously
  // * fetched session is served from its own cache entry — only a matching
  // * snapshot.provider is usable.
  const matchedSnapshot = accountUsageSnapshotMatchesProvider(snapshot, options.provider) ? snapshot : null
  const remaining = matchedSnapshot ? accountUsageMinRemaining(matchedSnapshot) : null
  const tightest = matchedSnapshot ? tightestAccountUsageWindow(matchedSnapshot) : null
  const quotaRatio = tightest ? formatCompactQuotaRatio(tightest.limit_remaining, tightest.limit) : null
  const creditsValue = matchedSnapshot ? finiteCredits(matchedSnapshot.credits_balance) : null
  const credits = creditsValue === null ? null : formatCreditsBalance(creditsValue, locale)
  const providerName = accountUsageProviderLabel(options.provider)
  const hasSession = Boolean(options.sessionId)
  const hasProvider = Boolean(options.provider.trim())

  return useMemo(
    () => ({
      className: cn(
        (error || (remaining !== null && remaining <= 20)) && 'text-amber-600 hover:text-amber-600',
        remaining !== null && remaining <= 5 && 'text-destructive hover:text-destructive'
      ),
      hidden: !hasSession || !hasProvider || methodUnavailable || unsupported || !matchedSnapshot,
      icon: loading && matchedSnapshot ? <Loader2 className="size-3 animate-spin" /> : <BarChart3 className="size-3" />,
      id: 'account-usage',
      label: accountUsageChipLabel({ copy, credits, providerName, quotaRatio, remaining }),
      menuAlign: 'end',
      menuClassName: 'w-auto border-(--ui-stroke-secondary) p-0',
      menuContent: matchedSnapshot ? (
        <AccountUsagePanel
          error={error}
          loading={loading}
          onRefresh={() => void refresh()}
          provider={options.provider}
          snapshot={matchedSnapshot}
          usage={options.usage}
        />
      ) : undefined,
      title: copy.accountUsage,
      toggleLabel: copy.toggleAccountUsage,
      variant: 'menu'
    }),
    [
      copy,
      credits,
      error,
      hasProvider,
      hasSession,
      loading,
      methodUnavailable,
      options.provider,
      options.usage,
      matchedSnapshot,
      providerName,
      quotaRatio,
      refresh,
      remaining,
      unsupported
    ]
  )
}

export function AccountUsagePanel({
  error,
  loading,
  onRefresh,
  provider,
  snapshot,
  usage
}: {
  error: boolean
  loading: boolean
  onRefresh: () => void
  provider: string
  snapshot: AccountUsageSnapshot
  usage: UsageStats
}) {
  const { locale, t } = useI18n()
  const copy = t.shell.statusbar.accountUsagePanel
  const fetchedAt = Date.parse(snapshot.fetched_at)
  const providerName = accountUsageProviderLabel(provider || snapshot.provider)
  const settingsUrl = accountUsageSettingsUrl(provider || snapshot.provider)
  const showSession = usage.total > 0
  const structuredRows = structuredAccountUsageRows(snapshot)

  return (
    <div className="flex w-72 min-w-0 flex-col gap-3 p-3 text-[0.75rem]" data-slot="account-usage-panel">
      <div className="min-w-0">
        <p className="font-medium text-foreground">{copy.title}</p>
        <p className="truncate text-[0.6875rem] text-muted-foreground">
          {providerName}
          {snapshot.plan ? ` · ${copy.plan(snapshot.plan)}` : ''}
        </p>
      </div>

      {snapshot.windows.length > 0 && (
        <ul className="flex min-w-0 flex-col gap-2">
          {snapshot.windows.map(window => {
            const used = finitePercent(window.used_percent)
            const remaining = accountUsageRemaining(window)

            return (
              <li className="flex min-w-0 flex-col gap-1" key={window.label}>
                <div className="flex min-w-0 items-baseline justify-between gap-2">
                  <span className="min-w-0 truncate font-medium text-foreground">
                    {accountUsageWindowTitle(window, copy)}
                  </span>
                  <span className="shrink-0 tabular-nums text-foreground">
                    {remaining === null ? copy.unavailable : copy.remaining(remaining)}
                  </span>
                </div>

                {used !== null && (
                  <div className="h-1.5 overflow-hidden rounded-full bg-(--ui-stroke-tertiary)">
                    <span
                      className="block h-full rounded-full bg-primary transition-[width]"
                      style={{ width: `${used}%` }}
                    />
                  </div>
                )}

                <div className="flex min-w-0 items-start justify-between gap-2 text-[0.6875rem] text-muted-foreground">
                  {used !== null && <span className="shrink-0">({copy.used(Math.round(used))})</span>}
                  <span className={cn('min-w-0 break-words', used !== null && 'text-end')}>
                    {accountUsageWindowFooter(window, copy, locale)}
                  </span>
                </div>
              </li>
            )
          })}
        </ul>
      )}

      {structuredRows ? (
        <AccountUsageStructuredDetails copy={copy} locale={locale} rows={structuredRows} />
      ) : (
        snapshot.details.length > 0 && (
          <ul className="flex min-w-0 flex-col gap-1 text-[0.6875rem] text-muted-foreground">
            {snapshot.details.map(detail => (
              <li className="min-w-0 truncate" key={detail}>
                {detail}
              </li>
            ))}
          </ul>
        )
      )}

      {showSession && (
        <div className="flex min-w-0 flex-col gap-1">
          <p className="font-medium text-foreground">{copy.thisSession}</p>
          <p className="truncate tabular-nums text-[0.6875rem] text-muted-foreground">
            {copy.sessionLine(compactNumber(usage.input), compactNumber(usage.output), compactNumber(usage.total))}
          </p>
        </div>
      )}

      {error && (
        <div className="flex min-w-0 items-start gap-2 text-amber-600">
          <AlertCircle className="mt-0.5 size-3 shrink-0" />
          <span>{copy.stale}</span>
        </div>
      )}

      <div className="flex min-w-0 items-center justify-between gap-2 text-[0.6875rem] text-muted-foreground">
        <span className="min-w-0 truncate">
          {Number.isFinite(fetchedAt) ? copy.updated(relativeTime(fetchedAt)) : copy.updatedUnknown}
        </span>
        <div className="flex shrink-0 items-center gap-2">
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
          {settingsUrl ? (
            <ExternalLink href={settingsUrl} native showExternalIcon={false}>
              {copy.openUsageSettings}
            </ExternalLink>
          ) : null}
        </div>
      </div>
    </div>
  )
}
