import type { AccountUsagePanelCopy } from '@/i18n/types_account_usage'
import type { AccountUsageRow, AccountUsageSnapshot, AccountUsageWindow } from '@/types/account-usage'

interface AccountUsageDetail {
  label: string
  value: string
}

const PROVIDER_NAMES: Record<AccountUsageSnapshot['provider'], string> = {
  anthropic: 'Anthropic',
  'openai-codex': 'Codex',
  openrouter: 'OpenRouter'
}

export function accountUsageProviderLabel(provider: string): string {
  const slug = provider.trim().toLowerCase()

  return Object.hasOwn(PROVIDER_NAMES, slug) ? PROVIDER_NAMES[slug as keyof typeof PROVIDER_NAMES] : slug
}

function number(value: null | number | undefined): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

export function accountUsageRemaining(window: AccountUsageWindow): number | null {
  const remaining = number(window.remaining_percent)
  const used = number(window.used_percent)
  const value = remaining ?? (used === null ? null : 100 - used)

  return value === null ? null : Math.max(0, Math.min(100, value))
}

/** Model-family limits never imply that the whole provider account is depleted. */
export function accountUsageMinRemaining(snapshot: AccountUsageSnapshot): number | null {
  const remaining = snapshot.windows
    .filter(window => window.scope === 'account')
    .flatMap(window => {
      const value = accountUsageRemaining(window)

      return value === null ? [] : [value]
    })

  return remaining.length ? Math.min(...remaining) : null
}

/** A positive remainder never reads as zero, and a partial allowance never reads as full. */
export function accountUsagePercent(value: number, locale: string): string {
  const percent = Math.max(0, Math.min(100, value))

  return percent > 0 && percent < 1 ? '<1' : new Intl.NumberFormat(locale).format(Math.floor(percent))
}

export function formatCreditsBalance(value: number, locale: string, currency = 'USD'): string {
  try {
    return new Intl.NumberFormat(locale, {
      currency,
      maximumFractionDigits: 2,
      minimumFractionDigits: 2,
      style: 'currency'
    }).format(value)
  } catch {
    return `${value.toFixed(2)} ${currency}`
  }
}

export function accountUsageTimestamp(value: null | string | undefined, locale: string): string | null {
  const timestamp = value ? Date.parse(value) : Number.NaN

  return Number.isFinite(timestamp)
    ? new Intl.DateTimeFormat(locale, { dateStyle: 'medium', timeStyle: 'short' }).format(timestamp)
    : null
}

export function accountUsageWindowFooter(
  window: AccountUsageWindow,
  copy: AccountUsagePanelCopy,
  locale: string
): string {
  const parts: string[] = []
  const remaining = number(window.limit_remaining)
  const limit = number(window.limit)

  if (remaining !== null && limit !== null) {
    parts.push(copy.ofLimitRemaining(formatCreditsBalance(remaining, locale), formatCreditsBalance(limit, locale)))
  }

  if (window.reset_interval) {
    parts.push(copy.resetIntervals[window.reset_interval])
  }

  const reset = accountUsageTimestamp(window.reset_at, locale)

  if (reset) {
    parts.push(copy.resets(reset))
  }

  return parts.join(' · ')
}

type DetailFormatter = (
  args: AccountUsageRow['args'],
  copy: AccountUsagePanelCopy,
  locale: string
) => AccountUsageDetail | null

const DETAIL_FORMATTERS: Record<AccountUsageRow['key'], DetailFormatter> = {
  api_key_usage: (args, copy, locale) => {
    const total = number(args.total)

    if (total === null) {
      return null
    }

    const parts = [copy.usageTotal(formatCreditsBalance(total, locale))]

    const periods = [
      ['daily', copy.usageToday],
      ['weekly', copy.usageThisWeek],
      ['monthly', copy.usageThisMonth]
    ] as const

    for (const [key, format] of periods) {
      const value = number(args[key])

      if (value !== null) {
        parts.push(format(formatCreditsBalance(value, locale)))
      }
    }

    return { label: copy.apiKeyUsage, value: parts.join(' · ') }
  },
  banked_resets: (args, copy) => {
    const count = number(args.count)

    return count === null ? null : { label: '', value: copy.bankedResets(count) }
  },
  credits_balance: (args, copy, locale) => {
    const value = number(args.value)

    return value === null
      ? null
      : { label: copy.creditsBalance, value: formatCreditsBalance(value, locale, args.currency ?? 'USD') }
  },
  credits_unlimited: (_args, copy) => ({ label: copy.creditsBalance, value: copy.creditsUnlimited }),
  extra_usage: (args, copy, locale) => {
    const used = number(args.used)
    const limit = number(args.limit)

    return used === null || limit === null
      ? null
      : {
          label: copy.extraUsage,
          value: copy.extraUsageValue(
            formatCreditsBalance(used, locale, args.currency ?? 'USD'),
            formatCreditsBalance(limit, locale, args.currency ?? 'USD')
          )
        }
  }
}

export function accountUsageDetails(
  rows: AccountUsageRow[],
  copy: AccountUsagePanelCopy,
  locale: string
): AccountUsageDetail[] {
  return rows.flatMap(row => {
    const detail = Object.hasOwn(DETAIL_FORMATTERS, row.key) ? DETAIL_FORMATTERS[row.key](row.args, copy, locale) : null

    return detail ? [detail] : []
  })
}
