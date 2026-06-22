import { useQuery } from '@tanstack/react-query'
import type { FC } from 'react'

import { getCodexUsage } from '@/hermes'
import { cn } from '@/lib/utils'
import type { CodexUsageResponse } from '@/types/hermes'

interface CodexQuotaCardProps {
  /** Only fetch when the popover is open (lazy). */
  enabled: boolean
  /** Active Hermes profile; part of the cache key because Codex auth is per-profile. */
  profile: string
  /** Current model provider; only OpenAI Codex exposes this quota surface. */
  provider: string
}

const CODEX_PROVIDER = 'openai-codex'

/** Hover card body showing live OpenAI Codex rate-limit / quota. */
export const CodexQuotaCard: FC<CodexQuotaCardProps> = ({ enabled, profile, provider }) => {
  const normalizedProfile = normalizeProfile(profile)
  const normalizedProvider = normalizeProvider(provider)
  const canFetch = enabled && normalizedProvider === CODEX_PROVIDER

  const usage = useQuery<CodexUsageResponse>({
    queryKey: ['codex-usage', normalizedProfile, normalizedProvider],
    queryFn: getCodexUsage,
    enabled: canFetch,
    staleTime: 60_000, // cache 1 min — quota doesn't move that fast
    refetchOnWindowFocus: false
  })

  if (!enabled || normalizedProvider !== CODEX_PROVIDER) {
    return null
  }

  if (usage.isPending) {
    return (
      <div className="flex items-center gap-2 px-3 py-2 text-xs text-(--ui-text-tertiary)">
        <span className="inline-block size-3 animate-spin rounded-full border-2 border-(--ui-stroke-tertiary) border-t-(--ui-text-secondary)" />
        Loading quota…
      </div>
    )
  }

  const data = usage.data

  if (!data?.available) {
    return <div className="px-3 py-2 text-xs text-(--ui-text-tertiary)">{data?.error || 'Codex quota unavailable'}</div>
  }

  const windows = data.windows ?? []
  const plan = data.plan ? ` · ${data.plan}` : ''

  return (
    <div className="min-w-52 max-w-64 select-none px-3 py-2.5 text-xs">
      {/* Header */}
      <div className="mb-2 flex items-center gap-1.5 text-(--ui-text-secondary)">
        <span className="font-medium">OpenAI Codex{plan}</span>
      </div>

      {/* Rate-limit windows */}
      {windows.length === 0 ? (
        <div className="text-(--ui-text-tertiary)">No rate-limit data</div>
      ) : (
        <div className="space-y-2">
          {windows.map(w => {
            const used = clampPercent(w.used_percent ?? 0)
            // Codex IDE shows remaining quota, while the backend reports used_percent.
            const remaining = clampPercent(w.remaining_percent ?? 100 - used)

            const colorClass = remaining <= 15 ? 'bg-red-500' : remaining <= 40 ? 'bg-amber-400' : 'bg-emerald-500'

            return (
              <div key={w.label}>
                <div className="mb-0.5 flex items-center justify-between gap-3">
                  <span className="text-(--ui-text-tertiary)">{w.label}</span>
                  <span className="tabular-nums text-(--ui-text-secondary)">{remaining}% left</span>
                </div>
                {/* Remaining-quota bar, matching Codex IDE semantics. */}
                <div className="h-1.5 w-full overflow-hidden rounded-full bg-(--ui-control-background)">
                  <div
                    className={cn('h-full rounded-full transition-all', colorClass)}
                    style={{ width: `${remaining}%` }}
                  />
                </div>
                <div className="mt-0.5 text-[0.65rem] leading-none text-(--ui-text-tertiary)">
                  {used}% used{w.reset_at ? ` · resets ${formatReset(new Date(w.reset_at))}` : ''}
                </div>
              </div>
            )
          })}
        </div>
      )}

      {/* Details (credits, reset credits, etc.) */}
      {(data.details?.length ?? 0) > 0 && (
        <div className="mt-2 space-y-0.5 border-t border-(--ui-stroke-tertiary) pt-2">
          {data.details!.map((d, i) => (
            <div className="text-[0.65rem] leading-snug text-(--ui-text-tertiary)" key={i}>
              {d}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function normalizeProfile(value: string): string {
  const trimmed = String(value || '').trim()

  return trimmed || 'default'
}

function normalizeProvider(value: string): string {
  return String(value || '')
    .trim()
    .toLowerCase()
}

function clampPercent(value: number): number {
  if (!Number.isFinite(value)) {
    return 0
  }

  return Math.max(0, Math.min(100, Math.round(value)))
}

/** Human-friendly relative + absolute local time for quota reset. */
function formatReset(date: Date): string {
  const absolute = formatAbsoluteDateTime(date)
  const now = Date.now()
  const diff = date.getTime() - now

  if (diff <= 0) {
    return `now · ${absolute}`
  }

  const mins = Math.floor(diff / 60_000)
  const hours = Math.floor(mins / 60)
  const days = Math.floor(hours / 24)

  let relative: string

  if (days > 0) {
    relative = `in ${days}d ${hours % 24}h`
  } else if (hours > 0) {
    relative = `in ${hours}h ${mins % 60}m`
  } else {
    relative = `in ${mins}m`
  }

  return `${relative} · ${absolute}`
}

function formatAbsoluteDateTime(date: Date): string {
  const pad = (n: number) => String(n).padStart(2, '0')
  const yyyy = date.getFullYear()
  const mm = pad(date.getMonth() + 1)
  const dd = pad(date.getDate())
  const hh = pad(date.getHours())
  const min = pad(date.getMinutes())
  const tz = formatShortTimeZone(date)

  return `${yyyy}-${mm}-${dd} ${hh}:${min}${tz ? ` ${tz}` : ''}`
}

function formatShortTimeZone(date: Date): string {
  try {
    const tz = new Intl.DateTimeFormat(undefined, { timeZoneName: 'short' })
      .formatToParts(date)
      .find(part => part.type === 'timeZoneName')?.value

    return tz ?? ''
  } catch {
    return ''
  }
}
