import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useMemo, useState } from 'react'

import { PageLoader } from '@/components/page-loader'
import { Button } from '@/components/ui/button'
import { getUsageAnalytics } from '@/hermes'
import { type Locale, useI18n } from '@/i18n'
import { AlertTriangle, Package } from '@/lib/icons'
import { $authState } from '@/store/auth'
import type { AnalyticsResponse } from '@/types/hermes'

import { useRefreshHotkey } from '../hooks/use-refresh-hotkey'
import { OverlayView } from '../overlays/overlay-view'
import { EmptyState } from '../settings/primitives'

// ─────────────────────────────────────────────────────────────────────────────
// 个人资料 — the profile stats page (Claude Code Desktop profile layout).
//
// Every number on this page is REAL local data; nothing is hardcoded:
//   • Header (avatar / name / @handle / plan): the managed-account store
//     ($authState — email/name/plan from the desktop managed bridge). When not
//     signed in (or on a managed-disabled/BYOK build with no account) it shows
//     the 未登录 placeholder and hides handle + plan.
//   • Stat cards, heatmap, insights, top plugins: GET /api/analytics/usage
//     (getUsageAnalytics), served by the local backend off the profile's
//     sessions DB (per-day token sums, per-model sums, skill-usage counters).
//     One fetch per page open (plus the `r` refresh hotkey) — no polling.
//
// Availability gating (数据可得性铁律): each block renders only when its slice
// of the analytics payload exists — no daily rows → no heatmap; no skill
// counters → the whole 最常用的插件 column disappears; zero sessions → a single
// empty state instead of a wall of zeros.
// ─────────────────────────────────────────────────────────────────────────────

// The heatmap spans the trailing 52 weeks; fetch a hair more than a year so
// the oldest visible cells still have data behind them.
const ANALYTICS_DAYS = 370
const HEATMAP_WEEKS = 52
const DAY_MS = 86_400_000

// The backend keys days as UTC dates (sqlite date(…,'unixepoch')), so all grid
// math sticks to UTC day indices to line up with those keys.
function utcDayIndex(isoDay: string): number {
  return Math.floor(Date.parse(`${isoDay}T00:00:00Z`) / DAY_MS)
}

function dateOfDayIndex(index: number): Date {
  return new Date(index * DAY_MS)
}

// BCP-47 tags for Intl date/month/weekday labels per app locale.
const INTL_TAGS: Record<Locale, string> = { en: 'en-US', ja: 'ja-JP', zh: 'zh-CN', 'zh-hant': 'zh-TW' }

// Same display semantics as the command center's usage panel.
function formatTokens(value: null | number | undefined): string {
  const num = Number(value || 0)

  if (num >= 1_000_000) {
    return `${(num / 1_000_000).toFixed(1)}M`
  }

  if (num >= 1_000) {
    return `${(num / 1_000).toFixed(1)}K`
  }

  return num.toLocaleString()
}

function formatCost(value: null | number | undefined): string {
  const num = Number(value || 0)

  if (num < 0.01) {
    return '<$0.01'
  }

  return `$${num.toFixed(2)}`
}

function formatInteger(value: null | number | undefined): string {
  return Number(value ?? 0).toLocaleString()
}

function initialOf(name: string): string {
  const match = name.replace(/[^\p{L}\p{N}]/gu, '').charAt(0)

  return (match || '?').toUpperCase()
}

interface ProfileStatsViewProps {
  onClose: () => void
}

export function ProfileStatsView({ onClose }: ProfileStatsViewProps) {
  const { locale, t } = useI18n()
  const p = t.profileStats
  const { account, status } = useStore($authState)

  const [usage, setUsage] = useState<AnalyticsResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<null | string>(null)

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)

    try {
      setUsage(await getUsageAnalytics(ANALYTICS_DAYS))
    } catch (err) {
      setError(err instanceof Error ? err.message : p.failedLoad)
    } finally {
      setLoading(false)
    }
  }, [p])

  useEffect(() => {
    void load()
  }, [load])

  useRefreshHotkey(() => void load())

  // Header identity — real managed-account data or an explicit 未登录
  // placeholder; never a made-up name.
  const signedIn = status === 'signed-in' && Boolean(account.email.trim() || account.name.trim())
  const email = account.email.trim()
  const name = signedIn ? account.name.trim() || email.split('@')[0] || email : p.signedOut
  const handle = signedIn && email ? email.split('@')[0] : ''
  const plan = signedIn ? account.plan.trim() : ''

  return (
    <OverlayView closeLabel={p.close} onClose={onClose}>
      <section className="p5-settings flex h-full min-h-0 flex-col overflow-hidden">
        <div className="min-h-0 flex-1 overflow-y-auto px-6 pb-20 pt-[calc(var(--titlebar-height)+1.25rem)]">
          <div className="p5-profile-page">
            <header className="p5-profile-header">
              <span aria-hidden className="p5-profile-avatar">
                {signedIn ? initialOf(name) : '?'}
              </span>
              <h2 className="p5-profile-name">{name}</h2>
              {(handle || plan) && (
                <div className="p5-profile-meta">
                  {handle && <span className="truncate">@{handle}</span>}
                  {handle && plan && <span aria-hidden>·</span>}
                  {plan && <span className="p5-profile-plan">{plan}</span>}
                </div>
              )}
            </header>

            {loading ? (
              <PageLoader className="min-h-64" label={p.loading} />
            ) : error ? (
              <div className="p5-profile-section flex flex-col items-center gap-3 text-center">
                <div className="flex items-center gap-2 text-[length:var(--conversation-caption-font-size)] text-destructive">
                  <AlertTriangle className="size-3.5 shrink-0" />
                  <span>{error}</span>
                </div>
                <Button onClick={() => void load()} size="sm" variant="outline">
                  {t.common.retry}
                </Button>
              </div>
            ) : usage ? (
              <UsageBody locale={locale} t={t} usage={usage} />
            ) : null}
          </div>
        </div>
      </section>
    </OverlayView>
  )
}

function UsageBody({ locale, t, usage }: { locale: Locale; t: ReturnType<typeof useI18n>['t']; usage: AnalyticsResponse }) {
  const p = t.profileStats
  const intlTag = INTL_TAGS[locale]
  const totals = usage.totals
  const daily = usage.daily
  const topSkills = usage.skills?.top_skills ?? []

  const dateFormat = useMemo(() => new Intl.DateTimeFormat(intlTag, { day: 'numeric', month: 'short', timeZone: 'UTC' }), [intlTag])

  // Per-day token totals (input + output; cache reads and reasoning tokens are
  // deliberately excluded so the heatmap tracks what the user actually spent).
  const tokensByDay = useMemo(() => {
    const map = new Map<number, number>()

    for (const entry of daily) {
      const index = utcDayIndex(entry.day)

      if (Number.isFinite(index)) {
        map.set(index, (map.get(index) ?? 0) + (entry.input_tokens || 0) + (entry.output_tokens || 0))
      }
    }

    return map
  }, [daily])

  const activeDays = useMemo(() => [...tokensByDay.values()].filter(v => v > 0).length, [tokensByDay])

  const totalTokens = totals.total_input == null && totals.total_output == null
    ? null
    : (totals.total_input || 0) + (totals.total_output || 0)

  // Nothing recorded at all → one honest empty state, not a wall of zeros.
  if (!totals.total_sessions) {
    return (
      <div className="p5-profile-section">
        <EmptyState description={p.emptyDesc} title={p.emptyTitle} />
      </div>
    )
  }

  // Stat cards — each card exists only when its metric is genuinely present in
  // the analytics payload (SQL SUM() returns null when there are no rows).
  const cards: { label: string; value: string }[] = [
    { label: p.stats.sessions, value: formatInteger(totals.total_sessions) }
  ]

  if (totalTokens != null) {
    cards.push({ label: p.stats.tokens, value: formatTokens(totalTokens) })
  }

  if (totals.total_api_calls != null) {
    cards.push({ label: p.stats.apiCalls, value: formatInteger(totals.total_api_calls) })
  }

  if (activeDays > 0) {
    cards.push({ label: p.stats.activeDays, value: formatInteger(activeDays) })
  }

  if (usage.skills?.summary && usage.skills.summary.distinct_skills_used > 0) {
    cards.push({ label: p.stats.skillsUsed, value: formatInteger(usage.skills.summary.distinct_skills_used) })
  }

  // 活动洞察 — every row is derived from the same analytics payload; rows whose
  // inputs are missing simply don't appear.
  const insights: { label: string; value: string }[] = []

  if (tokensByDay.size > 0) {
    let busiestIndex = -1
    let busiestTokens = 0

    for (const [index, value] of tokensByDay) {
      if (value > busiestTokens) {
        busiestTokens = value
        busiestIndex = index
      }
    }

    if (busiestIndex >= 0 && busiestTokens > 0) {
      insights.push({
        label: p.insights.busiestDay,
        value: `${dateFormat.format(dateOfDayIndex(busiestIndex))} · ${formatTokens(busiestTokens)}`
      })
    }

    if (activeDays > 0 && totalTokens != null && totalTokens > 0) {
      insights.push({ label: p.insights.avgPerActiveDay, value: formatTokens(Math.round(totalTokens / activeDays)) })
    }

    const longestStreak = computeLongestStreak(tokensByDay)

    if (longestStreak > 1) {
      insights.push({ label: p.insights.longestStreak, value: p.insights.streakDays(longestStreak) })
    }
  }

  const topModel = usage.by_model[0]?.model

  if (topModel) {
    insights.push({ label: p.insights.topModel, value: topModel })
  }

  if (totals.total_estimated_cost > 0) {
    insights.push({ label: p.insights.estimatedCost, value: formatCost(totals.total_estimated_cost) })
  }

  const hasHeatmap = tokensByDay.size > 0
  const hasInsights = insights.length > 0
  const hasSkills = topSkills.length > 0

  return (
    <>
      <div className="p5-profile-stat-grid">
        {cards.slice(0, 5).map(card => (
          <div className="p5-profile-stat" key={card.label}>
            <div className="p5-profile-stat-value">{card.value}</div>
            <div className="p5-profile-stat-label">{card.label}</div>
          </div>
        ))}
      </div>

      {/* Token 活动 — GitHub-style week × day heatmap off the per-day rows.
          Hidden entirely when the backend returned no per-day activity. */}
      {hasHeatmap && <TokenHeatmap intlTag={intlTag} p={p} tokensByDay={tokensByDay} />}

      {(hasInsights || hasSkills) && (
        <div className="p5-profile-columns" data-two={hasInsights && hasSkills ? 'true' : undefined}>
          {hasInsights && (
            <section className="p5-profile-card">
              <div className="p5-profile-card-title">
                <span>{p.insights.title}</span>
              </div>
              <div className="p5-profile-rows">
                {insights.map(row => (
                  <div className="p5-profile-kv" key={row.label}>
                    <span className="p5-profile-kv-label">{row.label}</span>
                    <span className="p5-profile-kv-value">{row.value}</span>
                  </div>
                ))}
              </div>
            </section>
          )}

          {/* 最常用的插件 — the backend's skill-usage counters. When no skill
              activity was ever recorded the whole column is omitted. */}
          {hasSkills && (
            <section className="p5-profile-card">
              <div className="p5-profile-card-title">
                <span>{p.topSkills.title}</span>
              </div>
              <div className="p5-profile-rows">
                {topSkills.slice(0, 6).map(entry => (
                  <div className="p5-profile-skill" key={entry.skill}>
                    <span aria-hidden className="p5-profile-skill-icon">
                      <Package className="size-3.5" />
                    </span>
                    <span className="min-w-0 flex-1 truncate">{entry.skill}</span>
                    <span className="p5-profile-kv-label shrink-0">{p.topSkills.uses(formatInteger(entry.total_count))}</span>
                  </div>
                ))}
              </div>
            </section>
          )}
        </div>
      )}
    </>
  )
}

// Longest run of consecutive UTC days with any token activity.
function computeLongestStreak(tokensByDay: Map<number, number>): number {
  const days = [...tokensByDay.entries()]
    .filter(([, value]) => value > 0)
    .map(([index]) => index)
    .sort((a, b) => a - b)

  let longest = 0
  let run = 0
  let previous = Number.NaN

  for (const day of days) {
    run = day === previous + 1 ? run + 1 : 1
    previous = day
    longest = Math.max(longest, run)
  }

  return longest
}

type HeatmapMode = 'cumulative' | 'daily' | 'weekly'

interface HeatmapCell {
  dayIndex: number
  future: boolean
  value: number
}

function TokenHeatmap({
  intlTag,
  p,
  tokensByDay
}: {
  intlTag: string
  p: ReturnType<typeof useI18n>['t']['profileStats']
  tokensByDay: Map<number, number>
}) {
  const [mode, setMode] = useState<HeatmapMode>('daily')

  const monthFormat = useMemo(() => new Intl.DateTimeFormat(intlTag, { month: 'short', timeZone: 'UTC' }), [intlTag])

  const dayFormat = useMemo(
    () => new Intl.DateTimeFormat(intlTag, { day: 'numeric', month: 'short', timeZone: 'UTC', year: 'numeric' }),
    [intlTag]
  )

  const weekdayFormat = useMemo(() => new Intl.DateTimeFormat(intlTag, { timeZone: 'UTC', weekday: 'narrow' }), [intlTag])

  // Sunday-started columns covering the trailing HEATMAP_WEEKS, current week
  // last (GitHub layout). Cells run column-major, which is also chronological
  // order — handy for the cumulative mode.
  const { cells, max, monthLabels } = useMemo(() => {
    const todayIndex = Math.floor(Date.now() / DAY_MS)
    const todayDow = dateOfDayIndex(todayIndex).getUTCDay()
    const gridStart = todayIndex - todayDow - (HEATMAP_WEEKS - 1) * 7

    const cells: HeatmapCell[] = []
    let running = 0
    let max = 0

    for (let week = 0; week < HEATMAP_WEEKS; week += 1) {
      let weekTotal = 0

      for (let row = 0; row < 7; row += 1) {
        const dayIndex = gridStart + week * 7 + row

        if (dayIndex <= todayIndex) {
          weekTotal += tokensByDay.get(dayIndex) ?? 0
        }
      }

      for (let row = 0; row < 7; row += 1) {
        const dayIndex = gridStart + week * 7 + row
        const future = dayIndex > todayIndex
        const dayTokens = future ? 0 : tokensByDay.get(dayIndex) ?? 0
        running += dayTokens

        const value = mode === 'daily' ? dayTokens : mode === 'weekly' ? weekTotal : running

        if (!future) {
          max = Math.max(max, value)
        }

        cells.push({ dayIndex, future, value })
      }
    }

    // One label per month change, dropped when the previous label sits fewer
    // than three columns away (keeps the axis from colliding at the seams).
    const monthLabels: (null | string)[] = []
    let lastLabelAt = -3

    for (let week = 0; week < HEATMAP_WEEKS; week += 1) {
      const first = dateOfDayIndex(gridStart + week * 7)
      const previous = week > 0 ? dateOfDayIndex(gridStart + (week - 1) * 7) : null
      const changed = !previous || first.getUTCMonth() !== previous.getUTCMonth()

      if (changed && week - lastLabelAt >= 3) {
        monthLabels.push(monthFormat.format(first))
        lastLabelAt = week
      } else {
        monthLabels.push(null)
      }
    }

    return { cells, max, monthLabels }
  }, [mode, monthFormat, tokensByDay])

  const level = (cell: HeatmapCell): number => {
    if (cell.future || cell.value <= 0 || max <= 0) {
      return 0
    }

    return Math.min(4, Math.max(1, Math.ceil((cell.value / max) * 4)))
  }

  const modes: { id: HeatmapMode; label: string }[] = [
    { id: 'daily', label: p.heatmap.daily },
    { id: 'weekly', label: p.heatmap.weekly },
    { id: 'cumulative', label: p.heatmap.cumulative }
  ]

  return (
    <section className="p5-profile-section p5-profile-card">
      <div className="p5-profile-card-title">
        <span>{p.heatmap.title}</span>
        <div className="p5-profile-toggle" role="tablist">
          {modes.map(entry => (
            <button
              aria-selected={mode === entry.id}
              data-active={mode === entry.id || undefined}
              key={entry.id}
              onClick={() => setMode(entry.id)}
              role="tab"
              type="button"
            >
              {entry.label}
            </button>
          ))}
        </div>
      </div>

      <div className="p5-profile-heatmap-scroll">
        <div className="p5-profile-heatmap">
          <div aria-hidden className="p5-profile-weekdays">
            {Array.from({ length: 7 }, (_, row) => (
              <span key={row}>
                {row % 2 === 1 ? weekdayFormat.format(new Date(Date.UTC(2024, 0, 7 + row))) : ''}
              </span>
            ))}
          </div>

          <div>
            <div className="p5-profile-heatmap-grid">
              {cells.map(cell => (
                <span
                  className="p5-profile-heatmap-cell"
                  data-future={cell.future || undefined}
                  data-level={level(cell)}
                  key={cell.dayIndex}
                  title={
                    cell.future
                      ? undefined
                      : p.heatmap.cellTitle(dayFormat.format(dateOfDayIndex(cell.dayIndex)), formatTokens(cell.value))
                  }
                />
              ))}
            </div>
            <div aria-hidden className="p5-profile-heatmap-months">
              {monthLabels.map((label, index) => (
                <span key={index}>{label ?? ''}</span>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div aria-hidden className="p5-profile-legend">
        <span>{p.heatmap.less}</span>
        {[0, 1, 2, 3, 4].map(value => (
          <span className="p5-profile-heatmap-cell" data-level={value} key={value} />
        ))}
        <span>{p.heatmap.more}</span>
      </div>
    </section>
  )
}
