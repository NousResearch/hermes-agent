import { useStore } from '@nanostores/react'
import { type MouseEvent, type ReactNode, useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { LogTail } from '@/components/chat/log-tail'
import { PageLoader } from '@/components/page-loader'
import { Button } from '@/components/ui/button'
import { SearchField } from '@/components/ui/search-field'
import { SegmentedControl } from '@/components/ui/segmented-control'
import { ResponsiveTabs } from '@/components/ui/tab-dropdown'
import { getActionStatus, getLogs, getStatus, getUsageAnalytics, restartGateway, updateHermes } from '@/hermes'
import type { ActionStatusResponse, AnalyticsResponse, StatusResponse } from '@/hermes'
import { useI18n } from '@/i18n'
import { sessionTitle } from '@/lib/chat-runtime'
import { compactNumber } from '@/lib/format'
import {
  Activity,
  AlertCircle,
  BarChart3,
  Bookmark,
  BookmarkFilled,
  Download,
  MessageCircle,
  Trash2,
  Wrench
} from '@/lib/icons'
import { exportSession } from '@/lib/session-export'
import { fmtDateTime } from '@/lib/time'
import { cn } from '@/lib/utils'
import { upsertDesktopActionTask } from '@/store/activity'
import { $pinnedSessionIds, pinSession, unpinSession } from '@/store/layout'
import { $sessions, sessionPinId } from '@/store/session'

import { useRefreshHotkey } from '../hooks/use-refresh-hotkey'
import { useRouteEnumParam } from '../hooks/use-route-enum-param'
import { OverlayMain, OverlayNav, OverlaySplitLayout } from '../overlays/overlay-split-layout'
import { OverlayView } from '../overlays/overlay-view'

import { MaintenancePanel } from './maintenance'

export type CommandCenterSection = 'maintenance' | 'sessions' | 'system' | 'usage'

const SECTIONS = ['sessions', 'system', 'usage', 'maintenance'] as const satisfies readonly CommandCenterSection[]

const LOG_FILES = ['agent', 'errors', 'gateway', 'desktop'] as const
const LOG_LEVELS = ['ALL', 'INFO', 'WARNING', 'ERROR'] as const

const USAGE_PERIODS = [7, 30, 90] as const
type UsagePeriod = (typeof USAGE_PERIODS)[number]

interface CommandCenterViewProps {
  initialSection?: CommandCenterSection
  onClose: () => void
  onDeleteSession: (sessionId: string) => Promise<void>
  // Accepted for call-site parity; navigation lives in the global Cmd+K palette.
  onNavigateRoute?: (path: string) => void
  onOpenSession: (sessionId: string) => void
}

function formatTimestamp(value?: number | null): string {
  if (!value) {
    return ''
  }

  const date = new Date(value * 1000)

  if (Number.isNaN(date.getTime())) {
    return ''
  }

  return fmtDateTime.format(date)
}

function useDebouncedValue<T>(value: T, delayMs: number): T {
  const [debounced, setDebounced] = useState(value)

  useEffect(() => {
    const id = window.setTimeout(() => setDebounced(value), delayMs)

    return () => window.clearTimeout(id)
  }, [delayMs, value])

  return debounced
}

function RowIconButton({
  children,
  className,
  onClick,
  title
}: {
  children: ReactNode
  className?: string
  onClick: (event: MouseEvent<HTMLButtonElement>) => void
  title: string
}) {
  return (
    <Button
      aria-label={title}
      className={cn('text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover) hover:text-foreground', className)}
      onClick={onClick}
      size="icon-xs"
      title={title}
      type="button"
      variant="ghost"
    >
      {children}
    </Button>
  )
}

function EmptyPanel({ action, description, title }: { action?: ReactNode; description: string; title?: string }) {
  return (
    <div className="grid min-h-48 place-items-center px-6 text-center">
      <div>
        {title && (
          <div className="text-[length:var(--conversation-text-font-size)] font-medium text-foreground">{title}</div>
        )}
        <div className="mt-1 text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
          {description}
        </div>
        {action && <div className="mt-3 flex justify-center">{action}</div>}
      </div>
    </div>
  )
}

export function CommandCenterView({ initialSection, onClose, onDeleteSession, onOpenSession }: CommandCenterViewProps) {
  const { t } = useI18n()
  const cc = t.commandCenter
  const sessions = useStore($sessions)
  const pinnedSessionIds = useStore($pinnedSessionIds)

  const [section, setSection] = useRouteEnumParam('section', SECTIONS, initialSection ?? 'sessions')

  const [query, setQuery] = useState('')
  const [status, setStatus] = useState<StatusResponse | null>(null)
  const [logs, setLogs] = useState<string[]>([])
  const [logFile, setLogFile] = useState<(typeof LOG_FILES)[number]>('agent')
  const [logLevel, setLogLevel] = useState<(typeof LOG_LEVELS)[number]>('ALL')
  const [logQuery, setLogQuery] = useState('')
  const [systemLoading, setSystemLoading] = useState(false)
  const [systemError, setSystemError] = useState('')
  const [systemAction, setSystemAction] = useState<ActionStatusResponse | null>(null)
  const [usagePeriod, setUsagePeriod] = useState<UsagePeriod>(30)
  const [usage, setUsage] = useState<AnalyticsResponse | null>(null)
  const [usageLoading, setUsageLoading] = useState(false)
  const [usageError, setUsageError] = useState('')
  const usageRequestRef = useRef(0)

  const debouncedQuery = useDebouncedValue(query.trim(), 180)

  const filteredSessions = useMemo(() => {
    const sorted = [...sessions].sort((a, b) => {
      const left = a.last_active || a.started_at || 0
      const right = b.last_active || b.started_at || 0

      return right - left
    })

    const needle = debouncedQuery.toLowerCase()

    if (!needle) {
      return sorted
    }

    return sorted.filter(session => {
      const haystack = `${sessionTitle(session)} ${session.id}`.toLowerCase()

      return haystack.includes(needle)
    })
  }, [debouncedQuery, sessions])

  const refreshSystem = useCallback(async () => {
    setSystemLoading(true)
    setSystemError('')

    try {
      const [nextStatus, nextLogs] = await Promise.all([
        getStatus(),
        getLogs({
          file: logFile,
          level: logLevel,
          lines: 200
        })
      ])

      setStatus(nextStatus)
      setLogs(nextLogs.lines)
    } catch (error) {
      setSystemError(error instanceof Error ? error.message : String(error))
    } finally {
      setSystemLoading(false)
    }
  }, [logFile, logLevel])

  const refreshUsage = useCallback(async (days: UsagePeriod) => {
    const requestId = usageRequestRef.current + 1
    usageRequestRef.current = requestId
    setUsageLoading(true)
    setUsageError('')

    try {
      const response = await getUsageAnalytics(days)

      if (usageRequestRef.current === requestId) {
        setUsage(response)
      }
    } catch (error) {
      if (usageRequestRef.current === requestId) {
        setUsageError(error instanceof Error ? error.message : String(error))
      }
    } finally {
      if (usageRequestRef.current === requestId) {
        setUsageLoading(false)
      }
    }
  }, [])

  useEffect(() => {
    // Refetch when the panel opens and whenever the log file/level filters
    // change (refreshSystem's identity tracks them).
    if (section === 'system') {
      void refreshSystem()
    }
  }, [refreshSystem, section])

  useEffect(() => {
    if (section === 'usage') {
      void refreshUsage(usagePeriod)
    }
  }, [refreshUsage, section, usagePeriod])

  useRefreshHotkey(() => {
    if (section === 'system') {
      void refreshSystem()
    } else if (section === 'usage') {
      void refreshUsage(usagePeriod)
    }
  })

  const sessionListHasResults = filteredSessions.length > 0

  // Client-side substring filter over the fetched tail (matches `hermes logs --search`).
  const visibleLogs = useMemo(() => {
    const needle = logQuery.trim().toLowerCase()

    if (!needle) {
      return logs
    }

    return logs.filter(line => line.toLowerCase().includes(needle))
  }, [logQuery, logs])

  const runSystemAction = useCallback(
    async (kind: 'restart' | 'update') => {
      setSystemError('')

      try {
        const started = kind === 'restart' ? await restartGateway() : await updateHermes()
        let nextStatus: ActionStatusResponse | null = null

        for (let attempt = 0; attempt < 18; attempt += 1) {
          await new Promise(resolve => window.setTimeout(resolve, 1200))
          const polled = await getActionStatus(started.name, 180)
          nextStatus = polled
          setSystemAction(polled)
          upsertDesktopActionTask(polled)

          if (!polled.running) {
            break
          }
        }

        if (!nextStatus) {
          const pendingStatus = {
            exit_code: null,
            lines: [cc.actionStartedWaiting],
            name: started.name,
            pid: started.pid,
            running: true
          }

          setSystemAction(pendingStatus)
          upsertDesktopActionTask(pendingStatus)
        }
      } catch (error) {
        setSystemError(error instanceof Error ? error.message : String(error))
      } finally {
        void refreshSystem()
      }
    },
    [cc, refreshSystem]
  )

  return (
    <OverlayView closeLabel={cc.close} onClose={onClose}>
      <OverlaySplitLayout>
        <OverlayNav
          groups={SECTIONS.map(value => ({
            active: section === value,
            icon:
              value === 'sessions'
                ? MessageCircle
                : value === 'system'
                  ? Activity
                  : value === 'maintenance'
                    ? Wrench
                    : BarChart3,
            id: value,
            label: cc.sections[value],
            onSelect: () => setSection(value)
          }))}
        />

        <OverlayMain>
          <header className="mb-4 flex items-center justify-between gap-3 max-[47.5rem]:mb-2">
            {/* Redundant on narrow — the nav dropdown already names the section. */}
            <div className="min-w-0 max-[47.5rem]:hidden">
              <h2 className="text-[length:var(--conversation-text-font-size)] font-semibold text-foreground">
                {cc.sections[section]}
              </h2>
              <p className="mt-0.5 text-[length:var(--conversation-caption-font-size)] leading-(--conversation-caption-line-height) text-(--ui-text-tertiary)">
                {cc.sectionDescriptions[section]}
              </p>
            </div>
            <div className="flex shrink-0 items-center gap-2">
              {section === 'sessions' && (
                <SearchField
                  containerClassName="max-w-[40vw]"
                  onChange={next => setQuery(next)}
                  placeholder={cc.searchPlaceholder}
                  value={query}
                />
              )}
              {section === 'usage' && (
                <SegmentedControl
                  onChange={id => setUsagePeriod(Number(id) as UsagePeriod)}
                  options={USAGE_PERIODS.map(value => ({ id: String(value), label: cc.days(value) }))}
                  value={String(usagePeriod)}
                />
              )}
            </div>
          </header>

          {section === 'sessions' ? (
            <div className="min-h-0 flex-1 overflow-y-auto">
              {!sessionListHasResults ? (
                <EmptyPanel description={debouncedQuery ? cc.noResults : cc.noSessions} />
              ) : (
                <ul>
                  {filteredSessions.map(session => {
                    const pinId = sessionPinId(session)
                    const pinned = pinnedSessionIds.includes(pinId)

                    return (
                      <li className="group flex items-center gap-2 py-2" key={session.id}>
                        <button
                          className="min-w-0 flex-1 text-left"
                          onClick={() => onOpenSession(session.id)}
                          type="button"
                        >
                          <div className="truncate text-[length:var(--conversation-text-font-size)] font-medium text-foreground">
                            {sessionTitle(session)}
                          </div>
                          <div className="truncate text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
                            {formatTimestamp(session.last_active || session.started_at)}
                          </div>
                        </button>
                        <div className="flex shrink-0 items-center gap-0.5 opacity-0 transition-opacity group-hover:opacity-100 focus-within:opacity-100">
                          <RowIconButton
                            onClick={() => void (pinned ? unpinSession(pinId) : pinSession(pinId))}
                            title={pinned ? cc.unpinSession : cc.pinSession}
                          >
                            {pinned ? <BookmarkFilled className="size-3.5" /> : <Bookmark className="size-3.5" />}
                          </RowIconButton>
                          <RowIconButton
                            onClick={() => void exportSession(session.id, { session, title: sessionTitle(session) })}
                            title={cc.exportSession}
                          >
                            <Download className="size-3.5" />
                          </RowIconButton>
                          <RowIconButton
                            className="hover:text-destructive"
                            onClick={() => void onDeleteSession(session.id)}
                            title={cc.deleteSession}
                          >
                            <Trash2 className="size-3.5" />
                          </RowIconButton>
                        </div>
                      </li>
                    )
                  })}
                </ul>
              )}
            </div>
          ) : section === 'usage' ? (
            <UsagePanel
              error={usageError}
              loading={usageLoading}
              onRefresh={() => void refreshUsage(usagePeriod)}
              period={usagePeriod}
              usage={usage}
            />
          ) : section === 'maintenance' ? (
            <MaintenancePanel />
          ) : (
            <div className="grid min-h-0 flex-1 grid-rows-[auto_minmax(0,1fr)] gap-4">
              <div>
                {status ? (
                  <div className="grid gap-2">
                    <div className="flex items-start justify-between gap-3 max-[47.5rem]:flex-col max-[47.5rem]:gap-2">
                      <div className="min-w-0">
                        <div className="flex items-center gap-2">
                          <span
                            className={cn(
                              'size-2 shrink-0 rounded-full',
                              status.gateway_running ? 'bg-emerald-500' : 'bg-amber-500'
                            )}
                          />
                          <span className="text-[length:var(--conversation-text-font-size)] font-medium text-foreground">
                            {status.gateway_running ? cc.gatewayRunning : cc.gatewayStopped}
                          </span>
                        </div>
                        <div className="mt-1 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
                          {cc.hermesActiveSessions(status.version, status.active_sessions)}
                        </div>
                      </div>
                      <div className="flex shrink-0 flex-wrap items-center gap-x-3 gap-y-1 whitespace-nowrap max-[47.5rem]:whitespace-normal">
                        <Button onClick={() => void runSystemAction('restart')} size="xs" variant="text">
                          {cc.restartGateway}
                        </Button>
                        <Button onClick={() => void runSystemAction('update')} size="xs" variant="textStrong">
                          {cc.updateHermes}
                        </Button>
                      </div>
                    </div>
                    {systemAction && (
                      <div className="text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
                        {systemAction.name} ·{' '}
                        {systemAction.running
                          ? cc.actionRunning
                          : systemAction.exit_code === 0
                            ? cc.actionDone
                            : cc.actionFailed}
                      </div>
                    )}
                  </div>
                ) : (
                  <PageLoader className="min-h-32" label={cc.loadingStatus} />
                )}
              </div>

              <div className="flex min-h-0 flex-col pt-2">
                <div className="mb-2 flex flex-wrap items-center justify-between gap-x-3 gap-y-1">
                  <span className="text-[0.625rem] font-medium uppercase tracking-[0.08em] text-(--ui-text-tertiary)">
                    {cc.recentLogs}
                  </span>
                  <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
                    <ResponsiveTabs
                      align="end"
                      onChange={id => setLogFile(id as (typeof LOG_FILES)[number])}
                      tabs={LOG_FILES.map(value => ({ id: value, label: value }))}
                      value={logFile}
                    />
                    <ResponsiveTabs
                      align="end"
                      onChange={id => setLogLevel(id as (typeof LOG_LEVELS)[number])}
                      tabs={LOG_LEVELS.map(value => ({
                        id: value,
                        label: value === 'ALL' ? 'all' : value.toLowerCase()
                      }))}
                      value={logLevel}
                    />
                    <SearchField
                      containerClassName="w-44"
                      onChange={next => setLogQuery(next)}
                      placeholder={cc.logSearchPlaceholder}
                      value={logQuery}
                    />
                  </div>
                  {systemError && (
                    <span className="inline-flex items-center gap-1 text-[length:var(--conversation-caption-font-size)] text-destructive">
                      <AlertCircle className="size-3.5" />
                      {systemError}
                    </span>
                  )}
                </div>
                <LogTail
                  className="flex-1 rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-bg-quinary)"
                  emptyLabel={cc.noLogs}
                  lines={systemLoading && logs.length === 0 ? null : visibleLogs}
                />
              </div>
            </div>
          )}
        </OverlayMain>
      </OverlaySplitLayout>
    </OverlayView>
  )
}

interface UsagePanelProps {
  error: string
  loading: boolean
  onRefresh: () => void
  period: UsagePeriod
  usage: AnalyticsResponse | null
}

function UsagePanel({ error, loading, onRefresh, period, usage }: UsagePanelProps) {
  const { t } = useI18n()
  const cc = t.commandCenter
  const daily = useMemo(() => usage?.daily ?? [], [usage])
  const totals = usage?.totals
  const byModel = usage?.by_model ?? []
  const topSkills = usage?.skills?.top_skills ?? []

  const maxTokens = useMemo(() => {
    if (!daily.length) {
      return 1
    }

    return daily.reduce((acc, entry) => Math.max(acc, (entry.input_tokens || 0) + (entry.output_tokens || 0)), 1)
  }, [daily])

  if (!totals) {
    return (
      <div className="min-h-0 flex-1">
        {loading ? (
          <PageLoader className="min-h-48" label={cc.loadingUsage} />
        ) : (
          <EmptyPanel
            action={
              <Button onClick={onRefresh} size="xs" variant="text">
                {cc.retry}
              </Button>
            }
            description={cc.noUsage(period)}
          />
        )}
      </div>
    )
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-5 overflow-y-auto pb-2">
      {error && (
        <span className="inline-flex items-center gap-1 text-[length:var(--conversation-caption-font-size)] text-destructive">
          <AlertCircle className="size-3.5" />
          {error}
        </span>
      )}

      <div className="grid grid-cols-2 gap-x-4 gap-y-4 py-2 sm:grid-cols-3">
        <UsageStat label={cc.statSessions} value={compactNumber(totals.total_sessions)} />
        <UsageStat label={cc.statApiCalls} value={compactNumber(totals.total_api_calls)} />
        <UsageStat
          label={cc.statTokens}
          value={`${compactNumber(totals.total_input)} / ${compactNumber(totals.total_output)}`}
        />
      </div>

      <section>
        <div className="mb-2 flex items-baseline justify-between">
          <span className="text-[0.625rem] font-medium uppercase tracking-[0.08em] text-(--ui-text-tertiary)">
            {cc.dailyTokens}
          </span>
          <span className="flex items-center gap-3 text-[0.65rem] text-(--ui-text-tertiary)">
            <span className="inline-flex items-center gap-1">
              <span className="size-2 rounded-[1px] bg-[color:var(--dt-primary)]/60" /> {cc.input}
            </span>
            <span className="inline-flex items-center gap-1">
              <span className="size-2 rounded-[1px] bg-emerald-500/70" /> {cc.output}
            </span>
          </span>
        </div>
        {daily.length === 0 ? (
          <div className="grid h-24 place-items-center text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
            {cc.noDailyActivity}
          </div>
        ) : (
          <>
            <div className="flex h-24 items-end gap-px">
              {daily.map(entry => {
                const inputH = Math.round(((entry.input_tokens || 0) / maxTokens) * 96)
                const outputH = Math.round(((entry.output_tokens || 0) / maxTokens) * 96)

                return (
                  <div
                    className="group relative flex h-24 min-w-0 flex-1 flex-col justify-end"
                    key={entry.day}
                    title={`${entry.day} · in ${compactNumber(entry.input_tokens)} · out ${compactNumber(entry.output_tokens)}`}
                  >
                    <div
                      className="w-full rounded-t-[1px] bg-[color:var(--dt-primary)]/50"
                      style={{ height: Math.max(inputH, entry.input_tokens > 0 ? 1 : 0) }}
                    />
                    <div
                      className="w-full bg-emerald-500/60"
                      style={{ height: Math.max(outputH, entry.output_tokens > 0 ? 1 : 0) }}
                    />
                  </div>
                )
              })}
            </div>
            <div className="mt-1 flex justify-between text-[0.6rem] text-(--ui-text-tertiary)">
              <span>{daily[0]?.day}</span>
              <span>{daily[daily.length - 1]?.day}</span>
            </div>
          </>
        )}
      </section>

      <div className="grid min-h-0 gap-x-8 gap-y-5 pt-1 sm:grid-cols-2">
        <UsageList
          emptyLabel={cc.noModelUsage}
          rows={byModel.slice(0, 6).map(entry => ({
            key: entry.model,
            label: entry.model,
            value: `${compactNumber((entry.input_tokens || 0) + (entry.output_tokens || 0))}`
          }))}
          title={cc.topModels}
        />
        <UsageList
          emptyLabel={cc.noSkillActivity}
          rows={topSkills.slice(0, 6).map(entry => ({
            key: entry.skill,
            label: entry.skill,
            value: cc.actions(compactNumber(entry.total_count))
          }))}
          title={cc.topSkills}
        />
      </div>
    </div>
  )
}

function UsageList({
  emptyLabel,
  rows,
  title
}: {
  emptyLabel: string
  rows: Array<{ key: string; label: string; value: string }>
  title: string
}) {
  return (
    <section className="min-w-0">
      <div className="mb-1.5 text-[0.625rem] font-medium uppercase tracking-[0.08em] text-(--ui-text-tertiary)">
        {title}
      </div>
      {rows.length === 0 ? (
        <div className="text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
          {emptyLabel}
        </div>
      ) : (
        <ul>
          {rows.map(row => (
            <li className="flex items-center justify-between gap-2 py-1.5" key={row.key}>
              <span className="min-w-0 truncate font-mono text-[0.7rem] text-foreground">{row.label}</span>
              <span className="shrink-0 text-[0.65rem] text-(--ui-text-tertiary)">{row.value}</span>
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}

function UsageStat({ hint, label, value }: { hint?: string; label: string; value: string }) {
  return (
    <div className="min-w-0">
      <div className="text-[0.625rem] font-medium uppercase tracking-[0.12em] text-(--ui-text-tertiary)">{label}</div>
      <div className="mt-1 truncate text-base font-semibold tracking-tight text-foreground">{value}</div>
      {hint && <div className="mt-0.5 truncate text-[0.62rem] text-(--ui-text-tertiary)">{hint}</div>}
    </div>
  )
}
