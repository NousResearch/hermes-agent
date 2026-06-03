import { useStore } from '@nanostores/react'
import {
  IconBookmark,
  IconBookmarkFilled,
  IconDownload,
  IconRefresh,
  IconTrash
} from '@tabler/icons-react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import {
  getActionStatus,
  getLogs,
  getStatus,
  getUsageAnalytics,
  restartGateway,
  searchSessions,
  updateHermes
} from '@/hermes'
import type {
  ActionStatusResponse,
  AnalyticsResponse,
  SessionInfo,
  SessionSearchResult as SessionSearchApiResult,
  StatusResponse
} from '@/hermes'
import { sessionTitle } from '@/lib/chat-runtime'
import { Activity, AlertCircle, BarChart3, Pin } from '@/lib/icons'
import { exportSession } from '@/lib/session-export'
import { cn } from '@/lib/utils'
import { upsertDesktopActionTask } from '@/store/activity'
import { $pinnedSessionIds, pinSession, unpinSession } from '@/store/layout'
import { $sessions, sessionPinId } from '@/store/session'

import { useRouteEnumParam } from '../hooks/use-route-enum-param'
import { OverlayActionButton, OverlayCard, overlayCardClass, OverlayIconButton } from '../overlays/overlay-chrome'
import { OverlaySearchInput } from '../overlays/overlay-search-input'
import { OverlayMain, OverlayNavItem, OverlaySidebar, OverlaySplitLayout } from '../overlays/overlay-split-layout'
import { OverlayView } from '../overlays/overlay-view'
import { ARTIFACTS_ROUTE, MESSAGING_ROUTE, NEW_CHAT_ROUTE, SETTINGS_ROUTE, SKILLS_ROUTE } from '../routes'

export type CommandCenterSection = 'sessions' | 'system' | 'usage'

const SECTIONS = ['sessions', 'system', 'usage'] as const satisfies readonly CommandCenterSection[]

const USAGE_PERIODS = [7, 30, 90] as const
type UsagePeriod = (typeof USAGE_PERIODS)[number]

interface CommandCenterViewProps {
  initialSection?: CommandCenterSection
  onClose: () => void
  onDeleteSession: (sessionId: string) => Promise<void>
  onNavigateRoute: (path: string) => void
  onOpenSession: (sessionId: string) => void
}

const SECTION_LABELS: Record<CommandCenterSection, string> = {
  sessions: 'Sessions',
  system: 'System',
  usage: 'Usage'
}

const SECTION_DESCRIPTIONS: Record<CommandCenterSection, string> = {
  sessions: 'Search and manage sessions',
  system: 'Status, logs, and system actions',
  usage: 'Token, cost, and skill activity over time'
}

interface NavigationSearchEntry {
  detail?: string
  id: string
  route: string
  title: string
}

interface SectionSearchEntry {
  detail?: string
  id: string
  section: CommandCenterSection
  title: string
}

const NAVIGATION_SEARCH_ENTRIES: readonly NavigationSearchEntry[] = [
  { id: 'nav-new-chat', route: NEW_CHAT_ROUTE, title: 'New session', detail: 'Start a fresh session' },
  { id: 'nav-settings', route: SETTINGS_ROUTE, title: 'Settings', detail: 'Configure Hermes desktop' },
  { id: 'nav-skills', route: SKILLS_ROUTE, title: 'Skills & Tools', detail: 'Enable skills, toolsets, and providers' },
  {
    id: 'nav-messaging',
    route: MESSAGING_ROUTE,
    title: 'Messaging',
    detail: 'Set up Telegram, Slack, Discord, and more'
  },
  { id: 'nav-artifacts', route: ARTIFACTS_ROUTE, title: 'Artifacts', detail: 'Browse generated outputs' }
]

const SECTION_SEARCH_ENTRIES: readonly SectionSearchEntry[] = [
  { id: 'section-sessions', section: 'sessions', title: 'Sessions panel', detail: 'Search, pin, and manage sessions' },
  { id: 'section-system', section: 'system', title: 'System panel', detail: 'Gateway status, logs, restart/update' },
  { id: 'section-usage', section: 'usage', title: 'Usage panel', detail: 'Token, cost, and skill activity' }
]

interface SessionSearchHit {
  detail?: string
  kind: 'session'
  /** Durable lineage-root id used for pinning so the pin survives compression. */
  pinId: string
  sessionId: string
  snippet: string
  title: string
}

interface RouteSearchHit {
  detail?: string
  kind: 'route'
  route: string
  title: string
}

interface SectionSearchHit {
  detail?: string
  kind: 'section'
  section: CommandCenterSection
  title: string
}

type CommandCenterSearchResult = RouteSearchHit | SectionSearchHit | SessionSearchHit

interface CommandCenterSearchProvider {
  id: string
  label: string
  search: (query: string) => Promise<CommandCenterSearchResult[]>
}

interface CommandCenterSearchGroup {
  id: string
  label: string
  results: CommandCenterSearchResult[]
}

function formatTimestamp(value?: number | null): string {
  if (!value) {
    return ''
  }

  const date = new Date(value * 1000)

  if (Number.isNaN(date.getTime())) {
    return ''
  }

  return new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'short' }).format(date)
}

function splitSessionSearchResult(result: SessionSearchApiResult, sessionsById: Map<string, SessionInfo>) {
  const row = sessionsById.get(result.session_id)
  const title = row ? sessionTitle(row) : result.session_id
  const detail = [result.model, result.source].filter(Boolean).join(' · ')

  return { detail, title }
}

function matchesSearchQuery(query: string, ...values: Array<string | undefined>): boolean {
  const normalized = query.trim().toLowerCase()

  if (!normalized) {
    return true
  }

  return values.some(value => value?.toLowerCase().includes(normalized))
}

function useDebouncedValue<T>(value: T, delayMs: number): T {
  const [debounced, setDebounced] = useState(value)

  useEffect(() => {
    const id = window.setTimeout(() => setDebounced(value), delayMs)

    return () => window.clearTimeout(id)
  }, [delayMs, value])

  return debounced
}

export function CommandCenterView({
  initialSection,
  onClose,
  onDeleteSession,
  onNavigateRoute,
  onOpenSession
}: CommandCenterViewProps) {
  const sessions = useStore($sessions)
  const pinnedSessionIds = useStore($pinnedSessionIds)

  const [section, setSection] = useRouteEnumParam('section', SECTIONS, initialSection ?? 'sessions')

  const [query, setQuery] = useState('')
  const [searchLoading, setSearchLoading] = useState(false)
  const [searchGroups, setSearchGroups] = useState<CommandCenterSearchGroup[]>([])
  const [status, setStatus] = useState<StatusResponse | null>(null)
  const [logs, setLogs] = useState<string[]>([])
  const [systemLoading, setSystemLoading] = useState(false)
  const [systemError, setSystemError] = useState('')
  const [systemAction, setSystemAction] = useState<ActionStatusResponse | null>(null)
  const [usagePeriod, setUsagePeriod] = useState<UsagePeriod>(30)
  const [usage, setUsage] = useState<AnalyticsResponse | null>(null)
  const [usageLoading, setUsageLoading] = useState(false)
  const [usageError, setUsageError] = useState('')
  const searchRequestRef = useRef(0)
  const usageRequestRef = useRef(0)

  const debouncedQuery = useDebouncedValue(query.trim(), 180)

  const sessionsById = useMemo(() => new Map(sessions.map(session => [session.id, session])), [sessions])

  const filteredSessions = useMemo(
    () =>
      [...sessions].sort((a, b) => {
        const left = a.last_active || a.started_at || 0
        const right = b.last_active || b.started_at || 0

        return right - left
      }),
    [sessions]
  )

  const searchProviders = useMemo<readonly CommandCenterSearchProvider[]>(
    () => [
      {
        id: 'navigation',
        label: 'Navigate',
        search: async searchQuery => {
          const routeHits: RouteSearchHit[] = NAVIGATION_SEARCH_ENTRIES.filter(entry =>
            matchesSearchQuery(searchQuery, entry.title, entry.detail, entry.route)
          ).map(entry => ({
            detail: entry.detail,
            kind: 'route',
            route: entry.route,
            title: entry.title
          }))

          const sectionHits: SectionSearchHit[] = SECTION_SEARCH_ENTRIES.filter(entry =>
            matchesSearchQuery(searchQuery, entry.title, entry.detail, SECTION_LABELS[entry.section])
          ).map(entry => ({
            detail: entry.detail,
            kind: 'section',
            section: entry.section,
            title: entry.title
          }))

          return [...routeHits, ...sectionHits]
        }
      },
      {
        id: 'sessions',
        label: 'Sessions',
        search: async searchQuery => {
          const response = await searchSessions(searchQuery)

          return response.results.map(result => {
            const { detail, title } = splitSessionSearchResult(result, sessionsById)

            return {
              detail,
              kind: 'session',
              pinId: result.lineage_root || result.session_id,
              sessionId: result.session_id,
              snippet: result.snippet || '',
              title
            } satisfies SessionSearchHit
          })
        }
      }
    ],
    [sessionsById]
  )

  const refreshSystem = useCallback(async () => {
    setSystemLoading(true)
    setSystemError('')

    try {
      const [nextStatus, nextLogs] = await Promise.all([
        getStatus(),
        getLogs({
          file: 'agent',
          lines: 120
        })
      ])

      setStatus(nextStatus)
      setLogs(nextLogs.lines)
    } catch (error) {
      setSystemError(error instanceof Error ? error.message : String(error))
    } finally {
      setSystemLoading(false)
    }
  }, [])

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
    if (!debouncedQuery) {
      setSearchGroups([])
      setSearchLoading(false)

      return
    }

    const requestId = searchRequestRef.current + 1
    searchRequestRef.current = requestId
    setSearchLoading(true)

    void Promise.all(
      searchProviders.map(async provider => ({
        id: provider.id,
        label: provider.label,
        results: await provider.search(debouncedQuery)
      }))
    )
      .then(groups => {
        if (searchRequestRef.current === requestId) {
          setSearchGroups(groups.filter(group => group.results.length > 0))
        }
      })
      .catch(() => {
        if (searchRequestRef.current === requestId) {
          setSearchGroups([])
        }
      })
      .finally(() => {
        if (searchRequestRef.current === requestId) {
          setSearchLoading(false)
        }
      })
  }, [debouncedQuery, searchProviders])

  useEffect(() => {
    if (section === 'system' && !status && !systemLoading) {
      void refreshSystem()
    }
  }, [refreshSystem, section, status, systemLoading])

  useEffect(() => {
    if (section === 'usage') {
      void refreshUsage(usagePeriod)
    }
  }, [refreshUsage, section, usagePeriod])

  const showGlobalSearchResults = debouncedQuery.length > 0
  const hasGlobalSearchResults = searchGroups.length > 0
  const sessionListHasResults = filteredSessions.length > 0

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
            lines: ['Action started, waiting for status...'],
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
    [refreshSystem]
  )

  const handleSearchSelect = useCallback(
    (result: CommandCenterSearchResult) => {
      if (result.kind === 'route') {
        onNavigateRoute(result.route)

        return
      }

      if (result.kind === 'section') {
        setSection(result.section)
        setQuery('')

        return
      }

      onOpenSession(result.sessionId)
    },
    [onNavigateRoute, onOpenSession, setSection]
  )

  return (
    <OverlayView
      closeLabel="Close command center"
      headerContent={
        <OverlaySearchInput
          containerClassName="w-[min(36rem,calc(100vw-32rem))] min-w-80"
          loading={searchLoading}
          onChange={next => setQuery(next)}
          placeholder="Search sessions, views, and actions"
          value={query}
        />
      }
      onClose={onClose}
    >
      <OverlaySplitLayout>
        <OverlaySidebar>
          {SECTIONS.map(value => (
            <OverlayNavItem
              active={section === value}
              icon={value === 'sessions' ? Pin : value === 'system' ? Activity : BarChart3}
              key={value}
              label={SECTION_LABELS[value]}
              onClick={() => setSection(value)}
            />
          ))}
        </OverlaySidebar>

        <OverlayMain>
          <header className="mb-4 flex items-center justify-between gap-2">
            <div>
              <h2 className="text-sm font-semibold text-foreground">{SECTION_LABELS[section]}</h2>
              <p className="text-xs text-muted-foreground">{SECTION_DESCRIPTIONS[section]}</p>
            </div>
            {section === 'system' && (
              <OverlayActionButton disabled={systemLoading} onClick={() => void refreshSystem()}>
                <IconRefresh className={cn('mr-1.5 size-3.5', systemLoading && 'animate-spin')} />
                {systemLoading ? 'Refreshing...' : 'Refresh'}
              </OverlayActionButton>
            )}
            {section === 'usage' && (
              <OverlayActionButton disabled={usageLoading} onClick={() => void refreshUsage(usagePeriod)}>
                <IconRefresh className={cn('mr-1.5 size-3.5', usageLoading && 'animate-spin')} />
                {usageLoading ? 'Refreshing...' : 'Refresh'}
              </OverlayActionButton>
            )}
          </header>

          {showGlobalSearchResults ? (
            <div className="min-h-0 flex-1 overflow-y-auto pr-1">
              {!hasGlobalSearchResults ? (
                <OverlayCard className="px-3 py-4 text-sm text-muted-foreground">
                  No matching results found.
                </OverlayCard>
              ) : (
                <div className="grid gap-3">
                  {searchGroups.map(group => (
                    <section className="grid gap-1.5" key={group.id}>
                      <h3 className="px-0.5 text-xs font-semibold tracking-[0.08em] text-muted-foreground/80 uppercase">
                        {group.label}
                      </h3>
                      {group.results.map(result => {
                        if (result.kind === 'session') {
                          const pinned = pinnedSessionIds.includes(result.pinId)

                          return (
                            <OverlayCard className="p-2.5" key={`${group.id}:${result.sessionId}:${result.snippet}`}>
                              <button
                                className="w-full text-left"
                                onClick={() => handleSearchSelect(result)}
                                type="button"
                              >
                                <div className="truncate text-sm font-medium text-foreground">{result.title}</div>
                                <div className="mt-0.5 text-xs text-muted-foreground">
                                  {result.detail || result.sessionId}
                                </div>
                                {result.snippet && (
                                  <div className="mt-1 whitespace-pre-wrap text-xs text-muted-foreground/85">
                                    {result.snippet}
                                  </div>
                                )}
                              </button>
                              <div className="mt-2 flex gap-1">
                                <OverlayIconButton
                                  onClick={event => {
                                    event.preventDefault()
                                    event.stopPropagation()
                                    pinned ? unpinSession(result.pinId) : pinSession(result.pinId)
                                  }}
                                  title={pinned ? 'Unpin session' : 'Pin session'}
                                >
                                  {pinned ? (
                                    <IconBookmarkFilled className="size-3.5" />
                                  ) : (
                                    <IconBookmark className="size-3.5" />
                                  )}
                                </OverlayIconButton>
                                <OverlayIconButton
                                  onClick={event => {
                                    event.preventDefault()
                                    event.stopPropagation()
                                    void exportSession(result.sessionId, { title: result.title })
                                  }}
                                  title="Export session"
                                >
                                  <IconDownload className="size-3.5" />
                                </OverlayIconButton>
                                <OverlayIconButton
                                  className="hover:text-destructive"
                                  onClick={event => {
                                    event.preventDefault()
                                    event.stopPropagation()
                                    void onDeleteSession(result.sessionId)
                                  }}
                                  title="Delete session"
                                >
                                  <IconTrash className="size-3.5" />
                                </OverlayIconButton>
                              </div>
                            </OverlayCard>
                          )
                        }

                        return (
                          <button
                            className={cn(
                              overlayCardClass,
                              'w-full px-3 py-2 text-left transition-colors hover:bg-[color-mix(in_srgb,var(--dt-muted)_48%,var(--dt-card))]'
                            )}
                            key={`${group.id}:${result.kind}:${result.title}`}
                            onClick={() => handleSearchSelect(result)}
                            type="button"
                          >
                            <div className="text-sm font-medium text-foreground">{result.title}</div>
                            {result.detail && (
                              <div className="mt-0.5 text-xs text-muted-foreground">{result.detail}</div>
                            )}
                          </button>
                        )
                      })}
                    </section>
                  ))}
                </div>
              )}
            </div>
          ) : section === 'sessions' ? (
            <div className="min-h-0 flex-1 overflow-y-auto">
              {!sessionListHasResults ? (
                <OverlayCard className="px-3 py-4 text-sm text-muted-foreground">No sessions yet.</OverlayCard>
              ) : (
                <div className="grid gap-1.5">
                  {filteredSessions.map(session => {
                    const pinId = sessionPinId(session)
                    const pinned = pinnedSessionIds.includes(pinId)

                    return (
                      <OverlayCard className="flex items-center gap-2 px-2.5 py-2" key={session.id}>
                        <button
                          className="min-w-0 flex-1 text-left"
                          onClick={() => onOpenSession(session.id)}
                          type="button"
                        >
                          <div className="truncate text-sm font-medium text-foreground">{sessionTitle(session)}</div>
                          <div className="truncate text-xs text-muted-foreground">
                            {formatTimestamp(session.last_active || session.started_at)}
                          </div>
                        </button>
                        <OverlayIconButton
                          onClick={() => (pinned ? unpinSession(pinId) : pinSession(pinId))}
                          title={pinned ? 'Unpin session' : 'Pin session'}
                        >
                          {pinned ? <IconBookmarkFilled className="size-3.5" /> : <IconBookmark className="size-3.5" />}
                        </OverlayIconButton>
                        <OverlayIconButton
                          onClick={() => void exportSession(session.id, { session, title: sessionTitle(session) })}
                          title="Export session"
                        >
                          <IconDownload className="size-3.5" />
                        </OverlayIconButton>
                        <OverlayIconButton
                          className="hover:text-destructive"
                          onClick={() => void onDeleteSession(session.id)}
                          title="Delete session"
                        >
                          <IconTrash className="size-3.5" />
                        </OverlayIconButton>
                      </OverlayCard>
                    )
                  })}
                </div>
              )}
            </div>
          ) : section === 'usage' ? (
            <UsagePanel
              error={usageError}
              loading={usageLoading}
              onPeriodChange={setUsagePeriod}
              onRefresh={() => void refreshUsage(usagePeriod)}
              period={usagePeriod}
              usage={usage}
            />
          ) : (
            <div className="grid min-h-0 flex-1 grid-rows-[auto_minmax(0,1fr)] gap-3">
              <OverlayCard className="p-3 text-sm">
                {status ? (
                  <div className="grid gap-2">
                    <div className="flex items-start justify-between gap-3">
                      <div className="min-w-0">
                        <div className="flex items-center gap-2">
                          <span
                            className={cn(
                              'size-2 rounded-full',
                              status.gateway_running ? 'bg-emerald-500' : 'bg-amber-500'
                            )}
                          />
                          <span className="font-medium text-foreground">
                            {status.gateway_running ? 'Messaging gateway running' : 'Messaging gateway stopped'}
                          </span>
                        </div>
                        <div className="mt-1 text-xs text-muted-foreground">
                          Hermes {status.version} · Active sessions {status.active_sessions}
                        </div>
                      </div>
                      <div className="flex shrink-0 items-center gap-1.5 whitespace-nowrap">
                        <OverlayActionButton className="h-7 px-2.5" onClick={() => void runSystemAction('restart')}>
                          Restart messaging
                        </OverlayActionButton>
                        <OverlayActionButton className="h-7 px-2.5" onClick={() => void runSystemAction('update')}>
                          Update Hermes
                        </OverlayActionButton>
                      </div>
                    </div>
                    {systemAction && (
                      <div className="text-xs text-muted-foreground">
                        {systemAction.name} ·{' '}
                        {systemAction.running ? 'running' : systemAction.exit_code === 0 ? 'done' : 'failed'}
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="text-xs text-muted-foreground">Loading status...</div>
                )}
              </OverlayCard>

              <OverlayCard className="min-h-0 overflow-hidden p-2">
                <div className="mb-2 flex items-center justify-between">
                  <span className="text-xs font-medium text-muted-foreground">Recent logs</span>
                  {systemError && (
                    <span className="inline-flex items-center gap-1 text-xs text-destructive">
                      <AlertCircle className="size-3.5" />
                      {systemError}
                    </span>
                  )}
                </div>
                <pre className="h-full min-h-0 overflow-auto whitespace-pre-wrap wrap-break-word font-mono text-[0.65rem] leading-relaxed text-muted-foreground">
                  {logs.length ? logs.join('\n') : 'No logs loaded yet.'}
                </pre>
              </OverlayCard>
            </div>
          )}
        </OverlayMain>
      </OverlaySplitLayout>
    </OverlayView>
  )
}

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

  if (num === 0) {
    return '$0.00'
  }

  if (num < 0.01) {
    return '<$0.01'
  }

  return `$${num.toFixed(2)}`
}

function formatInteger(value: null | number | undefined): string {
  return Number(value ?? 0).toLocaleString()
}

interface UsagePanelProps {
  error: string
  loading: boolean
  onPeriodChange: (period: UsagePeriod) => void
  onRefresh: () => void
  period: UsagePeriod
  usage: AnalyticsResponse | null
}

function UsagePanel({ error, loading, onPeriodChange, onRefresh, period, usage }: UsagePanelProps) {
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

  return (
    <div className="grid min-h-0 flex-1 grid-rows-[auto_auto_minmax(0,1fr)] gap-3">
      <OverlayCard className="flex flex-wrap items-center justify-between gap-2 p-3">
        <div className="flex items-center gap-1">
          {USAGE_PERIODS.map(value => (
            <button
              className={cn(
                'h-7 rounded-md px-2.5 text-xs transition-colors',
                value === period
                  ? 'bg-foreground text-background'
                  : 'text-muted-foreground hover:bg-muted/40 hover:text-foreground'
              )}
              key={value}
              onClick={() => onPeriodChange(value)}
              type="button"
            >
              {value}d
            </button>
          ))}
        </div>
        {error && (
          <span className="inline-flex items-center gap-1 text-xs text-destructive">
            <AlertCircle className="size-3.5" />
            {error}
          </span>
        )}
      </OverlayCard>

      <OverlayCard className="p-3">
        {totals ? (
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <UsageStat label="Sessions" value={formatInteger(totals.total_sessions)} />
            <UsageStat label="API calls" value={formatInteger(totals.total_api_calls)} />
            <UsageStat
              label="Tokens in/out"
              value={`${formatTokens(totals.total_input)} / ${formatTokens(totals.total_output)}`}
            />
            <UsageStat
              hint={totals.total_actual_cost > 0 ? `actual ${formatCost(totals.total_actual_cost)}` : undefined}
              label="Est. cost"
              value={formatCost(totals.total_estimated_cost)}
            />
          </div>
        ) : loading ? (
          <div className="text-xs text-muted-foreground">Loading usage...</div>
        ) : (
          <div className="text-xs text-muted-foreground">
            No usage in the last {period} days.{' '}
            <button className="underline underline-offset-4 decoration-current/20" onClick={onRefresh} type="button">
              Retry
            </button>
          </div>
        )}
      </OverlayCard>

      <div className="grid min-h-0 grid-rows-[auto_minmax(0,1fr)] gap-3">
        <OverlayCard className="p-3">
          <div className="mb-2 flex items-baseline justify-between">
            <span className="text-xs font-medium text-muted-foreground">Daily tokens</span>
            <span className="flex items-center gap-3 text-[0.65rem] text-muted-foreground">
              <span className="inline-flex items-center gap-1">
                <span className="size-2 bg-[color:var(--dt-primary)]/60" /> input
              </span>
              <span className="inline-flex items-center gap-1">
                <span className="size-2 bg-emerald-500/70" /> output
              </span>
            </span>
          </div>
          {daily.length === 0 ? (
            <div className="grid h-24 place-items-center text-xs text-muted-foreground">No daily activity.</div>
          ) : (
            <>
              <div className="flex h-24 items-end gap-px">
                {daily.map(entry => {
                  const total = (entry.input_tokens || 0) + (entry.output_tokens || 0)
                  const inputH = Math.round(((entry.input_tokens || 0) / maxTokens) * 96)
                  const outputH = Math.round(((entry.output_tokens || 0) / maxTokens) * 96)

                  return (
                    <div
                      className="group relative flex h-24 min-w-0 flex-1 flex-col justify-end"
                      key={entry.day}
                      title={`${entry.day} · in ${formatTokens(entry.input_tokens)} · out ${formatTokens(entry.output_tokens)}`}
                    >
                      <div
                        className="w-full bg-[color:var(--dt-primary)]/50"
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
              <div className="mt-1 flex justify-between text-[0.6rem] text-muted-foreground/70">
                <span>{daily[0]?.day}</span>
                <span>{daily[daily.length - 1]?.day}</span>
              </div>
            </>
          )}
        </OverlayCard>

        <OverlayCard className="min-h-0 overflow-auto p-2">
          <div className="grid gap-3 sm:grid-cols-2">
            <section className="min-w-0">
              <div className="mb-1.5 text-[0.7rem] font-medium uppercase tracking-wide text-muted-foreground">
                Top models
              </div>
              {byModel.length === 0 ? (
                <div className="text-xs text-muted-foreground">No model usage yet.</div>
              ) : (
                <ul className="space-y-1">
                  {byModel.slice(0, 6).map(entry => (
                    <li
                      className="flex items-center justify-between gap-2 rounded px-1.5 py-1 text-xs hover:bg-muted/40"
                      key={entry.model}
                    >
                      <span className="min-w-0 truncate font-mono text-[0.7rem] text-foreground">{entry.model}</span>
                      <span className="shrink-0 text-[0.65rem] text-muted-foreground">
                        {formatTokens((entry.input_tokens || 0) + (entry.output_tokens || 0))} ·{' '}
                        {formatCost(entry.estimated_cost)}
                      </span>
                    </li>
                  ))}
                </ul>
              )}
            </section>

            <section className="min-w-0">
              <div className="mb-1.5 text-[0.7rem] font-medium uppercase tracking-wide text-muted-foreground">
                Top skills
              </div>
              {topSkills.length === 0 ? (
                <div className="text-xs text-muted-foreground">No skill activity yet.</div>
              ) : (
                <ul className="space-y-1">
                  {topSkills.slice(0, 6).map(entry => (
                    <li
                      className="flex items-center justify-between gap-2 rounded px-1.5 py-1 text-xs hover:bg-muted/40"
                      key={entry.skill}
                    >
                      <span className="min-w-0 truncate font-mono text-[0.7rem] text-foreground">{entry.skill}</span>
                      <span className="shrink-0 text-[0.65rem] text-muted-foreground">
                        {entry.total_count.toLocaleString()} actions
                      </span>
                    </li>
                  ))}
                </ul>
              )}
            </section>
          </div>
        </OverlayCard>
      </div>
    </div>
  )
}

function UsageStat({ hint, label, value }: { hint?: string; label: string; value: string }) {
  return (
    <div className="min-w-0">
      <div className="text-[0.65rem] font-medium uppercase tracking-[0.12em] text-muted-foreground">{label}</div>
      <div className="mt-0.5 truncate text-sm font-semibold tracking-tight text-foreground">{value}</div>
      {hint && <div className="mt-0.5 truncate text-[0.62rem] text-muted-foreground/80">{hint}</div>}
    </div>
  )
}
