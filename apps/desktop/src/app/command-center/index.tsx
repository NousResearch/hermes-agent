import { useStore } from '@nanostores/react'
import {
  IconBookmark,
  IconBookmarkFilled,
  IconDownload,
  IconLoader2,
  IconRefresh,
  IconSparkles,
  IconTrash
} from '@tabler/icons-react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import {
  getActionStatus,
  getAuxiliaryModels,
  getGlobalModelInfo,
  getGlobalModelOptions,
  getLogs,
  getStatus,
  restartGateway,
  searchSessions,
  setModelAssignment,
  updateHermes
} from '@/hermes'
import type {
  ActionStatusResponse,
  AuxiliaryModelsResponse,
  ModelOptionProvider,
  SessionInfo,
  SessionSearchResult as SessionSearchApiResult,
  StatusResponse
} from '@/hermes'
import { sessionTitle } from '@/lib/chat-runtime'
import { triggerHaptic } from '@/lib/haptics'
import { Activity, AlertCircle, Cpu, Pin } from '@/lib/icons'
import { exportSession } from '@/lib/session-export'
import { cn } from '@/lib/utils'
import { upsertDesktopActionTask } from '@/store/activity'
import { $pinnedSessionIds, pinSession, unpinSession } from '@/store/layout'
import { $sessions } from '@/store/session'

import { OverlayActionButton, OverlayCard, overlayCardClass, OverlayIconButton } from '../overlays/overlay-chrome'
import { OverlaySearchInput } from '../overlays/overlay-search-input'
import { OverlayMain, OverlayNavItem, OverlaySidebar, OverlaySplitLayout } from '../overlays/overlay-split-layout'
import { OverlayView } from '../overlays/overlay-view'
import { ARTIFACTS_ROUTE, MESSAGING_ROUTE, NEW_CHAT_ROUTE, SETTINGS_ROUTE, SKILLS_ROUTE } from '../routes'

export type CommandCenterSection = 'models' | 'sessions' | 'system'

interface CommandCenterViewProps {
  initialSection?: CommandCenterSection
  onClose: () => void
  onDeleteSession: (sessionId: string) => Promise<void>
  onMainModelChanged?: (provider: string, model: string) => void
  onNavigateRoute: (path: string) => void
  onOpenSession: (sessionId: string) => void
}

const SECTION_LABELS: Record<CommandCenterSection, string> = {
  sessions: 'Sessions',
  system: 'System',
  models: 'Models'
}

const SECTION_DESCRIPTIONS: Record<CommandCenterSection, string> = {
  sessions: 'Search and manage sessions',
  system: 'Status, logs, and system actions',
  models: 'Global and auxiliary model controls'
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
  { id: 'nav-new-chat', route: NEW_CHAT_ROUTE, title: 'New chat', detail: 'Start a fresh session' },
  { id: 'nav-settings', route: SETTINGS_ROUTE, title: 'Settings', detail: 'Configure 元话 Agent Desktop' },
  { id: 'nav-skills', route: SKILLS_ROUTE, title: 'Skills', detail: 'Enable and inspect skills' },
  { id: 'nav-messaging', route: MESSAGING_ROUTE, title: 'Messaging', detail: 'Set up Telegram, Slack, Discord, and more' },
  { id: 'nav-artifacts', route: ARTIFACTS_ROUTE, title: 'Artifacts', detail: 'Browse generated outputs' }
]

const SECTION_SEARCH_ENTRIES: readonly SectionSearchEntry[] = [
  { id: 'section-sessions', section: 'sessions', title: 'Sessions panel', detail: 'Search, pin, and manage sessions' },
  { id: 'section-system', section: 'system', title: 'System panel', detail: 'Gateway status, logs, restart/update' },
  { id: 'section-models', section: 'models', title: 'Models panel', detail: 'Main and auxiliary model assignments' }
]

interface SessionSearchHit {
  detail?: string
  kind: 'session'
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
  onMainModelChanged,
  onNavigateRoute,
  onOpenSession
}: CommandCenterViewProps) {
  const sessions = useStore($sessions)
  const pinnedSessionIds = useStore($pinnedSessionIds)
  const [section, setSection] = useState<CommandCenterSection>(initialSection ?? 'sessions')
  const [query, setQuery] = useState('')
  const [searchLoading, setSearchLoading] = useState(false)
  const [searchGroups, setSearchGroups] = useState<CommandCenterSearchGroup[]>([])
  const [status, setStatus] = useState<StatusResponse | null>(null)
  const [logs, setLogs] = useState<string[]>([])
  const [systemLoading, setSystemLoading] = useState(false)
  const [systemError, setSystemError] = useState('')
  const [systemAction, setSystemAction] = useState<ActionStatusResponse | null>(null)
  const [modelsLoading, setModelsLoading] = useState(false)
  const [modelsError, setModelsError] = useState('')
  const [mainModel, setMainModel] = useState<{ model: string; provider: string } | null>(null)
  const [providers, setProviders] = useState<ModelOptionProvider[]>([])
  const [selectedProvider, setSelectedProvider] = useState('')
  const [selectedModel, setSelectedModel] = useState('')
  const [auxiliary, setAuxiliary] = useState<AuxiliaryModelsResponse | null>(null)
  const [applyingModel, setApplyingModel] = useState(false)
  const searchRequestRef = useRef(0)

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

  const selectedProviderModels = useMemo(
    () => providers.find(provider => provider.slug === selectedProvider)?.models ?? [],
    [providers, selectedProvider]
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

  const refreshModels = useCallback(async () => {
    setModelsLoading(true)
    setModelsError('')

    try {
      const [modelInfo, modelOptions, auxiliaryModels] = await Promise.all([
        getGlobalModelInfo(),
        getGlobalModelOptions(),
        getAuxiliaryModels()
      ])

      setMainModel({ model: modelInfo.model, provider: modelInfo.provider })
      setProviders(modelOptions.providers || [])
      setSelectedProvider(prev => prev || modelInfo.provider)
      setSelectedModel(prev => prev || modelInfo.model)
      setAuxiliary(auxiliaryModels)
    } catch (error) {
      setModelsError(error instanceof Error ? error.message : String(error))
    } finally {
      setModelsLoading(false)
    }
  }, [])

  useEffect(() => {
    if (initialSection && initialSection !== section) {
      setSection(initialSection)
    }
  }, [initialSection, section])

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault()
        triggerHaptic('close')
        onClose()
      }
    }

    window.addEventListener('keydown', onKeyDown)

    return () => window.removeEventListener('keydown', onKeyDown)
  }, [onClose])

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
    if (section === 'models' && !mainModel && !modelsLoading) {
      void refreshModels()
    }
  }, [mainModel, modelsLoading, refreshModels, section])

  useEffect(() => {
    if (!selectedProviderModels.length) {
      return
    }

    if (!selectedProviderModels.includes(selectedModel)) {
      setSelectedModel(selectedProviderModels[0])
    }
  }, [selectedModel, selectedProviderModels])

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

  const applyMainModel = useCallback(async () => {
    if (!selectedProvider || !selectedModel) {
      return
    }

    setApplyingModel(true)
    setModelsError('')

    try {
      const result = await setModelAssignment({
        model: selectedModel,
        provider: selectedProvider,
        scope: 'main'
      })

      const provider = result.provider || selectedProvider
      const model = result.model || selectedModel
      setMainModel({ provider, model })
      onMainModelChanged?.(provider, model)
      await refreshModels()
    } catch (error) {
      setModelsError(error instanceof Error ? error.message : String(error))
    } finally {
      setApplyingModel(false)
    }
  }, [onMainModelChanged, refreshModels, selectedModel, selectedProvider])

  const setAuxiliaryToMain = useCallback(
    async (task: string) => {
      if (!mainModel) {
        return
      }

      setApplyingModel(true)
      setModelsError('')

      try {
        await setModelAssignment({
          model: mainModel.model,
          provider: mainModel.provider,
          scope: 'auxiliary',
          task
        })
        await refreshModels()
      } catch (error) {
        setModelsError(error instanceof Error ? error.message : String(error))
      } finally {
        setApplyingModel(false)
      }
    },
    [mainModel, refreshModels]
  )

  const resetAuxiliaryModels = useCallback(async () => {
    if (!mainModel) {
      return
    }

    setApplyingModel(true)
    setModelsError('')

    try {
      await setModelAssignment({
        model: mainModel.model,
        provider: mainModel.provider,
        scope: 'auxiliary',
        task: '__reset__'
      })
      await refreshModels()
    } catch (error) {
      setModelsError(error instanceof Error ? error.message : String(error))
    } finally {
      setApplyingModel(false)
    }
  }, [mainModel, refreshModels])

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
    [onNavigateRoute, onOpenSession]
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
          {(['sessions', 'system', 'models'] as const).map(value => (
            <OverlayNavItem
              active={section === value}
              icon={value === 'sessions' ? Pin : value === 'system' ? Activity : Cpu}
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
            {section === 'models' && (
              <OverlayActionButton disabled={modelsLoading} onClick={() => void refreshModels()}>
                <IconRefresh className={cn('mr-1.5 size-3.5', modelsLoading && 'animate-spin')} />
                {modelsLoading ? 'Refreshing...' : 'Refresh'}
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
                          const pinned = pinnedSessionIds.includes(result.sessionId)

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
                                    pinned ? unpinSession(result.sessionId) : pinSession(result.sessionId)
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
                    const pinned = pinnedSessionIds.includes(session.id)

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
                          onClick={() => (pinned ? unpinSession(session.id) : pinSession(session.id))}
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
          ) : section === 'system' ? (
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
                            {status.gateway_running ? 'Gateway running' : 'Gateway not running'}
                          </span>
                        </div>
                        <div className="mt-1 text-xs text-muted-foreground">
                          Hermes {status.version} · Active sessions {status.active_sessions}
                        </div>
                      </div>
                      <div className="flex shrink-0 items-center gap-1.5 whitespace-nowrap">
                        <OverlayActionButton className="h-7 px-2.5" onClick={() => void runSystemAction('restart')}>
                          Restart gateway
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
          ) : (
            <div className="grid min-h-0 flex-1 grid-rows-[auto_auto_minmax(0,1fr)] gap-3">
              <OverlayCard className="p-3">
                {mainModel ? (
                  <>
                    <div className="text-sm font-medium text-foreground">Main model</div>
                    <div className="text-xs text-muted-foreground">
                      {mainModel.provider} / {mainModel.model}
                    </div>
                  </>
                ) : (
                  <div className="text-xs text-muted-foreground">Loading model state...</div>
                )}
              </OverlayCard>

              <OverlayCard className="p-3">
                <div className="mb-2 text-xs font-medium text-muted-foreground">Set global main model</div>
                <div className="flex flex-wrap items-center gap-2">
                  <select
                    className="h-8 min-w-36 rounded-md border border-border bg-background px-2 text-xs text-foreground"
                    onChange={event => setSelectedProvider(event.target.value)}
                    value={selectedProvider}
                  >
                    {(providers.length ? providers : [{ name: '—', slug: '', models: [] }]).map(provider => (
                      <option key={provider.slug || 'none'} value={provider.slug}>
                        {provider.name}
                      </option>
                    ))}
                  </select>
                  <select
                    className="h-8 min-w-58 rounded-md border border-border bg-background px-2 text-xs text-foreground"
                    onChange={event => setSelectedModel(event.target.value)}
                    value={selectedModel}
                  >
                    {(selectedProviderModels.length ? selectedProviderModels : ['']).map(model => (
                      <option key={model || 'none'} value={model}>
                        {model || 'No models available'}
                      </option>
                    ))}
                  </select>
                  <OverlayActionButton
                    disabled={!selectedProvider || !selectedModel || applyingModel}
                    onClick={() => void applyMainModel()}
                  >
                    {applyingModel ? (
                      <IconLoader2 className="mr-1.5 size-3.5 animate-spin" />
                    ) : (
                      <IconSparkles className="mr-1.5 size-3.5" />
                    )}
                    {applyingModel ? 'Applying...' : 'Apply'}
                  </OverlayActionButton>
                </div>
                {modelsError && <div className="mt-2 text-xs text-destructive">{modelsError}</div>}
              </OverlayCard>

              <OverlayCard className="min-h-0 overflow-auto p-2">
                <div className="mb-2 flex items-center justify-between">
                  <span className="text-xs font-medium text-muted-foreground">Auxiliary assignments</span>
                  <OverlayActionButton
                    disabled={!mainModel || applyingModel}
                    onClick={() => void resetAuxiliaryModels()}
                    tone="subtle"
                  >
                    Reset all
                  </OverlayActionButton>
                </div>
                <div className="grid gap-1.5">
                  {(auxiliary?.tasks || []).map(task => (
                    <OverlayCard className="flex items-center gap-2 px-2 py-1.5" key={task.task}>
                      <div className="min-w-0 flex-1">
                        <div className="truncate text-xs font-medium text-foreground">{task.task}</div>
                        <div className="truncate text-[0.65rem] text-muted-foreground">
                          {task.provider} / {task.model}
                        </div>
                      </div>
                      <OverlayActionButton
                        disabled={!mainModel || applyingModel}
                        onClick={() => void setAuxiliaryToMain(task.task)}
                      >
                        Set to main
                      </OverlayActionButton>
                    </OverlayCard>
                  ))}
                  {!auxiliary?.tasks?.length && (
                    <div className="text-xs text-muted-foreground">No auxiliary assignments reported.</div>
                  )}
                </div>
              </OverlayCard>
            </div>
          )}
        </OverlayMain>
      </OverlaySplitLayout>
    </OverlayView>
  )
}
