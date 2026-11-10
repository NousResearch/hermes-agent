import './agent-settings.css'

import { useStore } from '@nanostores/react'
import { useInfiniteQuery, useQuery } from '@tanstack/react-query'
import { useEffect, useMemo, useState } from 'react'
import type { ReactNode } from 'react'
import { useLocation, useNavigate } from 'react-router'

import { $apiRequestScope } from '@/api/client'
import { PageLoader } from '@/components/page-loader'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { FigmaAgentIcon } from '@/components/ui/figma-agent-icon'
import { RowButton } from '@/components/ui/row-button'
import { type CronJob, getProfileSoul } from '@/hermes'
import { useI18n } from '@/i18n'
import { sessionTitle } from '@/lib/chat-runtime'
import { resolveProfileColor } from '@/lib/profile-color'
import { asText } from '@/lib/text'
import { openRosterBot } from '@/plugins/hermes-bots/roster-actions'
import { $connectionsRegistry } from '@/store/connection-registry-state'
import { $activeConnectionId, selectConnection } from '@/store/connections'
import { $fleetRoster } from '@/store/fleet-roster'
import { $activeGatewayProfile, $profileColors, $profiles, ensureGatewayProfile } from '@/store/profile'
import { setSettingsScope } from '@/store/settings-scope'

import { CapabilitiesView } from '../capabilities'
import { useSkillsQuery } from '../capabilities/skills/skills-data'
import { useToolsetsQuery } from '../capabilities/toolsets/toolsets-data'
import { gatewayLabel } from '../chat/sidebar/gateway-group-model'
import { useHermesConfigRecord } from '../hooks/use-config-record'
import { MessagingView } from '../messaging'
import { openSession } from '../open-session'
import { Panel, PanelHeader } from '../overlays/panel'
import { PROFILES_ROUTE, SETTINGS_ROUTE } from '../routes'
import { SettingsView as ClassicSettingsView } from '../settings'
import { getNested, toolNames } from '../settings/helpers'
import { settingsScopeLabel } from '../settings/profile-scope'
import type { SettingsPageProps } from '../settings/types'

import { useAgentSettingsCopy } from './copy'
import {
  agentSettingsKey,
  type AgentSettingsOwner,
  getAgentMessaging,
  getAgentRecentSessions,
  getAgentRoutines
} from './data'
import { AgentNavigation } from './directory'
import { RoutineDialog } from './routine-dialog'

interface AgentSettingsViewProps extends SettingsPageProps {
  owner: AgentSettingsOwner
}

/** Existing editors use the active gateway. Gate them before rendering, never silently retarget. */
export function AgentSettingsView(props: AgentSettingsViewProps) {
  const c = useAgentSettingsCopy()
  const { t } = useI18n()
  const apiScope = useStore($apiRequestScope)
  const active = useStore($activeConnectionId)
  const profile = useStore($activeGatewayProfile)
  const registry = useStore($connectionsRegistry)
  const [error, setError] = useState('')
  const [opening, setOpening] = useState(false)
  const [scopeReady, setScopeReady] = useState(false)

  const matches =
    (!props.owner.connectionId || props.owner.connectionId === (apiScope.connectionId ?? active)) &&
    profile === props.owner.profile

  useEffect(() => {
    if (matches) {setSettingsScope(props.owner.profile)}
    setScopeReady(matches)
  }, [matches, props.owner.profile])

  async function activate() {
    setOpening(true)
    setError('')

    try {
      if (props.owner.connectionId && registry?.connections.some(item => item.id === props.owner.connectionId)) {
        await selectConnection(props.owner.connectionId, { profile: props.owner.profile })
      } else if (!props.owner.connectionId || props.owner.connectionId === (apiScope.connectionId ?? active)) {
        await ensureGatewayProfile(props.owner.profile)
      } else {
        throw new Error(c.unavailable)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : t.errors.genericFailure)
    } finally {
      setOpening(false)
    }
  }

  if (!matches)
    {return (
      <Panel onClose={props.onClose}>
        <PanelHeader subtitle={props.owner.profile} title={t.simplifiedAgents.settings} />
        <Button disabled={opening} onClick={() => void activate()}>
          {c.openAgent}
        </Button>
        {error && <p role="alert">{error}</p>}
      </Panel>
    )}

  return scopeReady ? <AgentSettingsContent {...props} /> : <PageLoader />
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="agent-overview-section">
      <h2>{title}</h2>
      <div className="agent-overview-card">{children}</div>
    </section>
  )
}

function SettingRow({ title, detail, subtitle, onClick }: { title: string; detail?: ReactNode; subtitle?: string; onClick: () => void }) {
  return (
    <RowButton
      className="agent-overview-row"
      onClick={onClick}
    >
      <span className="min-w-0 flex-1">
        <span className="block truncate">{title}</span>
        {subtitle && <span className="mt-1 block truncate text-xs text-(--ui-text-tertiary)">{subtitle}</span>}
      </span>
      <span className="max-w-[50%] truncate text-(--ui-text-tertiary)">{detail}</span>
      <Codicon name="chevron-right" />
    </RowButton>
  )
}

function AgentSettingsContent({ owner, ...props }: AgentSettingsViewProps) {
  const { t, locale } = useI18n()
  const c = useAgentSettingsCopy()
  const navigate = useNavigate()
  const { search } = useLocation()
  const params = new URLSearchParams(search)
  const panel = params.get('panel')
  const profiles = useStore($profiles)
  const colors = useStore($profileColors)
  const roster = useStore($fleetRoster)
  const registry = useStore($connectionsRegistry)
  const profile = profiles.find(item => item.name === owner.profile)
  const label = profile ? settingsScopeLabel(profile) : owner.profile

  const gateway =
    roster?.sources.find(source => source.connectionId === owner.connectionId)?.label ??
    gatewayLabel(owner.connectionId ?? 'local', registry, t.settings.gateway.webappHostTitle)

  const config = useHermesConfigRecord(owner)
  const skills = useSkillsQuery(owner)
  const tools = useToolsetsQuery(owner)
  const jobs = useQuery({ queryKey: [...agentSettingsKey(owner), 'routines'], queryFn: () => getAgentRoutines(owner) })

  const messaging = useQuery({
    queryKey: [...agentSettingsKey(owner), 'messaging'],
    queryFn: () => getAgentMessaging(owner)
  })

  const history = panel === 'history'

  const recent = useInfiniteQuery({
    queryKey: [...agentSettingsKey(owner), 'recent', history ? 25 : 5],
    initialPageParam: 0,
    queryFn: ({ pageParam }) => getAgentRecentSessions(owner, history ? 25 : 5, pageParam),
    getNextPageParam: page =>
      page.sessions.length >= page.limit && page.offset + page.limit < page.total
        ? page.offset + page.limit
        : undefined
  })

  const soul = useQuery({
    queryKey: [...agentSettingsKey(owner), 'soul'],
    queryFn: () => getProfileSoul(owner.profile, owner)
  })

  const apiScope = useStore($apiRequestScope)
  const activeConnectionId = useStore($activeConnectionId)

  const [routine, setRoutine] = useState<CronJob | 'new' | null>(null)
  const value = (path: string) => (config.isError ? c.unavailable : asText(getNested(config.data ?? {}, path)) || '—')

  const pending = (failed: boolean, loading: boolean, detail: ReactNode) =>
    failed ? c.unavailable : loading ? t.sidebar.loading : detail

  function openPanel(next: string | null, tab?: string, page?: string) {
    setSettingsScope(owner.profile)
    const query = new URLSearchParams({ agent: owner.profile })

    if (owner.connectionId) {query.set('connection', owner.connectionId)}

    if (next) {query.set('panel', next)}

    if (tab) {query.set('tab', tab)}

    if (page) {query.set('page', page)}
    navigate(`${SETTINGS_ROUTE}?${query}`)
  }

  const settings = (tab: string, page?: string) => openPanel('settings', `config:${tab}`, page)
  const back = () => openPanel(null)

  const connectionId = owner.connectionId ?? apiScope.connectionId ?? activeConnectionId ?? 'local'
  const fallbacks = getNested(config.data ?? {}, 'fallback_providers')
  const failed = [config, skills, tools, jobs, messaging, recent, soul].some(query => query.isError)

  // The endpoint backfills pinned sessions on each page; render each session once.
  const sessions = useMemo(() => {
    const unique = new Map(recent.data?.pages.flatMap(page => page.sessions).map(session => [session.id, session]))

    return [...unique.values()].filter(session => !session.hidden && !session.archived)
  }, [recent.data])

  const selectedTab = history ? 'history' : panel ? 'settings' : 'overview'

  function openChat() {
    // Dismiss the management route before the canonical opener owns navigation.
    props.onClose()
    void openRosterBot({
      name: owner.profile,
      connectionId,
      connectionLabel: gateway,
      targetProfile: roster?.agents.find(agent =>
        agent.connectionId === connectionId && agent.profile === owner.profile
      )?.targetProfile || owner.profile,
      sourceScoped: true
    })
  }

  const sessionRows = sessions.map(session => (
    <SettingRow
      key={session.id}
      onClick={() => {
        props.onClose()
        openSession(session.id, navigate, 'in-place', {
          workspaceMode: 'sessions',
          ownerRoute: { connectionId, profile: owner.profile }
        })
      }}
      subtitle={session.last_active ? new Date(session.last_active * 1000).toLocaleString(locale, { dateStyle: 'medium', timeStyle: 'short' }) : undefined}
      title={sessionTitle(session)}
    />
  ))

  const emptySessions = (
    <p className="agent-overview-empty">{pending(recent.isError, recent.isPending, c.noSessions)}</p>
  )

  return (
    <Panel contentClassName="agent-management" onClose={props.onClose}>
      <AgentNavigation connectionId={connectionId} profile={owner.profile} />
      <header className="agent-management-header">
        <FigmaAgentIcon className="agent-management-avatar" color={resolveProfileColor(owner.profile, colors)} />
        <div className="agent-management-identity">
          <h1>{label}</h1>
          <div className="agent-management-meta">
            <span>{gateway}</span>
            {profile?.path && <span className="truncate">{profile.path}</span>}
            {profile?.is_default && <span className="agent-management-default">{t.profiles.defaultBadge}</span>}
          </div>
        </div>
        <div className="agent-management-actions">
          <Button onClick={() => navigate(`${PROFILES_ROUTE}?manage=true`)} variant="outline">
            {t.profiles.manageProfiles}
          </Button>
          <Button onClick={openChat}>
            {c.openChat}
          </Button>
        </div>
      </header>
      <nav aria-label={t.simplifiedAgents.settings} className="agent-management-tabs">
        {([
          ['overview', c.overview, () => openPanel(null)],
          ['history', c.history, () => openPanel('history')],
          ['settings', t.commandCenter.settings, () => settings('model', 'main')]
        ] as const).map(([id, title, onClick]) => (
          <button aria-current={selectedTab === id ? 'page' : undefined} key={id} onClick={onClick} type="button">
            {title}
          </button>
        ))}
      </nav>
      {panel === 'settings' ? (
        <div className="agent-management-editor">
          <ClassicSettingsView {...props} onClose={back} />
        </div>
      ) : panel === 'capabilities' || panel === 'messaging' ? (
        <div className="agent-management-editor">
          {panel === 'capabilities' ? (
            <CapabilitiesView fixedConnection={owner.connectionId} fixedProfile={owner.profile} />
          ) : <MessagingView />}
        </div>
      ) : (
        <div className="agent-management-scroll">
          {failed && (
            <div className="agent-management-error" role="alert">
              <span>{c.unavailable}</span>
              <Button
                onClick={() => {
                  void Promise.all([
                    config.refetch(), skills.refetch(), tools.refetch(), jobs.refetch(),
                    messaging.refetch(), recent.refetch(), soul.refetch()
                  ])
                }}
                variant="secondary"
              >
                {c.retry}
              </Button>
            </div>
          )}
          {history ? (
            <Section title={c.history}>
              {sessionRows.length ? sessionRows : emptySessions}
              {recent.hasNextPage && (
                <div className="agent-overview-footer">
                  <Button disabled={recent.isFetchingNextPage} onClick={() => void recent.fetchNextPage()} variant="textStrong">
                    {t.sidebar.loadMore}
                  </Button>
                </div>
              )}
            </Section>
          ) : (
            <div className="agent-overview-grid">
              <div className="agent-overview-column">
                {soul.data?.content && (
                  <Section title={c.instructions}>
                    <div className="agent-instructions-preview">
                      <p>{soul.data.content}</p>
                      <span>SOUL.md{profile?.path ? ` · ${profile.path}` : ''}</span>
                    </div>
                  </Section>
                )}
                <Section title={c.behavior}>
                  <SettingRow
                    detail={value('model.default') === '—' ? value('model') : value('model.default')}
                    onClick={() => settings('model', 'main')}
                    title={c.model}
                  />
                  <SettingRow
                    detail={config.isError ? c.unavailable : Array.isArray(fallbacks) ? fallbacks.length : '—'}
                    onClick={() => settings('model', 'fallbacks')}
                    title={c.fallbacks}
                  />
                  <SettingRow detail={value('memory.provider')} onClick={() => settings('memory')} title={c.memory} />
                </Section>
                <Section title={c.capabilities}>
                  <SettingRow
                    detail={pending(tools.isError, tools.isPending, new Set(tools.data?.filter(tool => tool.enabled).flatMap(toolNames)).size)}
                    onClick={() => openPanel('capabilities', 'toolsets')}
                    title={c.tools}
                  />
                  <SettingRow
                    detail={pending(skills.isError, skills.isPending, skills.data?.length)}
                    onClick={() => openPanel('capabilities', 'skills')}
                    title={c.skills}
                  />
                  <SettingRow onClick={() => openPanel('capabilities', 'connectors')} title={c.connectors} />
                  <SettingRow onClick={() => openPanel('capabilities', 'plugins')} title={t.skills.tabPlugins} />
                  <SettingRow
                    detail={pending(jobs.isError, jobs.isPending, jobs.data?.filter(job => job.enabled).length)}
                    onClick={() => document.getElementById('agent-routines')?.scrollIntoView({ block: 'nearest' })}
                    title={c.routines}
                  />
                </Section>
                <div id="agent-routines">
                  <Section title={c.routines}>
                    {jobs.data?.map(job => (
                      <SettingRow
                        detail={job.schedule_display || job.schedule?.display || job.schedule?.expr}
                        key={job.id}
                        onClick={() => setRoutine(job)}
                        title={job.name || job.id}
                      />
                    ))}
                    {!jobs.data?.length && (
                      <p className="agent-overview-empty">{pending(jobs.isError, jobs.isPending, c.noRoutines)}</p>
                    )}
                    <div className="agent-overview-footer">
                      <Button onClick={() => setRoutine('new')} variant="textStrong">
                        <Codicon name="add" />
                        {c.addRoutine}
                      </Button>
                    </div>
                  </Section>
                </div>
              </div>
              <div className="agent-overview-column">
                <Section title={c.channels}>
                  {(messaging.data?.platforms ?? [])
                    .filter(platform => platform.id === 'telegram' || platform.id === 'discord' || platform.configured)
                    .map(platform => (
                      <SettingRow
                        detail={platform.state === 'connected' ? c.connected : platform.configured ? c.configured : c.disconnected}
                        key={platform.id}
                        onClick={() => openPanel('messaging')}
                        title={platform.name}
                      />
                    ))}
                  {!messaging.data && (
                    <SettingRow
                      detail={pending(messaging.isError, messaging.isPending, '—')}
                      onClick={() => openPanel('messaging')}
                      title={c.connections}
                    />
                  )}
                  <SettingRow detail={value('tts.provider')} onClick={() => settings('voice')} title={c.voice} />
                </Section>
                <Section title={c.approvals}>
                  <SettingRow detail={value('approvals.mode')} onClick={() => settings('safety', 'approvals')} title={c.approvalMode} />
                  <SettingRow onClick={() => openPanel('settings', 'web-access')} title={c.access} />
                </Section>
                <Section title={c.recent}>
                  {sessionRows.length ? sessionRows : emptySessions}
                  <div className="agent-overview-footer">
                    <Button onClick={() => openPanel('history')} variant="textStrong">{c.viewHistory}</Button>
                  </div>
                </Section>
              </div>
            </div>
          )}
        </div>
      )}
      {routine && (
        <RoutineDialog
          job={routine === 'new' ? undefined : routine}
          onClose={() => setRoutine(null)}
          onSaved={() => {
            void jobs.refetch()
          }}
          owner={owner}
        />
      )}
    </Panel>
  )
}
