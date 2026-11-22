import './agent-settings.css'

import { useStore } from '@nanostores/react'
import { useQuery, type UseQueryResult } from '@tanstack/react-query'
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import { useLocation, useNavigate } from 'react-router'

import { $apiRequestScope } from '@/api/client'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { EmptyState } from '@/components/ui/empty-state'
import { FigmaAgentIcon } from '@/components/ui/figma-agent-icon'
import { PathText } from '@/components/ui/path-text'
import { RowButton } from '@/components/ui/row-button'
import { CountSkeleton } from '@/components/ui/skeleton'
import type { CronJob, MessagingPlatformsResponse } from '@/hermes'
import { useI18n } from '@/i18n'
import { sessionTitle } from '@/lib/chat-runtime'
import { platformStatusTone } from '@/lib/platform-status'
import { profileColorSoft, resolveProfileColor } from '@/lib/profile-color'
import { asText } from '@/lib/text'
import { openRosterBot } from '@/plugins/hermes-bots/roster-actions'
import { $connectionsRegistry } from '@/store/connection-registry-state'
import { $activeConnectionId, selectConnection } from '@/store/connections'
import { $fleetRoster } from '@/store/fleet-roster'
import { $activeGatewayProfile, $profileColors, $profiles, ensureGatewayProfile } from '@/store/profile'
import { $gatewayState } from '@/store/session'
import type { HermesConfigRecord, ProfileInfo } from '@/types/hermes'

import { useConnectorsCount, usePluginsCount } from '../capabilities/capability-counts'
import { useSkillsQuery } from '../capabilities/skills/skills-data'
import { useToolsetsQuery } from '../capabilities/toolsets/toolsets-data'
import { gatewayLabel } from '../chat/sidebar/gateway-group-model'
import { useSimplifiedAgents } from '../chat/sidebar/use-simplified-agents'
import { useHermesConfigRecord } from '../hooks/use-config-record'
import { openSession } from '../open-session'
import { Panel } from '../overlays/panel'
import { PageHeader, PageTabs } from '../redesign/page-header'
import { PageHero } from '../redesign/page-hero'
import { SETTINGS_ROUTE } from '../routes'
import { useRoutinesPageLinks } from '../routines/agent-routines'
import { getNested } from '../settings/helpers'
import { SettingsOwnerContext, settingsScopeLabel } from '../settings/profile-scope'
import type { SettingsPageProps } from '../settings/types'

import { AgentChannelWarnings } from './channel-warning'
import { useAgentSettingsCopy } from './copy'
import type { AgentRouteOwner } from './data'
import { AgentNavigation } from './directory'
import { AgentHistory } from './history'
import { agentMachine, rememberViewedAgent } from './landing'
import { AgentOverviewActions } from './overview-actions'
import { agentOverviewQueries } from './overview-queries'
import { useAgentSettingsEditorCopy } from './settings-copy'
import { agentCapabilityCounts } from './settings-counts'
import { AgentSettingsEditor } from './settings-editor'
import { AgentSettingsPanel, useInlineAgentSettingsMenu } from './settings-panel'
import { resolveAgentSettingsSection } from './settings-routes'

interface AgentSettingsViewProps extends SettingsPageProps {
  /** Resolved once, on entry (entry.tsx): every read, write and cache key below uses this identity. */
  owner: AgentRouteOwner
}

/** The page pins every read and write to its owner (SettingsOwnerContext, owner-scoped helpers,
 *  requestGatewayForAgent), so any agent on this window's machine renders without being activated.
 *  Another machine stays gated behind an explicit "Open agent": never silently retarget. */
export function AgentSettingsView(props: AgentSettingsViewProps) {
  const c = useAgentSettingsCopy()
  const { t } = useI18n()
  const apiScope = useStore($apiRequestScope)
  const active = useStore($activeConnectionId)
  const profile = useStore($activeGatewayProfile)
  const location = useLocation()
  const { search } = location
  const navigate = useNavigate()
  const route = location.pathname + search
  const activeRoute = useRef<string | null>(route)
  useLayoutEffect(() => {
    activeRoute.current = route

    return () => {
      activeRoute.current = null
    }
  }, [route])
  const registry = useStore($connectionsRegistry)
  const [error, setError] = useState('')
  const [opening, setOpening] = useState(false)
  const { owner } = props

  const sameConnection = owner.connectionId === agentMachine(apiScope.connectionId ?? active)
  const ownerActive = sameConnection && profile === owner.profile

  async function activate(targetProfile = owner.profile) {
    setOpening(true)
    setError('')

    try {
      if (registry?.connections.some(item => item.id === owner.connectionId)) {
        await selectConnection(owner.connectionId, { profile: targetProfile, preserveRoute: true })
      } else if (sameConnection) {
        await ensureGatewayProfile(targetProfile)
      } else {
        throw new Error(c.unavailable)
      }

      return true
    } catch (err) {
      setError(err instanceof Error ? err.message : t.errors.genericFailure)

      return false
    } finally {
      setOpening(false)
    }
  }

  if (!sameConnection) {
    // Opening another agent swaps this window's gateway profile, so it waits for a click; the page
    // keeps its frame (the Agents panel, the agent's name) around that one action.
    return (
      <Panel contentClassName="agent-management page-stage" onClose={props.onClose}>
        <AgentNavigation connectionId={owner.connectionId} profile={owner.profile} />
        <EmptyState
          action={
            <Button disabled={opening} onClick={() => void activate()}>
              {c.openAgent}
            </Button>
          }
          className="mt-6"
          description={error ? <span role="alert">{error}</span> : c.openAgentDescription}
          title={owner.profile}
        />
      </Panel>
    )
  }

  // Captured when the rename starts (the dialog holds this render's callback): a local rename
  // retires the old name's gateway, so a window that was ON this agent reopens it under the new
  // name. An agent the window was only viewing is followed by route alone, never activated.
  const reopenAfterRename = ownerActive

  async function onProfileRenamed(name: string) {
    // Only the route that initiated the rename may follow the new owner.
    if (activeRoute.current !== route || (reopenAfterRename && !(await activate(name)))) {
      return
    }

    if (activeRoute.current !== route) {
      return
    }

    const params = new URLSearchParams(search)
    params.set('agentProfile', name)
    navigate({ pathname: location.pathname, search: `?${params}` }, { replace: true })
  }

  return (
    <SettingsOwnerContext.Provider value={owner}>
      <AgentSettingsContent
        {...props}
        // The composer shows the ACTIVE agent's model: another agent's model change must not repaint it.
        onMainModelChanged={ownerActive ? props.onMainModelChanged : undefined}
        onProfileRenamed={onProfileRenamed}
      />
    </SettingsOwnerContext.Provider>
  )
}

function Section({ title, children, onOpen }: { title: string; children: ReactNode; onOpen?: () => void }) {
  return (
    <section className="agent-overview-section">
      <h2>
        {onOpen ? (
          <button className="agent-overview-section-link" onClick={onOpen} type="button">
            {title}
            <Codicon name="chevron-right" />
          </button>
        ) : (
          title
        )}
      </h2>
      <div className="agent-overview-card">{children}</div>
    </section>
  )
}

/** A value that has not arrived yet: a bar in the value's slot, so the row keeps its final size. */
const pendingValue = <CountSkeleton aria-hidden className="w-10" />

/** Rows standing in for a list that is still loading, at the rows' exact geometry. */
function SkeletonRows({ count, subtitle = false }: { count: number; subtitle?: boolean }) {
  return Array.from({ length: count }, (_, index) => (
    <div aria-hidden className="agent-overview-row" key={index}>
      <span className="min-w-0 flex-1">
        <span className="block">
          <CountSkeleton className="w-32" />
        </span>
        {subtitle && (
          <span className="agent-overview-row-subtitle mt-1 block text-xs">
            <CountSkeleton className="w-20" />
          </span>
        )}
      </span>
      <Codicon name="chevron-right" />
    </div>
  ))
}

function SettingRow({
  title,
  detail,
  subtitle,
  onClick
}: {
  title: string
  detail?: ReactNode
  subtitle?: string
  onClick: () => void
}) {
  return (
    <RowButton className="agent-overview-row" onClick={onClick}>
      <span className="min-w-0 flex-1">
        <span className="block truncate">{title}</span>
        {subtitle && (
          <span className="agent-overview-row-subtitle mt-1 block truncate text-xs text-(--ui-text-tertiary)">
            {subtitle}
          </span>
        )}
      </span>
      <span className="max-w-[50%] truncate text-(--ui-text-tertiary)">{detail}</span>
      <Codicon name="chevron-right" />
    </RowButton>
  )
}

interface AgentManagementHeaderProps {
  gateway: string
  onOpenChat: () => void
  owner: AgentRouteOwner
  profile?: ProfileInfo
  tabs: ReactNode
}

function AgentManagementHeader({ gateway, onOpenChat, owner, profile, tabs }: AgentManagementHeaderProps) {
  const c = useAgentSettingsCopy()
  const { generalPage } = useAgentSettingsEditorCopy()
  const colors = useStore($profileColors)
  const connected = useStore($gatewayState) === 'open'

  return (
    <PageHeader
      actions={
        <>
          <Button onClick={onOpenChat} size="lg">
            {c.openChat}
          </Button>
          <AgentOverviewActions owner={owner} profile={profile} />
        </>
      }
      // The machine beside the name (18563:12863): green while this window's gateway is connected.
      badge={
        <span className="agent-management-machine" data-connected={connected}>
          {gateway}
          <span className="sr-only">{connected ? c.connected : c.disconnected}</span>
        </span>
      }
      description={
        profile?.is_default && <span className="agent-management-role">{generalPage.defaultAgent}</span>
      }
      eyebrow={profile?.path && <PathText className="agent-management-path" path={profile.path} />}
      leading={
        <FigmaAgentIcon
          className="agent-management-avatar [&>img]:brightness-0 dark:[&>img]:brightness-100"
          color={profileColorSoft(resolveProfileColor(owner.profile, colors) ?? 'var(--portal-brand-fill)', 30)}
        />
      }
      tabs={tabs}
      title={profile ? settingsScopeLabel(profile) : owner.profile}
    />
  )
}

/** The agent's machine: the roster's label (what the chat opener records) and the label the chat
 *  sidebar shows ("Hermes host"), never the raw serving address. */
function useAgentMachine(owner: AgentRouteOwner) {
  const { t } = useI18n()
  const roster = useStore($fleetRoster)
  const registry = useStore($connectionsRegistry)
  const { agents } = useSimplifiedAgents([])

  const gateway =
    roster?.sources.find(source => source.connectionId === owner.connectionId)?.label ??
    gatewayLabel(owner.connectionId, registry, t.settings.gateway.webappHostTitle)

  const machine =
    agents.find(agent => agent.profile === owner.profile && agentMachine(agent.connectionId) === owner.connectionId)
      ?.gateway ?? gateway

  return { gateway, machine }
}

/** Recent activity before (or without) rows: the query asks for five, so loading reserves five and
 *  the column below never jumps when they land. */
function RecentPlaceholder({ recent }: { recent: UseQueryResult<unknown> }) {
  const c = useAgentSettingsCopy()

  return recent.isPending ? (
    <SkeletonRows count={5} subtitle />
  ) : (
    <p className="agent-overview-empty">{recent.isError ? c.unavailable : c.noSessions}</p>
  )
}

/** The agent page's URL for a tab (`panel`) and settings page; unset parts are left out. */
function agentPanelHref(owner: AgentRouteOwner, parts: Record<'page' | 'panel' | 'tab', null | string | undefined>) {
  const query = new URLSearchParams({ agentProfile: owner.profile })
  const entries = Object.entries({ connection: owner.connectionId, ...parts })

  for (const [key, value] of entries) {
    if (value) {
      query.set(key, value)
    }
  }

  return `${SETTINGS_ROUTE}?${query}`
}

/** Everything the agent page reads about its agent, each read enabled only for the tabs that
 *  show it (the settings menu counts routines and channels; History needs neither). */
function useAgentPageData(owner: AgentRouteOwner, panel: null | string) {
  const config = useHermesConfigRecord(owner)
  const skills = useSkillsQuery(owner)
  const tools = useToolsetsQuery(owner)
  const connectors = useConnectorsCount(owner)
  const plugins = usePluginsCount(owner)

  const jobs = useQuery({
    ...agentOverviewQueries.routines(owner),
    // The settings sub-nav counts routines and channels too; History needs neither.
    enabled: panel !== 'history'
  })

  const messaging = useQuery({
    ...agentOverviewQueries.messaging(owner),
    enabled: panel !== 'history'
  })

  const recent = useQuery({
    ...agentOverviewQueries.recent(owner),
    enabled: !panel
  })

  const soul = useQuery({
    ...agentOverviewQueries.soul(owner),
    enabled: !panel
  })

  return { config, connectors, jobs, messaging, plugins, recent, skills, soul, tools }
}

function AgentSettingsContent({
  owner,
  ...props
}: AgentSettingsViewProps & { onProfileRenamed: (name: string) => Promise<void> }) {
  const { t, locale } = useI18n()
  const c = useAgentSettingsCopy()
  const navigate = useNavigate()
  const { search } = useLocation()
  const params = new URLSearchParams(search)
  const panel = params.get('panel')
  const profiles = useStore($profiles)
  const profile = profiles.find(item => item.name === owner.profile)
  const roster = useStore($fleetRoster)

  const { config, connectors, jobs, messaging, plugins, recent, skills, soul, tools } = useAgentPageData(owner, panel)
  const history = panel === 'history'

  const configText = (path: string) => asText(getNested(config.data ?? {}, path))

  const pending = (failed: boolean, loading: boolean, detail: ReactNode) =>
    failed ? c.unavailable : loading ? pendingValue : detail

  const value = (path: string) => pending(config.isError, config.isPending, configText(path) || '—')

  function openPanel(next: string | null, tab?: string, page?: string) {
    navigate(agentPanelHref(owner, { panel: next, tab, page }))
  }

  const settings = (tab: string, page?: string) => openPanel('settings', `config:${tab}`, page)
  const back = () => openPanel(null)

  const { connectionId } = owner
  const { gateway, machine } = useAgentMachine(owner)

  // The Agents destination reopens the agent this window last showed.
  useEffect(() => rememberViewedAgent(owner), [owner])

  const failed =
    [config, skills, tools, jobs, messaging, recent, soul].some(query => query.isError) ||
    connectors.failed ||
    plugins.failed

  // The endpoint backfills pinned sessions on each page; render each session once.
  const sessions = useMemo(() => {
    const unique = new Map(recent.data?.sessions.map(session => [session.id, session]))

    return [...unique.values()].filter(session => !session.hidden && !session.archived)
  }, [recent.data])

  const selectedTab = history ? 'history' : panel ? 'settings' : 'overview'
  const inlineMenu = useInlineAgentSettingsMenu()

  const counts = agentCapabilityCounts({
    connectors: connectors.value,
    plugins: plugins.value,
    jobs: jobs.data,
    messaging: messaging.data,
    skills: skills.data,
    tools: tools.data
  })

  function openChat() {
    // Dismiss the management route before the canonical opener owns navigation.
    props.onClose()
    void openRosterBot({
      name: owner.profile,
      connectionId,
      connectionLabel: gateway,
      targetProfile:
        roster?.agents.find(agent => agent.connectionId === connectionId && agent.profile === owner.profile)
          ?.targetProfile || owner.profile,
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
      subtitle={
        session.last_active
          ? new Date(session.last_active * 1000).toLocaleString(locale, { dateStyle: 'medium', timeStyle: 'short' })
          : undefined
      }
      title={sessionTitle(session)}
    />
  ))

  const emptySessions = <RecentPlaceholder recent={recent} />

  return (
    <Panel contentClassName="agent-management page-stage" onClose={props.onClose}>
      {selectedTab === 'settings' ? (
        <AgentSettingsPanel
          counts={counts.nav}
          isDefault={Boolean(profile?.is_default)}
          label={profile ? settingsScopeLabel(profile) : owner.profile}
          onBack={back}
          profile={owner.profile}
          section={resolveAgentSettingsSection(params)}
        />
      ) : (
        <AgentNavigation connectionId={connectionId} profile={owner.profile} />
      )}
      <PageHero art="tide" />
      <AgentManagementHeader
        gateway={machine}
        onOpenChat={openChat}
        owner={owner}
        profile={profile}
        tabs={
          <PageTabs
            label={t.simplifiedAgents.settings}
            onChange={id => openPanel(id === 'overview' ? null : id)}
            tabs={[
              { id: 'overview', label: c.overview },
              { id: 'history', label: c.history },
              { id: 'settings', label: t.commandCenter.settings }
            ]}
            value={selectedTab}
            variant="segmented"
          />
        }
      />
      {history ? (
        <AgentHistory key={JSON.stringify(owner)} onClose={props.onClose} owner={owner} />
      ) : panel === 'settings' || panel === 'capabilities' || panel === 'messaging' ? (
        <div className="agent-management-editor">
          <AgentSettingsEditor {...props} counts={counts} inlineNavigation={inlineMenu} onClose={back} owner={owner} />
        </div>
      ) : (
        <div className="agent-management-scroll scroll-fade-y">
          {failed && (
            <div className="agent-management-error" role="alert">
              <span>{c.unavailable}</span>
              <Button
                onClick={() => {
                  void Promise.all([
                    config.refetch(),
                    skills.refetch(),
                    tools.refetch(),
                    jobs.refetch(),
                    messaging.refetch(),
                    recent.refetch(),
                    soul.refetch()
                  ])
                  connectors.retry()
                  plugins.retry()
                }}
                variant="secondary"
              >
                {c.retry}
              </Button>
            </div>
          )}
          <AgentChannelWarnings messaging={messaging.data} />
          <div className="agent-overview-grid">
            <div className="agent-overview-column">
              <AgentInstructions
                content={soul.data?.content}
                loading={soul.isPending}
                onOpen={() => openPanel('settings', 'agent:instructions')}
                path={profile?.path}
                status={soul.isError ? c.unavailable : t.profiles.notSet}
              />
              <AgentBehavior
                config={config}
                model={value(configText('model.default') ? 'model.default' : 'model')}
                onSettings={settings}
                value={value}
              />
              <Section onOpen={() => openPanel('capabilities')} title={c.capabilities}>
                {/* One count per capability, the same number the settings menu shows (settings-counts.ts). */}
                <SettingRow
                  detail={pending(tools.isError, tools.isPending, counts.tools?.toolsets)}
                  onClick={() => openPanel('capabilities', 'toolsets')}
                  title={c.tools}
                />
                <SettingRow
                  detail={pending(skills.isError, skills.isPending, counts.skills)}
                  onClick={() => openPanel('capabilities', 'skills')}
                  title={c.skills}
                />
                <SettingRow
                  detail={pending(connectors.failed, counts.nav.connectors === undefined, counts.nav.connectors)}
                  onClick={() => openPanel('capabilities', 'connectors')}
                  title={c.connectors}
                />
                <SettingRow
                  detail={pending(plugins.failed, counts.nav.plugins === undefined, counts.nav.plugins)}
                  onClick={() => openPanel('capabilities', 'plugins')}
                  title={t.skills.tabPlugins}
                />
                <SettingRow
                  detail={pending(jobs.isError, jobs.isPending, counts.nav.schedules)}
                  onClick={() => document.getElementById('agent-routines')?.scrollIntoView({ block: 'nearest' })}
                  title={t.cron.title}
                />
              </Section>
              <AgentRoutines jobs={jobs} owner={owner} />
            </div>
            <div className="agent-overview-column">
              <AgentChannels messaging={messaging} onOpen={() => openPanel('messaging')}>
                <SettingRow detail={value('tts.provider')} onClick={() => settings('voice')} title={c.voice} />
              </AgentChannels>
              <Section onOpen={() => settings('safety')} title={c.approvals}>
                <SettingRow
                  detail={value('approvals.mode')}
                  onClick={() => settings('safety', 'approvals')}
                  title={c.approvalMode}
                />
              </Section>
              <Section onOpen={() => openPanel('history')} title={c.recent}>
                {sessionRows.length ? sessionRows : emptySessions}
                <div className="agent-overview-footer">
                  <Button onClick={() => openPanel('history')} variant="textStrong">
                    {c.viewHistory}
                  </Button>
                </div>
              </Section>
            </div>
          </div>
        </div>
      )}
    </Panel>
  )
}

interface AgentInstructionsProps {
  content?: string
  loading: boolean
  path?: string
  status: ReactNode
  onOpen: () => void
}

function AgentInstructions({ content, loading, path, status, onOpen }: AgentInstructionsProps) {
  const c = useAgentSettingsCopy()

  return (
    <Section onOpen={onOpen} title={c.instructions}>
      <div className="agent-instructions-preview">
        {loading ? (
          // The preview clamps to three lines; hold all three while SOUL.md loads.
          <p aria-hidden className="agent-instructions-skeleton">
            <CountSkeleton className="w-full" />
            <CountSkeleton className="w-full" />
            <CountSkeleton className="w-2/3" />
          </p>
        ) : (
          <p>{content || status}</p>
        )}
        <span className="agent-instructions-source">
          SOUL.md
          {path && (
            <>
              <span aria-hidden>·</span>
              <PathText path={path} />
            </>
          )}
        </span>
      </div>
    </Section>
  )
}

interface AgentRoutinesProps {
  jobs: UseQueryResult<CronJob[]>
  owner: AgentRouteOwner
}

/** The overview's routines, read only: each opens on the Routines page, where routines are created and
 *  changed, and New routine starts one there for this agent. */
function AgentRoutines({ jobs, owner }: AgentRoutinesProps) {
  const { t } = useI18n()
  const c = useAgentSettingsCopy()
  const links = useRoutinesPageLinks(owner)

  return (
    <div id="agent-routines">
      <Section title={t.cron.title}>
        {jobs.data?.map(job => (
          <SettingRow
            detail={job.schedule_display || job.schedule?.display || job.schedule?.expr}
            key={job.id}
            onClick={() => links.open(job)}
            title={job.name || job.id}
          />
        ))}
        {jobs.isPending ? (
          <SkeletonRows count={1} />
        ) : (
          !jobs.data?.length && (
            <p className="agent-overview-empty">{jobs.isError ? c.unavailable : t.cron.emptyTitleNew}</p>
          )
        )}
        <div className="agent-overview-footer">
          <Button onClick={links.create} variant="textStrong">
            <Codicon name="add" />
            {t.cron.newCron}
          </Button>
        </div>
      </Section>
    </div>
  )
}

interface AgentChannelsProps {
  messaging: UseQueryResult<MessagingPlatformsResponse>
  onOpen: () => void
  children: ReactNode
}

/** Messaging status stays owner-scoped; the caller supplies the separate voice setting. */
function AgentChannels({ messaging, onOpen, children }: AgentChannelsProps) {
  const { t } = useI18n()
  const c = useAgentSettingsCopy()

  return (
    <Section onOpen={onOpen} title={c.channels}>
      {(messaging.data?.platforms ?? [])
        .filter(platform => platform.id === 'telegram' || platform.id === 'discord' || platform.configured)
        .map(platform => (
          <SettingRow
            detail={
              platform.state === 'connected' ? (
                <span className="agent-channel-status" data-tone="connected">
                  {c.connected}
                </span>
              ) : platform.configured && platform.enabled && ['bad', 'warn'].includes(platformStatusTone(platform)) ? (
                <span className="agent-channel-status" data-tone="disconnected">
                  {c.disconnected}
                </span>
              ) : platform.configured ? (
                c.configured
              ) : (
                <span className="agent-channel-status" data-tone="disconnected">
                  {c.disconnected}
                </span>
              )
            }
            key={platform.id}
            onClick={onOpen}
            title={platform.name}
          />
        ))}
      {/* Telegram and Discord are always listed, so loading holds their two rows. */}
      {messaging.isPending && <SkeletonRows count={2} />}
      {messaging.isError && <SettingRow detail={c.unavailable} onClick={onOpen} title={c.connections} />}
      {children}
    </Section>
  )
}

interface AgentBehaviorProps {
  config: UseQueryResult<HermesConfigRecord>
  model: ReactNode
  value: (path: string) => ReactNode
  onSettings: (tab: string, page?: string) => void
}

function AgentBehavior({ config, model, value, onSettings }: AgentBehaviorProps) {
  const c = useAgentSettingsCopy()
  const fallbacks = getNested(config.data ?? {}, 'fallback_providers')

  return (
    <Section onOpen={() => onSettings('model')} title={c.behavior}>
      <SettingRow detail={model} onClick={() => onSettings('model', 'main')} title={c.model} />
      <SettingRow
        detail={
          config.isError
            ? c.unavailable
            : config.isPending
              ? pendingValue
              : Array.isArray(fallbacks)
                ? fallbacks.length
                : '—'
        }
        onClick={() => onSettings('model', 'fallbacks')}
        title={c.fallbacks}
      />
      <SettingRow detail={value('memory.provider')} onClick={() => onSettings('memory')} title={c.memory} />
    </Section>
  )
}
