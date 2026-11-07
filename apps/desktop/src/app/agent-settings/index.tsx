import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'
import { useEffect, useState } from 'react'
import type { ReactNode } from 'react'
import { useLocation, useNavigate } from 'react-router'

import { $apiRequestScope } from '@/api/client'
import { PageLoader } from '@/components/page-loader'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { ProfileGlyph } from '@/components/ui/profile-glyph'
import { RowButton } from '@/components/ui/row-button'
import type { CronJob } from '@/hermes'
import { useI18n } from '@/i18n'
import { sessionTitle } from '@/lib/chat-runtime'
import { resolveProfileColor } from '@/lib/profile-color'
import { asText } from '@/lib/text'
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
import { Panel, PanelHeader, PanelSectionLabel } from '../overlays/panel'
import { SETTINGS_ROUTE } from '../routes'
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
    <section className="grid content-start gap-2">
      <PanelSectionLabel>{title}</PanelSectionLabel>
      <div className="rounded-md bg-(--ui-bg-tertiary)/40 p-1">{children}</div>
    </section>
  )
}

function SettingRow({ title, detail, onClick }: { title: string; detail?: ReactNode; onClick: () => void }) {
  return (
    <RowButton
      className="flex w-full items-center gap-3 rounded-sm px-3 py-2.5 text-left text-xs hover:bg-(--chrome-action-hover)"
      onClick={onClick}
    >
      <span className="min-w-0 flex-1 truncate">{title}</span>
      <span className="max-w-[50%] truncate text-(--ui-text-tertiary)">{detail}</span>
      <Codicon name="chevron-right" />
    </RowButton>
  )
}

function AgentSettingsContent({ owner, ...props }: AgentSettingsViewProps) {
  const { t } = useI18n()
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

  const recent = useQuery({
    queryKey: [...agentSettingsKey(owner), 'recent'],
    queryFn: () => getAgentRecentSessions(owner)
  })

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

  if (panel === 'settings') {return <ClassicSettingsView {...props} onClose={back} />}

  if (panel === 'capabilities' || panel === 'messaging')
    {return (
      <Panel onClose={back}>
        <PanelHeader subtitle={gateway} title={label} />
        <Button className="mb-3 w-fit" onClick={back} variant="ghost">
          <Codicon name="arrow-left" />
          {t.simplifiedAgents.settings}
        </Button>
        <div className="min-h-0 flex-1 overflow-auto">
          {panel === 'capabilities' ? (
            <CapabilitiesView fixedConnection={owner.connectionId} fixedProfile={owner.profile} />
          ) : (
            <MessagingView />
          )}
        </div>
      </Panel>
    )}

  const fallbacks = getNested(config.data ?? {}, 'fallback_providers')
  const failed = [config, skills, tools, jobs, messaging, recent].some(query => query.isError)

  return (
    <Panel onClose={props.onClose}>
      <Button className="mb-3 w-fit" onClick={props.onClose} variant="ghost">
        <Codicon name="arrow-left" />
        {c.back}
      </Button>
      <PanelHeader
        subtitle={gateway}
        title={
          <span className="flex items-center gap-2">
            <ProfileGlyph color={resolveProfileColor(owner.profile, colors)} isDefault={owner.profile === 'default'} name={label} />
            <span>{label}</span>
          </span>
        }
      />
      {failed && (
        <div className="mb-4 flex items-center gap-2" role="alert">
          <span className="text-xs">{c.unavailable}</span>
          <Button
            onClick={() => {
              void Promise.all([
                config.refetch(),
                skills.refetch(),
                tools.refetch(),
                jobs.refetch(),
                messaging.refetch(),
                recent.refetch()
              ])
            }}
            variant="secondary"
          >
            {c.retry}
          </Button>
        </div>
      )}
      <div className="grid min-h-0 flex-1 gap-8 overflow-auto py-4 md:grid-cols-2">
        <div className="grid content-start gap-6">
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
          </Section>
          <Section title={c.memory}>
            <SettingRow detail={value('memory.provider')} onClick={() => settings('memory')} title={c.provider} />
          </Section>
          <Section title={c.connections}>
            {(messaging.data?.platforms ?? [])
              .filter(platform => platform.id === 'telegram' || platform.id === 'discord' || platform.configured)
              .map(platform => (
                <SettingRow
                  detail={
                    platform.state === 'connected' ? c.connected : platform.configured ? c.configured : c.disconnected
                  }
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
          <Section title={c.model}>
            <SettingRow
              detail={value('model.default') === '—' ? value('model') : value('model.default')}
              onClick={() => settings('model', 'main')}
              title={c.primary}
            />
            <SettingRow
              detail={config.isError ? c.unavailable : Array.isArray(fallbacks) ? fallbacks.length : '—'}
              onClick={() => settings('model', 'fallbacks')}
              title={c.fallbacks}
            />
          </Section>
        </div>
        <div className="grid content-start gap-6">
          <Section title={c.routines}>
            <Button className="mb-2" onClick={() => setRoutine('new')} variant="secondary">
              <Codicon name="add" />
              {c.addRoutine}
            </Button>
            {jobs.data?.map(job => (
              <SettingRow
                detail={job.schedule_display || job.schedule?.display || job.schedule?.expr}
                key={job.id}
                onClick={() => setRoutine(job)}
                title={job.name || job.id}
              />
            ))}
            {!jobs.data?.length && (
              <p className="px-3 py-2 text-xs text-(--ui-text-tertiary)">
                {pending(jobs.isError, jobs.isPending, c.noRoutines)}
              </p>
            )}
          </Section>
          <Section title={c.approvals}>
            <SettingRow
              detail={value('approvals.mode')}
              onClick={() => settings('safety', 'approvals')}
              title={c.approvalMode}
            />
            <SettingRow onClick={() => openPanel('settings', 'web-access')} title={c.access} />
          </Section>
          <Section title={c.recent}>
            {recent.data?.sessions
              .filter(session => !session.hidden && !session.archived)
              .map(session => (
                <SettingRow
                  key={session.id}
                  onClick={() =>
                    openSession(session.id, navigate, 'in-place', {
                      workspaceMode: 'sessions',
                      ...(owner.connectionId
                        ? { ownerRoute: { connectionId: owner.connectionId, profile: owner.profile } }
                        : {})
                    })
                  }
                  title={sessionTitle(session)}
                />
              ))}
            {!recent.data?.sessions.filter(session => !session.hidden && !session.archived).length && (
              <p className="px-3 py-2 text-xs text-(--ui-text-tertiary)">
                {pending(recent.isError, recent.isPending, c.noSessions)}
              </p>
            )}
          </Section>
        </div>
      </div>
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
