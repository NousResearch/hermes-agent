import './settings-editor.css'

import { type ReactNode, useMemo, useRef } from 'react'
import { useLocation, useNavigate } from 'react-router'

import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'

import { CapabilitiesView } from '../capabilities'
import { toolsetKeyGroup, useToolsetsQuery } from '../capabilities/toolsets/toolsets-data'
import { useAgentCapabilityCopy } from '../capabilities/ui/agent-capability-copy'
import { MessagingView } from '../messaging'
import { BrowserRealProfilePanel } from '../settings/browser-real-profile-panel'
import { ConfigSettings } from '../settings/config-settings'
import { KeysSettings } from '../settings/keys-settings'
import { PluginSettingsPane, usePluginSettingsRoute } from '../settings/plugin-settings'
import { SettingsBreadcrumbContext } from '../settings/primitives'
import { SettingsOwnerContext } from '../settings/profile-scope'
import { PROVIDER_VIEWS, ProvidersSettings, type ProviderView } from '../settings/providers-settings'
import { SettingsDisclosure } from '../settings/settings-card'
import type { SettingsPageProps } from '../settings/types'
import { VaultSettings } from '../settings/vault-settings'

import { useAgentSettingsCopy } from './copy'
import type { AgentSettingsOwner } from './data'
import { useAgentSettingsEditorCopy } from './settings-copy'
import type { AgentCapabilityCounts } from './settings-counts'
import { AgentGeneralSettings, AgentInstructionsSettings } from './settings-general'
import { useAgentSettingsLayoutCopy } from './settings-layout-copy'
import {
  AGENT_SETTINGS_GROUPS,
  agentSettingsSearch,
  type AgentSettingsSection,
  resolveAgentSettingsSection
} from './settings-routes'
import { AgentSchedulesSettings } from './settings-schedules'

interface AgentSettingsEditorProps extends SettingsPageProps {
  owner: AgentSettingsOwner
  onProfileRenamed: (name: string) => Promise<void>
  /** Counts from data the agent page already holds; absent ones stay blank. */
  counts?: AgentCapabilityCounts
}

type AgentSettingsLabels = Record<
  AgentSettingsSection | 'agent' | 'behavior' | 'capabilities' | 'access' | 'runtime',
  string
>

/** Mounted only by the redesign's owner gate. All embedded controls share that immutable owner. */
export function AgentSettingsEditor(props: AgentSettingsEditorProps) {
  const owner = useMemo(
    () => ({ connectionId: props.owner.connectionId ?? 'local', profile: props.owner.profile }),
    [props.owner.connectionId, props.owner.profile]
  )

  return (
    <SettingsOwnerContext.Provider value={owner}>
      <SettingsBreadcrumbContext.Provider value>
        <AgentSettingsEditorContent {...props} key={JSON.stringify(owner)} owner={owner} />
      </SettingsBreadcrumbContext.Provider>
    </SettingsOwnerContext.Provider>
  )
}

function AgentSettingsEditorContent({
  owner,
  counts,
  onClose,
  onConfigSaved,
  onMainModelChanged,
  onProfileRenamed
}: AgentSettingsEditorProps) {
  const { t } = useI18n()
  const c = useAgentSettingsEditorCopy()
  const a = useAgentSettingsCopy()
  const layout = useAgentSettingsLayoutCopy()
  const location = useLocation()
  const navigate = useNavigate()
  const params = new URLSearchParams(location.search)
  const section = resolveAgentSettingsSection(params)
  const page = params.get('page') ?? undefined
  const tab = params.get('tab')
  const importInputRef = useRef<HTMLInputElement | null>(null)
  const bodyRef = useRef<HTMLElement | null>(null)

  const labels: AgentSettingsLabels = {
    agent: t.sidebar.filter.profile,
    behavior: a.behavior,
    capabilities: a.capabilities,
    access: c.access,
    runtime: c.runtime,
    general: c.general,
    instructions: a.instructions,
    model: a.model,
    chat: t.settings.sections.chat,
    memory: t.settings.sections.memory,
    voice: a.voice,
    tools: a.tools,
    skills: a.skills,
    connectors: a.connectors,
    plugins: t.skills.tabPlugins,
    messaging: t.sidebar.nav.messaging,
    schedules: t.cron.title,
    safety: c.safety,
    browser: c.browser,
    providers: c.providers,
    workspace: t.settings.sections.workspace,
    limits: c.limits,
    'tools-advanced': a.tools
  }

  function open(next: AgentSettingsSection, preserveScroll = false) {
    if (bodyRef.current && !preserveScroll) {
      bodyRef.current.scrollTop = 0
    }

    navigate({ pathname: location.pathname, search: agentSettingsSearch(location.search, next) }, { replace: true })
  }

  function openPage(key: string, value: string) {
    if (bodyRef.current) {
      bodyRef.current.scrollTop = 0
    }

    const next = new URLSearchParams(location.search)
    next.set(key, value)
    navigate({ pathname: location.pathname, search: `?${next}` }, { replace: true })
  }

  const configSection = resolveConfigSection(section)

  return (
    <div className="agent-settings-editor">
      <AgentSettingsNavigation
        ariaLabel={t.commandCenter.settings}
        counts={counts?.nav}
        labels={labels}
        onOpen={open}
        section={section}
      />
      <section aria-label={t.commandCenter.settings} className="agent-settings-body" ref={bodyRef}>
        <AgentSettingsHeading counts={counts} labels={labels} section={section} />
        {section === 'general' && <AgentGeneralSettings onProfileRenamed={onProfileRenamed} owner={owner} />}
        {section === 'instructions' && <AgentInstructionsSettings owner={owner} />}
        {configSection && (
          <ConfigSettings
            activeSectionId={configSection}
            agentLayout
            agentModel={section === 'model'}
            importInputRef={importInputRef}
            onConfigSaved={onConfigSaved}
            onMainModelChanged={onMainModelChanged}
            subpage={section === 'model' ? page : undefined}
          />
        )}
        {section && ['tools', 'tools-advanced', 'skills', 'connectors', 'plugins'].includes(section) && (
          <AgentCapabilitiesSettings
            onOpenAdvanced={() => open(section === 'tools-advanced' ? 'tools' : 'tools-advanced', true)}
            owner={owner}
            params={params}
            section={section}
          >
            {(section === 'tools' || section === 'tools-advanced') && (
              <>
                <AgentToolKeys owner={owner} />
                <ConfigSettings
                  activeSectionId="advanced"
                  importInputRef={importInputRef}
                  onConfigSaved={onConfigSaved}
                  subpage="tools"
                />
              </>
            )}
          </AgentCapabilitiesSettings>
        )}
        {section === 'messaging' && <MessagingView className="agent-settings-messaging" />}
        {section === 'schedules' && <AgentSchedulesSettings owner={owner} />}
        {section === 'browser' && (
          <>
            <VaultSettings subpage={tab === 'vault' ? page : undefined} />
            <SettingsDisclosure
              description={layout.summaries.browser}
              open={tab === 'config:browser' && Boolean(page)}
              title={t.settings.sections.advanced}
            >
              <BrowserRealProfilePanel profile={owner} />
              <ConfigSettings
                activeSectionId="browser"
                importInputRef={importInputRef}
                onConfigSaved={onConfigSaved}
                subpage="network"
              />
            </SettingsDisclosure>
          </>
        )}
        {section === 'providers' && (
          <AgentProviderSettings
            onClose={onClose}
            onConfigSaved={onConfigSaved}
            onMainModelChanged={onMainModelChanged}
            onViewChange={view =>
              view === 'local' ? navigate('/settings?tab=providers&pview=local') : openPage('pview', view)
            }
            params={params}
          />
        )}
      </section>
    </div>
  )
}

interface AgentSettingsHeadingProps {
  labels: AgentSettingsLabels
  section: AgentSettingsSection | null
  counts?: AgentCapabilityCounts
}

function AgentSettingsHeading({ labels, section, counts }: AgentSettingsHeadingProps) {
  const { t } = useI18n()
  const c = useAgentSettingsEditorCopy()
  const capability = useAgentCapabilityCopy()
  const page = section === 'tools-advanced' ? 'tools' : section

  // Tools and Skills lead with what this agent has on (895:14659); the rest keep their fixed line.
  const summary =
    page === 'tools' && counts?.tools
      ? capability.tools.summary(counts.tools.toolsets, counts.tools.tools)
      : page === 'skills' && counts?.skills !== undefined
        ? capability.skills.summary(counts.skills)
        : page && c.descriptions[page]

  return (
    <>
      <header>
        <h2>{section ? labels[section] : t.commandCenter.settings}</h2>
        {summary && <p>{summary}</p>}
      </header>
      {!section && <p role="alert">{c.invalid}</p>}
    </>
  )
}

function resolveConfigSection(section: AgentSettingsSection | null) {
  return section && ['model', 'chat', 'memory', 'voice', 'safety', 'workspace', 'limits'].includes(section)
    ? section === 'limits'
      ? 'advanced'
      : section
    : null
}

interface AgentCapabilitiesSettingsProps {
  owner: AgentSettingsOwner
  section: AgentSettingsSection
  params: URLSearchParams
  onOpenAdvanced: () => void
  children: ReactNode
}

function AgentCapabilitiesSettings({
  owner,
  section,
  params,
  onOpenAdvanced,
  children
}: AgentCapabilitiesSettingsProps) {
  const { t } = useI18n()
  const layout = useAgentSettingsLayoutCopy()
  const tools = section === 'tools' || section === 'tools-advanced'

  return (
    <>
      <CapabilitiesView
        className="agent-settings-capabilities"
        embedded
        fixedConnection={owner.connectionId}
        fixedMode={tools ? 'toolsets' : (section as 'skills' | 'connectors' | 'plugins')}
        fixedProfile={owner.profile}
      />
      {tools && (
        // The route owns the expanded state (tab=agent:tools-advanced): opening follows a deep link,
        // and the user's toggle writes back to it.
        <SettingsDisclosure
          description={layout.summaries.tools}
          onOpenChange={next => next !== (section === 'tools-advanced') && onOpenAdvanced()}
          open={section === 'tools-advanced'}
          title={t.settings.sections.advanced}
        >
          {children}
        </SettingsDisclosure>
      )}
      {section === 'plugins' && (params.get('plugin') || params.get('agent')) && <AgentPluginSettings />}
    </>
  )
}

/** The tool API keys, grouped by the toolset that calls each (895:23456). */
function AgentToolKeys({ owner }: { owner: AgentSettingsOwner }) {
  const copy = useAgentCapabilityCopy()
  const { data: toolsets } = useToolsetsQuery(owner)

  const groupBy = useMemo(
    () => (toolsets ? toolsetKeyGroup(toolsets, copy.tools.otherKeys) : undefined),
    [copy.tools.otherKeys, toolsets]
  )

  return <KeysSettings groupBy={groupBy} view="tools" />
}

interface AgentProviderSettingsProps extends SettingsPageProps {
  params: URLSearchParams
  onViewChange: (view: ProviderView) => void
}

// Providers & Keys (Figma 895:21016): accounts, then API keys, then custom endpoints under Advanced.
const PROVIDER_SECTIONS = ['accounts', 'keys'] as const

function AgentProviderSettings({
  params,
  onClose,
  onConfigSaved,
  onMainModelChanged,
  onViewChange
}: AgentProviderSettingsProps) {
  const { t } = useI18n()
  const c = useAgentSettingsEditorCopy()
  const layout = useAgentSettingsLayoutCopy()
  const requested = params.get('pview')
  const known = !requested || PROVIDER_VIEWS.includes(requested as ProviderView)

  const titles: Record<(typeof PROVIDER_SECTIONS)[number], string> = {
    accounts: t.settings.nav.providerAccounts,
    keys: t.settings.nav.providerApiKeys
  }

  // Every view is on this page, so a cross-link ("Have an API key instead?") moves to its section;
  // local models keep their own global page.
  const showView = (view: ProviderView) =>
    view === 'local'
      ? onViewChange(view)
      : document.getElementById(`agent-providers-${view}`)?.scrollIntoView({ block: 'start' })

  const providers = (view: ProviderView) => (
    <ProvidersSettings
      onClose={onClose}
      onConfigSaved={onConfigSaved}
      onMainModelChanged={onMainModelChanged}
      onViewChange={showView}
      view={view}
    />
  )

  if (!known) {
    return <p role="alert">{c.invalid}</p>
  }

  return (
    <>
      {PROVIDER_SECTIONS.map(view => (
        <section className="agent-settings-group" id={`agent-providers-${view}`} key={view}>
          <h3>{titles[view]}</h3>
          {providers(view)}
        </section>
      ))}
      <SettingsDisclosure
        description={layout.summaries.providers}
        open={requested === 'custom-endpoints'}
        title={t.settings.sections.advanced}
      >
        {providers('custom-endpoints')}
      </SettingsDisclosure>
      {requested === 'local' && (
        <Button onClick={() => onViewChange('local')}>{t.settings.nav.providerLocalModels}</Button>
      )}
    </>
  )
}

function AgentPluginSettings() {
  const plugins = usePluginSettingsRoute(true)

  return (
    <PluginSettingsPane
      entries={plugins.entries}
      missing={plugins.missing}
      onOpen={plugins.open}
      pending={plugins.pending}
      target={plugins.target}
    />
  )
}

interface AgentSettingsNavigationProps {
  ariaLabel: string
  labels: AgentSettingsLabels
  section: AgentSettingsSection | null
  counts?: AgentCapabilityCounts['nav']
  onOpen: (section: AgentSettingsSection) => void
}

function AgentSettingsNavigation({ ariaLabel, labels, section, counts, onOpen }: AgentSettingsNavigationProps) {
  return (
    <nav aria-label={ariaLabel} className="agent-settings-navigation">
      {AGENT_SETTINGS_GROUPS.map(([group, sections]) => (
        <div className="agent-settings-nav-group" key={group}>
          <h2>{labels[group]}</h2>
          {sections.map(id => (
            <button
              aria-current={section === id || (id === 'tools' && section === 'tools-advanced') ? 'page' : undefined}
              aria-label={counts?.[id] ? `${labels[id]}, ${counts[id]}` : undefined}
              key={id}
              onClick={() => onOpen(id)}
              type="button"
            >
              <span>{labels[id]}</span>
              {/* Zero reads as noise in a nav; the page itself says what is off. */}
              {Boolean(counts?.[id]) && <span className="agent-settings-nav-count">{counts?.[id]}</span>}
            </button>
          ))}
        </div>
      ))}
    </nav>
  )
}
