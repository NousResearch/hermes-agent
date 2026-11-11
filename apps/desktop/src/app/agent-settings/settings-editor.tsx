import './settings-editor.css'

import { type ReactNode, useMemo, useRef } from 'react'
import { useLocation, useNavigate } from 'react-router'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { useI18n } from '@/i18n'

import { CapabilitiesView } from '../capabilities'
import { MessagingView } from '../messaging'
import { BrowserRealProfilePanel } from '../settings/browser-real-profile-panel'
import { ConfigSettings } from '../settings/config-settings'
import { CONFIG_SUBPAGES } from '../settings/config-subpages'
import { KeysSettings } from '../settings/keys-settings'
import { PluginSettingsPane, usePluginSettingsRoute } from '../settings/plugin-settings'
import { SettingsBreadcrumbContext } from '../settings/primitives'
import { SettingsOwnerContext } from '../settings/profile-scope'
import { PROVIDER_VIEWS, ProvidersSettings, type ProviderView } from '../settings/providers-settings'
import type { SettingsPageProps } from '../settings/types'
import { VaultSettings } from '../settings/vault-settings'

import { useAgentSettingsCopy } from './copy'
import type { AgentSettingsOwner } from './data'
import { useAgentSettingsEditorCopy } from './settings-copy'
import { AgentGeneralSettings, AgentInstructionsSettings } from './settings-general'
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
  onClose,
  onConfigSaved,
  onMainModelChanged,
  onProfileRenamed
}: AgentSettingsEditorProps) {
  const { t } = useI18n()
  const c = useAgentSettingsEditorCopy()
  const a = useAgentSettingsCopy()
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

  const runtimePage = page ?? 'runtime'

  return (
    <div className="agent-settings-editor">
      <AgentSettingsNavigation ariaLabel={t.commandCenter.settings} labels={labels} onOpen={open} section={section} />
      <section aria-label={t.commandCenter.settings} className="agent-settings-body" ref={bodyRef}>
        <AgentSettingsHeading labels={labels} section={section} />
        {section === 'general' && <AgentGeneralSettings onProfileRenamed={onProfileRenamed} owner={owner} />}
        {section === 'instructions' && <AgentInstructionsSettings owner={owner} />}
        {section === 'limits' && (
          <div className="agent-settings-subpages">
            {CONFIG_SUBPAGES.advanced
              .filter(item => ['runtime', 'terminal', 'output'].includes(item.id))
              .map(item => (
                <Button
                  aria-pressed={runtimePage === item.id}
                  key={item.id}
                  onClick={() => openPage('page', item.id)}
                  variant="secondary"
                >
                  {t.settings.subpages[item.labelKey]}
                </Button>
              ))}
          </div>
        )}
        {configSection && (
          <ConfigSettings
            activeSectionId={configSection}
            agentLayout
            agentModel={section === 'model'}
            importInputRef={importInputRef}
            onConfigSaved={onConfigSaved}
            onMainModelChanged={onMainModelChanged}
            subpage={section === 'limits' ? runtimePage : section === 'model' ? page : undefined}
          />
        )}
        {section && ['tools', 'tools-advanced', 'skills', 'connectors', 'plugins'].includes(section) && (
          <AgentCapabilitiesSettings
            onOpenAdvanced={() => open(section === 'tools-advanced' ? 'tools' : 'tools-advanced', true)}
            owner={owner}
            params={params}
            section={section}
          >
            {section === 'tools-advanced' && (
              <>
                <KeysSettings view="tools" />
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
            <BrowserRealProfilePanel profile={owner} />
            <ConfigSettings
              activeSectionId="browser"
              importInputRef={importInputRef}
              onConfigSaved={onConfigSaved}
              subpage="network"
            />
            <h3>{t.settings.nav.vault}</h3>
            <VaultSettings subpage={tab === 'vault' ? page : undefined} />
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
}

function AgentSettingsHeading({ labels, section }: AgentSettingsHeadingProps) {
  const { t } = useI18n()
  const c = useAgentSettingsEditorCopy()

  return (
    <>
      <header>
        <h2>{section ? labels[section] : t.commandCenter.settings}</h2>
        {section && <p>{c.descriptions[section === 'tools-advanced' ? 'tools' : section]}</p>}
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
        <div className="agent-settings-advanced">
          <button
            aria-expanded={section === 'tools-advanced'}
            className="agent-settings-advanced-toggle"
            onClick={onOpenAdvanced}
            type="button"
          >
            <span>{t.settings.sections.advanced}</span>
            <Codicon name={section === 'tools-advanced' ? 'chevron-up' : 'chevron-down'} />
          </button>
          {section === 'tools-advanced' && <div>{children}</div>}
        </div>
      )}
      {section === 'plugins' && (params.get('plugin') || params.get('agent')) && <AgentPluginSettings />}
    </>
  )
}

interface AgentProviderSettingsProps extends SettingsPageProps {
  params: URLSearchParams
  onViewChange: (view: ProviderView) => void
}

function AgentProviderSettings({
  params,
  onClose,
  onConfigSaved,
  onMainModelChanged,
  onViewChange
}: AgentProviderSettingsProps) {
  const { t } = useI18n()
  const c = useAgentSettingsEditorCopy()
  const providerView = params.get('pview') ?? 'keys'
  const validProvider = providerView !== 'local' && PROVIDER_VIEWS.includes(providerView as ProviderView)

  return (
    <>
      <div className="agent-settings-subpages">
        {(['keys', 'accounts', 'custom-endpoints'] as const).map(view => (
          <Button
            aria-pressed={providerView === view}
            key={view}
            onClick={() => onViewChange(view)}
            variant="secondary"
          >
            {view === 'keys'
              ? t.settings.nav.providerApiKeys
              : view === 'accounts'
                ? t.settings.nav.providerAccounts
                : t.settings.nav.providerCustomEndpoints}
          </Button>
        ))}
      </div>
      {validProvider ? (
        <ProvidersSettings
          onClose={onClose}
          onConfigSaved={onConfigSaved}
          onMainModelChanged={onMainModelChanged}
          onViewChange={onViewChange}
          view={providerView as ProviderView}
        />
      ) : (
        <p role="alert">{c.invalid}</p>
      )}
      {providerView === 'local' && (
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
  onOpen: (section: AgentSettingsSection) => void
}

function AgentSettingsNavigation({ ariaLabel, labels, section, onOpen }: AgentSettingsNavigationProps) {
  return (
    <nav aria-label={ariaLabel} className="agent-settings-navigation">
      {AGENT_SETTINGS_GROUPS.map(([group, sections]) => (
        <div className="agent-settings-nav-group" key={group}>
          <h2>{labels[group]}</h2>
          {sections.map(id => (
            <button
              aria-current={section === id || (id === 'tools' && section === 'tools-advanced') ? 'page' : undefined}
              key={id}
              onClick={() => onOpen(id)}
              type="button"
            >
              {labels[id]}
            </button>
          ))}
        </div>
      ))}
    </nav>
  )
}
