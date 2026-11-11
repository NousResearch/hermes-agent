import './settings-editor.css'

import { useMemo, useRef } from 'react'
import { useLocation, useNavigate } from 'react-router'

import { Button } from '@/components/ui/button'
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

/** Mounted only by the redesign's owner gate. All embedded controls share that immutable owner. */
export function AgentSettingsEditor(props: SettingsPageProps & { owner: AgentSettingsOwner }) {
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
  onMainModelChanged
}: SettingsPageProps & { owner: AgentSettingsOwner }) {
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

  const labels: Record<AgentSettingsSection | 'agent' | 'behavior' | 'capabilities' | 'access' | 'runtime', string> = {
    agent: t.simplifiedAgents.agents,
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
    'tools-advanced': `${a.tools} · ${t.settings.sections.advanced}`
  }

  function open(next: AgentSettingsSection) {
    navigate({ pathname: location.pathname, search: agentSettingsSearch(location.search, next) }, { replace: true })
  }

  function openPage(key: string, value: string) {
    const next = new URLSearchParams(location.search)
    next.set(key, value)
    navigate({ pathname: location.pathname, search: `?${next}` }, { replace: true })
  }

  const configSection = resolveConfigSection(section)

  const runtimePage = page ?? 'runtime'

  return (
    <div className="agent-settings-editor">
      <nav aria-label={t.commandCenter.settings} className="agent-settings-navigation">
        {AGENT_SETTINGS_GROUPS.map(([group, sections]) => (
          <div className="agent-settings-nav-group" key={group}>
            <h2>{labels[group]}</h2>
            {sections.map(id => (
              <button
                aria-current={section === id || (id === 'tools' && section === 'tools-advanced') ? 'page' : undefined}
                key={id}
                onClick={() => open(id)}
                type="button"
              >
                {labels[id]}
              </button>
            ))}
          </div>
        ))}
      </nav>
      <section className="agent-settings-body">
        <header>
          <h2>{section ? labels[section] : t.commandCenter.settings}</h2>
          {section === 'model' && <p>{c.modelDescription}</p>}
        </header>
        {!section && <p role="alert">{c.invalid}</p>}
        {section === 'general' && <AgentGeneralSettings owner={owner} />}
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
            agentModel={section === 'model'}
            importInputRef={importInputRef}
            onConfigSaved={onConfigSaved}
            onMainModelChanged={onMainModelChanged}
            subpage={section === 'limits' ? runtimePage : section === 'model' ? page : undefined}
          />
        )}
        {section && ['tools', 'skills', 'connectors', 'plugins'].includes(section) && (
          <AgentCapabilitiesSettings
            onOpenAdvanced={() => open('tools-advanced')}
            owner={owner}
            params={params}
            section={section}
          />
        )}
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
}

function AgentCapabilitiesSettings({ owner, section, params, onOpenAdvanced }: AgentCapabilitiesSettingsProps) {
  const { t } = useI18n()

  return (
    <>
      <CapabilitiesView
        className="agent-settings-capabilities"
        embedded
        fixedConnection={owner.connectionId}
        fixedMode={section === 'tools' ? 'toolsets' : (section as 'skills' | 'connectors' | 'plugins')}
        fixedProfile={owner.profile}
      />
      {section === 'tools' && (
        <Button onClick={onOpenAdvanced} variant="secondary">
          {t.settings.sections.advanced}
        </Button>
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
