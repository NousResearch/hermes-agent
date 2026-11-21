import { CONFIG_SUBPAGES } from '../settings/config-subpages'

export const AGENT_SETTINGS_GROUPS = [
  ['agent', ['general', 'instructions']],
  ['behavior', ['model', 'chat', 'memory', 'voice']],
  ['capabilities', ['tools', 'skills', 'connectors', 'plugins', 'messaging', 'schedules']],
  ['access', ['safety', 'browser', 'providers']],
  ['runtime', ['workspace', 'limits']]
] as const

export type AgentSettingsSection = (typeof AGENT_SETTINGS_GROUPS)[number][1][number] | 'tools-advanced'

export interface AgentSettingsPage {
  /** Links address it as `tab=config:<section>`, the config page it shares with app Settings. */
  configLink?: true
  /** The ConfigSettings section its page renders. */
  config?: string
  /** The Capabilities tab its page pins. */
  capability?: 'connectors' | 'plugins' | 'skills' | 'toolsets'
}

/** What each section's page is: every route, link and editor branch below reads this one table. */
export const AGENT_SETTINGS_PAGES: Record<AgentSettingsSection, AgentSettingsPage> = {
  general: {},
  instructions: {},
  model: { configLink: true, config: 'model' },
  chat: { configLink: true, config: 'chat' },
  memory: { configLink: true, config: 'memory' },
  voice: { configLink: true, config: 'voice' },
  tools: { capability: 'toolsets' },
  skills: { capability: 'skills' },
  connectors: { capability: 'connectors' },
  plugins: { capability: 'plugins' },
  messaging: {},
  schedules: {},
  safety: { configLink: true, config: 'safety' },
  // Browser shares config:browser's links but draws its own page (the vault, then the config).
  browser: { configLink: true },
  providers: {},
  workspace: { configLink: true, config: 'workspace' },
  limits: { config: 'advanced' },
  'tools-advanced': { capability: 'toolsets' }
}

const SECTIONS = Object.keys(AGENT_SETTINGS_PAGES) as AgentSettingsSection[]

const isSection = (id: string): id is AgentSettingsSection => Object.hasOwn(AGENT_SETTINGS_PAGES, id)

/** Legacy config links retain their target, rather than coercing unrelated/global pages to Model. */
export function resolveAgentSettingsSection(params: URLSearchParams): AgentSettingsSection | null {
  const tab = params.get('tab')
  const page = params.get('page')

  if (params.get('panel') === 'messaging') {
    return 'messaging'
  }

  if (!tab) {
    // The Overview's Capabilities heading opens the capabilities group, whose first page is Tools.
    return params.get('panel') === 'capabilities' ? 'tools' : 'general'
  }

  if (tab.startsWith('agent:')) {
    const section = tab.slice(6)

    if (section === 'limits' && page && !['runtime', 'terminal', 'output'].includes(page)) {
      return null
    }

    return isSection(section) ? section : null
  }

  if (tab.startsWith('config:')) {
    return resolveConfigSettingsSection(tab.slice(7), page)
  }

  if (tab === 'vault') {
    return ['credentials', 'sources'].includes(page ?? 'credentials') ? 'browser' : null
  }

  if (tab === 'keys') {
    return params.get('kview') === 'settings' ? null : 'tools-advanced'
  }

  if (tab === 'providers') {
    return 'providers'
  }

  // The Capabilities page's own tabs; its old MCP tab is Connectors now.
  const capability = tab === 'mcp' ? 'connectors' : tab

  return SECTIONS.find(section => AGENT_SETTINGS_PAGES[section].capability === capability) ?? null
}

function resolveConfigSettingsSection(section: string, page: string | null): AgentSettingsSection | null {
  if (page && !CONFIG_SUBPAGES[section]?.some(candidate => candidate.id === page)) {
    return null
  }

  if (section === 'advanced') {
    return page === 'tools' ? 'tools-advanced' : page === 'desktop' ? null : 'limits'
  }

  return isSection(section) && AGENT_SETTINGS_PAGES[section].configLink ? section : null
}

export function agentSettingsSearch(search: string, section: AgentSettingsSection): string {
  const params = new URLSearchParams(search)

  for (const key of ['page', 'field', 'setting', 'key', 'pview', 'kview', 'plugin', 'agent', 'toolset']) {
    params.delete(key)
  }

  params.set('panel', 'settings')
  params.set('tab', AGENT_SETTINGS_PAGES[section].configLink ? `config:${section}` : `agent:${section}`)

  return `?${params}`
}
