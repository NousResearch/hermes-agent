import { CONFIG_SUBPAGES } from '../settings/config-subpages'

export const AGENT_SETTINGS_GROUPS = [
  ['agent', ['general', 'instructions']],
  ['behavior', ['model', 'chat', 'memory', 'voice']],
  ['capabilities', ['tools', 'skills', 'connectors', 'plugins', 'messaging', 'schedules']],
  ['access', ['safety', 'browser', 'providers']],
  ['runtime', ['workspace', 'limits']]
] as const

export type AgentSettingsSection = (typeof AGENT_SETTINGS_GROUPS)[number][1][number] | 'tools-advanced'

/** Legacy config links retain their target, rather than coercing unrelated/global pages to Model. */
export function resolveAgentSettingsSection(params: URLSearchParams): AgentSettingsSection | null {
  const tab = params.get('tab')
  const page = params.get('page')

  if (params.get('panel') === 'messaging') {
    return 'messaging'
  }

  if (!tab) {
    return 'general'
  }

  if (tab.startsWith('agent:')) {
    const section = tab.slice(6)

    if (section === 'limits' && page && !['runtime', 'terminal', 'output'].includes(page)) {
      return null
    }

    return section === 'tools-advanced' || AGENT_SETTINGS_GROUPS.some(([, pages]) => pages.some(id => id === section))
      ? (section as AgentSettingsSection)
      : null
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

  if (['toolsets', 'skills', 'connectors', 'mcp', 'plugins'].includes(tab)) {
    return tab === 'toolsets' ? 'tools' : tab === 'mcp' ? 'connectors' : (tab as AgentSettingsSection)
  }

  return null
}

function resolveConfigSettingsSection(section: string, page: string | null): AgentSettingsSection | null {
  if (page && !CONFIG_SUBPAGES[section]?.some(candidate => candidate.id === page)) {
    return null
  }

  if (section === 'advanced') {
    return page === 'tools' ? 'tools-advanced' : page === 'desktop' ? null : 'limits'
  }

  return ['model', 'chat', 'memory', 'voice', 'safety', 'browser', 'workspace'].includes(section)
    ? (section as AgentSettingsSection)
    : null
}

export function agentSettingsSearch(search: string, section: AgentSettingsSection): string {
  const params = new URLSearchParams(search)

  for (const key of ['page', 'field', 'setting', 'key', 'pview', 'kview', 'plugin', 'agent', 'toolset']) {
    params.delete(key)
  }

  params.set('panel', 'settings')
  params.set(
    'tab',
    ['model', 'chat', 'memory', 'voice', 'safety', 'browser', 'workspace'].includes(section)
      ? `config:${section}`
      : `agent:${section}`
  )

  return `?${params}`
}
