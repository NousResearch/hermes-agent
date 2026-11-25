import { expect, it } from 'vitest'

import { resolveAgentSettingsSection } from '@/app/agent-settings/settings-routes'

import { toolsRoute } from './tools-route'

it('sends the redesign to the agent\'s own Tools page, not to a tab Capabilities no longer has', () => {
  const [path, search] = toolsRoute(true, 'ops').split('?')

  expect(path).toBe('/settings')
  expect(new URLSearchParams(search).get('agentProfile')).toBe('ops')
  expect(resolveAgentSettingsSection(new URLSearchParams(search))).toBe('tools')
})

it('keeps classic on its Capabilities tab', () => {
  expect(toolsRoute(false, 'ops')).toBe('/capabilities?tab=toolsets')
})
