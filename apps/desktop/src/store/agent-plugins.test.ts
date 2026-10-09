import { describe, expect, it } from 'vitest'

import { type AgentPluginRow, isDesktopRelevantPlugin } from './agent-plugins'

const row = (partial: Partial<AgentPluginRow>): AgentPluginRow =>
  ({ name: partial.key ?? 'x', status: 'enabled', ...partial }) as AgentPluginRow

describe('isDesktopRelevantPlugin (#98861)', () => {
  it('lists the manageable bundled lifecycle plugins and keeps every other built-in hidden', () => {
    expect(isDesktopRelevantPlugin(row({ key: 'disk-cleanup', source: 'bundled' }))).toBe(true)
    expect(isDesktopRelevantPlugin(row({ key: 'security-guidance', source: 'bundled' }))).toBe(true)

    for (const key of [
      'platforms/discord',
      'model-providers/openai',
      'web/firecrawl',
      'browser/agent-browser',
      'kanban'
    ]) {
      expect(isDesktopRelevantPlugin(row({ key, source: 'bundled' }))).toBe(false)
    }

    // User installs are unaffected either way.
    expect(isDesktopRelevantPlugin(row({ key: 'my-plugin', source: 'user' }))).toBe(true)
  })
})
