import { describe, expect, it } from 'vitest'

import { agentSettingsSearch, resolveAgentSettingsSection } from './settings-routes'

describe('agent settings routes', () => {
  it('keeps the owner when moving between config, capability and runtime controls', () => {
    const original = '?agentProfile=research&connection=remote-a&panel=capabilities&tab=toolsets&field=old'
    const query = new URLSearchParams(agentSettingsSearch(original, 'model'))
    expect(query.get('agentProfile')).toBe('research')
    expect(query.get('connection')).toBe('remote-a')
    expect(query.has('field')).toBe(false)
    expect(resolveAgentSettingsSection(query)).toBe('model')
    const runtime = new URLSearchParams(agentSettingsSearch(query.toString(), 'limits'))
    expect(runtime.get('agentProfile')).toBe('research')
    expect(runtime.get('connection')).toBe('remote-a')
    expect(resolveAgentSettingsSection(runtime)).toBe('limits')
  })

  it('resolves existing deep links without admitting global or nonexistent config pages', () => {
    expect(resolveAgentSettingsSection(new URLSearchParams('tab=config:model&page=fallbacks'))).toBe('model')
    expect(resolveAgentSettingsSection(new URLSearchParams('panel=capabilities&tab=toolsets'))).toBe('tools')
    expect(resolveAgentSettingsSection(new URLSearchParams('panel=messaging'))).toBe('messaging')

    for (const route of [
      'tab=config:appearance',
      'tab=billing',
      'tab=config:model&page=unknown',
      'tab=agent:limits&page=desktop'
    ]) {
      expect(resolveAgentSettingsSection(new URLSearchParams(route))).toBeNull()
    }
  })
})
